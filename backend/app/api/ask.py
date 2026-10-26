import json
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.deps import get_current_db
from backend.app.db.models import Workspace, DocumentChunk, Document
from backend.app.db.session import SessionLocal
from backend.app.core.errors import NotFoundError, LLMProviderError
from backend.app.core.security import sanitize_workspace_id
from backend.app.core.logging import logger
from backend.app.config import settings
from backend.app.retrieval.embedder import embed_texts
from backend.app.retrieval.vectorstore import vector_store
from backend.app.retrieval.bm25 import search_bm25
from backend.app.generation.llm_provider import get_llm_provider

router = APIRouter(tags=["Ask & RAG"])

# ---------------------------------------------------------------------------
# Reranker (lazy-loaded, unloaded after idle timeout)
# ---------------------------------------------------------------------------
_reranker_cache: Dict[str, Any] = {}

def _get_reranker():
    if settings.LOW_MEMORY_MODE:
        return None
    model_name = settings.RERANKER_MODEL
    if model_name in _reranker_cache:
        return _reranker_cache[model_name]
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker '%s'...", model_name)
        try:
            model = CrossEncoder(model_name, device="cpu", local_files_only=True)
        except Exception:
            model = CrossEncoder(model_name, device="cpu", local_files_only=False)
        _reranker_cache[model_name] = model
        logger.info("Reranker loaded.")
        return model
    except Exception as e:
        logger.warning("Could not load reranker: %s. Skipping reranking.", e)
        _reranker_cache[model_name] = None
        return None

def warmup_reranker():
    """Pre-warm reranker in memory so queries don't hit a cold-start penalty."""
    try:
        r = _get_reranker()
        if r is not None:
            r.predict([("warmup", "warmup passage")], batch_size=1)
            logger.info("Reranker pre-warmed successfully.")
    except Exception as e:
        logger.warning("Reranker warmup error: %s", e)


def _reciprocal_rank_fusion(
    dense_results: List[Dict], bm25_results: List[Dict], k: int = 60
) -> List[Dict]:
    """Combine dense and sparse result lists using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for rank, r in enumerate(dense_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = r

    for rank, r in enumerate(bm25_results):
        cid = r["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in chunk_map:
            chunk_map[cid] = r

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"] = round(scores[cid], 6)
        merged.append(chunk)
    return merged


def _build_rag_prompt(question: str, chunks: List[Dict], answer_style: str) -> tuple[str, str]:
    """Build system + user prompt from retrieved chunks."""

    style_instructions = {
        "concise": (
            "Give a direct, concise answer in 3–5 sentences covering the core idea. "
            "Add at most 1–2 citation numbers only for the most important claim."
        ),
        "detailed": (
            "Write a thorough, well-structured academic answer. "
            "Start with a clear definition or overview paragraph. "
            "Then elaborate with key concepts, explanations, and examples found in the material — use paragraphs or a short numbered list where it improves clarity. "
            "Aim for 150–300 words. Cite sources sparingly — only use [1], [2] etc. at the end of a paragraph or for a critical fact, not after every sentence."
        ),
        "exam-5-mark": (
            "Answer in the format of a university exam 5-mark answer:\n"
            "1. Definition (1 mark)\n2. Key Points / Explanation (3 marks)\n3. Conclusion / Summary (1 mark).\n"
            "Write at least 3–4 sentences per section. Cite the source once per section at most."
        ),
        "explain-like-beginner": (
            "Explain in very simple terms using everyday analogies. Avoid jargon. "
            "Write at least 3 paragraphs so the reader truly understands. "
            "Make it easy for someone with absolutely no background in the subject."
        ),
    }.get(answer_style, "Answer the question clearly and accurately based on the provided context.")

    system_prompt = (
        "You are ScholarRAG, a knowledgeable academic assistant. "
        "You ONLY answer from the provided course material context — never from outside knowledge. "
        "If the answer is not in the context, say: 'This information is not present in your uploaded materials.' "
        "Use citations like [1], [2] sparingly — only once per key claim or at the end of a paragraph, "
        "NOT after every sentence. Avoid repeating the same citation multiple times in a row."
    )

    context_blocks = []
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        doc_name = meta.get("document_name", "Unknown")
        page_start = meta.get("page_start", "?")
        page_end = meta.get("page_end", "?")
        heading = meta.get("heading_path", "")
        unit = meta.get("unit", "")

        header = f"[{i}] {doc_name} | Page {page_start}–{page_end}"
        if unit and unit != "General":
            header += f" | {unit}"
        if heading:
            header += f" | {heading}"

        context_blocks.append(f"{header}\n{c['content']}")

    context_text = "\n\n---\n\n".join(context_blocks)
    user_prompt = (
        f"CONTEXT FROM COURSE MATERIAL:\n\n{context_text}\n\n"
        f"---\n\nQUESTION: {question}\n\n"
        f"ANSWER STYLE: {style_instructions}\n\n"
        "Answer (cite sources as [1], [2], etc.):"
    )

    return system_prompt, user_prompt


@router.post("/workspaces/{workspace_id}/ask")
async def ask_workspace(
    workspace_id: str,
    body: dict,
    db: Session = Depends(get_current_db),
):
    """
    SSE-streaming RAG endpoint.

    Body fields:
      - question (str): The user's question
      - answer_style (str): concise | detailed | exam-5-mark | explain-like-beginner
      - unit_filter (str|null): optional unit to restrict search
      - top_k_final (int): number of chunks to pass to LLM (default 5)
    """
    clean_ws_id = sanitize_workspace_id(workspace_id)
    ws = db.query(Workspace).filter(Workspace.id == clean_ws_id).first()
    if not ws:
        raise NotFoundError(f"Workspace '{workspace_id}' not found.")

    question: str = body.get("question", "").strip()
    answer_style: str = body.get("answer_style", "detailed")
    unit_filter: Optional[str] = body.get("unit_filter") or None
    top_k_final: int = int(body.get("top_k_final", settings.TOP_K_RERANK))

    if not question:
        raise ValueError("Question cannot be empty.")

    # ── 1. Embed query ──────────────────────────────────────────────────────
    emb_model = (
        settings.EMBEDDING_MODEL_MULTILINGUAL
        if ws.mode == "multilingual"
        else settings.EMBEDDING_MODEL_DEFAULT
    )
    query_embeddings = await asyncio.to_thread(embed_texts, [question], emb_model)
    query_vec = query_embeddings[0]

    # ── 2. Dense retrieval ──────────────────────────────────────────────────
    chroma_filter = None
    if unit_filter:
        chroma_filter = {"unit": unit_filter}

    dense_results = await asyncio.to_thread(
        vector_store.query, clean_ws_id, query_vec, settings.TOP_K_DENSE, chroma_filter
    )

    # ── 3. BM25 keyword retrieval ───────────────────────────────────────────
    bm25_results = search_bm25(clean_ws_id, question, settings.TOP_K_BM25)
    if unit_filter:
        bm25_results = [
            r for r in bm25_results
            if r.get("metadata", {}).get("unit", "") == unit_filter
        ]

    # ── 4. Reciprocal Rank Fusion ───────────────────────────────────────────
    fused = _reciprocal_rank_fusion(dense_results, bm25_results)

    # ── 5. Cross-encoder reranking ──────────────────────────────────────────
    if not settings.LOW_MEMORY_MODE and len(fused) > 1:
        reranker = await asyncio.to_thread(_get_reranker)
        if reranker is not None:
            try:
                # Score top 10 fused candidates to keep CPU latency under 0.8s
                candidates = fused[:10]
                remaining = fused[10:]
                pairs = [(question, c["content"]) for c in candidates]
                cross_scores = await asyncio.to_thread(reranker.predict, pairs, batch_size=16)
                for c, sc in zip(candidates, cross_scores):
                    c["cross_score"] = float(sc)
                candidates.sort(key=lambda x: x.get("cross_score", 0.0), reverse=True)
                fused = candidates + remaining
            except Exception as e:
                logger.warning("Reranker inference failed: %s", e)

    # ── 6. Relevance threshold filter ──────────────────────────────────────
    top_chunks = fused[:top_k_final]
    if not top_chunks:
        async def _no_results():
            yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': 'No relevant content found in your uploaded materials for this question. Please ensure relevant documents are indexed.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(_no_results(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── 7. Build citation metadata for frontend ─────────────────────────────
    citations = []
    for i, c in enumerate(top_chunks, 1):
        meta = c.get("metadata", {})
        citations.append({
            "num": i,
            "chunk_id": c["id"],
            "document_id": meta.get("document_id", ""),
            "document_name": meta.get("document_name", ""),
            "unit": meta.get("unit", "General"),
            "heading_path": meta.get("heading_path", ""),
            "page_start": meta.get("page_start", 1),
            "page_end": meta.get("page_end", 1),
            "chunk_type": meta.get("chunk_type", "text"),
            "content_preview": c["content"][:280],
            "rrf_score": round(c.get("rrf_score", 0.0), 4),
            "cross_score": round(c.get("cross_score", 0.0), 4) if "cross_score" in c else None,
        })

    # ── 8. Build prompt ─────────────────────────────────────────────────────
    system_prompt, user_prompt = _build_rag_prompt(question, top_chunks, answer_style)

    # ── 9. Stream LLM response ──────────────────────────────────────────────
    llm = get_llm_provider()

    async def event_stream():
        # First: emit citations metadata
        yield f"data: {json.dumps({'type': 'citations', 'citations': citations})}\n\n"
        await asyncio.sleep(0)

        # Then: stream LLM tokens
        try:
            async for token in llm.generate_stream(user_prompt, system_prompt=system_prompt):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        except LLMProviderError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.exception("LLM streaming error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': 'LLM generation failed: ' + str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

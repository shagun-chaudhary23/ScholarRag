import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Sparkles,
  BookOpen,
  FileText,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Loader2,
  RotateCcw,
  MessageSquare,
  User,
  Copy,
  Check,
  ExternalLink,
} from "lucide-react";
import { API_BASE_URL, ANSWER_STYLES } from "../../lib/constants";

// ─── Types ───────────────────────────────────────────────────────────────────

interface Citation {
  num: number;
  chunk_id: string;
  document_id: string;
  document_name: string;
  unit: string;
  heading_path: string;
  page_start: number;
  page_end: number;
  chunk_type: string;
  content_preview: string;
  rrf_score: number;
  cross_score: number | null;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
  error?: string;
  timestamp: Date;
}

interface Props {
  workspaceId: string;
  workspaceName: string;
  documents: { id: string; original_filename: string; unit?: string }[];
}

// ─── Citation Card ────────────────────────────────────────────────────────────

const CitationCard: React.FC<{
  citation: Citation;
  workspaceId: string;
  isExpanded: boolean;
  onToggle: () => void;
}> = ({ citation, workspaceId, isExpanded, onToggle }) => {
  const pageUrl = `${API_BASE_URL}/workspaces/${workspaceId}/documents/${citation.document_id}/pages/${citation.page_start}`;

  return (
    <div className="rounded-xl border border-border bg-bg-surface-2 overflow-hidden transition-all duration-200">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-bg-surface-3 transition-colors"
      >
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-brand-violet/20 text-brand-violet text-xs font-bold flex items-center justify-center border border-brand-violet/30">
          {citation.num}
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-fg truncate">{citation.document_name}</p>
          <p className="text-[10px] text-fg-subtle truncate">
            {citation.unit !== "General" && (
              <span className="text-brand-amber font-medium">{citation.unit} · </span>
            )}
            {citation.heading_path && `${citation.heading_path} · `}
            Page {citation.page_start}
            {citation.page_end !== citation.page_start && `–${citation.page_end}`}
          </p>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="hidden sm:inline text-[10px] font-mono px-1.5 py-0.5 rounded bg-bg text-fg-subtle border border-border capitalize">
            {citation.chunk_type}
          </span>
          {isExpanded ? (
            <ChevronUp className="w-3.5 h-3.5 text-fg-subtle" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-fg-subtle" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-3 border-t border-border/50 space-y-2">
          <p className="text-xs text-fg-muted leading-relaxed pt-2 line-clamp-6">
            {citation.content_preview}
            {citation.content_preview.length >= 280 && (
              <span className="text-fg-subtle italic">… (truncated)</span>
            )}
          </p>
          <a
            href={pageUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-brand-violet hover:underline"
          >
            <ExternalLink className="w-3 h-3" />
            View Page {citation.page_start} in PDF
          </a>
        </div>
      )}
    </div>
  );
};

// ─── Message Bubble ───────────────────────────────────────────────────────────

const MessageBubble: React.FC<{
  msg: ChatMessage;
  workspaceId: string;
}> = ({ msg, workspaceId }) => {
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());
  const [copied, setCopied] = useState(false);

  const toggleCitation = (num: number) => {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      next.has(num) ? next.delete(num) : next.add(num);
      return next;
    });
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (msg.role === "user") {
    return (
      <div className="flex justify-end gap-3">
        <div className="max-w-[78%] px-4 py-3 rounded-2xl rounded-tr-sm bg-gradient-to-br from-brand-violet to-brand-sky text-white text-sm leading-relaxed shadow-glow-violet">
          {msg.content}
        </div>
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-brand-violet/20 border border-brand-violet/30 flex items-center justify-center">
          <User className="w-4 h-4 text-brand-violet" />
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-brand-violet to-brand-sky flex items-center justify-center shadow-glow-violet">
        <Sparkles className="w-4 h-4 text-white" />
      </div>

      <div className="flex-1 min-w-0 space-y-3">
        {/* Answer text */}
        <div className="relative group">
          <div
            className={`px-4 py-3.5 rounded-2xl rounded-tl-sm border border-border bg-bg-surface shadow-dark-card text-sm text-fg leading-relaxed whitespace-pre-wrap ${
              msg.isStreaming ? "animate-pulse-glow" : ""
            }`}
          >
            {msg.error ? (
              <div className="flex items-start gap-2 text-brand-coral">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span>{msg.error}</span>
              </div>
            ) : (
              <>
                {msg.content || (msg.isStreaming && <span className="text-fg-subtle italic">Thinking…</span>)}
                {msg.isStreaming && (
                  <span className="inline-block w-1.5 h-4 bg-brand-violet ml-0.5 animate-pulse rounded-sm align-middle" />
                )}
              </>
            )}
          </div>

          {/* Copy button */}
          {!msg.isStreaming && msg.content && !msg.error && (
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 bg-bg-surface-2 border border-border text-fg-subtle hover:text-fg transition-all"
              title="Copy answer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-brand-mint" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          )}
        </div>

        {/* Citations */}
        {msg.citations && msg.citations.length > 0 && !msg.isStreaming && (
          <div className="space-y-1.5">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-fg-subtle flex items-center gap-1.5 px-1">
              <BookOpen className="w-3 h-3" />
              Source Citations ({msg.citations.length})
            </p>
            {msg.citations.map((c) => (
              <CitationCard
                key={c.chunk_id}
                citation={c}
                workspaceId={workspaceId}
                isExpanded={expandedCitations.has(c.num)}
                onToggle={() => toggleCitation(c.num)}
              />
            ))}
          </div>
        )}

        <p className="text-[10px] text-fg-subtle px-1">
          {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </p>
      </div>
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

export const AskNotesView: React.FC<Props> = ({ workspaceId, workspaceName, documents }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [answerStyle, setAnswerStyle] = useState("detailed");
  const [unitFilter, setUnitFilter] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Unique units from indexed documents
  const units = Array.from(
    new Set(documents.map((d) => d.unit).filter(Boolean))
  ).sort();

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    const question = input.trim();
    if (!question || isLoading) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
      timestamp: new Date(),
    };

    const assistantMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      citations: [],
      isStreaming: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE_URL}/workspaces/${workspaceId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          question,
          answer_style: answerStyle,
          unit_filter: unitFilter || null,
          top_k_final: 3,
        }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`Server error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let citations: Citation[] = [];
      let answerText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === "citations") {
              citations = data.citations;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id ? { ...m, citations } : m
                )
              );
            } else if (data.type === "token") {
              answerText += data.token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id ? { ...m, content: answerText } : m
                )
              );
            } else if (data.type === "error") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, error: data.message, isStreaming: false }
                    : m
                )
              );
            } else if (data.type === "done") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id ? { ...m, isStreaming: false } : m
                )
              );
            }
          } catch {
            // skip malformed SSE line
          }
        }
      }

      // Ensure streaming state is cleared
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsg.id ? { ...m, isStreaming: false } : m
        )
      );
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? {
                  ...m,
                  isStreaming: false,
                  error: err.message || "Failed to get response. Is the LLM configured?",
                }
              : m
          )
        );
      }
    } finally {
      setIsLoading(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleClear = () => {
    if (isLoading) abortRef.current?.abort();
    setMessages([]);
    setIsLoading(false);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)] min-h-[500px] rounded-2xl border border-border bg-bg-surface shadow-dark-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-bg-surface-2/80">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-brand-violet" />
          <span className="text-sm font-semibold text-fg">Ask Notes</span>
          <span className="text-xs text-fg-subtle">· {workspaceName}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Unit filter */}
          {units.length > 0 && (
            <select
              value={unitFilter}
              onChange={(e) => setUnitFilter(e.target.value)}
              className="text-xs px-2 py-1 rounded-lg bg-bg border border-border text-fg-muted focus:outline-none focus:border-brand-violet"
            >
              <option value="">All Units</option>
              {units.map((u) => (
                <option key={u} value={u}>
                  {u}
                </option>
              ))}
            </select>
          )}

          {/* Answer style */}
          <select
            value={answerStyle}
            onChange={(e) => setAnswerStyle(e.target.value)}
            className="text-xs px-2 py-1 rounded-lg bg-bg border border-border text-fg-muted focus:outline-none focus:border-brand-violet"
          >
            {ANSWER_STYLES.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>

          {messages.length > 0 && (
            <button
              onClick={handleClear}
              className="p-1.5 rounded-lg hover:bg-bg-surface-3 text-fg-subtle hover:text-fg transition-colors"
              title="Clear conversation"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-5 space-y-6 scroll-smooth">
        {isEmpty ? (
          /* Empty state */
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 px-4">
            <div className="w-16 h-16 rounded-2xl bg-brand-violet/10 flex items-center justify-center shadow-glow-violet">
              <Sparkles className="w-8 h-8 text-brand-violet" />
            </div>
            <div className="space-y-2 max-w-lg">
              <h3 className="font-serif-heading font-bold text-xl text-fg">Ask anything about your course material</h3>
              <p className="text-sm text-fg-muted leading-relaxed">
                ScholarRAG retrieves exact passages from your uploaded PDFs, answers strictly from your material, and shows page-level citations you can verify.
              </p>
            </div>

            {/* Suggested questions */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {[
                "Explain the key concepts covered in Unit 1",
                "What are the main formulas I need to remember?",
                "Summarize the most important topics for exams",
                "What is the definition of [topic from your notes]?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                  className="text-left px-3 py-2.5 rounded-xl border border-border bg-bg-surface-2 hover:bg-bg-surface-3 hover:border-brand-violet/40 text-xs text-fg-muted hover:text-fg transition-all"
                >
                  <FileText className="w-3 h-3 inline mr-1.5 text-brand-violet" />
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} workspaceId={workspaceId} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 py-3 border-t border-border bg-bg-surface-2/60">
        <form onSubmit={handleSubmit} className="flex items-end gap-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              // Auto-resize
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your course material… (Enter to send, Shift+Enter for new line)"
            rows={1}
            disabled={isLoading}
            className="flex-1 resize-none px-4 py-2.5 rounded-xl border border-border bg-bg text-sm text-fg placeholder:text-fg-subtle focus:outline-none focus:border-brand-violet transition-colors disabled:opacity-50 leading-relaxed"
            style={{ minHeight: "42px", maxHeight: "120px" }}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-brand-violet to-brand-sky text-white flex items-center justify-center shadow-glow-violet hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            title="Send (Enter)"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>

        <p className="text-[10px] text-fg-subtle text-center mt-2">
          ScholarRAG answers <strong>only</strong> from your uploaded PDFs — never from outside knowledge
        </p>
      </div>
    </div>
  );
};

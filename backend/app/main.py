import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.core.errors import ScholarRAGError, app_exception_handler
from backend.app.db.session import init_db

# Routers
from backend.app.api.health import router as health_router
from backend.app.api.settings import router as settings_router
from backend.app.api.workspaces import router as workspaces_router
from backend.app.api.documents import router as documents_router
from backend.app.api.ask import router as ask_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting %s v%s...", settings.APP_NAME, settings.APP_VERSION)
    init_db()
    
    # Pre-warm AI models in background thread so user queries are instant without cold-start delay
    async def _warmup_background():
        try:
            loop = asyncio.get_running_loop()
            from backend.app.retrieval.embedder import warmup_embedder
            from backend.app.api.ask import warmup_reranker
            await loop.run_in_executor(None, warmup_embedder)
            await loop.run_in_executor(None, warmup_reranker)
            logger.info("All AI models pre-warmed and ready for instant Q&A.")
        except Exception as e:
            logger.warning("Background warmup notice: %s", e)

    asyncio.create_task(_warmup_background())
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Academic Question Answering System using Retrieval-Augmented Generation (RAG) with page-level visual citations.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Exception Handler
app.add_exception_handler(ScholarRAGError, app_exception_handler)

@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
    return response

# Register API v1 Routers
api_v1_prefix = "/api/v1"
app.include_router(health_router, prefix=api_v1_prefix)
app.include_router(settings_router, prefix=api_v1_prefix)
app.include_router(workspaces_router, prefix=api_v1_prefix)
app.include_router(documents_router, prefix=api_v1_prefix)
app.include_router(ask_router, prefix=api_v1_prefix)

@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "tagline": "Ask your syllabus anything. Verify it in one click.",
        "docs": "/docs",
        "api_v1": api_v1_prefix
    }

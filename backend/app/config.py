from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    APP_NAME: str = "ScholarRAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Storage Paths
    DATA_DIR: Path = WORKSPACE_DATA_DIR
    SQLITE_DB_PATH: Path = WORKSPACE_DATA_DIR / "scholarrag.db"
    CHROMA_PERSIST_DIR: Path = WORKSPACE_DATA_DIR / "chroma"
    
    # Hardware & Performance
    LOW_MEMORY_MODE: bool = False  # If True, bypass cross-encoder reranker, smaller context
    MAX_CONCURRENT_INGESTION: int = 1
    
    # LLM Settings
    DEFAULT_LLM_PROVIDER: Literal["ollama", "openai", "gemini"] = "gemini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"
    
    # Embedding Models
    EMBEDDING_MODEL_DEFAULT: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_MULTILINGUAL: str = "intfloat/multilingual-e5-small"
    
    # Reranker Model
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_IDLE_TIMEOUT_SECONDS: int = 120  # Unload from RAM after 2 minutes of idle
    
    # Retrieval Hyperparameters
    TOP_K_DENSE: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 3
    RELEVANCE_SCORE_THRESHOLD: float = 0.35  # Threshold below which abstention triggers
    
    # Ingestion Constraints
    MAX_UPLOAD_SIZE_MB: int = 100
    MAX_PAGES_PER_DOC: int = 1000
    CHUNK_MIN_TOKENS: int = 400
    CHUNK_MAX_TOKENS: int = 800
    CHUNK_OVERLAP_PERCENT: float = 0.12
    
    # Security
    ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

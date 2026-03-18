from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    source_dir: Path = Path(os.getenv("RAG_SOURCE_DIR", "data/knowledge_base"))
    persist_dir: Path = Path(os.getenv("RAG_PERSIST_DIR", "vectorstore"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "4"))
    fetch_k: int = int(os.getenv("RAG_FETCH_K", "8"))
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = 0.0

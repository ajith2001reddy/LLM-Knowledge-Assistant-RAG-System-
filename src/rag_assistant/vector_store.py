from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag_assistant.config import Settings


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)



def build_vector_store(chunks: list[Document], settings: Settings) -> Path:
    embeddings = get_embeddings(settings)
    vector_store = FAISS.from_documents(chunks, embeddings)
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(settings.persist_dir))
    return settings.persist_dir



def load_vector_store(settings: Settings) -> FAISS:
    if not settings.persist_dir.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{settings.persist_dir}'. Run the ingestion step first."
        )

    return FAISS.load_local(
        str(settings.persist_dir),
        get_embeddings(settings),
        allow_dangerous_deserialization=True,
    )

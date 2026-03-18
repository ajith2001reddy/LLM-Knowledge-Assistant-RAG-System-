from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_assistant.config import Settings

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(source_dir: Path) -> list[Document]:
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported knowledge files were found in '{source_dir}'. "
            f"Add markdown or text files with one of: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    documents: list[Document] = []
    for path in files:
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={
                    "source": str(path.relative_to(source_dir.parent)),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                },
            )
        )
    return documents



def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)

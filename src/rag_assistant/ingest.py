from __future__ import annotations

import argparse

from rag_assistant.config import Settings
from rag_assistant.documents import load_documents, split_documents
from rag_assistant.vector_store import build_vector_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a FAISS vector index from local knowledge files.")
    parser.add_argument("--source-dir", help="Override the knowledge base directory.")
    parser.add_argument("--persist-dir", help="Override the vector store output directory.")
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings()
    if args.source_dir:
        settings.source_dir = settings.source_dir.__class__(args.source_dir)
    if args.persist_dir:
        settings.persist_dir = settings.persist_dir.__class__(args.persist_dir)

    documents = load_documents(settings.source_dir)
    chunks = split_documents(documents, settings)
    build_vector_store(chunks, settings)

    print(f"Loaded {len(documents)} documents from {settings.source_dir}.")
    print(f"Created {len(chunks)} chunks and saved the FAISS index to {settings.persist_dir}.")


if __name__ == "__main__":
    main()

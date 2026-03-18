from __future__ import annotations

import argparse

from rag_assistant.chain import ask
from rag_assistant.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions against the local RAG knowledge base.")
    parser.add_argument("--question", help="Single question to ask the assistant.")
    parser.add_argument("--persist-dir", help="Override the vector store directory.")
    return parser



def run_single_question(question: str, settings: Settings) -> None:
    result = ask(question, settings)
    print("\nAnswer:\n")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")



def run_interactive(settings: Settings) -> None:
    print("Interactive RAG chat. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue
        run_single_question(question, settings)



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings()
    if args.persist_dir:
        settings.persist_dir = settings.persist_dir.__class__(args.persist_dir)

    if args.question:
        run_single_question(args.question, settings)
        return

    run_interactive(settings)


if __name__ == "__main__":
    main()

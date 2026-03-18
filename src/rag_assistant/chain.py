from __future__ import annotations

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag_assistant.config import Settings
from rag_assistant.vector_store import load_vector_store

SYSTEM_PROMPT = """You are a helpful knowledge assistant.
Answer the user's question using only the retrieved context.
If the answer is not supported by the context, say you do not know.
Keep the response concise, factual, and easy to understand.

Context:
{context}
"""


def build_rag_chain(settings: Settings):
    vector_store = load_vector_store(settings)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_k, "fetch_k": settings.fetch_k},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)



def ask(question: str, settings: Settings) -> dict:
    chain = build_rag_chain(settings)
    result = chain.invoke({"input": question})
    sources = format_sources(result.get("context", []))
    return {
        "answer": result.get("answer", ""),
        "sources": sources,
    }



def format_sources(documents: list[Document]) -> list[str]:
    seen: set[str] = set()
    ordered_sources: list[str] = []

    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        if source not in seen:
            seen.add(source)
            ordered_sources.append(source)

    return ordered_sources

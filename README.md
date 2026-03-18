# LLM Knowledge Assistant (RAG System)

This repository now contains an actual Retrieval-Augmented Generation (RAG) demo project built with **LangChain + FAISS**. It ingests local knowledge files, generates vector embeddings, stores them in a FAISS index, and answers user questions with retrieved context so responses stay grounded in source data.

## Features

- LangChain-based RAG pipeline with modular ingestion and chat entry points.
- FAISS vector store for semantic retrieval over local markdown/text knowledge files.
- Prompt-engineered answer chain that instructs the model to stay within retrieved context.
- Demo knowledge base included under `data/knowledge_base/` so the project is easy to understand and extend.
- CLI workflow for both indexing content and asking questions.

## Project Structure

```text
.
├── .env.example
├── data/knowledge_base/
├── pyproject.toml
├── README.md
└── src/rag_assistant/
    ├── chain.py
    ├── chat.py
    ├── config.py
    ├── documents.py
    ├── ingest.py
    └── vector_store.py
```

## Setup

1. Create and activate a virtual environment.
2. Install the project:
   ```bash
   pip install -e .
   ```
3. Copy the environment template and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
4. Update `.env` if you want to change the chat model, embedding model, chunking, or persistence paths.

## Usage

### 1) Build the vector index

```bash
rag-ingest
```

You can also point at a custom dataset:

```bash
rag-ingest --source-dir path/to/knowledge --persist-dir vectorstore
```

### 2) Ask a single question

```bash
rag-chat --question "Which plan includes API access?"
```

### 3) Start interactive chat

```bash
rag-chat
```

## How It Works

1. The ingestion pipeline reads `.md` and `.txt` knowledge files.
2. Documents are split into overlapping chunks for better retrieval quality.
3. OpenAI embeddings are generated for each chunk.
4. FAISS stores the vectors locally for semantic similarity search.
5. At query time, the retriever selects the most relevant chunks.
6. The prompt instructs the chat model to answer only from retrieved context.

## Example Knowledge Use Cases

- Internal company knowledge assistants.
- Product and support copilots.
- Document-grounded Q&A demos.
- Semantic search prototypes for operations or customer success teams.

## Notes

- This demo expects `OPENAI_API_KEY` to be set before running ingestion or chat.
- The sample knowledge base is intentionally small so you can replace it with your own files quickly.
- The FAISS loader uses local deserialization for the saved index, so only load vector stores you trust.

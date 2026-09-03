# langchain-examples

Standalone LangChain sample scripts: document loading, text splitting, embeddings, vector stores, AWS Bedrock connectivity, and RAG.

## Requirements

- Python 3.14 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with your AWS credentials/region if you want to run the Bedrock samples (`check_connection.py`, `embeddings.py`, `embeddings_chroma.py`, `embedding_chroma_retrieval.py`, `embedding_chroma_persistence.py`, and eventually `rag_pipeline.py`). `documents.py` and `text_splitter.py` work without any configuration.

## Usage

```bash
uv run .\documents.py                       # document loading samples (text, web, directory, structured data, PDF)
uv run .\text_splitter.py                   # text splitting samples (character-based and recursive)
uv run .\embeddings.py                      # generate text embeddings via Bedrock, with a local cache
uv run .\embeddings_chroma.py               # store embeddings in Chroma and run similarity search
uv run .\embedding_chroma_retrieval.py      # use a Chroma vector store as a retriever (similarity and MMR)
uv run .\embedding_chroma_persistence.py    # show a Chroma collection persisting to disk across runs (run it twice)
uv run .\check_connection.py                # verify AWS Bedrock credentials/model access
```

## Files

- `documents.py` — loads content into `langchain_core.documents.Document` objects from text files, web pages, directories, in-memory structured data, and PDFs. Builds on the underlying libraries (`pypdf`, `requests`, `bs4`) directly rather than `langchain-community` loaders, which are being sunset.
- `text_splitter.py` — splits text/`Document`s into smaller chunks with `langchain-text-splitters` (`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`), for feeding into an LLM or embedding pipeline.
- `embeddings.py` — generates text embeddings via `langchain_aws.BedrockEmbeddings`, including a simple local JSON cache to avoid repeat calls for known text.
- `embeddings_chroma.py` — stores `Document`s and their embeddings in a `langchain_chroma.Chroma` vector store and runs similarity search against it.
- `embedding_chroma_retrieval.py` — wraps a Chroma vector store with `as_retriever()` and calls it via the standard `.invoke()` Runnable interface, comparing plain similarity search against MMR.
- `embedding_chroma_persistence.py` — builds a Chroma collection in a stable (non-temp) directory and, run a second time, reloads it from disk without re-embedding, to show the collection actually persists across runs.
- `config.py` — loads AWS credentials/region and Bedrock model ids from environment variables (via `.env`) into typed config objects.
- `check_connection.py` — sends a trivial prompt to Bedrock via `ChatBedrockConverse` to confirm credentials and model access work.
- `rag_pipeline.py` — not yet implemented (currently commented-out stubs for a basic RAG demo tying text splitting + embeddings + retrieval together).
- `utils/embeddings_client.py` — shared factory for the `BedrockEmbeddings` client, used by every sample above that needs to embed text.
- `utils/chroma_temp_dir.py` — shared context manager for samples that build a throwaway Chroma collection in a temp directory (used by `embeddings_chroma.py` and `embedding_chroma_retrieval.py`).

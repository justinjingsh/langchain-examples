# langchain-examples

Standalone LangChain sample scripts: document loading, AWS Bedrock connectivity, and RAG.

## Requirements

- Python 3.14 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env
```

Fill in `.env` with your AWS credentials/region if you want to run the Bedrock samples (`check_connection.py`, and eventually `rag_pipeline.py`). `documents.py` works without any configuration.

## Usage

```bash
uv run python documents.py         # document loading samples (text, web, directory, structured data, PDF)
uv run python check_connection.py  # verify AWS Bedrock credentials/model access
```

## Files

- `documents.py` — loads content into `langchain_core.documents.Document` objects from text files, web pages, directories, in-memory structured data, and PDFs. Builds on the underlying libraries (`pypdf`, `requests`, `bs4`) directly rather than `langchain-community` loaders, which are being sunset.
- `config.py` — loads AWS credentials/region and Bedrock model ids from environment variables (via `.env`) into typed config objects.
- `check_connection.py` — sends a trivial prompt to Bedrock via `ChatBedrockConverse` to confirm credentials and model access work.
- `rag_pipeline.py` — not yet implemented (currently commented-out stubs for a basic RAG demo).

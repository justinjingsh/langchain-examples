# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small collection of standalone LangChain sample scripts (document loading, AWS Bedrock connectivity, RAG). There is no app entry point, package, or test suite — each `.py` file at the repo root is a self-contained runnable demo with its own `main()`.

## Commands

Dependencies are managed with `uv` (see `pyproject.toml` / `uv.lock`).

```bash
uv sync                      # install/update dependencies
uv run python documents.py   # run the document-loading samples
uv run python check_connection.py  # verify AWS Bedrock credentials/model access
```

There is no build step, lint config, or test suite in this repo.

## Configuration

- Config is loaded from environment variables via `config.py`, backed by a local `.env` file (`load_dotenv()` in `config.py`). Copy `.env.example` to `.env` and fill in AWS credentials/region to run `check_connection.py` or anything hitting Bedrock.
- `config.load_config()` is the single entry point other scripts use — it returns a `Config` dataclass (`AWSConfig` + `ModelConfig`) rather than scripts calling `os.getenv()` directly. Missing AWS fields resolve to `None` rather than raising; missing model ids fall back to defaults (`amazon.nova-micro-v1:0` for chat, `amazon.titan-embed-text-v2:0` for embeddings).
- Required Python version is pinned tightly: `>=3.14` (see `.python-version` and `pyproject.toml`).

## Architecture notes

**`langchain-community` is intentionally avoided.** It's being sunset (see the deprecation note in `documents.py`), so document loading in this repo does NOT use `TextLoader`, `WebBaseLoader`, `DirectoryLoader`, or `PyPDFLoader` from `langchain_community`. Instead, `documents.py` builds `langchain_core.documents.Document` objects directly on top of the underlying libraries those loaders used to wrap:
- text files → `pathlib`
- web pages → `requests` + `bs4` (`BeautifulSoup`)
- directories of files → a hand-rolled generator (`lazy_load_directory`) for lazy loading
- PDFs → `pypdf.PdfReader` directly, producing one `Document` per page (mirroring `PyPDFLoader`'s per-page behavior)

When adding a new loader-style sample, follow this same pattern (wrap the underlying library directly into a `Document`) rather than reaching for a `langchain_community` loader.

Bedrock access goes through `langchain_aws.ChatBedrockConverse` (Bedrock's Converse API), not a model-specific client — this gives a consistent request/response shape regardless of which foundation model is configured via `BEDROCK_MODEL_ID`.

`rag_pipeline.py` is currently all commented-out stub code (a RAG demo that hasn't been implemented yet) — don't treat it as working reference code.

"""
Sample code demonstrating how to generate text embeddings via AWS Bedrock
using `langchain_aws.BedrockEmbeddings`.

An embedding is a list of floats representing the meaning of a piece of
text, positioned so that texts with similar meaning end up with similar
vectors - the basis for semantic search and RAG retrieval.

Run directly with: uv run .\\embeddings.py
"""

import hashlib
import json
from pathlib import Path

from langchain_aws import BedrockEmbeddings

from config import Config, load_config
from utils.embeddings_client import build_embeddings_client

CACHE_FILE = Path(__file__).parent / ".embedding_cache.json"


def embed_single_text(embeddings: BedrockEmbeddings) -> None:
    """Embed one piece of text and show the resulting vector."""
    text = "LangChain makes it easier to build LLM-powered applications."
    vector = embeddings.embed_query(text)
    print(f"Embedded text: {text!r}")
    print(f"Vector length: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
    print()


def embed_multiple_documents(embeddings: BedrockEmbeddings) -> None:
    """Embed a batch of texts at once, e.g. before storing them in a vector store."""
    texts = [
        "Bedrock hosts foundation models from providers like Amazon and Anthropic.",
        "Embeddings are used for semantic search and retrieval-augmented generation.",
        "LangChain provides a common interface across many embedding providers.",
    ]
    vectors = embeddings.embed_documents(texts)
    print(f"Embedded {len(texts)} documents:")
    for text, vector in zip(texts, vectors):
        print(f"  {text!r} -> vector of length {len(vector)}")
    print()


def _cache_key(model_id: str, text: str) -> str:
    """Hash the model id + text so cache entries don't collide across models."""
    return hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest()


def embed_with_local_cache(embeddings: BedrockEmbeddings, config: Config) -> None:
    """Embed text through a local JSON cache, avoiding repeat calls for known text."""
    cache: dict[str, list[float]] = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    text = "LangChain makes it easier to build LLM-powered applications."
    key = _cache_key(config.model.embedding_model_id, text)

    if key in cache:
        print(f"Cache hit for: {text!r}")
        vector = cache[key]
    else:
        print(f"Cache miss for: {text!r} - calling Bedrock")
        vector = embeddings.embed_query(text)
        cache[key] = vector
        CACHE_FILE.write_text(json.dumps(cache))

    print(f"Vector length: {len(vector)}")

    # Re-run the same text to demonstrate the cache being used on the second call.
    print(f"Re-embedding same text: {text!r}")
    key = _cache_key(config.model.embedding_model_id, text)
    if key in cache:
        print("Cache hit for: (same text) - no Bedrock call made")
    print()


def main() -> None:
    print("=== Langchain embeddings sample ===")
    config = load_config()
    print(f"Embedding model id: {config.model.embedding_model_id}")

    embeddings = build_embeddings_client(config)
    embed_single_text(embeddings)
    embed_multiple_documents(embeddings)
    embed_with_local_cache(embeddings, config)


if __name__ == "__main__":
    main()

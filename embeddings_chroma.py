"""
Sample code demonstrating how to store embeddings in a Chroma vector store
using `langchain_chroma.Chroma`, and how to query it via similarity search.

This builds on embeddings.py: instead of just computing raw vectors,
Documents and their embeddings are persisted in a vector store so they can
be searched later by semantic similarity - the retrieval half of a RAG
pipeline (see rag_pipeline.py).

Run directly with: uv run .\\embeddings_chroma.py
"""

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Config, load_config
from utils.chroma_temp_dir import temp_persist_directory
from utils.embeddings_client import build_embeddings_client


def build_sample_documents() -> list[Document]:
    """A handful of Documents to embed and store, covering a few unrelated
    topics so similarity search results are easy to eyeball."""
    return [
        Document(
            page_content="LangChain provides a common interface across many LLM providers.",
            metadata={"source": "langchain_overview.txt"},
        ),
        Document(
            page_content="Bedrock hosts foundation models from providers like Amazon and Anthropic.",
            metadata={"source": "bedrock_overview.txt"},
        ),
        Document(
            page_content="Chroma is an open-source embedding database used as a vector store.",
            metadata={"source": "chroma_overview.txt"},
        ),
        Document(
            page_content="Sourdough bread relies on a wild yeast starter instead of commercial yeast.",
            metadata={"source": "baking_notes.txt"},
        ),
    ]


def build_vector_store(embeddings: BedrockEmbeddings, documents: list[Document], persist_directory: str) -> Chroma:
    """Embed `documents` and store the vectors in a Chroma collection on disk.

    `from_documents` embeds every Document's page_content (via
    `embeddings.embed_documents` under the hood) and writes the resulting
    vectors, text, and metadata into the collection at `persist_directory`.
    """
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


def similarity_search(vector_store: Chroma) -> None:
    """Embed a query and return the stored Documents whose vectors are
    closest to it - the core operation a RAG retriever performs."""
    query = "What LLM providers does Bedrock support?"
    # Embeds `query` and returns the k Documents whose vectors are nearest
    # to it, ordered most-similar first. No scores here - see
    # similarity_search_with_score() below if you need the distance values.
    results = vector_store.similarity_search(query, k=2)

    print(f"Query: {query!r}")
    print(f"Top {len(results)} matches:")
    for doc in results:
        print(f"  {doc.page_content!r} (source: {doc.metadata['source']})")
    print()


def similarity_search_with_score(vector_store: Chroma) -> None:
    """Same as similarity_search, but also returns a distance score per
    result - lower means more similar for Chroma's default distance metric."""
    query = "database for storing embeddings"
    # Same nearest-neighbour search as similarity_search(), but returns
    # (Document, score) tuples instead of bare Documents - useful when you
    # need to filter out weak matches rather than always taking the top k.
    results = vector_store.similarity_search_with_score(query, k=2)

    print(f"Query: {query!r}")
    print(f"Top {len(results)} matches with scores:")
    for doc, score in results:
        print(f"  score={score:.4f}  {doc.page_content!r} (source: {doc.metadata['source']})")
    print()


def main() -> None:
    print("=== Langchain + Chroma vector store sample ===")
    config: Config = load_config()
    embeddings = build_embeddings_client(config)
    documents = build_sample_documents()

    # A temp directory keeps this sample self-contained - real usage would
    # point persist_directory at a stable path so the collection survives
    # across runs instead of being recreated (and discarded) every time.
    # See embedding_chroma_persistence.py for that stable-path case.
    with temp_persist_directory() as persist_directory:
        vector_store = build_vector_store(embeddings, documents, persist_directory)
        similarity_search(vector_store)
        similarity_search_with_score(vector_store)


if __name__ == "__main__":
    main()

"""
Sample code demonstrating that a Chroma vector store persists to disk
across separate runs of a script, rather than only within one process.

embeddings_chroma.py builds its vector store in a temp directory that gets
deleted (shutil.rmtree) at the end of every run, so it never actually shows
data surviving between runs - it only proves the store works within a
single process. This sample uses a stable, non-temp directory instead and
is deliberately run twice to show the difference:

  - 1st run: PERSIST_DIRECTORY doesn't exist yet, so the sample Documents
    are embedded (a Bedrock call per Document) and written to disk.
  - 2nd+ run: PERSIST_DIRECTORY already has a Chroma collection in it, so
    the existing vectors are loaded straight from disk - no embedding call
    is made for the stored Documents, only for the query at search time.

Run directly with: uv run .\\embedding_chroma_persistence.py
Run it a second time to see the "loaded existing" path instead.
Delete the .chroma_persistence_demo directory to reset back to a fresh start.
"""

from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma

from config import Config, load_config
from embeddings_chroma import build_sample_documents, build_vector_store
from utils.embeddings_client import build_embeddings_client

# A stable directory (unlike embeddings_chroma.py's tempfile.mkdtemp()) so
# the Chroma collection written here still exists the next time this
# script is run. Chroma writes a chroma.sqlite3 file into this directory
# as part of from_documents()/add_documents() - that's what makes the
# collection persistent rather than in-memory-only.
PERSIST_DIRECTORY = Path(__file__).parent / ".chroma_persistence_demo"


def has_persisted_data(persist_directory: Path) -> bool:
    """Check whether a Chroma collection already exists on disk at
    `persist_directory`, so main() can decide whether to embed the sample
    Documents from scratch or just load what's already there."""
    return (persist_directory / "chroma.sqlite3").exists()


def load_existing_vector_store(embeddings: BedrockEmbeddings, persist_directory: Path) -> Chroma:
    """Re-open a Chroma collection that was already written to disk by a
    previous run, without embedding or re-adding any Documents.

    This is the counterpart to embeddings_chroma.build_vector_store(),
    which always calls Chroma.from_documents() and therefore always
    embeds its input Documents. Here, passing persist_directory to the
    Chroma constructor directly (with no documents/texts argument) instead
    just points it at the existing collection database on disk. The
    collection_name argument defaults to "langchain" in both this
    constructor and from_documents(), so the two paths agree on which
    collection inside chroma.sqlite3 they're reading/writing without
    either one needing to say so explicitly.
    """
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def query_vector_store(vector_store: Chroma) -> None:
    """Run one similarity search so both the "freshly created" and
    "loaded from disk" paths in main() can be shown returning the same
    results, proving the reloaded store is equivalent to the original."""
    query = "What LLM providers does Bedrock support?"
    # Only the query text is embedded here - the stored Document vectors
    # were either just computed (1st run) or loaded from disk (later
    # runs), not recomputed by this call either way.
    results = vector_store.similarity_search(query, k=2)

    print(f"Query: {query!r}")
    print(f"Top {len(results)} matches:")
    for doc in results:
        print(f"  {doc.page_content!r} (source: {doc.metadata['source']})")
    print()


def main() -> None:
    print("=== Langchain + Chroma persistence sample ===")
    config: Config = load_config()
    embeddings = build_embeddings_client(config)

    if has_persisted_data(PERSIST_DIRECTORY):
        print(f"Found existing Chroma data at {PERSIST_DIRECTORY} - loading without re-embedding.")
        vector_store = load_existing_vector_store(embeddings, PERSIST_DIRECTORY)
    else:
        print(f"No existing Chroma data at {PERSIST_DIRECTORY} - embedding sample Documents and creating it.")
        documents = build_sample_documents()
        # Note: unlike embeddings_chroma.py, PERSIST_DIRECTORY is never
        # deleted after this - that's the whole point of this sample.
        vector_store = build_vector_store(embeddings, documents, str(PERSIST_DIRECTORY))

    query_vector_store(vector_store)
    print("Run this script again to see the persisted-data path instead of this one.")
    print(f"Delete {PERSIST_DIRECTORY} to reset back to a fresh start.")


if __name__ == "__main__":
    main()

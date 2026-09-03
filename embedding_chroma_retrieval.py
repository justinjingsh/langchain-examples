"""
Sample code demonstrating how to use a Chroma vector store as a retriever
via `VectorStore.as_retriever()`.

This builds on embeddings_chroma.py, which queries the vector store
directly (similarity_search / similarity_search_with_score). Wrapping it
with as_retriever() instead turns it into a `BaseRetriever` - a Runnable
with a standard .invoke(query) -> list[Document] interface, which is what
lets a vector store be plugged straight into an LCEL chain alongside a
prompt and a chat model (see rag_pipeline.py).

Run directly with: uv run .\\embedding_chroma_retrieval.py
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from config import Config, load_config
from embeddings_chroma import build_sample_documents, build_vector_store
from utils.chroma_temp_dir import temp_persist_directory
from utils.embeddings_client import build_embeddings_client


def as_similarity_retriever(vector_store: Chroma) -> VectorStoreRetriever:
    """The default retriever: plain nearest-neighbour similarity search,
    wrapped so it can be called via .invoke() instead of
    vector_store.similarity_search() directly."""
    # as_retriever() doesn't embed or search anything itself - it just
    # returns a VectorStoreRetriever object that closes over vector_store
    # and these search_kwargs. The actual embed-the-query-and-search work
    # happens later, inside run_retriever(), when .invoke() is called.
    #
    # search_kwargs is passed straight through to the underlying search
    # call (here, vector_store.similarity_search(query, k=2)), so any
    # keyword that method accepts can go in this dict.
    return vector_store.as_retriever(search_kwargs={"k": 2})


def as_mmr_retriever(vector_store: Chroma) -> VectorStoreRetriever:
    """A retriever using Maximal Marginal Relevance instead of plain
    similarity - it still favours documents close to the query, but
    penalises ones that are too similar to results already picked, which
    reduces near-duplicate results in the returned set."""
    # search_type="mmr" swaps the retriever's internal search call from
    # similarity_search to max_marginal_relevance_search. That method
    # first fetches fetch_k candidates by similarity (a wider net than
    # what's finally returned), then greedily selects k of them, at each
    # step trading off "close to the query" against "different from what's
    # already been picked" - hence fetch_k >= k. With only 4 unrelated
    # sample Documents in the store there's little redundancy to remove,
    # so the result here happens to match the plain similarity retriever;
    # the difference shows up more with a larger, more repetitive corpus.
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 4},
    )


def run_retriever(name: str, retriever: VectorStoreRetriever, query: str) -> None:
    """Retrievers are Runnables, so .invoke(query) is the standard way to
    call one - the same interface a prompt or chat model uses, which is
    what lets a retriever slot into an LCEL chain."""
    # This single call does everything similarity_search() did in
    # embeddings_chroma.py under the hood: embed `query` via the vector
    # store's embedding function, run the configured search
    # (similarity or mmr, per how the retriever was built above), and
    # return the matching Documents in ranked order. Because .invoke() is
    # the standard Runnable entrypoint, this same line would still work if
    # `retriever` were swapped for a prompt, a chat model, or any other
    # LCEL-compatible component.
    results: list[Document] = retriever.invoke(query)

    print(f"[{name}] Query: {query!r}")
    print(f"Top {len(results)} matches:")
    for doc in results:
        print(f"  {doc.page_content!r} (source: {doc.metadata['source']})")
    print()


def main() -> None:
    print("=== Langchain + Chroma retrieval sample ===")
    config: Config = load_config()
    embeddings = build_embeddings_client(config)
    # Same 4 unrelated-topic sample Documents used in embeddings_chroma.py,
    # reused here (rather than redefined) so both samples search over
    # identical data and their results are directly comparable.
    documents = build_sample_documents()

    # A fresh temp directory per run keeps this sample self-contained: the
    # vector store is built from scratch every time rather than persisting
    # across runs. See embedding_chroma_persistence.py for the stable-path,
    # persists-across-runs case.
    with temp_persist_directory() as persist_directory:
        # Embeds every Document's page_content and writes the resulting
        # vectors into a Chroma collection on disk at persist_directory -
        # see embeddings_chroma.build_vector_store for details.
        vector_store = build_vector_store(embeddings, documents, persist_directory)
        # Same query for both retrievers below, so the only variable
        # between the two printed result sets is the search strategy.
        query = "What LLM providers does Bedrock support?"

        run_retriever("similarity", as_similarity_retriever(vector_store), query)
        run_retriever("mmr", as_mmr_retriever(vector_store), query)


if __name__ == "__main__":
    main()

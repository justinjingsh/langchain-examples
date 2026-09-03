"""
Sample code demonstrating how to split text into chunks using
`langchain-text-splitters`, for feeding long documents into an LLM or an
embedding/retrieval pipeline in smaller pieces.

Splitters take one or more `langchain_core.documents.Document` objects (or
raw strings) and return a list of smaller Documents, preserving metadata
from the source Document on each chunk.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


def split_by_character():
    """Split text on a single fixed separator (blank lines, by default).

    `CharacterTextSplitter` looks only for one separator string. If a chunk
    between separators is still longer than `chunk_size`, it is left as-is
    (it does not fall back to splitting on anything smaller) - see
    `split_recursively` below for a splitter that does.
    """
    text = (
        "LangChain is a framework for building LLM-powered applications.\n\n"
        "It provides abstractions for chains, agents, and tools.\n\n"
        "Text splitters break long documents into smaller chunks so they fit "
        "within a model's context window or an embedding model's input limit."
    )

    splitter = CharacterTextSplitter(
        # separator is the ONLY split point this splitter looks for - here it
        # splits on blank lines, so each paragraph starts as its own chunk.
        separator="\n\n",
        # chunk_size is a target max length (in characters, since we're not
        # passing length_function) - paragraphs are merged into a chunk until
        # adding the next one would exceed this.
        chunk_size=100,
        # chunk_overlap repeats this many trailing characters from the
        # previous chunk at the start of the next one, so context isn't lost
        # right at a chunk boundary (e.g. a sentence split across chunks).
        chunk_overlap=20,
    )
    # split_text takes a raw string and returns a list of strings - there's no
    # Document/metadata involved here (see split_documents() for that case).
    chunks = splitter.split_text(text)

    print("Splitting by a fixed separator...")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk!r}")
    print()


def split_recursively():
    """Split text using an ordered list of separators, falling back to the
    next one whenever a chunk is still too big.

    `RecursiveCharacterTextSplitter` tries each separator in `separators` in
    order (paragraphs, then lines, then words, then characters) until every
    chunk fits within `chunk_size`. This is the generally recommended
    splitter for plain text, since it keeps related text together as long as
    possible instead of cutting at a fixed point.
    """
    text = (
        "LangChain is a framework for building LLM-powered applications. "
        "It provides abstractions for chains, agents, and tools.\n"
        "Text splitters break long documents into smaller chunks so they fit "
        "within a model's context window or an embedding model's input limit. "
        "Choosing a good chunk size and overlap matters a lot for retrieval "
        "quality in a RAG pipeline."
    )

    splitter = RecursiveCharacterTextSplitter(
        # chunk_size/chunk_overlap mean the same thing as for
        # CharacterTextSplitter above, but here they're a target this
        # splitter actively works to satisfy, rather than just a threshold
        # for merging pre-split pieces.
        chunk_size=100,
        chunk_overlap=20,
        # separators defaults to ["\n\n", "\n", " ", ""] - not passed here,
        # but that's what drives the "recursive" behaviour: try splitting on
        # paragraphs first, and only if a resulting chunk is still bigger
        # than chunk_size, re-split that chunk on the next separator in the
        # list (lines, then words, then finally individual characters).
        # This is why every chunk below fits within chunk_size, unlike
        # CharacterTextSplitter's single long chunk 2 in split_by_character().
    )
    chunks = splitter.split_text(text)

    print("Splitting recursively on paragraphs, then lines, then words...")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk!r}")
    print()


def split_documents():
    """Split a `Document` (rather than a raw string) into smaller Documents.

    `split_documents` chunks `page_content` the same way `split_text` does,
    but copies the source Document's `metadata` onto every resulting chunk -
    useful for keeping track of which original file/page a chunk came from.
    """
    doc = Document(
        page_content=(
            "Retrieval-augmented generation (RAG) combines a retriever with "
            "an LLM. The retriever fetches relevant chunks from a vector "
            "store, and the LLM uses them as context to answer a question."
        ),
        metadata={"source": "rag_overview.txt"},
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
    chunks = splitter.split_documents([doc])

    print("Splitting a Document (metadata is preserved on each chunk)...")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} content: {chunk.page_content!r}")
        print(f"Chunk {i} metadata: {chunk.metadata}")
        print()


def main():
    print("=== Langchain text splitter samples ===")
    split_by_character()
    split_recursively()
    split_documents()


if __name__ == "__main__":
    main()

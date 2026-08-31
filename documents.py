"""
Sample code demonstrating how to load data into LangChain `Document` objects
from a variety of sources, WITHOUT depending on `langchain-community`.

`langchain-community` is being sunset (see
https://github.com/langchain-ai/langchain-community/issues/674), so instead
of importing loaders like `TextLoader`, `WebBaseLoader`, `DirectoryLoader` or
`PyPDFLoader` from it, each function below builds `Document` objects directly
using the underlying library the community loader would have wrapped
(`pypdf`, `requests` + `bs4`, `pathlib`) and `langchain_core.documents.Document`.

A `Document` is just a small container with two fields:
  - page_content: the extracted text
  - metadata: a dict describing where the text came from (source, page, etc.)
"""

import tempfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from pypdf import PdfReader


def load_text_file():
    """Load a single local text file into one Document.

    This mirrors what `TextLoader` used to do: read the whole file into
    `page_content` and record the file path as `metadata["source"]`.

    The file is created in a temp directory purely so this sample is
    self-contained and has no external file dependency - in real usage you'd
    just pass the path to an existing file.
    """
    # delete=False: on Windows the file must be closed before we can
    # re-open/delete it, so we can't rely on the context manager to clean it up.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = Path(temp_file.name)

    try:
        text = temp_file_path.read_text()
        doc = Document(page_content=text, metadata={"source": str(temp_file_path)})
        print(f"Document content: {doc.page_content}")
        print(f"Document metadata: {doc.metadata}")
        print()
    finally:
        # Clean up the temp file now that we're done reading it.
        temp_file_path.unlink()


def load_web_content():
    """Fetch a web page and load its visible text into one Document.

    This replaces `WebBaseLoader`: fetch the raw HTML with `requests`, strip
    the markup with `BeautifulSoup` to get plain text, and wrap it in a
    Document with the page URL as `metadata["source"]`.
    """
    url = "https://python.langchain.com/docs/introduction/"
    # Some sites reject requests with no User-Agent header, so set one.
    response = requests.get(url, headers={"User-Agent": "langchain-examples"})
    response.raise_for_status()  # fail fast on 4xx/5xx instead of loading an error page

    # "html.parser" is Python's built-in parser - no extra dependency needed.
    soup = BeautifulSoup(response.text, "html.parser")
    # get_text collapses all the tags away, leaving just the readable text.
    # separator=" " avoids words from adjacent tags getting jammed together.
    text = soup.get_text(separator=" ", strip=True)

    doc = Document(page_content=text, metadata={"source": url})
    # Content is truncated for the demo print - real pipelines would keep it all.
    print(f"Document content: {doc.page_content[:200]}...")
    print(f"Document metadata: {doc.metadata}")
    print()


def lazy_load_directory(directory):
    """Yield one Document per .txt file in a directory, one at a time.

    This is a generator (uses `yield`) rather than returning a list, which
    replaces what `DirectoryLoader(..., loader_cls=TextLoader).lazy_load()`
    used to do: files are read from disk on demand as the caller iterates,
    instead of loading everything into memory up front.
    """
    # sorted() gives a stable, predictable order (glob() order isn't guaranteed).
    for path in sorted(Path(directory).glob("*.txt")):
        yield Document(page_content=path.read_text(), metadata={"source": str(path)})


def lazy_loader():
    """Demonstrate lazy-loading multiple documents from a directory.

    Creates 5 sample .txt files in a temporary directory, then loads them one
    at a time via `lazy_load_directory` to show the generator only reads each
    file when the loop actually asks for the next one.
    """
    # TemporaryDirectory cleans up the whole folder automatically on exit.
    with tempfile.TemporaryDirectory() as tempdir:
        for i in range(5):
            path = Path(tempdir) / f"document_{i}.txt"
            path.write_text(f"This is the content of document {i}.")

        print("Loading documents lazily...")
        for doc in lazy_load_directory(tempdir):
            print(f"Document content: {doc.page_content}")
            print(f"Document metadata: {doc.metadata}")
            print()


def load_structure():
    """Build a Document directly from structured/in-memory data (e.g. a dict
    that came from a JSON API or database row), rather than from a file.

    Shows that Documents don't have to come from a "loader" at all - any code
    that produces text and some descriptive fields can construct one, which
    is useful when your source data isn't file-based.
    """
    structured_data = {
        "title": "Sample Structured Document",
        "content": "This is a sample structured document for testing.",
        "author": "LangChain Team"
    }
    # Only the free-text field goes into page_content; the rest becomes metadata
    # so it's still queryable/filterable later without being mixed into the text.
    doc = Document(page_content=structured_data["content"],
                   metadata={"title": structured_data["title"], "author": structured_data["author"]})
    print(f"Document content: {doc.page_content}")
    print(f"Document metadata: {doc.metadata}")
    print()


def load_pdf(pdf_path=r"C:\exports\langchain_demo.pdf"):
    """Load a PDF and create one Document per page.

    This replaces `PyPDFLoader`, which itself is a thin wrapper around
    `pypdf.PdfReader`. Splitting into one Document per page (rather than one
    Document for the whole PDF) mirrors PyPDFLoader's behaviour and makes it
    easy to cite/retrieve individual pages later (e.g. in a RAG pipeline).
    """
    print("Loading PDF document...")

    reader = PdfReader(pdf_path)
    for page_number, page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text(),
            # page_number is 0-indexed here, matching PyPDFLoader's convention.
            metadata={"source": str(pdf_path), "page": page_number},
        )
        print(f"Document content: {doc.page_content[:200]!r}")
        print(f"Document metadata: {doc.metadata}")
        print()


def main():
    print("Langchain document loading samples")
    load_text_file()
    load_web_content()
    lazy_loader()
    load_structure()
    load_pdf()

if __name__ == "__main__":
    main()

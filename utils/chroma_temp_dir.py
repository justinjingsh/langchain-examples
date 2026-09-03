"""
Shared helper for samples that build a throwaway Chroma collection: a
context manager that creates a temp directory and guarantees its removal,
used by embeddings_chroma.py and embedding_chroma_retrieval.py.

Pulled out because both samples had this exact create/try/finally-cleanup
sequence, including the Windows-specific reason for the manual rmtree,
duplicated in full.
"""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def temp_persist_directory() -> Iterator[str]:
    """Yield a fresh temp directory path, deleting it on exit.

    Cleaned up manually with shutil.rmtree rather than
    tempfile.TemporaryDirectory's own context manager: Chroma keeps its
    sqlite file open for the life of the process, and on Windows that open
    handle makes TemporaryDirectory's cleanup-on-exit raise PermissionError.
    """
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

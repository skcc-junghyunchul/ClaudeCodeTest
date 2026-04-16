"""
Document loading and splitting for Korean-language market intelligence documents.

Supports: .md, .txt, .pdf, .docx
Korean-aware splitting uses sentence-boundary separators common in Korean text.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import config


# Korean sentence-boundary separators (ordered from coarsest to finest)
KOREAN_SEPARATORS = [
    "\n\n\n",   # Multiple blank lines (section breaks)
    "\n\n",     # Paragraph breaks
    "\n",       # Line breaks
    "。",       # CJK full stop
    "．",       # Full-width period
    "！",       # Full-width exclamation
    "？",       # Full-width question mark
    ". ",       # ASCII period + space
    "! ",
    "? ",
    "다. ",     # Common Korean sentence ending
    "요. ",
    "죠. ",
    "며. ",
    "고. ",
    "가. ",
    " ",
    "",
]


def load_documents(data_dir: str = "./data") -> List[Document]:
    """Load all supported documents from a directory."""
    docs: List[Document] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in sorted(data_path.rglob("*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()
        loader = None

        if suffix in (".md", ".markdown"):
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(file_path))

        if loader:
            loaded = loader.load()
            # Tag each doc with its source filename
            for doc in loaded:
                doc.metadata.setdefault("source", file_path.name)
            docs.extend(loaded)
            print(f"  Loaded {len(loaded)} page(s) from {file_path.name}")

    print(f"Total documents loaded: {len(docs)}")
    return docs


def build_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small chunks stored in the vector DB for precise retrieval."""
    return RecursiveCharacterTextSplitter(
        separators=KOREAN_SEPARATORS,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


def build_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Larger chunks that provide richer context when returned as the answer source."""
    return RecursiveCharacterTextSplitter(
        separators=KOREAN_SEPARATORS,
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP * 2,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    docs: List[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> List[Document]:
    """Split documents with the given splitter (defaults to child splitter)."""
    if splitter is None:
        splitter = build_child_splitter()

    chunks = splitter.split_documents(docs)

    # Attach chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Split into {len(chunks)} chunks")
    return chunks

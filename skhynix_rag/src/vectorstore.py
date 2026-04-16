"""
Chroma vector store management with Azure OpenAI embeddings.

Uses a parent-document pattern:
  - Child chunks (small) are stored in Chroma for precise semantic search.
  - Parent chunks (large) are stored in a side dict and returned as context
    so the LLM has more surrounding text to work with.
"""

import json
import os
from typing import List, Dict

import chromadb
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document

from .config import config
from .document_processor import (
    load_documents,
    split_documents,
    build_child_splitter,
    build_parent_splitter,
)


def build_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        openai_api_version=config.AZURE_OPENAI_EMBEDDING_API_VERSION,
    )


def build_vectorstore(embeddings: AzureOpenAIEmbeddings | None = None) -> Chroma:
    """Load (or create) the persisted Chroma collection."""
    if embeddings is None:
        embeddings = build_embeddings()

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def ingest(data_dir: str = "./data") -> tuple[Chroma, Dict[str, str]]:
    """
    Load, split, embed, and persist documents.

    Returns
    -------
    vectorstore : Chroma
    parent_store : dict[chunk_id -> parent_text]
        Maps each child chunk_id to the text of its parent chunk so the agent
        can retrieve richer context.
    """
    print("=== Loading documents ===")
    raw_docs = load_documents(data_dir)

    print("\n=== Creating parent chunks ===")
    parent_chunks = split_documents(raw_docs, build_parent_splitter())

    print("\n=== Creating child chunks ===")
    child_chunks: List[Document] = []
    parent_store: Dict[str, str] = {}

    child_splitter = build_child_splitter()
    for p_idx, parent in enumerate(parent_chunks):
        children = child_splitter.split_documents([parent])
        for c_idx, child in enumerate(children):
            uid = f"p{p_idx}_c{c_idx}"
            child.metadata["chunk_id"] = uid
            child.metadata["parent_id"] = p_idx
            child_chunks.append(child)
            parent_store[uid] = parent.page_content

    print(f"  {len(parent_chunks)} parent chunks → {len(child_chunks)} child chunks")

    print("\n=== Embedding & persisting to Chroma ===")
    embeddings = build_embeddings()

    # Delete existing collection to avoid duplicates on re-ingestion
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    existing = [c.name for c in client.list_collections()]
    if config.COLLECTION_NAME in existing:
        client.delete_collection(config.COLLECTION_NAME)
        print(f"  Deleted existing collection '{config.COLLECTION_NAME}'")

    vectorstore = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    print(f"  Stored {len(child_chunks)} child chunks in Chroma")

    # Persist parent store alongside the vector DB
    parent_store_path = os.path.join(config.CHROMA_PERSIST_DIR, "parent_store.json")
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
    with open(parent_store_path, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)
    print(f"  Saved parent store to {parent_store_path}")

    return vectorstore, parent_store


def load_parent_store() -> Dict[str, str]:
    parent_store_path = os.path.join(config.CHROMA_PERSIST_DIR, "parent_store.json")
    if not os.path.exists(parent_store_path):
        return {}
    with open(parent_store_path, encoding="utf-8") as f:
        return json.load(f)

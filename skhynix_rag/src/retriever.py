"""
Advanced hybrid retrieval with cross-encoder reranking.

Pipeline:
  1. Dense retrieval  – Chroma (Azure OpenAI embeddings)
  2. Sparse retrieval – BM25 (token-level keyword matching, good for Korean nouns)
  3. Ensemble        – weighted combination via EnsembleRetriever
  4. Reranking       – cross-encoder (BAAI/bge-reranker-v2-m3, multilingual)
  5. Parent upgrade  – swap child text for its larger parent chunk

The cross-encoder is optional; the retriever degrades gracefully if
sentence-transformers is unavailable.
"""

from typing import List, Dict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .config import config


# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional dependency)
# ---------------------------------------------------------------------------

def _build_reranker():
    """Return a CrossEncoder model or None if sentence-transformers is absent."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        print("  Cross-encoder reranker: BAAI/bge-reranker-v2-m3 (multilingual)")
        return CrossEncoder("BAAI/bge-reranker-v2-m3")
    except Exception as e:
        print(f"  Cross-encoder unavailable ({e}); skipping reranking step.")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Wraps dense + BM25 ensemble retrieval with optional cross-encoder reranking
    and parent-document expansion.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        all_docs: List[Document],
        parent_store: Dict[str, str],
    ):
        self._vectorstore = vectorstore
        self._parent_store = parent_store
        self._reranker = _build_reranker()

        # Dense retriever (Chroma similarity search)
        dense_retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": config.RETRIEVAL_K,
                "fetch_k": config.RETRIEVAL_K * 3,
                "lambda_mult": 0.7,   # 0 = max diversity, 1 = max relevance
            },
        )

        # Sparse retriever (BM25 over child-chunk texts)
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = config.RETRIEVAL_K

        # Ensemble: weighted combination
        dense_weight = 1.0 - config.BM25_WEIGHT
        self._ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[dense_weight, config.BM25_WEIGHT],
        )

    def retrieve(self, query: str) -> List[Document]:
        """Full pipeline: ensemble → rerank → parent-expand."""
        # 1. Hybrid ensemble retrieval
        docs = self._ensemble.invoke(query)

        # 2. Cross-encoder reranking
        if self._reranker and docs:
            docs = self._rerank(query, docs)

        # 3. Parent-document expansion
        docs = self._expand_to_parents(docs)

        return docs[: config.RERANK_TOP_K]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]

    def _expand_to_parents(self, docs: List[Document]) -> List[Document]:
        """Replace each child chunk with its parent chunk (if available)."""
        expanded: List[Document] = []
        seen_parents: set = set()

        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            parent_text = self._parent_store.get(str(chunk_id)) if chunk_id else None

            if parent_text and chunk_id not in seen_parents:
                seen_parents.add(chunk_id)
                expanded.append(
                    Document(
                        page_content=parent_text,
                        metadata={**doc.metadata, "expanded": True},
                    )
                )
            elif chunk_id not in seen_parents:
                seen_parents.add(str(chunk_id))
                expanded.append(doc)

        return expanded

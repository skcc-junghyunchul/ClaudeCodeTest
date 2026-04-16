"""
Corrective RAG (CRAG) + Self-RAG agent implemented with LangGraph.

Graph flow
----------

    START
      │
      ▼
  retrieve          ← hybrid retrieval (dense + BM25 + rerank)
      │
      ▼
  grade_documents   ← LLM grades each doc for relevance
      │
      ├─ all relevant ──────────────────────────────┐
      │                                             ▼
      └─ some/all irrelevant → transform_query → retrieve (loop)
                                                   │
                                                   ▼
                                              generate   ← builds answer from context
                                                   │
                                                   ▼
                                         check_hallucination
                                                   │
                                    ┌── grounded ──┴── not grounded ──┐
                                    ▼                                  ▼
                              grade_answer                         generate (retry)
                                    │
                         ┌── useful ─┴── not useful ──┐
                         ▼                            ▼
                        END                    transform_query → retrieve (loop)

Iteration guard: MAX_ITERATIONS prevents infinite loops.
"""

import json
from typing import List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from .config import config
from .prompts import (
    GENERATE_PROMPT,
    GRADE_DOCUMENT_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_GRADE_PROMPT,
    REWRITE_PROMPT,
)
from .retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    iteration: int          # guards against infinite loops


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _parse_score(text: str) -> str:
    """Extract 'yes'/'no' from a JSON or plain-text LLM response."""
    try:
        data = json.loads(text.strip())
        return data.get("score", "no").lower()
    except json.JSONDecodeError:
        lower = text.lower()
        if "yes" in lower:
            return "yes"
        return "no"


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_agent(retriever: HybridRetriever):
    """
    Compile and return the LangGraph CRAG agent.

    Parameters
    ----------
    retriever : HybridRetriever
        Pre-built hybrid retriever (owns the vectorstore + parent store).
    """

    llm = AzureChatOpenAI(
        azure_deployment=config.AZURE_OPENAI_CHAT_DEPLOYMENT,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        openai_api_version=config.AZURE_OPENAI_API_VERSION,
        temperature=0,
        streaming=True,
    )

    # -----------------------------------------------------------------------
    # Node: retrieve
    # -----------------------------------------------------------------------

    def retrieve(state: GraphState) -> GraphState:
        print(f"\n[retrieve] query: {state['question']}")
        docs = retriever.retrieve(state["question"])
        print(f"  → {len(docs)} document(s) retrieved")
        return {**state, "documents": docs}

    # -----------------------------------------------------------------------
    # Node: grade_documents
    # -----------------------------------------------------------------------

    def grade_documents(state: GraphState) -> GraphState:
        question = state["question"]
        docs = state["documents"]
        grader = GRADE_DOCUMENT_PROMPT | llm

        relevant: List[Document] = []
        for doc in docs:
            result = grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            score = _parse_score(result.content)
            if score == "yes":
                relevant.append(doc)

        print(f"[grade_documents] {len(relevant)}/{len(docs)} relevant")
        return {**state, "documents": relevant}

    # -----------------------------------------------------------------------
    # Node: generate
    # -----------------------------------------------------------------------

    def generate(state: GraphState) -> GraphState:
        question = state["question"]
        docs = state["documents"]
        context = _format_docs(docs)

        chain = GENERATE_PROMPT | llm
        result = chain.invoke({"context": context, "question": question})
        generation = result.content
        print(f"[generate] answer length: {len(generation)} chars")
        return {**state, "generation": generation}

    # -----------------------------------------------------------------------
    # Node: transform_query
    # -----------------------------------------------------------------------

    def transform_query(state: GraphState) -> GraphState:
        chain = REWRITE_PROMPT | llm
        result = chain.invoke({"question": state["question"]})
        new_question = result.content.strip()
        iteration = state.get("iteration", 0) + 1
        print(f"[transform_query] iteration {iteration}: '{new_question}'")
        return {**state, "question": new_question, "iteration": iteration}

    # -----------------------------------------------------------------------
    # Conditional edges
    # -----------------------------------------------------------------------

    def decide_after_grading(
        state: GraphState,
    ) -> Literal["generate", "transform_query"]:
        if state["documents"]:
            return "generate"
        if state.get("iteration", 0) >= config.MAX_ITERATIONS:
            print("[decide] max iterations reached — generating with empty context")
            return "generate"
        return "transform_query"

    def decide_after_generation(
        state: GraphState,
    ) -> Literal["useful", "not_supported", "transform_query"]:
        # -- Hallucination check --
        hal_chain = HALLUCINATION_PROMPT | llm
        docs_text = _format_docs(state["documents"]) if state["documents"] else "(없음)"
        hal_result = hal_chain.invoke(
            {"documents": docs_text, "generation": state["generation"]}
        )
        hal_score = _parse_score(hal_result.content)
        print(f"[hallucination_check] grounded: {hal_score}")

        if hal_score == "no":
            if state.get("iteration", 0) >= config.MAX_ITERATIONS:
                return "useful"  # give up retrying; return best effort
            return "not_supported"

        # -- Answer quality check --
        qa_chain = ANSWER_GRADE_PROMPT | llm
        qa_result = qa_chain.invoke(
            {"question": state["question"], "generation": state["generation"]}
        )
        qa_score = _parse_score(qa_result.content)
        print(f"[answer_quality_check] useful: {qa_score}")

        if qa_score == "yes":
            return "useful"
        if state.get("iteration", 0) >= config.MAX_ITERATIONS:
            return "useful"
        return "transform_query"

    # -----------------------------------------------------------------------
    # Build graph
    # -----------------------------------------------------------------------

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "useful": END,
            "not_supported": "generate",       # retry with same docs
            "transform_query": "transform_query",
        },
    )

    workflow.add_edge("transform_query", "retrieve")

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience: run a single question through a compiled agent
# ---------------------------------------------------------------------------

def ask(agent, question: str) -> str:
    """Invoke the compiled agent and return the final answer string."""
    result = agent.invoke(
        {"question": question, "documents": [], "generation": "", "iteration": 0}
    )
    return result.get("generation", "답변을 생성하지 못했습니다.")

"""
Interactive QnA chatbot for SK하이닉스 market intelligence.

Usage
-----
    python main.py                  # interactive REPL
    python main.py -q "HBM 시장 점유율은?"   # single question
    python main.py --demo           # run built-in demo questions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.vectorstore import build_vectorstore, load_parent_store
from src.document_processor import load_documents, split_documents, build_child_splitter
from src.retriever import HybridRetriever
from src.agent_graph import build_agent, ask

DEMO_QUESTIONS = [
    "SK하이닉스의 HBM 시장 점유율은 얼마이며, 경쟁사와 비교하면 어떤가요?",
    "2025년 DRAM 시장에서 DDR5 수요 현황과 전망을 설명해 주세요.",
    "SK하이닉스의 HBM4 개발 현황과 주요 기술 혁신은 무엇인가요?",
    "SK하이닉스의 주요 고객사(NVIDIA, Apple 등)와의 관계를 설명해 주세요.",
    "SK하이닉스의 2025년 재무 전망과 분기별 실적 예상치는?",
    "HBM 시장의 주요 리스크 요인은 무엇인가요?",
]


def print_banner():
    print("=" * 60)
    print("  SK하이닉스 시장 인텔리전스 QnA 에이전트")
    print("  (CRAG + Self-RAG | LangGraph + Azure OpenAI)")
    print("=" * 60)
    print()


def build_components():
    """Load vector store, build retriever and agent."""
    print(">>> 시스템 초기화 중...")
    config.validate()

    vectorstore = build_vectorstore()
    parent_store = load_parent_store()

    # BM25 needs the full document list
    raw_docs = load_documents("./data")
    child_chunks = split_documents(raw_docs, build_child_splitter())

    retriever = HybridRetriever(
        vectorstore=vectorstore,
        all_docs=child_chunks,
        parent_store=parent_store,
    )

    agent = build_agent(retriever)
    print(">>> 준비 완료\n")
    return agent


def interactive_loop(agent):
    print("질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    while True:
        try:
            question = input("질문> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "종료"):
            print("종료합니다.")
            break

        print()
        answer = ask(agent, question)
        print("\n" + "─" * 60)
        print(answer)
        print("─" * 60 + "\n")


def run_demo(agent):
    print("=== 데모 질문 실행 ===\n")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"[질문 {i}] {q}")
        print()
        answer = ask(agent, q)
        print("─" * 60)
        print(answer)
        print("─" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SK하이닉스 시장 인텔리전스 QnA 에이전트"
    )
    parser.add_argument("-q", "--question", help="단일 질문 모드")
    parser.add_argument("--demo", action="store_true", help="데모 질문 실행")
    args = parser.parse_args()

    print_banner()
    agent = build_components()

    if args.question:
        answer = ask(agent, args.question)
        print(answer)
    elif args.demo:
        run_demo(agent)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()

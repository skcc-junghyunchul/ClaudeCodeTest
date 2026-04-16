"""
Document ingestion script.

Run once (or whenever documents change) to build/rebuild the vector store:

    python ingest.py
    python ingest.py --data-dir ./data
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.vectorstore import ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing source documents (default: ./data)",
    )
    args = parser.parse_args()

    print("SK하이닉스 시장 인텔리전스 RAG — 문서 인제스트\n")
    config.validate()

    vectorstore, parent_store = ingest(data_dir=args.data_dir)

    print(f"\n✓ 인제스트 완료")
    print(f"  Vector store : {config.CHROMA_PERSIST_DIR}")
    print(f"  Collection   : {config.COLLECTION_NAME}")
    print(f"  Parent chunks: {len(parent_store)}")


if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Azure OpenAI - Chat
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")

    # Azure OpenAI - Embeddings
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
    )
    AZURE_OPENAI_EMBEDDING_API_VERSION: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-02-01"
    )

    # Vector Store
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "skhynix_market_intel")

    # RAG Settings
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

    # BM25 weight in hybrid retrieval (0.0–1.0; dense = 1 - this)
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))

    def validate(self) -> None:
        missing = [
            name
            for name, val in [
                ("AZURE_OPENAI_API_KEY", self.AZURE_OPENAI_API_KEY),
                ("AZURE_OPENAI_ENDPOINT", self.AZURE_OPENAI_ENDPOINT),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill in the values."
            )


config = Config()

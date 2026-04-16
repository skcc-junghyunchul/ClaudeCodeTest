# SK하이닉스 시장 인텔리전스 QnA 에이전트

한국어 반도체 시장 인텔리전스 문서를 대상으로 하는 고정밀 QnA 챗봇입니다.  
LangGraph 기반의 Corrective RAG(CRAG) + Self-RAG 아키텍처를 적용하여 환각(hallucination) 없는 정확한 답변을 제공합니다.

---

## 목차

1. [시스템 아키텍처](#시스템-아키텍처)
2. [사전 준비 사항](#사전-준비-사항)
3. [설치 방법](#설치-방법)
4. [환경 변수 설정](#환경-변수-설정)
5. [문서 인제스트](#문서-인제스트)
6. [실행 방법](#실행-방법)
7. [프로젝트 구조](#프로젝트-구조)

---

## 시스템 아키텍처

```
질문 입력
    │
    ▼
[retrieve]  ── 하이브리드 검색 (Dense MMR + BM25) → 크로스인코더 재순위화 → 부모 청크 확장
    │
    ▼
[grade_documents]  ── LLM이 각 문서의 관련성 평가
    │
    ├─ 관련 문서 있음 ─────────────────────────────────────────┐
    │                                                         ▼
    └─ 관련 문서 없음 → [transform_query] → [retrieve] → [generate]  ── 답변 생성
                                                              │
                                                    [check_hallucination]  ── 환각 여부 검증
                                                              │
                                              ┌─ 근거 있음 ───┴─── 근거 없음 ─┐
                                              ▼                               ▼
                                       [grade_answer]                   [generate] (재시도)
                                              │
                                  ┌─ 유용함 ──┴── 유용하지 않음 ─┐
                                  ▼                              ▼
                                 END                     [transform_query] → [retrieve]
```

### 핵심 기술 스택

| 구성 요소 | 기술 |
|---|---|
| 에이전트 프레임워크 | LangGraph (StateGraph) |
| LLM | Azure OpenAI (GPT-4o) |
| 임베딩 | Azure OpenAI (text-embedding-3-large) |
| 벡터 DB | ChromaDB (로컬 영구 저장) |
| Dense 검색 | Chroma MMR (Maximal Marginal Relevance) |
| Sparse 검색 | BM25 (키워드 기반, 한국어 명사 검색에 효과적) |
| 재순위화 | Cross-Encoder (BAAI/bge-reranker-v2-m3, 다국어 지원) |
| 청크 전략 | Parent-Child 계층적 청킹 |

---

## 사전 준비 사항

### 1. Python 환경

Python **3.10 이상**이 필요합니다.

```bash
python --version
# Python 3.10.x 이상이어야 합니다.
```

### 2. Azure OpenAI 리소스

Azure Portal에서 다음 두 가지 배포(Deployment)가 생성되어 있어야 합니다.

| 용도 | 권장 모델 |
|---|---|
| 채팅(LLM) | `gpt-4o` 또는 `gpt-4-turbo` |
| 임베딩 | `text-embedding-3-large` |

Azure OpenAI 리소스 생성 방법:
1. [Azure Portal](https://portal.azure.com) 접속
2. **Azure OpenAI** 서비스 생성
3. **모델 배포(Model Deployments)** 메뉴에서 위 두 모델 배포
4. **엔드포인트 URL**과 **API 키** 확인 (설정 → 키 및 엔드포인트)

---

## 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/skcc-junghyunchul/ClaudeCodeTest.git
cd ClaudeCodeTest/skhynix_rag
```

### 2. 가상 환경 생성 (권장)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

> **참고**: `sentence-transformers` 패키지 설치 시 처음 실행 때 크로스인코더 모델(`BAAI/bge-reranker-v2-m3`, 약 1GB)이 자동으로 다운로드됩니다. 인터넷 연결이 필요합니다.

---

## 환경 변수 설정

`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 Azure OpenAI 정보를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 항목을 실제 값으로 채웁니다.

```dotenv
# Azure OpenAI - 채팅 모델
AZURE_OPENAI_API_KEY=여기에_API_키_입력
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o          # 실제 배포 이름으로 변경

# Azure OpenAI - 임베딩 모델
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large   # 실제 배포 이름으로 변경
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-02-01

# 벡터 DB 경로 (기본값 사용 권장)
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=skhynix_market_intel
```

> **주의**: `.env` 파일에는 민감한 정보가 포함되므로 절대 Git에 커밋하지 마세요.

---

## 문서 인제스트

QnA를 실행하기 전에 **반드시 한 번** 문서 인제스트를 수행해야 합니다.  
인제스트는 문서를 청킹하고 임베딩하여 ChromaDB에 저장하는 과정입니다.

```bash
python ingest.py
```

기본적으로 `./data` 디렉토리의 모든 문서(`.md`, `.txt`, `.pdf`, `.docx`)를 처리합니다.  
다른 디렉토리를 지정하려면:

```bash
python ingest.py --data-dir ./your_documents
```

### 나만의 문서 추가하기

`./data` 디렉토리에 파일을 추가한 후 `ingest.py`를 다시 실행하면 됩니다.  
지원 형식: `.md`, `.txt`, `.pdf`, `.docx`

```bash
cp 내_보고서.pdf data/
python ingest.py
```

---

## 실행 방법

### 대화형 모드 (기본)

```bash
python main.py
```

실행 후 프롬프트에 질문을 입력합니다. 종료하려면 `exit` 또는 `quit`를 입력합니다.

```
질문> HBM 시장에서 SK하이닉스의 경쟁 우위는 무엇인가요?

──────────────────────────────────────────────────────────
SK하이닉스는 HBM 시장에서 약 53%의 점유율을 기록하며...
──────────────────────────────────────────────────────────
```

### 단일 질문 모드

```bash
python main.py -q "2025년 SK하이닉스 영업이익률 전망은?"
```

### 데모 모드 (6개 샘플 질문 자동 실행)

```bash
python main.py --demo
```

---

## 프로젝트 구조

```
skhynix_rag/
├── data/
│   └── skhynix_market_intelligence.md   # 샘플 한국어 시장 인텔리전스 문서
├── src/
│   ├── config.py            # 환경 변수 및 RAG 파라미터 설정
│   ├── document_processor.py # 문서 로딩 및 한국어 청킹
│   ├── vectorstore.py       # ChromaDB 관리 (Parent-Child 인제스트)
│   ├── retriever.py         # 하이브리드 검색 + 재순위화
│   ├── prompts.py           # 에이전트 노드별 한국어 프롬프트
│   └── agent_graph.py       # LangGraph CRAG 에이전트 그래프
├── ingest.py                # 문서 인제스트 실행 스크립트
├── main.py                  # QnA 에이전트 실행 진입점
├── requirements.txt
├── .env.example             # 환경 변수 템플릿
└── README.md
```

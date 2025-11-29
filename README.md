# 🤖 Multi-Doc Pro: Multi-Modal RAG Agent v2.0

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?logo=langchain&logoColor=white)
![LlamaParse](https://img.shields.io/badge/LlamaParse-Vision%20AI-000000)
![Ragas](https://img.shields.io/badge/Ragas-Evaluation-orange)
![License](https://img.shields.io/badge/License-Proprietary-red)

> **"Standard RAG is blind to charts. Multi-Doc Pro can see."**

A production-grade Retrieval-Augmented Generation (RAG) system engineered to process **complex financial documents** (like IMF Reports) containing dense tables, charts, and figures. 

Unlike traditional pipelines that strip away visual context, this system uses **Vision-Language Models (GPT-4o-mini)** to parse visual data and **FlashRank Re-ranking** to ensure high-precision retrieval.

---

## 🏗️ System Architecture

The pipeline follows a **"Parse-Then-Chunk"** architecture optimized for preserving document structure.
---<img width="2816" height="1536" alt="Gemini_Generated_Image_2lty912lty912lty" src="https://github.com/user-attachments/assets/f7b34336-d969-49c9-af3e-725a1d6942c8" />

## 📂 Project Structure

```text
MULTI-DOC-PRO/
│
├── 📁 .streamlit/            # Streamlit UI configuration (Theme, Colors)
├── 📁 data/                  # Raw Data Storage (PDFs/Images)
│   └── 📄 qatar_imf.pdf      # (Sample) Financial Report for Testing
├── 📁 persist/               # ChromaDB Vector Store (Persistent Embeddings)
├── 📁 venv/                  # Python Virtual Environment
│
├── 📄 app.py                 # 🚀 Main Application Entry Point (Streamlit UI)
├── 📄 setup_db.py            # 🛠️ Database Initialization Script (Runs Ingestion)
│
├── 🧠 Core Logic Modules
│   ├── 📄 chain_handler.py        # RAG Logic: Query Rewriting, Re-ranking, & Generation
│   ├── 📄 data_loader.py          # Smart Chunking: Markdown & Recursive Splitters
│   ├── 📄 vector_store_handler.py # Database: ChromaDB Management & Embedding
│   ├── 📄 file_handler.py         # Router: Determines file types (PDF vs Text)
│   └── 📄 llama_parser_handler.py # Vision AI: LlamaParse + GPT-4o-mini Integration
│
├── 🔧 Utilities
│   └── 📄 multimodal_utils.py     # Helpers: Markdown Cleanup & Filename Sanitization
│
├── ⚖️ Evaluation Suite
│   ├── 📄 evaluate.py             # Ragas Config: Automated Grading Logic
│   ├── 📄 finish_grading.py       # Safe-Mode Grader: Runs metrics with rate-limiting
│   ├── 📄 pre_eval_backup.csv     # Intermediate results cache
│   └── 📄 evaluation_report.csv   # Final Accuracy Scores
│
├── 📄 requirements.txt       # Project Dependencies
└── 📄 .env                   # API Keys (Excluded from Git)
```
---

## 🌟 Key Features
| **Feature**                 | **Description**                                                    | **Tech Stack**             |
| --------------------------- | ------------------------------------------------------------------ | -------------------------- |
| ``👀 Multi-Modal Vision``   | Parses PDFs as images to extract charts & tables into Markdown.    | LlamaParse, GPT-4o-mini    |
| ``🧠 Smart Chunking ``      | Splits text by Markdown headers to preserve logical sections.      | MarkdownHeaderTextSplitter |
| ``🎯 Precision Retrieval ``| Uses a Two-Stage retrieval process (Vector Search + Reranking).    | ChromaDB, FlashRank        |
|`` 🛡️ Hallucination Guard``| Answers are strictly grounded in retrieved context with citations. | LangChain, Llama-3.3-70b   |
|`` ⚖️ Automated Eval``     | Self-grading pipeline to measure Faithfulness and Accuracy.        | Ragas Framework            |

---
##🚀 Getting Started(local deployment)

### 🛠️ Requirements
* Python 3.10+
* A `GROQ_API_KEY` (for fast inference)
* A `GOOGLE_API_KEY` (for embeddings)
* A `LLAMA_CLOUD_API_KEY` (for llama parse)

### Setup

1.  **Clone the repository & install dependencies:**
    ```bash
    git clone [https://github.com/Madhan-mohan14/Multi-Modal-Rag.git](https://github.com/Madhan-mohan14/Multi-Modal-Rag.git)
    cd Multi-Modal-Rag
    pip install -r requirements.txt 
    ```

2.  **Set Environment Variables:** Create a `.env` file in the root directory:
       ```
    GROQ_API_KEY="YOUR_GROQ_KEY"
    GOOGLE_API_KEY="YOUR_GOOGLE_KEY"
    LLAMA_CLOUD_API_KEY="YOUR_GOOGLE_KEY"
    ```
3.  **Run the App:**
4.  first you have to run the setup_db.py to load the documents which we have and pre index them 
    ```bash
     python setup_db.py
    streamlit run app.py
    ```
5. for evaluation
   ```bash
   python finish_grading.py
   ```
---
## 📊 Evaluation Results
This system was rigorously tested using the Ragas framework against a "Golden Dataset" derived from the IMF Qatar Article IV Report.

🏆 Score: 78% Faithfulness
The system demonstrates high reliability (0.78), correctly refusing to answer when data is missing (e.g., specific regulatory measures vs. objectives), effectively mitigating hallucinations.

## 📜 License
Proprietary & Confidential. Copyright (c) 2025 Madhan Mohan. All Rights Reserved.

Permitted: Viewing for educational/evaluation purposes.

Prohibited: Commercial use, modification, or redistribution without permission.
## 🔮 Future Roadmap

* **🗣️ Audio Querying:** Integrate OpenAI Whisper to allow users to ask questions via voice instead of typing.
* **🔍 Hybrid Search:** Implement BM25 + Vector Search (Reciprocal Rank Fusion) to better capture specific acronyms and keywords.
* **☁️ Cloud Native:** Dockerize the application and migrate to serverless vector databases (Pinecone/Weaviate) for enterprise scaling.
* **🕸️ GraphRAG:** Add Knowledge Graphs to link disconnected entities across documents for complex multi-hop reasoning.
* **🕵️ Agentic Workflow:** Upgrade to LangGraph agents that can self-correct poor retrieval results and validate data with external tools and also user can ask related to the previous questions and maintain history.

# Chat-AI-AGENT"# Chat-AI-AGENT" 
# 🤖 Multi Utility Chatbot — LangGraph + RAG + Tools

A conversational AI assistant built with **LangGraph**, **LangChain** and **Streamlit**. It can answer questions about your own PDFs (RAG), search the web, fetch live stock prices and do arithmetic, while keeping a separate, persistent memory for every chat thread.

---

## ✨ Features

- **Chat with your PDF**: upload a PDF per conversation and ask questions; answers are grounded in retrieved chunks (FAISS + MMR retrieval).
- **Tool-calling agent**: the LLM decides when to call a tool, using a LangGraph `ToolNode` loop:
  - 🔎 **Web search** (DuckDuckGo)
  - 📈 **Stock prices** (Alpha Vantage)
  - 🧮 **Calculator** (add, sub, mul, div)
  - 📄 **RAG tool** (per-thread document retrieval)
- **Multi-thread conversations**: every chat has its own thread ID, and its own indexed document.
- **Persistent memory**: conversation state is stored in SQLite via LangGraph's `SqliteSaver`, so past chats can be reloaded from the sidebar.
- **Streaming responses** with a live status indicator when a tool is running.

---

## 🏗️ Architecture

```
            ┌────────────┐
 START ───▶ │  chat_node │ ◀──────────────┐
            │ (LLM + tools bound)          │
            └─────┬──────┘                 │
        tool call?│                        │
         ┌────────┴────────┐               │
         ▼                 ▼               │
        END          ┌───────────┐         │
                     │ tool_node │ ────────┘
                     └───────────┘
   search · stock price · calculator · rag_tool
```

**PDF ingestion flow:** PDF upload → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap) → `all-MiniLM-L6-v2` embeddings → FAISS index → MMR retriever stored per thread.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph, LangChain |
| LLM | Google Gemini (`ChatGoogleGenerativeAI`); Hugging Face Llama 3.1 8B optional |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| Memory | SQLite (`SqliteSaver`) |
| Frontend | Streamlit |
| Tools | DuckDuckGo Search, Alpha Vantage API |

---

## 📁 Project Structure

```
├── langgraph_rag_backend.py     # LLM, tools, graph, PDF ingestion, checkpointer
├── streamlit_rag_frontend.py    # Streamlit UI (chat, sidebar, PDF upload, threads)
├── chatbot.db                   # SQLite conversation store (auto-created)
├── requirements.txt
├── .env                         # API keys (not committed)
└── README.md
```

> Rename the files above to match your actual filenames.

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install streamlit langgraph langgraph-checkpoint-sqlite \
  langchain langchain-core langchain-community langchain-text-splitters \
  langchain-huggingface langchain-google-genai \
  faiss-cpu pypdf sentence-transformers duckduckgo-search \
  python-dotenv requests
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
```

### 5. Run the app
```bash
streamlit run streamlit_rag_frontend.py
```

---

## 💡 Usage

1. Open the app and click **New Chat** (or continue a past conversation from the sidebar).
2. **Upload a PDF** in the sidebar and wait for "PDF indexed".
3. Ask things like:
   - *"Summarize the key points of this document."*
   - *"What is the current stock price of TSLA?"*
   - *"What is 245 × 18?"*
   - *"Search the web for the latest news on LangGraph."*

---

## 🔮 Future Improvements

- Persist FAISS indexes to disk so documents survive restarts
- Support multiple documents per thread and more file types
- Show source chunks/page numbers alongside answers
- Add more tools (weather, email, code execution)
- Docker setup and cloud deployment

---

## 👤 Author

**Sayan Paul**: CSE student at Heritage Institute of Technology, Kolkata, focused on deep learning, LLMs and NLP.

---

## 📄 License

This project is licensed under the MIT License.
# 🧮 RAG-Powered Math Problem Solver & Tutor

> Built with: Python · LangChain · LangGraph · FAISS · ChromaDB · HuggingFace · Gemini · FastAPI · Streamlit

---

## 🏗️ Project Architecture

```
Math-problem-solver/
│
├── knowledge_base/          # Curated math knowledge (Algebra, Calculus, Statistics)
│   ├── algebra.txt
│   ├── calculus.txt
│   └── statistics.txt
│
├── rag_pipeline.py          # FAISS vector store + HuggingFace embeddings
├── agent.py                 # LangGraph agentic reasoning (decompose → solve → synthesize)
├── memory.py                # ChromaDB conversation memory
├── main.py                  # FastAPI backend (REST API)
├── streamlit_app.py         # Streamlit frontend (interactive UI)
│
├── faiss_index/             # Auto-created: FAISS vector index
├── chroma_memory/           # Auto-created: ChromaDB persistent storage
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Instructions

### Step 1 — Install Python dependencies
Make sure you have Python 3.8+ installed on your system.

### Step 2 — Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your Gemini API key
```bash

cp .env


```

### Step 5 — Build the FAISS index (first time only)
```bash
python rag_pipeline.py
```
This embeds the math knowledge base using HuggingFace and saves a FAISS index locally.

---

## 🚀 Running the Project

You need **two terminals** running simultaneously:

### Terminal 1 — Start FastAPI backend
```bash
python main.py
```
API will be live at: http://localhost:8000
API docs at: http://localhost:8000/docs

### Terminal 2 — Start Streamlit frontend
```bash
streamlit run streamlit_app.py
```
UI will open at: http://localhost:8501

---

## 🧠 How It Works

### RAG Pipeline (rag_pipeline.py)
1. Loads math knowledge from `.txt` files (algebra, calculus, statistics)
2. Embeds each chunk using `all-MiniLM-L6-v2` (HuggingFace, free, runs locally)
3. Stores in FAISS vector index for fast similarity search
4. On each query, retrieves top-4 most relevant chunks

### LangGraph Agent (agent.py)
The graph has 4 nodes:
```
retrieve → decompose → solve_steps → synthesize
```
- **retrieve**: Gets relevant formulas/theorems from FAISS
- **decompose**: Asks Gemini to break the problem into sub-steps
- **solve_steps**: Solves each sub-step individually with Gemini
- **synthesize**: Combines everything into a final clear answer

### Memory (memory.py)
- Uses ChromaDB to persist conversation history
- Each session gets a unique ID
- Relevant past exchanges are retrieved and passed to the agent as context
- Enables coherent multi-turn tutoring sessions

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/solve` | Solve a math problem |
| GET | `/history/{session_id}` | Get session chat history |
| POST | `/clear` | Clear session memory |
| GET | `/health` | Health check |

### Example API call:
```python
import requests

response = requests.post("http://localhost:8000/solve", json={
    "question": "Solve x^2 - 5x + 6 = 0",
    "session_id": "student_001"
})

data = response.json()
print(data["final_answer"])
```

---

## 💡 Example Questions to Try

- `Solve x² - 5x + 6 = 0 using the quadratic formula`
- `Find the derivative of x³ sin(x)`
- `What is the probability of getting exactly 3 heads in 5 coin flips?`
- `Integrate x² from 0 to 3`
- `Find the sum of the first 20 terms of AP: 2, 5, 8...`
- `Explain the Central Limit Theorem with an example`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Conversation Memory | ChromaDB |
| Agentic Workflow | LangGraph |
| Orchestration | LangChain |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Language | Python 3.8+ |

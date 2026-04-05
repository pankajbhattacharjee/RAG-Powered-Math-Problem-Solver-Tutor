"""
main.py  —  FastAPI Backend
Endpoints:
  POST /solve        - solve a math problem (with session memory)
  GET  /history      - get chat history for a session
  POST /clear        - clear session memory
  GET  /health       - health check
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from agent import solve_math_problem
from memory import ConversationMemory

app = FastAPI(
    title="RAG Math Tutor API",
    description="AI-powered math tutor using RAG + LangGraph + Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (maps session_id -> ConversationMemory)
sessions: dict[str, ConversationMemory] = {}


def get_or_create_session(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory(session_id=session_id)
    return sessions[session_id]


# ── Request / Response Models ─────────────────────────────────────────────────

class SolveRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class StepSolution(BaseModel):
    step_number: int
    step_description: str
    solution: str

class SolveResponse(BaseModel):
    question: str
    sub_steps: List[str]
    step_solutions: List[str]
    final_answer: str
    session_id: str
    total_exchanges: int

class HistoryResponse(BaseModel):
    session_id: str
    history: List[dict]
    total_exchanges: int

class ClearRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "message": "Math Tutor API is running"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session = get_or_create_session(req.session_id)
    chat_history = session.get_chat_history()

    try:
        result = solve_math_problem(req.question, chat_history=chat_history)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        error_text = str(e)
        if "API key not valid" in error_text or "API_KEY_INVALID" in error_text:
            error_text = (
                "Agent error: Invalid Google API key. "
                "Please set a valid GOOGLE_API_KEY in your .env or environment."
            )
        raise HTTPException(status_code=500, detail=error_text)

    # Store in memory
    session.add_exchange(req.question, result["final_answer"])

    return SolveResponse(
        question=req.question,
        sub_steps=result["sub_steps"],
        step_solutions=result["step_solutions"],
        final_answer=result["final_answer"],
        session_id=req.session_id,
        total_exchanges=session.message_count
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    session = get_or_create_session(session_id)
    return HistoryResponse(
        session_id=session_id,
        history=session.get_chat_history(),
        total_exchanges=session.message_count
    )


@app.post("/clear")
def clear_session(req: ClearRequest):
    session = get_or_create_session(req.session_id)
    session.clear_session()
    return {"message": f"Session '{req.session_id}' cleared successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

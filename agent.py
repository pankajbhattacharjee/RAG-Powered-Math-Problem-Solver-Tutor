"""
agent.py
LangGraph agentic reasoning workflow.
Breaks complex math problems into sub-steps, solves each with Gemini,
synthesizes final explained solution.
"""

import os
from typing import TypedDict, List, Annotated
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from rag_pipeline import build_faiss_index, retrieve_context


# ── State definition ──────────────────────────────────────────────────────────

class MathState(TypedDict):
    question: str
    context: str
    sub_steps: List[str]
    step_solutions: Annotated[List[str], operator.add]
    final_answer: str
    chat_history: List[dict]


# ── LLM setup ─────────────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Please add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )


# ── Graph nodes ───────────────────────────────────────────────────────────────

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_faiss_index()
    return _vectorstore


def retrieve_node(state: MathState) -> dict:
    """Node 1: Retrieve relevant math knowledge from FAISS."""
    print("[Agent] Retrieving context from FAISS...")
    vs = get_vectorstore()
    context = retrieve_context(state["question"], vs, k=4)
    return {"context": context}


def decompose_node(state: MathState) -> dict:
    """Node 2: Break the problem into logical sub-steps."""
    print("[Agent] Decomposing problem into sub-steps...")
    llm = get_llm()

    system = SystemMessage(content="""You are a math tutor. 
Your job is to break a math problem into clear, sequential sub-steps.
Return ONLY a numbered list of sub-steps (no solutions yet).
Each sub-step should be one clear action.
Example:
1. Identify the type of equation
2. Rearrange to standard form
3. Apply the quadratic formula
4. Simplify the result""")

    history_text = ""
    if state.get("chat_history"):
        recent = state["chat_history"][-4:]
        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
        history_text = f"\nConversation context:\n{history_text}\n"

    user = HumanMessage(content=f"""
{history_text}
Math Problem: {state['question']}

Relevant formulas and theorems:
{state['context']}

Break this problem into numbered sub-steps:""")

    response = llm.invoke([system, user])
    raw = response.content.strip()

    # Parse numbered list
    sub_steps = []
    for line in raw.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                sub_steps.append(cleaned)

    if not sub_steps:
        sub_steps = [state["question"]]

    print(f"[Agent] Decomposed into {len(sub_steps)} sub-steps")
    return {"sub_steps": sub_steps}


def solve_steps_node(state: MathState) -> dict:
    """Node 3: Solve all sub-steps in one call to reduce API usage."""
    print("[Agent] Solving all sub-steps...")
    llm = get_llm()

    sub_steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(state["sub_steps"])])

    system = SystemMessage(content="""You are an expert math tutor.
Solve each sub-step clearly and concisely, showing your working for each.
Use plain text math notation. Number your solutions to match the sub-steps.""")

    user = HumanMessage(content=f"""
Original problem: {state['question']}

Relevant formulas:
{state['context']}

Sub-steps to solve:
{sub_steps_text}

Solve each sub-step:""")

    response = llm.invoke([system, user])
    full_solution = response.content.strip()

    # Split the response into individual step solutions
    solutions = []
    lines = full_solution.split('\n')
    current_step = []
    for line in lines:
        if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
            if current_step:
                solutions.append('\n'.join(current_step))
            current_step = [line]
        else:
            current_step.append(line)
    if current_step:
        solutions.append('\n'.join(current_step))

    # If parsing failed, treat as one solution
    if len(solutions) != len(state["sub_steps"]):
        solutions = [f"Step {i+1} — {step}:\n{full_solution}" for i, step in enumerate(state["sub_steps"])]

    print(f"[Agent] Solved {len(solutions)} steps in one call")
    return {"step_solutions": solutions}


def synthesize_node(state: MathState) -> dict:
    """Node 4: Synthesize all steps into a final explained answer."""
    print("[Agent] Synthesizing final answer...")
    llm = get_llm()

    steps_text = "\n\n".join(state["step_solutions"])

    system = SystemMessage(content="""You are a friendly math tutor.
Synthesize the step-by-step solutions into one clear, well-structured final answer.
Format:
- Brief explanation of the approach
- Key steps (summarized)
- Final Answer clearly stated
- Any important notes or common mistakes to avoid""")

    user = HumanMessage(content=f"""
Problem: {state['question']}

Step-by-step solutions:
{steps_text}

Synthesize into a clear final answer:""")

    response = llm.invoke([system, user])
    return {"final_answer": response.content.strip()}


# ── Build the graph ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(MathState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("solve_steps", solve_steps_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decompose")
    graph.add_edge("decompose", "solve_steps")
    graph.add_edge("solve_steps", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ── Public interface ───────────────────────────────────────────────────────────

_compiled_graph = None

def solve_math_problem(question: str, chat_history: list = None) -> dict:
    """
    Main entry point. Takes a math question and optional chat history.
    Returns dict with sub_steps, step_solutions, and final_answer.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()

    initial_state: MathState = {
        "question": question,
        "context": "",
        "sub_steps": [],
        "step_solutions": [],
        "final_answer": "",
        "chat_history": chat_history or []
    }

    result = _compiled_graph.invoke(initial_state)
    return {
        "question": question,
        "sub_steps": result["sub_steps"],
        "step_solutions": result["step_solutions"],
        "final_answer": result["final_answer"],
        "context_used": result["context"]
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    result = solve_math_problem("Solve x^2 - 5x + 6 = 0 using the quadratic formula")
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
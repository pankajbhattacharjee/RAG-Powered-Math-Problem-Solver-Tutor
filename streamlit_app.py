"""
streamlit_app.py  —  Streamlit Frontend
Real-time interactive math tutoring UI.
"""

import streamlit as st
import requests
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Math Tutor AI",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ─────────────────────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "total_exchanges" not in st.session_state:
    st.session_state.total_exchanges = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📐 Math Tutor AI")
    st.caption("Powered by RAG + LangGraph + Gemini")
    st.divider()

    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.markdown(f"**Questions asked:** {st.session_state.total_exchanges}")

    st.divider()
    st.markdown("### 💡 Try these examples")

    examples = [
        "Solve x² - 5x + 6 = 0",
        "Find the derivative of x³ sin(x)",
        "What is the probability of getting 3 heads in 5 coin flips?",
        "Integrate x² from 0 to 3",
        "Find the sum of first 20 terms of AP: 2, 5, 8...",
        "Solve using logarithm: 2^x = 32",
    ]

    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🗑️ Clear Session", use_container_width=True, type="secondary"):
        try:
            requests.post(f"{API_URL}/clear", json={"session_id": st.session_state.session_id})
        except:
            pass
        st.session_state.chat_history = []
        st.session_state.total_exchanges = 0
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.divider()
    st.markdown("### 📚 Topics Covered")
    st.markdown("- 🔢 **Algebra** (quadratics, logs, sequences)")
    st.markdown("- 📈 **Calculus** (derivatives, integrals, limits)")
    st.markdown("- 📊 **Statistics** (probability, distributions)")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🧮 RAG-Powered Math Problem Solver")
st.caption("Ask any math question — I'll break it into steps and explain each one!")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{msg['content']}**")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Show sub-steps if present
            if "sub_steps" in msg:
                with st.expander("🔍 Problem Breakdown (Sub-steps)", expanded=False):
                    for i, step in enumerate(msg["sub_steps"], 1):
                        st.markdown(f"**Step {i}:** {step}")

            if "step_solutions" in msg:
                with st.expander("📝 Detailed Step-by-Step Solutions", expanded=False):
                    for sol in msg["step_solutions"]:
                        st.markdown(sol)
                        st.divider()

            st.markdown("### ✅ Final Answer")
            st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────────────────────

# Handle sidebar example button click
pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a math question... e.g. 'Solve x² - 5x + 6 = 0'") or pending

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(f"**{question}**")

    # Call API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking step by step..."):
            try:
                response = requests.post(
                    f"{API_URL}/solve",
                    json={"question": question, "session_id": st.session_state.session_id},
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()

                sub_steps = data.get("sub_steps", [])
                step_solutions = data.get("step_solutions", [])
                final_answer = data.get("final_answer", "")
                st.session_state.total_exchanges = data.get("total_exchanges", 0)

                # Display
                with st.expander("🔍 Problem Breakdown (Sub-steps)", expanded=True):
                    for i, step in enumerate(sub_steps, 1):
                        st.markdown(f"**Step {i}:** {step}")

                with st.expander("📝 Detailed Step-by-Step Solutions", expanded=False):
                    for sol in step_solutions:
                        st.markdown(sol)
                        st.divider()

                st.markdown("### ✅ Final Answer")
                st.markdown(final_answer)

                # Save to history
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_answer,
                    "sub_steps": sub_steps,
                    "step_solutions": step_solutions
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running: `python main.py`")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The problem may be complex — try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.rerun()

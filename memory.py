"""
memory.py
ChromaDB-based conversation memory.
Stores and retrieves past messages for multi-turn tutoring sessions.
"""

import os
import uuid
import chromadb
from datetime import datetime
from pathlib import Path

CHROMA_PATH = str(Path(__file__).parent / "chroma_memory")


class ConversationMemory:
    """Manages multi-turn conversation context using ChromaDB."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="math_tutor_memory",
            metadata={"hnsw:space": "cosine"}
        )
        self._chat_history = []  # in-memory list for current session
        print(f"[Memory] Session ID: {self.session_id}")

    def add_exchange(self, question: str, answer: str):
        """Store a Q&A exchange in ChromaDB."""
        exchange_id = f"{self.session_id}_{len(self._chat_history)}"
        doc = f"Q: {question}\nA: {answer}"

        try:
            self.collection.add(
                documents=[doc],
                ids=[exchange_id],
                metadatas=[{
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "question": question[:200]
                }]
            )
            print(f"[Memory] Stored exchange {exchange_id}")
        except Exception as e:
            print(f"[Memory] Failed to store exchange (continuing): {e}")

        # Always update in-memory history
        self._chat_history.append({"role": "user", "content": question})
        self._chat_history.append({"role": "assistant", "content": answer})

    def get_relevant_history(self, current_question: str, n_results: int = 3) -> str:
        """Retrieve past exchanges relevant to the current question."""
        total = self.collection.count()
        if total == 0:
            return ""

        results = self.collection.query(
            query_texts=[current_question],
            n_results=min(n_results, total),
            where={"session_id": self.session_id}
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        history_parts = []
        for doc in results["documents"][0]:
            history_parts.append(doc)

        return "\n\n".join(history_parts)

    def get_chat_history(self) -> list:
        """Return the current session's chat history as a list of dicts."""
        return self._chat_history.copy()

    def clear_session(self):
        """Clear current session from ChromaDB."""
        results = self.collection.get(where={"session_id": self.session_id})
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        self._chat_history = []
        print(f"[Memory] Session {self.session_id} cleared.")

    @property
    def message_count(self) -> int:
        return len(self._chat_history) // 2  # pairs of Q&A


if __name__ == "__main__":
    mem = ConversationMemory(session_id="test123")
    mem.add_exchange(
        "What is the quadratic formula?",
        "x = (-b ± sqrt(b^2 - 4ac)) / 2a"
    )
    mem.add_exchange(
        "How do I find the discriminant?",
        "The discriminant is D = b^2 - 4ac"
    )
    relevant = mem.get_relevant_history("solve x^2 - 4 = 0")
    print("Relevant history:", relevant)

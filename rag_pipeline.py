"""
rag_pipeline.py
Core RAG pipeline: loads math knowledge base, embeds with HuggingFace,
stores in FAISS vector store, retrieves relevant context for queries.
"""

import os
from pathlib import Path
from typing import List
import shutil

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge_base"
FAISS_INDEX_PATH = Path(__file__).parent / "faiss_index"


def load_knowledge_base() -> List[Document]:
    """Load all .txt files from the knowledge_base directory as Documents."""
    docs = []
    for txt_file in KNOWLEDGE_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        # Split by double newline to create chunks
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata={"source": txt_file.stem, "topic": txt_file.stem.capitalize()}
            ))
    print(f"[RAG] Loaded {len(docs)} knowledge chunks from {KNOWLEDGE_DIR}")
    return docs


def get_embeddings():
    """Return HuggingFace sentence-transformer embeddings (free, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_faiss_index(force_rebuild: bool = False) -> FAISS:
    """Build or load FAISS vector index from knowledge base."""
    embeddings = get_embeddings()

    if FAISS_INDEX_PATH.exists() and not force_rebuild:
        print("[RAG] Loading existing FAISS index...")
        try:
            return FAISS.load_local(
                str(FAISS_INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[RAG] Failed to load existing FAISS index: {e}")
            print("[RAG] Removing corrupt/incompatible index and rebuilding...")
            try:
                if FAISS_INDEX_PATH.is_dir():
                    shutil.rmtree(FAISS_INDEX_PATH)
                else:
                    FAISS_INDEX_PATH.unlink()
            except Exception as cleanup_error:
                print(f"[RAG] Warning: could not delete old FAISS index: {cleanup_error}")

    print("[RAG] Building new FAISS index...")
    docs = load_knowledge_base()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"[RAG] FAISS index saved to {FAISS_INDEX_PATH}")
    return vectorstore


def retrieve_context(query: str, vectorstore: FAISS, k: int = 4) -> str:
    """Retrieve top-k relevant chunks for a math query."""
    results = vectorstore.similarity_search(query, k=k)
    context_parts = []
    for doc in results:
        context_parts.append(f"[{doc.metadata['topic']}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Quick test
    vs = build_faiss_index(force_rebuild=True)
    ctx = retrieve_context("quadratic equation formula", vs)
    print("\n=== Retrieved Context ===")
    print(ctx[:500])
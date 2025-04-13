from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGStore:
    """
    RAG system using FAISS for fast vector search and sentence-transformers for embeddings.
    Stores embeddings per session and retrieves top-k chunks per user query.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load a pre-trained sentence transformer (MiniLM output = 384-d)
        self.model = SentenceTransformer(model_name)
        self.session_indexes: Dict[str, faiss.IndexFlatL2] = {}
        self.session_chunks: Dict[str, List[str]] = {}

    def add_context(self, session_id: str, text: str, chunk_size: int = 500):
        """
        Split and embed raw text, and store in session-specific FAISS index.
        Args:
            session_id (str): Active chat session
            text (str): Raw uploaded context (e.g., .txt content)
            chunk_size (int): Approx chunk size in characters
        """
        if session_id not in self.session_indexes:
            self.session_indexes[session_id] = faiss.IndexFlatL2(384)
            self.session_chunks[session_id] = []

        chunks = self._chunk_text(text, chunk_size)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        self.session_indexes[session_id].add(embeddings)
        self.session_chunks[session_id].extend(chunks)

    def retrieve(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k relevant context chunks for a given session and query.
        Returns:
            List[str]: Ranked chunks most similar to query
        """
        if session_id not in self.session_indexes:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.session_indexes[session_id].search(query_vec, top_k)
        return [self.session_chunks[session_id][i] for i in I[0] if i < len(self.session_chunks[session_id])]

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Chunk text into approx. fixed-size segments by paragraph.
        Returns:
            List[str]: Clean text chunks
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
                current += para + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = para + "\n\n"

        if current:
            chunks.append(current.strip())

        return chunks


# Global store to use across app (singleton pattern)
rag_store = RAGStore()

# app/services/rag.py
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGStore:
    """
    RAG system using FAISS for fast vector search and sentence-transformers for embeddings.
    Stores embeddings per session and retrieves top-k chunks with source metadata for referencing.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        # Store FAISS index and associated metadata-rich chunks per session
        self.session_indexes: Dict[str, faiss.IndexFlatL2] = {}
        self.session_chunks: Dict[str, List[Dict[str, str]]] = {}  # text + source metadata
        self.session_files: Dict[str, str] = {}  # latest uploaded file name per session

    def add_context(self, session_id: str, text: str, filename: str = "user_uploaded_file", chunk_size: int = 500):
        """
        Splits and embeds raw uploaded text, stores it in a session-specific FAISS index.

        Args:
            session_id (str): Unique session identifier (e.g. chat ID)
            text (str): Raw text extracted from the uploaded document
            filename (str): Name of the uploaded file (used for reference)
            chunk_size (int): Max characters per chunk (approx.)
        """
        if session_id not in self.session_indexes:
            self.session_indexes[session_id] = faiss.IndexFlatL2(384)
            self.session_chunks[session_id] = []

        # Step 1: Chunk and embed
        chunks = self._chunk_text(text, chunk_size)
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        # Step 2: Add metadata to each chunk
        chunk_objs = [{"text": chunk, "source": filename} for chunk in chunks]

        # Step 3: Store
        self.session_indexes[session_id].add(embeddings)
        self.session_chunks[session_id].extend(chunk_objs)
        self.session_files[session_id] = filename

    def retrieve(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k relevant context chunks with source references.

        Returns:
            List[str]: Formatted context with document references for LLM to cite.
        """
        if session_id not in self.session_indexes:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.session_indexes[session_id].search(query_vec, top_k)

        results = []
        for i in I[0]:
            if i < len(self.session_chunks[session_id]):
                chunk_obj = self.session_chunks[session_id][i]
                formatted = f"[Context from {chunk_obj['source']}]\n{chunk_obj['text']}"
                results.append(formatted)
        return results

    def get_latest_doc_name(self, session_id: str) -> str:
        """
        Returns the most recently uploaded document name for a given session.
        """
        return self.session_files.get(session_id, "user_uploaded_document")

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Splits text into paragraph-based chunks of approx. the given size.
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
    
    def get_brief_summary(self, session_id: str, max_lines: int = 3) -> str:
        """
        Returns a short, readable summary of the most recently uploaded document for session.
        This is *not* a true semantic summary — just a token-efficient preview.

        Args:
            session_id (str): Session identifier
            max_lines (int): Max number of lines to include in the summary

        Returns:
            str: Short summary with filename reference
        """
        chunks = self.session_chunks.get(session_id)
        if not chunks:
            return ""

        # Grab the first chunk (most representative intro)
        first_chunk = chunks[0]["text"]
        lines = first_chunk.strip().splitlines()
        snippet = "\n".join(lines[:max_lines])

        source = chunks[0]["source"]
        return f"[Summary from {source}]\n{snippet}"

# Singleton instance
rag_store = RAGStore()

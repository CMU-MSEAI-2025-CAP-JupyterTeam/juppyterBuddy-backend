from typing import Dict, List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGStore:
    """
    RAG system using FAISS for fast vector search and sentence-transformers for embeddings.
    Stores embeddings per session and retrieves top-k chunks per user query.

    Now also tracks the filename of the most recently uploaded document per session
    to allow referencing it in the response.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load a pre-trained sentence transformer model (MiniLM gives 384-d embeddings)
        self.model = SentenceTransformer(model_name)

        # Dictionary to store a FAISS index for each session
        self.session_indexes: Dict[str, faiss.IndexFlatL2] = {}

        # Dictionary to store the corresponding text chunks per session
        self.session_chunks: Dict[str, List[str]] = {}

        # ✅ NEW: Dictionary to store the last uploaded filename per session
        self.session_files: Dict[str, str] = {}

    def add_context(self, session_id: str, text: str, filename: str = "user_uploaded.pdf", chunk_size: int = 500):
        """
        Splits and embeds raw uploaded text, stores it in a session-specific FAISS index.

        Args:
            session_id (str): Unique session identifier (e.g. chat ID)
            text (str): Raw text extracted from the uploaded document
            filename (str): Name of the uploaded file (used for reference)
            chunk_size (int): Max characters per chunk (approx.)
        """
        # Initialize new FAISS index and chunk list if session is new
        if session_id not in self.session_indexes:
            self.session_indexes[session_id] = faiss.IndexFlatL2(384)
            self.session_chunks[session_id] = []

        # Break the text into paragraphs (approximate size)
        chunks = self._chunk_text(text, chunk_size)

        # Embed all chunks using the sentence transformer model
        embeddings = self.model.encode(chunks, convert_to_numpy=True)

        # Store embeddings and chunks in the FAISS index and internal memory
        self.session_indexes[session_id].add(embeddings)
        self.session_chunks[session_id].extend(chunks)

        # ✅ Store the name of the uploaded file
        self.session_files[session_id] = filename

    def retrieve(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k relevant context chunks for a given session and query.

        Args:
            session_id (str): Session ID for document context
            query (str): User query
            top_k (int): Number of most relevant chunks to retrieve

        Returns:
            List[str]: List of context strings ranked by semantic similarity
        """
        if session_id not in self.session_indexes:
            return []

        # Convert query into vector
        query_vec = self.model.encode([query], convert_to_numpy=True)

        # Search the index for the most relevant chunks
        D, I = self.session_indexes[session_id].search(query_vec, top_k)

        # Return the text chunks for the top-k results
        return [self.session_chunks[session_id][i] for i in I[0] if i < len(self.session_chunks[session_id])]

    def get_latest_doc_name(self, session_id: str) -> str:
        """
        Returns the most recently uploaded document name for a given session.

        Args:
            session_id (str): Session ID

        Returns:
            str: Filename of the latest uploaded document (fallback: 'user_uploaded.pdf')
        """
        return self.session_files.get(session_id, "user_uploaded.pdf")

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Splits text into paragraph-based chunks of approx. the given size.

        Args:
            text (str): Raw input text
            chunk_size (int): Approx max size per chunk

        Returns:
            List[str]: Cleaned and sized chunks
        """
        paragraphs = text.split("\n\n")  # paragraph separation
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


# ✅ Singleton instance that can be used across the backend
rag_store = RAGStore()

from langchain_core.tools import tool
from app.services.rag import rag_store 

@tool
def retrieve_context(query: str, session_id: str) -> str:
    """
    Retrieve relevant uploaded context documents for a given query using RAG store.
    This internal tool is available to the LLM and used only when appropriate.

    Args:
        query (str): The user's query or question.
        session_id (str): The user's session identifier to look up stored context.

    Returns:
        str: Top retrieved snippets from the user's uploaded documents.
    """
    results = rag_store.retrieve(session_id, query, top_k=3)  # ✅ FIXED METHOD NAME
    if not results:
        return "No relevant context was found for this query."
    return "\n\n".join(f"[Context]\n{r}" for r in results)


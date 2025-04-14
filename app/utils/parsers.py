# app/utils/parsers.py

import fitz  # PyMuPDF
import io

def extract_text_from_pdf(binary_data: bytes) -> str:
    """
    Extract text from a binary PDF using PyMuPDF (fitz).

    Args:
        binary_data (bytes): The raw PDF file content.

    Returns:
        str: Extracted text from all pages.
    """
    text = []
    try:
        with fitz.open(stream=io.BytesIO(binary_data), filetype="pdf") as doc:
            for page in doc:
                text.append(page.get_text())
    except Exception as e:
        raise RuntimeError(f"PDF parsing failed: {str(e)}")

    return "\n\n".join(text)

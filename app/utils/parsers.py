<<<<<<< HEAD
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
=======
import fitz  # PyMuPDF import io import base64 import json import yaml import nbformat from docx import Document from typing import Optional def extract_text(content: str, mime: str) -> Optional[str]: """ Generic text extractor from base64 content based on MIME type. Args: content (str): The base64-encoded content. mime (str): The MIME type of the file. Returns: Optional[str]: Extracted plain text, or None if unsupported or failed. """ try: if mime.startswith("text/"): # Already plain text, decode if it's base64 try: decoded_content = base64.b64decode(content).decode('utf-8') return decoded_content except (base64.binascii.Error, UnicodeDecodeError): return content  # Return as is if it's not base64 encoded elif mime == "application/pdf": binary_data = base64.b64decode(content) with fitz.open(stream=io.BytesIO(binary_data), filetype="pdf") as doc: return "\n\n".join([page.get_text() for page in doc]) elif mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]: # DOCX parsing binary_data = base64.b64decode(content) doc = Document(io.BytesIO(binary_data)) return "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text]) elif mime == "application/json": # JSON parsing try: binary_data = base64.b64decode(content) data = json.loads(binary_data) return json.dumps(data, indent=2) except base64.binascii.Error: # Try direct parsing if not base64 data = json.loads(content) return json.dumps(data, indent=2) elif mime == "application/x-yaml": # YAML parsing try: binary_data = base64.b64decode(content) data = yaml.safe_load(binary_data) return yaml.dump(data, default_flow_style=False) except base64.binascii.Error: # Try direct parsing if not base64 data = yaml.safe_load(content) return yaml.dump(data, default_flow_style=False) elif mime == "application/x-ipynb+json": # Jupyter Notebook parsing using nbformat try: binary_data = base64.b64decode(content) notebook = nbformat.reads(binary_data.decode('utf-8'), as_version=4) except base64.binascii.Error: # Try direct parsing if not base64 notebook = nbformat.reads(content, as_version=4) # Extract text from markdown and code cells text_parts = [] for cell in notebook.cells: cell_type = cell.cell_type source = cell.source if cell_type == 'markdown': text_parts.append(f"[Markdown Cell]\n{source}") elif cell_type == 'code': text_parts.append(f"[Code Cell]\n{source}") # Include output if available if hasattr(cell, 'outputs'): for output in cell.outputs: if 'text' in output: text_content = ''.join(output.text) if isinstance(output.text, list) else output.text text_parts.append(f"[Output]\n{text_content}") elif 'data' in output and 'text/plain' in output.data: text_parts.append(f"[Output]\n{output.data['text/plain']}") return "\n\n".join(text_parts) except Exception as e: print(f"Error extracting text: {str(e)}") return None
>>>>>>> a815a3f (hold backend)

"""
JupyterBuddy Helper Utilities

This module provides utility functions for the JupyterBuddy application.
"""
import json
import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks from text.
    
    Args:
        text: Text to extract code blocks from
        
    Returns:
        List of tuples containing (language, code)
    """
    # Regular expression for code blocks: ```language\ncode\n```
    pattern = r'```(\w*)\n([\s\S]*?)\n```'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    return matches

def format_notebook_context(notebook_context: Optional[Dict[str, Any]]) -> str:
    """
    Format notebook context for inclusion in prompts.
    
    Args:
        notebook_context: Notebook context to format
        
    Returns:
        Formatted notebook context as a string
    """
    if not notebook_context:
        return "No active notebook"
    
    # Start with notebook metadata
    formatted = [
        f"Notebook: {notebook_context.get('title', 'Untitled')}",
        f"Path: {notebook_context.get('path', 'Unknown')}",
        f"Active Cell: {notebook_context.get('activeCell', 'None')}"
    ]
    
    # Add cell information
    cells = notebook_context.get('cells', [])
    formatted.append(f"Total Cells: {len(cells)}")
    
    # Add cell summaries
    cell_summaries = []
    for cell in cells:
        # Get cell type and content
        cell_type = cell.get('type', 'unknown')
        content = cell.get('content', '')
        
        # Limit content length for summary
        if len(content) > 50:
            content_summary = content[:47] + '...'
        else:
            content_summary = content
            
        # Add cell summary
        cell_summaries.append(
            f"Cell {cell.get('index')}: [{cell_type}] {content_summary.replace(chr(10), ' ')}"
        )
    
    # Add cell summaries to formatted output
    if cell_summaries:
        formatted.append("\nCells:")
        formatted.extend(cell_summaries)
    
    return "\n".join(formatted)

def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely load JSON from text.
    
    Args:
        text: Text to parse as JSON
        default: Default value to return if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Error parsing JSON: {e}")
        return default

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + '...'
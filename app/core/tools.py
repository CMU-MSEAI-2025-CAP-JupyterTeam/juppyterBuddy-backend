"""
JupyterBuddy Tools Module

This module defines the tools that can be used by the agent to interact with
the notebook, including creating cells, updating cells, and executing cells.
These tools are used within the agent loop to take actions on the notebook.
"""
import json
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.tools import BaseTool, StructuredTool, tool

# Set up logging
import logging
logger = logging.getLogger(__name__)

# Define tool schemas for validation and documentation
class CellType(str, Enum):
    """Cell types in Jupyter notebooks."""
    CODE = "code"
    MARKDOWN = "markdown"

class CellPosition(str, Enum):
    """Positions for new cell insertion."""
    START = "start"
    END = "end"
    AFTER_ACTIVE = "after_active"
    BEFORE_ACTIVE = "before_active"

class CreateCellInput(BaseModel):
    """Input schema for creating a new cell."""
    cell_type: CellType = Field(description="Type of cell to create (code or markdown)")
    content: str = Field(description="Content to put in the cell")
    position: Union[CellPosition, int] = Field(
        default=CellPosition.AFTER_ACTIVE,
        description="Position to insert the cell ('start', 'end', 'after_active', 'before_active', or a specific cell index)"
    )

class UpdateCellInput(BaseModel):
    """Input schema for updating an existing cell."""
    cell_index: int = Field(description="Index of the cell to update")
    content: str = Field(description="New content for the cell")
    
class ExecuteCellInput(BaseModel):
    """Input schema for executing a cell."""
    cell_index: int = Field(description="Index of the cell to execute")

class GetNotebookInfoInput(BaseModel):
    """Input schema for getting notebook information."""
    include_cell_content: bool = Field(
        default=True, 
        description="Whether to include cell content in the response"
    )

# Define the tools
@tool(args_schema=CreateCellInput)
def create_cell_tool(
    cell_type: CellType,
    content: str,
    position: Union[CellPosition, int] = CellPosition.AFTER_ACTIVE
) -> str:
    """
    Create a new cell in the notebook.
    
    This tool sends a request to the frontend to create a new cell of the specified type
    with the given content at the specified position.
    
    Args:
        cell_type: The type of cell to create ('code' or 'markdown')
        content: The content to put in the cell
        position: Where to insert the cell ('start', 'end', 'after_active', 'before_active', or a specific index)
        
    Returns:
        A message indicating the cell was created
    """
    # This function only creates a payload to be sent to the frontend
    # The actual cell creation happens in the frontend
    return json.dumps({
        "status": "tool_called",
        "message": f"Will create a {cell_type} cell with position {position}"
    })

@tool(args_schema=UpdateCellInput)
def update_cell_tool(cell_index: int, content: str) -> str:
    """
    Update an existing cell in the notebook.
    
    This tool sends a request to the frontend to update the content of an existing cell.
    
    Args:
        cell_index: The index of the cell to update
        content: The new content for the cell
        
    Returns:
        A message indicating the cell was updated
    """
    # This function only creates a payload to be sent to the frontend
    # The actual cell update happens in the frontend
    return json.dumps({
        "status": "tool_called",
        "message": f"Will update cell at index {cell_index}"
    })

@tool(args_schema=ExecuteCellInput)
def execute_cell_tool(cell_index: int) -> str:
    """
    Execute a cell in the notebook.
    
    This tool sends a request to the frontend to execute a specific cell.
    
    Args:
        cell_index: The index of the cell to execute
        
    Returns:
        A message indicating the cell execution was requested
    """
    # This function only creates a payload to be sent to the frontend
    # The actual cell execution happens in the frontend
    return json.dumps({
        "status": "tool_called",
        "message": f"Will execute cell at index {cell_index}"
    })

@tool(args_schema=GetNotebookInfoInput)
def get_notebook_info_tool(include_cell_content: bool = True) -> str:
    """
    Get information about the current notebook.
    
    This tool sends a request to the frontend to get information about the current notebook,
    including the cells, active cell index, and other metadata.
    
    Args:
        include_cell_content: Whether to include the content of each cell in the response
        
    Returns:
        A message indicating that notebook info was requested
    """
    # This function only creates a payload to be sent to the frontend
    # The actual notebook info retrieval happens in the frontend
    return json.dumps({
        "status": "tool_called",
        "message": f"Getting notebook info with include_cell_content={include_cell_content}"
    })
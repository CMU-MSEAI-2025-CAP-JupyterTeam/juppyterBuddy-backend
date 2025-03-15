"""
JupyterBuddy API Routes

This module defines the API routes for the JupyterBuddy backend,
including routes for managing API keys and other configuration.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

# Local imports
from app.config import get_settings

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Health check response model
class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str


# Define a route for the root endpoint
@router.get("/")
async def root():
    return {"message": "Welcome to JupyterBuddy API"}


# Add a new route for health check
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service is running.

    Returns:
        Health status and version information
    """
    settings = get_settings()
    return {"status": "healthy", "version": settings.APP_VERSION}


# # local imports
# from app.core.security import store_api_key, get_api_key

# # API key request model
# class APIKeyRequest(BaseModel):
#     """Request model for setting an API key."""

#     provider: str
#     api_key: str

# # API key response model
# class APIKeyResponse(BaseModel):
#     """Response model for API key operations."""

#     provider: str
#     status: str
#     message: str


# # Add a new route to set an API key for a specific provider
# @router.post("/apikey", response_model=APIKeyResponse)
# async def set_api_key(request: APIKeyRequest):
#     """
#     Set an API key for a specific provider.

#     Args:
#         request: The API key request containing the provider and key

#     Returns:
#         Status of the operation
#     """
#     try:
#         # Check if the provider is valid
#         valid_providers = ["openai", "anthropic", "google"]
#         if request.provider.lower() not in valid_providers:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
#             )

#         # Store the API key
#         result = store_api_key(request.provider, request.api_key)

#         if result:
#             return {
#                 "provider": request.provider,
#                 "status": "success",
#                 "message": f"API key for {request.provider} stored successfully",
#             }
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Failed to store API key for {request.provider}",
#             )
#     except Exception as e:
#         logger.exception(f"Error setting API key: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
#         )


# # Add a new route to check if an API key is set for a specific provider
# @router.get("/apikey/{provider}", response_model=APIKeyResponse)
# async def check_api_key(provider: str):
#     """
#     Check if an API key is set for a specific provider.

#     Args:
#         provider: The provider to check

#     Returns:
#         Status of the API key
#     """
#     try:
#         # Check if the provider is valid
#         valid_providers = ["openai", "anthropic", "google"]
#         if provider.lower() not in valid_providers:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
#             )

#         # Check if the API key exists
#         api_key = get_api_key(provider)

#         if api_key:
#             return {
#                 "provider": provider,
#                 "status": "exists",
#                 "message": f"API key for {provider} is set",
#             }
#         else:
#             return {
#                 "provider": provider,
#                 "status": "missing",
#                 "message": f"No API key found for {provider}",
#             }
#     except Exception as e:
#         logger.exception(f"Error checking API key: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
#         )
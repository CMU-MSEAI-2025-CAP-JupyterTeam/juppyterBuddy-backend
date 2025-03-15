# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

# Import routers from other modules
from app.api.routes import router as api_router
from app.api.websocket import websocket_endpoint

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
# Create a logger for the main module
logger = logging.getLogger(__name__)

# Define lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI application startup and shutdown events using async generators.
    i.e. code that runs before the application starts and after it shuts down.
    """
    # Startup code - runs before the application starts accepting requests
    logger.info("Starting up JupyterBuddy backend")
    # Yield control to FastAPI to handle requests
    yield  # The application starts running here
    # Shutdown code - runs when the application is shutting down
    logger.info("Shutting down JupyterBuddy backend")

# Initialize FastAPI application with lifespan handler
app = FastAPI(
    title="JupyterBuddy Backend",
    description="Backend service for JupyterBuddy - a conversational assistant for JupyterLab",
    version="0.1.0",
    lifespan=lifespan,
)

# Set up CORS middleware to allow frontend to connect to the backend
app.add_middleware(
    CORSMiddleware,
    # Allow all origins during development - restrict in production
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers with their prefixes
app.include_router(
    api_router,  # Include the API router
    prefix="/api/v1",  # All routes inside this router will start with `/api/v1`
    tags=["api"],  # Add tags to categorize the routes (for documentation in Swagger UI)
)

# Add WebSocket endpoint 
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/{session_id}")
async def websocket_route(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication with the frontend."""
    await websocket_endpoint(websocket, session_id)

# Run the server if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    # Start the server with hot-reload enabled for development
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug",
    )
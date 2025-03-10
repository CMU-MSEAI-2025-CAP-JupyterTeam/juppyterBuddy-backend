# JupyterBuddy Backend

This repository contains the backend service for JupyterBuddy, a Conversational JupyterLab Extension that enables users to perform AI/ML workflows seamlessly via a natural language interface.

## Features

- **FastAPI Backend**: Modern, high-performance API server
- **WebSocket Support**: Real-time communication with the JupyterLab extension
- **LangGraph Integration**: Complex conversational workflows and state management
- **RAG System**: Retrieval-augmented generation for context-aware responses
- **Security**: Secure API key management and token-based authentication
- **Containerization**: Docker and Docker Compose for easy deployment

## Project Structure

```
JupyterBuddy-backend/
├── app/                       # Main application package
│   ├── api/                   # API routes and WebSocket handlers
│   ├── core/                  # Core functionality (agent, LLM, security)
│   ├── models/                # Data models
│   ├── schemas/               # Pydantic schemas
│   ├── services/              # Business logic services
│   ├── utils/                 # Utility functions
│   ├── config.py              # Application configuration
│   └── main.py                # FastAPI application entry point
├── data/                      # Data directory for vector store
├── tests/                     # Test suite
├── .env                       # Environment variables (create this file)
├── .gitignore                 # Git ignore file
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Docker Compose configuration
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)
- An OpenAI API key

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/JupyterBuddy-backend.git
   cd JupyterBuddy-backend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   SECRET_KEY=your-secure-secret-key
   OPENAI_API_KEY=your-openai-api-key
   ```

### Running the Backend

#### Local Development

```bash
uvicorn app.main:app --reload
```

The server will start at http://localhost:8000, and the API documentation will be available at http://localhost:8000/docs.

#### Using Docker Compose

```bash
docker-compose up --build
```

## API Documentation

When the server is running, you can access the auto-generated API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Integration with JupyterLab Extension

The backend communicates with the JupyterLab extension through:

1. RESTful API endpoints for standard operations
2. WebSocket connections for real-time conversation

The frontend repository is available at: https://github.com/yourusername/JupyterBuddy-frontend

## Development

### Adding New Tools

To add new tools for the agent:

1. Implement the tool function in `app/core/agent.py`
2. Add the tool to the `_create_tools` method
3. Update the schemas as needed

### Extending RAG Capabilities

To enhance the RAG system:

1. Add new document loaders in `app/services/rag.py`
2. Implement additional indexing methods as needed

## Testing

Run the test suite:

```bash
pytest
```

## License

[MIT License](LICENSE)
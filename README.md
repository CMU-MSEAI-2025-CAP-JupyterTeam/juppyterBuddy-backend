# JupyterBuddy: AI-Powered Notebook Workflow Assistant

## Overview

JupyterBuddy is an AI-powered assistant that integrates directly into JupyterLab, enabling users to interact with their notebooks through natural language. It streamlines workflows for data scientists and ML engineers by handling tasks such as code execution, cell modification, and metadata retrieval.

![JupyterBuddy Architecture](https://i.imgur.com/example.png)

## Features

- **Natural Language Interface**: Communicate with your notebook using conversational language
- **Code Operations**: Create, update, and execute cells without manual intervention
- **Context Awareness**: Maintains conversation history and notebook state for intelligent responses
- **Error Handling**: Automatically detects and helps resolve errors in notebook execution
- **Persistent Memory**: Conversation history is preserved across sessions

## Architecture

JupyterBuddy follows a client-server architecture with a real-time communication layer:

### Frontend Components

- **React Interface**: Chat UI integrated within JupyterLab
- **Tool Implementations**: Notebook operations defined in TypeScript
- **WebSocket Client**: Real-time bidirectional communication with backend

### Backend Components

- **FastAPI Server**: Handles HTTP and WebSocket connections
- **WebSocketManager**: Routes messages between frontend and agent
- **JupyterBuddyAgent**: Orchestrates the workflow using LangGraph
- **LLM Integration**: Uses language models via LangChain
- **State Management**: Persists conversation history across sessions

### Communication Flow

1. User enters a message in the frontend chat interface
2. Frontend sends message and notebook context to backend via WebSocket
3. Agent processes the message through an LLM
4. LLM generates a response or tool calls
5. Tool calls are sent to frontend for execution
6. Execution results are returned to backend
7. Process repeats if additional actions are needed

## Design Patterns

JupyterBuddy implements several design patterns:

- **Observer Pattern**: WebSocket communication for real-time updates
- **Strategy Pattern**: Different tool implementations for notebook operations
- **Command Pattern**: Tool execution as encapsulated commands
- **Chain of Responsibility**: LangGraph workflow processing
- **Factory Pattern**: LLM creation and configuration
- **Singleton Pattern**: WebSocketManager for global connection tracking
- **State Pattern**: Conversation and execution state management

## Installation

### Prerequisites

- JupyterLab 4.0+
- Python 3.9+
- Node.js 18+

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/jupyterbuddy.git
   cd jupyterbuddy
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env with your LLM API keys
   ```

5. Start the backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. Install the JupyterLab extension:
   ```bash
   jupyter labextension install jupyterbuddy
   ```

2. Alternatively, build from source:
   ```bash
   cd frontend
   npm install
   npm run build
   jupyter labextension install .
   ```

## Configuration

JupyterBuddy can be configured through environment variables or the config.py file:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai, anthropic, google) | openai |
| `LLM_MODEL_NAME` | Specific model to use | (provider default) |
| `LLM_TEMPERATURE` | Temperature for generation | 0.7 |
| `AGENT_STATE_DIR` | Directory for state persistence | agent_states |

## API Reference

### WebSocket API

- **Connection**: `ws://localhost:8000/ws/{session_id}`
- **Message Types**:
  - `register_tools`: Register frontend tool definitions
  - `user_message`: Send user input
  - `action_result`: Return tool execution results

## Project Structure

```
jupyterbuddy/
├── app/                    # Backend application
│   ├── api/                # API endpoints
│   │   ├── routes.py       # HTTP routes
│   │   └── websocket.py    # WebSocket handling
│   ├── core/               # Core functionality
│   │   ├── agent.py        # LangGraph agent
│   │   ├── llm.py          # LLM integration
│   │   └── prompts.py      # System prompts
│   ├── services/           # Service layer
│   │   └── StateManagerService.py # State persistence
│   └── main.py             # Application entry point
├── frontend/               # JupyterLab extension
│   ├── src/                # Source code
│   │   ├── index.ts        # Extension entry point
│   │   ├── App.tsx         # Main component
│   │   └── jupyterbuddyTools/  # Tool implementations
│   │       └── tools.ts    # Tool definitions
│   └── package.json        # Dependencies
└── requirements.txt        # Python dependencies
```

## Development Guide

### Adding New Tools

1. Define the tool in `frontend/src/jupyterbuddyTools/tools.ts`:
   ```typescript
   // Add to jupyterBuddyTools array
   {
     name: "new_tool",
     description: "Description of what the tool does",
     parameters: {
       type: "object",
       properties: {
         param_name: {
           name: "param_name",
           type: "string",
           description: "Parameter description",
           required: true
         }
       },
       required: ["param_name"]
     }
   }
   
   // Add implementation to toolFunctions
   new_tool: (payload, notebookTracker, getNotebookContext) => {
     // Implementation
     return {
       action_type: 'NEW_TOOL',
       result: { /* result data */ },
       success: true
     };
   }
   ```

2. The tool will be automatically registered with the LLM when the WebSocket connection is established.

### Extending the Agent

To modify the agent's behavior:

1. Update the prompt in `app/core/prompts.py`
2. Add or modify nodes in the LangGraph workflow in `app/core/agent.py`
3. Update the state structure in `AgentState` if needed

## Troubleshooting

### Common Issues

- **WebSocket Connection Errors**: Ensure the backend server is running and accessible
- **Tool Execution Failures**: Check browser console for JavaScript errors
- **LLM API Errors**: Verify API keys are correctly set in environment variables

### Debugging

- Backend logs are available in the terminal running the server
- Frontend logs can be viewed in the browser console
- Set `logging.level` to `DEBUG` in `app/main.py` for verbose logging

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [JupyterLab](https://jupyterlab.readthedocs.io/) for the notebook environment

---

This documentation provides an overview of JupyterBuddy's architecture, installation instructions, and development guidelines. For detailed API documentation, refer to the docstrings in the source code.
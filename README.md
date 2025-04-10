# OwlAI

An intelligent AI agent system with RAG (Retrieval Augmented Generation) capabilities, text-to-speech integration, and extensible tools framework.

## 🦉 Overview

OwlAI is a versatile agent-based AI platform built on LangChain that provides:

- **Agent Architecture**: Customizable agents with different personalities and capabilities
- **RAG Integration**: Domain-specific knowledge via retrieval augmented generation
- **Tool Framework**: Extensible capabilities via a tool-based architecture
- **REST API**: FastAPI-based API for programmatic access
- **Memory System**: Persistent conversation history
- **Modular Design**: Components can be used independently or as a whole

## 🚀 Features

- **Multi-Agent System**: Manage and interact with various AI agents with different specializations
- **LLM Integration**: Supports multiple LLM providers (OpenAI, Anthropic)
- **RAG Capabilities**: Built-in retrieval augmented generation for domain-specific knowledge
- **Text-to-Speech**: Multiple TTS engines (Coqui-TTS, Edge TTS, ElevenLabs, etc.)
- **Spotify Integration**: Control Spotify playback
- **Extensible Tool Framework**: Easy to add new capabilities
- **REST API**: FastAPI-based REST API for programmatic access
- **Message History Management**: Smart FIFO message handling to manage context window
- **Advanced Embeddings**: Efficient vector search for knowledge retrieval
- **Telemetry**: Built-in performance tracking
- **Streaming Responses**: Real-time streaming of agent responses

## 📦 Project Structure

```
owlai/
├── core.py           # Core agent implementation
├── nest.py           # Agent manager for handling multiple agents
├── config/           # Configuration for agents, prompts, and tools
├── db/               # Database and memory persistence
├── services/         # Core services used by the agents
│   ├── datastore.py  # Data storage and retrieval
│   ├── embeddings.py # Vector embeddings
│   ├── rag.py        # Retrieval augmented generation
│   ├── system.py     # System utilities
│   ├── tools/        # Tool implementations
│       ├── box.py    # Tool registry
│       ├── interpreter.py # Code interpreter
│       ├── spotify.py # Spotify integration
│       └── ttsengine.py # Text-to-speech engines
├── ...
```

## 🛠️ Installation

### Requirements

- Python 3.8 or higher
- PyTorch (with CUDA support recommended for performance)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/owlai.git
cd owlai

# Install the package
pip install -e .
```

### Installing with Optional Dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Install with text-to-speech support
pip install -e ".[tts]"

# Install with Spotify support
pip install -e ".[spotify]"

# Install all optional dependencies
pip install -e ".[dev,tts,spotify]"
```

### Installing PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## ⚙️ Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
OWLAI_ENV=development  # or production
```

## 🔍 Usage

### Python API

```python
from owlai.core import OwlAgent
from owlai.services.tools.box import TOOLBOX

# Create an agent
agent = OwlAgent(
    name="my_agent",
    version="1.0",
    description="A helpful assistant",
    system_prompt="You are a helpful assistant.",
    llm_config={
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 2048,
        "context_size": 4096,
        "tools_names": ["tool1", "tool2"]
    }
)

# Initialize tools
agent.init_callable_tools([TOOLBOX[tool_name] for tool_name in agent.llm_config.tools_names])

# Run the agent with a query
response = agent.message_invoke("Tell me about quantum computing")
print(response)
```

### REST API

OwlAI provides a FastAPI-based REST API for programmatic access. To start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000` with the following endpoints:

#### List Available Agents
```bash
curl http://localhost:8000/agents
```

#### Get Agent Information
```bash
curl http://localhost:8000/agents/info
```

#### Invoke an Agent
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_id", "question": "Hello, how are you?", "query_id": "unique_id", "session_id": "session_id"}'
```

#### Stream Agent Response
```bash
curl -X POST http://localhost:8000/stream-query \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "agent_id", "question": "Hello, how are you?", "query_id": "unique_id", "session_id": "session_id"}'
```

Interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧩 Agent Configuration

Agents are defined in `owlai/config/agents.py`. Example configuration:

```python
{
    "agent_name": {
        "name": "agent_name",
        "version": "1.0",
        "description": "Agent description",
        "system_prompt": "System prompt for the agent",
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.1,
            "context_size": 4000,
            "tools_names": ["tool1", "tool2"],
        },
        "default_queries": ["Example query 1", "Example query 2"]
    }
}
```

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Code Style

This project uses Black for code formatting and isort for import sorting:

```bash
# Format code
black .

# Sort imports
isort .
```

## 📄 License

MIT

## 🙏 Acknowledgements

This project builds upon several open-source libraries and frameworks, including:
- LangChain
- PyTorch
- FastAPI
- FAISS
- Sentence-Transformers
- and many others 
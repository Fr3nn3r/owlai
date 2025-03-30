# OwlAI

An intelligent AI agent system with RAG (Retrieval Augmented Generation) capabilities, text-to-speech integration, and other tools.

## Features

- **LLM Integration**: Supports multiple LLM providers (OpenAI, Anthropic)
- **RAG Capabilities**: Built-in retrieval augmented generation for knowledge base access
- **Text-to-Speech**: Multiple TTS engines (Coqui-TTS, Edge TTS, ElevenLabs, etc.)
- **Spotify Integration**: Control Spotify playback
- **Extensible Tool Framework**: Easy to add new capabilities
- **REST API**: FastAPI-based REST API for programmatic access

## Installation

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

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
$env:PYTHONPATH = "C:\Users\fbrun\Documents\GitHub\owlai\src"
echo $env:OWLAI_ENV
```

## Usage

### Python API

Basic usage example:

```python
from owlai.core import OwlAgent

# Create an agent
agent = OwlAgent(
    model_provider="openai",
    model_name="gpt-4o",
    system_prompt="You are a helpful assistant."
)

# Run the agent with a query
response = agent.run("Tell me about quantum computing")
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
curl -X POST http://localhost:8000/agents/invoke \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "agent_name", "message": "Hello, how are you?"}'
```

You can also access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

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

## License

MIT

## Acknowledgements

This project builds upon several open-source libraries and frameworks, including LangChain, PyTorch, and many others. 
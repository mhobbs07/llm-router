# LLM Router

A high-performance, cost-optimized LLM routing service with multi-provider support and seamless Open Web UI integration via MCP (Model Context Protocol). Route requests across OpenAI, Google Gemini, xAI/Grok, and other providers with intelligent fallback, caching, and budget management.

## Features

- **Multi-Provider Routing**: Automatic routing across OpenAI GPT, Google Gemini, xAI Grok, and Anthropic Claude
- **Intelligent Fallback**: Seamless fallback between providers when one fails
- **Cost Optimization**: Smart model selection, response caching, and budget limits
- **Open Web UI Integration**: Full MCP server support for tool calling and enhanced AI capabilities
- **Real-time Monitoring**: Comprehensive metrics and cost analytics
- **Streaming Support**: Full streaming chat completions with cost tracking
- **Context Optimization**: Automatic context trimming to reduce costs
- **Tool Calling**: MCP-compatible tool execution and management

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llm-router
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   # Copy and edit the environment template
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Basic Configuration

Create a `.env` file with your API keys:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini Configuration
GOOGLE_GEMINI_API_KEY=your_gemini_api_key_here

# xAI/Grok Configuration
XAI_API_KEY=your_xai_api_key_here

# Cost Management (optional)
COST_BUDGET_DAILY_LIMIT=10.0
COST_BUDGET_MONTHLY_LIMIT=100.0
COST_CACHING_ENABLED=true
```

### Running the Service

Start the LLM Router API server:

```bash
python -m src.main
```

The server will start on `http://localhost:8000` with the following endpoints:
- `POST /chat/completions` - Chat completions
- `POST /chat/completions/stream` - Streaming chat completions
- `GET /metrics` - Service metrics
- `GET /cost-analytics` - Cost analytics
- `GET /health` - Health check

## Open Web UI Integration

The LLM Router provides seamless integration with Open Web UI through MCP (Model Context Protocol), enabling advanced features like tool calling, function execution, and enhanced AI capabilities.

### MCP Server Setup

1. **Enable MCP in configuration:**
   ```bash
   # Add to your .env file
   MCP_ENABLED=true
   MCP_TRANSPORT=stdio  # or 'http' for HTTP transport
   MCP_HOST=localhost
   MCP_PORT=8001
   ```

2. **Configure Open Web UI:**

   Add the MCP server configuration to your Open Web UI config:

   ```json
   {
     "mcpServers": {
       "llm-router": {
         "command": "python",
         "args": ["-m", "src.mcp_main"],
         "env": {
           "OPENAI_API_KEY": "your_openai_key",
           "GOOGLE_GEMINI_API_KEY": "your_gemini_key",
           "XAI_API_KEY": "your_xai_key"
         }
       }
     }
   }
   ```

3. **For Docker deployments:**

   If running Open Web UI in Docker, mount the MCP configuration:

   ```yaml
   version: '3.8'
   services:
     open-webui:
       image: ghcr.io/open-webui/open-webui:main
       volumes:
         - ./mcp-config.json:/app/mcp-config.json
       environment:
         - MCP_CONFIG_PATH=/app/mcp-config.json
   ```

### MCP Tools Available

The MCP server provides several built-in tools:

- **Calculator**: Mathematical expression evaluation
- **Weather**: Mock weather information service
- **File Reader**: Read contents of files on the system

### Custom MCP Tools

You can extend the MCP server with custom tools by modifying `src/mcp_main.py`:

```python
def create_sample_tools(mcp_server: MCPServer):
    # Add your custom tools here
    def my_custom_tool(args):
        # Tool implementation
        return "Tool result"

    mcp_server.register_tool(
        name="my_tool",
        description="Description of my tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param1"]
        },
        handler=my_custom_tool
    )
```

## Cost Optimization Features

### Smart Model Selection

The router automatically selects the most cost-effective model based on request characteristics:

```bash
# Cost-optimized hierarchy (default)
GPT-4 → Claude-3-Sonnet → Gemini-Pro → Grok-1 → GPT-3.5-Turbo → Claude-3-Haiku

# Performance-optimized hierarchy (for complex tasks)
GPT-4 → Claude-3-Sonnet → Gemini-Pro → Grok-1 → GPT-3.5-Turbo → Claude-3-Haiku
```

### Response Caching

Reduce costs with intelligent caching:

```bash
# Cache configuration
COST_CACHING_ENABLED=true
COST_CACHE_TTL_SECONDS=3600  # 1 hour
COST_CACHE_MAX_SIZE_MB=100   # 100MB cache limit
```

### Budget Management

Set spending limits and receive alerts:

```bash
# Daily and monthly budget limits
COST_BUDGET_DAILY_LIMIT=10.0
COST_BUDGET_MONTHLY_LIMIT=100.0
COST_BUDGET_ALERT_THRESHOLD=0.8  # Alert at 80% usage
```

### Context Optimization

Automatically trim conversation context to reduce token costs:

```bash
# Context optimization
COST_MAX_CONTEXT_TOKENS=8000  # Maximum context length
```

## API Usage Examples

### Basic Chat Completion

```python
import requests

response = requests.post("http://localhost:8000/chat/completions", json={
    "model": "gpt-4",  # Will be routed to available provider
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
})

print(response.json())
```

### Streaming Chat Completion

```python
import requests

response = requests.post("http://localhost:8000/chat/completions/stream", json={
    "model": "gemini-pro",
    "messages": [
        {"role": "user", "content": "Tell me a story"}
    ],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### With Tool Calling (MCP)

```python
response = requests.post("http://localhost:8000/chat/completions", json={
    "model": "gpt-4",
    "messages": [
        {"role": "user", "content": "Calculate 15 * 23"}
    ],
    "tools": [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }]
})
```

## Monitoring and Analytics

### Metrics Endpoint

Get real-time service metrics:

```bash
curl http://localhost:8000/metrics
```

Response includes:
- Total requests and errors
- Provider success rates
- Response time statistics
- Cost metrics by provider and model

### Cost Analytics

Detailed cost analysis and insights:

```bash
curl http://localhost:8000/cost-analytics
```

Includes:
- Current spending vs budget
- Cost efficiency metrics
- Savings potential analysis
- Budget alerts

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `GOOGLE_GEMINI_API_KEY` | Google Gemini API key | Required for Gemini |
| `XAI_API_KEY` | xAI API key | Required for Grok |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Claude |
| `MCP_ENABLED` | Enable MCP server | `false` |
| `MCP_TRANSPORT` | MCP transport (`stdio` or `http`) | `stdio` |
| `COST_CACHING_ENABLED` | Enable response caching | `true` |
| `COST_CACHE_TTL_SECONDS` | Cache TTL in seconds | `3600` |
| `COST_BUDGET_DAILY_LIMIT` | Daily budget limit ($) | `10.0` |
| `COST_BUDGET_MONTHLY_LIMIT` | Monthly budget limit ($) | `100.0` |

### Model Cost Configuration

Default model costs (per 1K tokens):

```json
{
  "gpt-4": {"input": 0.03, "output": 0.06},
  "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
  "gemini-pro": {"input": 0.00025, "output": 0.0005},
  "claude-3-sonnet": {"input": 0.015, "output": 0.075},
  "grok-1": {"input": 0.005, "output": 0.015}
}
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_integration.py::TestProviderRouting -v
pytest tests/test_integration.py::TestFallbackLogic -v
```

### Project Structure

```
llm-router/
├── src/
│   ├── main.py          # FastAPI server and routing logic
│   ├── providers.py     # LLM provider implementations
│   ├── config.py        # Configuration management
│   ├── models.py        # Pydantic models
│   ├── mcp_server.py   # MCP server implementation
│   └── mcp_main.py      # MCP server entry point
├── tests/
│   └── test_integration.py  # Integration tests
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Providers

1. Implement the `LLMProvider` protocol in `src/providers.py`
2. Add configuration class in `src/config.py`
3. Update routing logic in `src/main.py`
4. Add environment variable loading in `load_config_from_env()`

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
# Provider Setup Guide

This guide explains how to set up API keys for all supported LLM providers in the router system.

## Environment Variables

The system uses environment variables to configure API keys for different providers. You can set these in the `.env` file or as system environment variables.

### Required Environment Variables by Provider

#### Anthropic Claude
```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

**How to get an Anthropic API key:**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in to your account
3. Navigate to "API Keys" in the dashboard
4. Create a new API key
5. Copy the key (it starts with `sk-ant-`)

**Optional Anthropic configuration:**
```bash
ANTHROPIC_DEFAULT_MODEL=claude-3-sonnet-20240229
ANTHROPIC_TEMPERATURE=0.7
ANTHROPIC_MAX_TOKENS=4096
```

#### OpenAI
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

**Optional OpenAI configuration:**
```bash
OPENAI_DEFAULT_MODEL=gpt-4
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096
```

#### Google Gemini
```bash
GOOGLE_GEMINI_API_KEY=your-gemini-api-key-here
```

**Optional Gemini configuration:**
```bash
GOOGLE_GEMINI_DEFAULT_MODEL=gemini-pro
GOOGLE_GEMINI_TEMPERATURE=0.7
GOOGLE_GEMINI_MAX_TOKENS=4096
```

#### xAI Grok
```bash
XAI_API_KEY=your-xai-api-key-here
```

**Optional xAI configuration:**
```bash
XAI_DEFAULT_MODEL=grok-1
XAI_TEMPERATURE=0.7
XAI_MAX_TOKENS=4096
```

## Model Routing

The system automatically routes models to the appropriate provider based on the model name:

- **Claude models** → Anthropic provider
  - `claude-3-5-sonnet-20241022`
  - `claude-3-sonnet-20240229`
  - `claude-3-haiku-20240307`
  - `claude-3-opus-20240229`

- **GPT models** → OpenAI provider
  - `gpt-4`
  - `gpt-3.5-turbo`
  - `chatgpt`

- **Gemini models** → Google provider
  - `gemini-pro`
  - `gemini-pro-vision`

- **Grok models** → xAI provider
  - `grok-1`

## Testing the Setup

### 1. Test with curl

Test Anthropic Claude:
```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "claude-3-sonnet-20240229"
  }'
```

### 2. Run the test suite

Test all providers:
```bash
# From the project root directory
python -m pytest tests/ -v
```

Test only Anthropic integration:
```bash
python -m pytest tests/test_anthropic_integration.py -v
```

Alternative method using PYTHONPATH:
```bash
PYTHONPATH=. pytest tests/test_anthropic_integration.py -v
```

### 3. Check available models

```bash
curl http://localhost:8000/models
```

## Troubleshooting

### Common Issues

1. **"Provider 'anthropic' not configured"**
   - Make sure `ANTHROPIC_API_KEY` is set in your `.env` file
   - Restart the server after adding the key

2. **"Anthropic authentication error"**
   - Verify your API key is correct and starts with `sk-ant-`
   - Check that your Anthropic account has sufficient credits

3. **"All providers failed"**
   - At least one provider must be configured
   - Check that your API keys are valid

### Debugging

Enable debug logging by setting:
```bash
LOG_LEVEL=DEBUG
```

Check the server logs for detailed error messages when requests fail.

## Cost Management

The system includes cost tracking and budget management. Configure these optional settings:

```bash
# Budget limits
COST_BUDGET_DAILY_LIMIT=10.0
COST_BUDGET_MONTHLY_LIMIT=100.0
COST_BUDGET_ALERT_THRESHOLD=0.8

# Caching
COST_CACHING_ENABLED=true
COST_CACHE_TTL_SECONDS=3600

# Model policies
COST_ALLOWED_MODELS_JSON='["gpt-4", "claude-3-sonnet-20240229"]'
COST_BLOCKED_MODELS_JSON='["gpt-3.5-turbo"]'
COST_ALLOW_UNKNOWN_MODELS=true
```

## Running the Server

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your `.env` file with API keys

3. Start the server:
```bash
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

4. Test the setup:
```bash
curl http://localhost:8000/health
```

The server will be available at `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`.
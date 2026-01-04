import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure
from fastapi.testclient import TestClient

from src.models import ChatRequest, Message, ChatResponse, Tool, ToolCall
from src.main import chat_completions, app


class TestChatCompletions:
    """Test core chat completion functionality across providers."""

    def setup_method(self):
        """Reset global state before each test."""
        import datetime
        from src.main import metrics, cost_tracker
        metrics.clear()
        metrics.update({
            "total_requests": 0,
            "requests_by_endpoint": {},
            "total_errors": 0,
            "errors_by_endpoint": {},
            "provider_success": {},
            "response_time_sum": 0.0,
            "response_time_count": 0,
            "total_cost": 0.0,
            "cost_by_provider": {},
            "cost_by_model": {},
            "daily_cost": 0.0,
            "monthly_cost": 0.0,
            "last_cost_reset": datetime.datetime.utcnow().date()
        })
        cost_tracker.clear()
        cost_tracker.update({
            "daily_spend": 0.0,
            "monthly_spend": 0.0,
            "last_daily_reset": datetime.datetime.utcnow().date(),
            "last_monthly_reset": datetime.datetime.utcnow().replace(day=1).date(),
            "budget_alerts_sent": set()
        })

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_openai_chat_completion_success(self, mock_openai_client):
        """Test successful chat completion with OpenAI."""
        # Mock the OpenAI client and response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None  # Explicitly set to None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",
            temperature=0.7,
            max_tokens=100
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Test response"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30

    @patch.dict(os.environ, {
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_gemini_chat_completion_success(self, mock_genai_model):
        """Test successful chat completion with Gemini."""
        # Mock Gemini model and response
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_chat = Mock()
        mock_model_instance.start_chat.return_value = mock_chat

        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]
        mock_response.candidates[0].content.parts[0].text = "Gemini response"
        mock_response.candidates[0].finish_reason.name = "STOP"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 25
        mock_response.usage_metadata.total_token_count = 40

        mock_chat.send_message_async = AsyncMock(return_value=mock_response)

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gemini-pro"
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Gemini response"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 40

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    @patch('src.providers.AnthropicProvider.chat_completion')
    @pytest.mark.asyncio
    async def test_anthropic_chat_completion_success(self, mock_completion):
        """Test successful chat completion with Anthropic."""
        mock_completion.return_value = Success(ChatResponse(
            content="Claude response",
            finish_reason="stop",
            usage={"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}
        ))

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="claude-3-sonnet-20240229"
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Claude response"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_chat_completion_empty_messages(self):
        """Test chat completion with empty messages fails."""
        request = ChatRequest(
            messages=[],
            model="gpt-4"
        )

        result = await chat_completions(request)

        assert isinstance(result, Failure)
        assert "empty" in result.failure().lower()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_with_system_message(self, mock_openai_client):
        """Test chat completion with system message."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I'll be helpful and concise."
        mock_response.choices[0].message.tool_calls = None  # Explicitly set to None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        request = ChatRequest(
            messages=[
                Message(role="system", content="Be helpful and concise."),
                Message(role="user", content="Hello")
            ],
            model="gpt-4"
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "I'll be helpful and concise."

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_with_tools(self, mock_openai_client):
        """Test chat completion with tool calling."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        # Mock tool call response
        mock_tool_call = Mock()
        mock_tool_call.model_dump.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}'
            }
        }

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I'll check the weather for you."
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 40

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        request = ChatRequest(
            messages=[Message(role="user", content="What's the weather in San Francisco?")],
            model="gpt-4",
            tools=[
                Tool(
                    type="function",
                    function={
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            }
                        }
                    }
                )
            ]
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.finish_reason == "tool_calls"
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, mock_openai_client):
        """Test chat completion with API error."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4"
        )

        result = await chat_completions(request)

        assert isinstance(result, Failure)
        assert "failed" in result.failure().lower()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_chat_completion_endpoint_success(self, mock_openai_client):
        """Test chat completion endpoint with successful response."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Endpoint test response"
        mock_response.choices[0].message.tool_calls = None  # Explicitly set to None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Endpoint test message"}],
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 100
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 200
        data = response.json()
        # Check OpenAI-compatible format
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["content"] == "Endpoint test response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["total_tokens"] == 25

    def test_chat_completion_endpoint_validation_error(self):
        """Test chat completion endpoint with validation error."""
        client = TestClient(app)

        # Missing required fields
        request_data = {
            "model": "gpt-4"
            # Missing messages
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_chat_completion_endpoint_no_providers(self):
        """Test chat completion endpoint when no providers are configured."""
        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 400
        assert "failed" in response.json()["detail"].lower()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_context_optimization(self, mock_openai_client):
        """Test chat completion with context optimization."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Optimized response"
        mock_response.choices[0].message.tool_calls = None  # Explicitly set to None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 70

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create request with long context that should be optimized
        long_messages = [
            Message(role="user", content="A" * 1000),  # Long message
            Message(role="assistant", content="B" * 1000),
            Message(role="user", content="C" * 1000),
            Message(role="user", content="Final question")
        ]

        request = ChatRequest(
            messages=long_messages,
            model="gpt-4"
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Optimized response"

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_model_optimization(self, mock_openai_client):
        """Test chat completion with model optimization."""
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Model optimized response"
        mock_response.choices[0].message.tool_calls = None  # Explicitly set to None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 15
        mock_response.usage.total_tokens = 25

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        # Simple request that might be optimized to a cheaper model
        request = ChatRequest(
            messages=[Message(role="user", content="Hi")],
            model="gpt-4"  # Might be optimized to gpt-3.5-turbo
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Model optimized response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
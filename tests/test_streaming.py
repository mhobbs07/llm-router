import os
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure
from fastapi.testclient import TestClient

from src.models import ChatRequest, Message, ChatStreamResponse
from src.main import chat_completions_stream, app


class TestStreamingEndpoint:
    """Test streaming endpoint functionality."""

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
    async def test_chat_completions_stream_success(self, mock_openai_client):
        """Test successful streaming chat completion."""
        # Mock the OpenAI client and streaming response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        # Create mock streaming chunks
        mock_chunks = []
        for i, content in enumerate(["Hello", " ", "world", "!"]):
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.tool_calls = None  # Explicitly set to None
            mock_chunk.choices[0].finish_reason = "stop" if i == 3 else None
            mock_chunks.append(mock_chunk)

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_stream())

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",
            stream=True
        )

        result = await chat_completions_stream(request)

        assert isinstance(result, Success)
        stream = result.unwrap()

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 4
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " "
        assert chunks[2].delta == "world"
        assert chunks[3].delta == "!"
        assert chunks[3].finish_reason == "stop"

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    @patch('src.providers.AnthropicProvider.chat_completion_stream')
    @pytest.mark.asyncio
    async def test_anthropic_streaming_success(self, mock_stream):
        """Test successful streaming with Anthropic provider."""
        async def mock_generator():
            yield ChatStreamResponse(delta="Hello", finish_reason=None)
            yield ChatStreamResponse(delta=" from", finish_reason=None)
            yield ChatStreamResponse(delta=" Claude!", finish_reason="stop")

        mock_stream.return_value = Success(mock_generator())

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="claude-3-sonnet-20240229",
            stream=True
        )

        result = await chat_completions_stream(request)

        assert isinstance(result, Success)
        stream = result.unwrap()

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " from"
        assert chunks[2].delta == " Claude!"
        assert chunks[2].finish_reason == "stop"

    @patch.dict(os.environ, {
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_gemini_streaming_success(self, mock_genai_model):
        """Test successful streaming with Gemini provider."""
        # Mock Gemini streaming
        mock_model_instance = Mock()
        mock_genai_model.return_value = mock_model_instance

        mock_chat = Mock()
        mock_model_instance.start_chat.return_value = mock_chat

        # Create mock streaming chunks
        mock_chunks = []
        for i, content in enumerate(["Gemini", " streaming", " response"]):
            mock_chunk = Mock()
            mock_chunk.candidates = [Mock()]
            mock_chunk.candidates[0].content.parts = [Mock()]
            mock_chunk.candidates[0].content.parts[0].text = content
            mock_chunk.candidates[0].finish_reason = Mock()
            mock_chunk.candidates[0].finish_reason.name = "STOP" if i == 2 else None
            mock_chunks.append(mock_chunk)

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_chat.send_message_async = AsyncMock(return_value=mock_stream())

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gemini-pro",
            stream=True
        )

        result = await chat_completions_stream(request)

        assert isinstance(result, Success)
        stream = result.unwrap()

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Gemini"
        assert chunks[1].delta == " streaming"
        assert chunks[2].delta == " response"
        assert chunks[2].finish_reason == "stop"

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_streaming_endpoint_requires_stream_true(self):
        """Test that streaming endpoint requires stream=True."""
        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": False  # This should fail
        }

        response = client.post("/chat/completions/stream", json=request_data)
        assert response.status_code == 400
        assert "Stream must be set to true" in response.json()["detail"]

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_streaming_endpoint_integration(self, mock_openai_client):
        """Test the full streaming endpoint with HTTP client."""
        # Mock the OpenAI client and streaming response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        # Create mock streaming chunks
        mock_chunks = []
        for i, content in enumerate(["Hello", " world"]):
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.tool_calls = None  # Explicitly set to None
            mock_chunk.choices[0].finish_reason = "stop" if i == 1 else None
            mock_chunks.append(mock_chunk)

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_stream())

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": True
        }

        response = client.post("/chat/completions/stream", json=request_data)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE response
        response_lines = response.text.strip().split("\n\n")
        assert len(response_lines) == 4  # 2 content chunks + 1 final chunk + 1 done

        # Check first chunk
        first_data = json.loads(response_lines[0].replace("data: ", ""))
        # Check OpenAI-compatible format
        assert first_data["object"] == "chat.completion.chunk"
        assert len(first_data["choices"]) == 1
        assert first_data["choices"][0]["delta"]["content"] == "Hello"
        assert first_data["choices"][0]["finish_reason"] is None

        # Check second chunk
        second_data = json.loads(response_lines[1].replace("data: ", ""))
        assert second_data["choices"][0]["delta"]["content"] == " world"
        assert second_data["choices"][0]["finish_reason"] == "stop"

        # Check final chunk (empty delta with finish_reason)
        third_data = json.loads(response_lines[2].replace("data: ", ""))
        assert third_data["choices"][0]["delta"] == {}
        assert third_data["choices"][0]["finish_reason"] == "stop"

        # Check done message
        assert response_lines[3] == "data: [DONE]"

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self, mock_openai_client):
        """Test error handling in streaming responses."""
        # Mock OpenAI client to fail
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("Streaming failed")
        )

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",
            stream=True
        )

        result = await chat_completions_stream(request)

        assert isinstance(result, Failure)
        assert "failed" in result.failure().lower()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_streaming_endpoint_error_handling(self, mock_openai_client):
        """Test error handling in streaming endpoint."""
        # Mock OpenAI client to fail
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "stream": True
        }

        response = client.post("/chat/completions/stream", json=request_data)

        assert response.status_code == 400
        assert "failed" in response.json()["detail"].lower()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_streaming_with_usage_tracking(self, mock_openai_client):
        """Test streaming with usage information tracking."""
        # Mock the OpenAI client and streaming response with usage info
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        # Create mock streaming chunks with usage in the last chunk
        mock_chunks = []
        for i, content in enumerate(["Hello", " world"]):
            mock_chunk = Mock()
            mock_chunk.choices = [Mock()]
            mock_chunk.choices[0].delta = Mock()
            mock_chunk.choices[0].delta.content = content
            mock_chunk.choices[0].delta.tool_calls = None  # Explicitly set to None
            mock_chunk.choices[0].finish_reason = "stop" if i == 1 else None
            
            # Add usage info to the last chunk
            if i == 1:
                mock_chunk.usage = Mock()
                mock_chunk.usage.prompt_tokens = 10
                mock_chunk.usage.completion_tokens = 5
                mock_chunk.usage.total_tokens = 15
            else:
                mock_chunk.usage = None
                
            mock_chunks.append(mock_chunk)

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_stream())

        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",
            stream=True
        )

        result = await chat_completions_stream(request)

        assert isinstance(result, Success)
        stream = result.unwrap()

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[1].finish_reason == "stop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
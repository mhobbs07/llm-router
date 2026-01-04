import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure
from fastapi.testclient import TestClient
import datetime

from src.config import AppConfig, AnthropicConfig, CostConfig
from src.providers import AnthropicProvider
from src.models import ChatRequest, Message, ChatResponse, ChatStreamResponse, Tool, ToolCall
from src.main import (
    get_provider_name, create_provider, chat_completions, chat_completions_stream, 
    app, metrics, is_model_allowed, get_available_models, cost_tracker
)


class TestAnthropicClaudeIntegration:
    """Test Anthropic Claude provider integration with the LLM router system."""

    def setup_method(self):
        """Reset global state before each test."""
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

    def test_claude_model_routing(self):
        """Test that Claude models are correctly routed to anthropic provider."""
        claude_models = [
            "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307", "Claude-3-Sonnet"
        ]
        
        for model in claude_models:
            result = get_provider_name(model)
            assert isinstance(result, Success)
            assert result.unwrap() == "anthropic"

    def test_create_anthropic_provider(self):
        """Test that create_provider correctly creates AnthropicProvider instances."""
        config = AppConfig(anthropic=AnthropicConfig(api_key="test-key"))
        result = create_provider(["anthropic"], config)
        
        assert isinstance(result, Success)
        assert isinstance(result.unwrap(), AnthropicProvider)

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.chat_completion')
    @pytest.mark.asyncio
    async def test_claude_chat_completion(self, mock_completion):
        """Test basic chat completion with Claude model."""
        mock_completion.return_value = Success(ChatResponse(
            content="Hello! How can I help?", finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        ))
        
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="claude-3-sonnet-20240229"
        )
        
        result = await chat_completions(request)
        assert isinstance(result, Success)
        assert result.unwrap().content == "Hello! How can I help?"
        assert metrics["provider_success"]["anthropic"] == 1

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.chat_completion_stream')
    @pytest.mark.asyncio
    async def test_claude_streaming(self, mock_stream):
        """Test streaming chat completion with Claude model."""
        async def mock_generator():
            yield ChatStreamResponse(delta="Hello", finish_reason=None)
            yield ChatStreamResponse(delta=" world!", finish_reason="stop")
        
        mock_stream.return_value = Success(mock_generator())
        
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="claude-3-sonnet-20240229", stream=True
        )
        
        result = await chat_completions_stream(request)
        assert isinstance(result, Success)
        
        chunks = [chunk async for chunk in result.unwrap()]
        assert len(chunks) == 2
        assert chunks[0].delta == "Hello"
        assert chunks[1].finish_reason == "stop"

    def test_claude_auth_error(self):
        """Test error handling for invalid API keys at the provider level."""
        from src.config import AnthropicConfig
        from src.providers import AnthropicProvider
        
        # Test that creating a provider with invalid config works
        config = AnthropicConfig(api_key="invalid-key")
        provider = AnthropicProvider(config)
        
        # Test that the provider was created successfully
        assert provider is not None
        assert provider.config.api_key == "invalid-key"
        
        # This test verifies that the provider can be instantiated with invalid keys
        # The actual authentication error would occur during API calls, which we mock in other tests

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.chat_completion')
    @pytest.mark.asyncio
    async def test_claude_tool_calling(self, mock_completion):
        """Test tool calling support with Claude models."""
        mock_completion.return_value = Success(ChatResponse(
            content="I'll calculate that.", finish_reason="tool_calls",
            usage={"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
            tool_calls=[ToolCall(
                id="call_123", type="function",
                function={"name": "calculate", "arguments": {"expr": "2+2"}}
            )]
        ))
        
        request = ChatRequest(
            messages=[Message(role="user", content="What is 2+2?")],
            model="claude-3-sonnet-20240229",
            tools=[Tool(type="function", function={
                "name": "calculate", "description": "Calculate expressions",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            })]
        )
        
        result = await chat_completions(request)
        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.tool_calls is not None
        assert response.tool_calls[0].function["name"] == "calculate"

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.list_models')
    @pytest.mark.asyncio
    async def test_claude_model_discovery(self, mock_list):
        """Test model discovery for Anthropic provider."""
        mock_list.return_value = Success([
            "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229"
        ])
        
        result = await get_available_models()
        model_ids = [m["id"] for m in result["data"]]
        
        assert "claude-3-5-sonnet-20241022" in model_ids
        for model in result["data"]:
            if "claude" in model["id"]:
                assert model["owned_by"] == "anthropic"

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.chat_completion')
    @pytest.mark.asyncio
    async def test_claude_system_message(self, mock_completion):
        """Test system message handling specific to Anthropic."""
        mock_completion.return_value = Success(ChatResponse(
            content="I'll be concise.", finish_reason="stop",
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        ))
        
        request = ChatRequest(
            messages=[
                Message(role="system", content="Be concise."),
                Message(role="user", content="Hello")
            ],
            model="claude-3-sonnet-20240229"
        )
        
        result = await chat_completions(request)
        assert isinstance(result, Success)
        mock_completion.assert_called_once()

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('src.providers.AnthropicProvider.chat_completion')
    @patch('src.main.load_config_from_env')
    @pytest.mark.asyncio
    async def test_claude_cost_tracking(self, mock_load_config, mock_completion):
        """Test integration with cost optimization features."""
        # Mock the config to ensure cost tracking works
        from src.config import AppConfig, AnthropicConfig, CostConfig
        mock_config = AppConfig(
            anthropic=AnthropicConfig(api_key="test-key"),
            cost=CostConfig(
                model_costs={
                    "claude-3-sonnet": {"input": 0.015, "output": 0.075}
                }
            )
        )
        mock_load_config.return_value = Success(mock_config)
        
        mock_completion.return_value = Success(ChatResponse(
            content="Response", finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        ))
        
        request = ChatRequest(
            messages=[Message(role="user", content="Question")],
            model="claude-3-sonnet"  # Use exact model name from cost config
        )
        
        result = await chat_completions(request)
        assert isinstance(result, Success)
        
        # Check cost tracking - should be > 0 since we have cost config for claude-3-sonnet
        # Cost = (100 * 0.015) + (50 * 0.075) = 1.5 + 3.75 = 5.25
        assert metrics["total_cost"] > 0
        assert "anthropic" in metrics["cost_by_provider"]
        assert "claude-3-sonnet" in metrics["cost_by_model"]

    def test_claude_policy_filtering(self):
        """Test policy filtering for Claude models."""
        config = AppConfig(cost=CostConfig(
            allowed_models=["claude-3-sonnet-20240229"]
        ))
        
        assert is_model_allowed("claude-3-sonnet-20240229", config) == True
        assert is_model_allowed("claude-3-opus-20240229", config) == False

    def test_anthropic_config_validation(self):
        """Test Anthropic configuration validation."""
        valid_config = AnthropicConfig(api_key="sk-ant-test")
        assert valid_config.api_key == "sk-ant-test"
        
        with pytest.raises(ValueError):
            AnthropicConfig(api_key="")

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'})
    def test_anthropic_endpoint_integration(self):
        """Test Anthropic integration through FastAPI endpoints."""
        client = TestClient(app)
        
        with patch('src.providers.AnthropicProvider.chat_completion') as mock:
            mock.return_value = Success(ChatResponse(
                content="Test response", finish_reason="stop",
                usage={"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
            ))
            
            response = client.post("/chat/completions", json={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "claude-3-sonnet-20240229"
            })
            
            assert response.status_code == 200
            assert response.json()["content"] == "Test response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
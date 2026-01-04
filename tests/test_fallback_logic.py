import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure

from src.config import AppConfig, OpenAIConfig, GoogleGeminiConfig, load_config_from_env, get_provider_config
from src.providers import OpenAIProvider, GeminiProvider
from src.models import ChatRequest, Message, ChatResponse
from src.main import create_provider, chat_completions


class TestFallbackLogic:
    """Test fallback logic when providers are unavailable."""

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
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    def test_create_provider_success_first_choice(self):
        """Test creating provider when first choice succeeds."""
        config_result = load_config_from_env()
        config = config_result.unwrap()

        result = create_provider(["openai", "google_gemini"], config)
        assert isinstance(result, Success)
        assert isinstance(result.unwrap(), OpenAIProvider)

    @patch.dict(os.environ, {
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    def test_create_provider_fallback_to_second(self):
        """Test creating provider when first fails and second succeeds."""
        config_result = load_config_from_env()
        config = config_result.unwrap()

        result = create_provider(["openai", "google_gemini"], config)
        assert isinstance(result, Success)
        assert isinstance(result.unwrap(), GeminiProvider)

    @patch.dict(os.environ, {}, clear=True)
    def test_create_provider_all_fail(self):
        """Test creating provider when all providers fail."""
        config_result = load_config_from_env()
        config = config_result.unwrap()

        result = create_provider(["openai", "google_gemini"], config)
        assert isinstance(result, Failure)
        assert "All providers failed" in result.failure()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    def test_create_provider_invalid_provider_name(self):
        """Test creating provider with invalid provider name."""
        config_result = load_config_from_env()
        config = config_result.unwrap()

        result = create_provider(["invalid_provider"], config)
        assert isinstance(result, Failure)
        assert "not configured" in result.failure()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_chat_completions_openai_fails_gemini_succeeds(self, mock_genai_model, mock_openai_client):
        """Test successful fallback from OpenAI failure to Gemini success."""
        # Mock OpenAI to fail
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API down")
        )

        # Mock Gemini to succeed
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
            model="gpt-4"  # This will try OpenAI first, then fallback to Gemini
        )

        result = await chat_completions(request)

        assert isinstance(result, Success)
        response = result.unwrap()
        assert response.content == "Gemini response"
        assert response.finish_reason == "stop"

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'invalid-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completions_openai_fails_no_fallback(self, mock_openai_client):
        """Test fallback from OpenAI to Gemini when OpenAI fails but Gemini not configured."""
        # Mock OpenAI to fail with authentication error
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(
            side_effect=Exception("Authentication failed")
        )

        # Gemini is not configured, so should still fail
        request = ChatRequest(
            messages=[Message(role="user", content="Fallback test message")],
            model="gpt-4"
        )

        result = await chat_completions(request)

        # Should fail because Gemini is not configured as fallback
        assert isinstance(result, Failure)

    def test_get_provider_config_success(self):
        """Test getting provider config successfully."""
        config = AppConfig(
            openai=OpenAIConfig(api_key="test-key"),
            google_gemini=GoogleGeminiConfig(api_key="test-key")
        )

        result = get_provider_config("openai", config)
        assert isinstance(result, Success)
        assert result.unwrap().api_key == "test-key"

    def test_get_provider_config_not_found(self):
        """Test getting provider config for non-existent provider."""
        config = AppConfig()

        result = get_provider_config("nonexistent", config)
        assert isinstance(result, Failure)
        assert "not configured" in result.failure()

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_provider_initialization_error_handling(self, mock_openai_client):
        """Test error handling during provider initialization."""
        # Mock OpenAI client to raise exception during initialization
        mock_openai_client.side_effect = Exception("Failed to initialize OpenAI client")

        config_result = load_config_from_env()
        config = config_result.unwrap()

        result = create_provider(["openai"], config)
        assert isinstance(result, Failure)
        assert "Failed to initialize" in result.failure()


class TestConfigurationLoading:
    """Test configuration loading from environment variables."""

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
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    def test_load_config_with_openai_and_gemini(self):
        """Test loading config with both OpenAI and Gemini providers."""
        result = load_config_from_env()

        assert isinstance(result, Success)
        config = result.unwrap()

        assert config.openai is not None
        assert config.openai.api_key == 'test-openai-key'
        assert config.google_gemini is not None
        assert config.google_gemini.api_key == 'test-gemini-key'

    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_no_env_vars(self):
        """Test loading config when no environment variables are set."""
        result = load_config_from_env()

        assert isinstance(result, Success)
        config = result.unwrap()

        assert config.openai is None
        assert config.google_gemini is None

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_DEFAULT_MODEL': 'gpt-3.5-turbo',
        'OPENAI_TEMPERATURE': '0.5',
        'OPENAI_MAX_TOKENS': '2048'
    })
    def test_load_config_custom_values(self):
        """Test loading config with custom values."""
        result = load_config_from_env()

        assert isinstance(result, Success)
        config = result.unwrap()

        assert config.openai.default_model == 'gpt-3.5-turbo'
        assert config.openai.temperature == 0.5
        assert config.openai.max_tokens == 2048

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'ANTHROPIC_DEFAULT_MODEL': 'claude-3-haiku-20240307',
        'ANTHROPIC_TEMPERATURE': '0.3',
        'ANTHROPIC_MAX_TOKENS': '8192'
    })
    def test_load_config_anthropic_custom_values(self):
        """Test loading Anthropic config with custom values."""
        result = load_config_from_env()

        assert isinstance(result, Success)
        config = result.unwrap()

        assert config.anthropic is not None
        assert config.anthropic.api_key == 'test-anthropic-key'
        assert config.anthropic.default_model == 'claude-3-haiku-20240307'
        assert config.anthropic.temperature == 0.3
        assert config.anthropic.max_tokens == 8192


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
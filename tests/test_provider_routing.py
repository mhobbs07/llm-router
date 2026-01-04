import os
import pytest
from unittest.mock import Mock, patch
from returns.result import Success, Failure

from src.config import AppConfig, OpenAIConfig, GoogleGeminiConfig, AnthropicConfig
from src.providers import OpenAIProvider, GeminiProvider, AnthropicProvider
from src.main import get_provider_name, get_fallback_providers, create_provider


class TestProviderRouting:
    """Test provider routing logic."""

    def test_get_provider_name_openai(self):
        """Test routing for OpenAI models."""
        result = get_provider_name("gpt-4")
        assert isinstance(result, Success)
        assert result.unwrap() == "openai"

        result = get_provider_name("chatgpt")
        assert isinstance(result, Success)
        assert result.unwrap() == "openai"

    def test_get_provider_name_gemini(self):
        """Test routing for Gemini models."""
        result = get_provider_name("gemini-pro")
        assert isinstance(result, Success)
        assert result.unwrap() == "google_gemini"

    def test_get_provider_name_claude(self):
        """Test routing for Claude models."""
        result = get_provider_name("claude-3-sonnet-20240229")
        assert isinstance(result, Success)
        assert result.unwrap() == "anthropic"

        result = get_provider_name("claude-3-haiku-20240307")
        assert isinstance(result, Success)
        assert result.unwrap() == "anthropic"

    def test_get_provider_name_grok(self):
        """Test routing for Grok models."""
        result = get_provider_name("grok-1")
        assert isinstance(result, Success)
        assert result.unwrap() == "xai"

    def test_get_provider_name_unknown(self):
        """Test routing for unknown models defaults to huggingface."""
        result = get_provider_name("unknown-model")
        assert isinstance(result, Success)
        assert result.unwrap() == "huggingface"

    def test_get_fallback_providers_openai(self):
        """Test fallback providers for OpenAI."""
        fallbacks = get_fallback_providers("openai")
        assert fallbacks == ["openai", "google_gemini"]

    def test_get_fallback_providers_gemini(self):
        """Test fallback providers for Gemini."""
        fallbacks = get_fallback_providers("google_gemini")
        assert fallbacks == ["google_gemini", "openai"]

    def test_get_fallback_providers_anthropic(self):
        """Test fallback providers for Anthropic."""
        fallbacks = get_fallback_providers("anthropic")
        assert fallbacks == ["anthropic"]

    def test_get_fallback_providers_unknown(self):
        """Test fallback providers for unknown provider."""
        fallbacks = get_fallback_providers("unknown")
        assert fallbacks == ["unknown"]


class TestProviderInstantiation:
    """Test provider instantiation with valid configurations."""

    def test_openai_provider_instantiation(self):
        """Test that OpenAI provider can be instantiated with valid config."""
        config = OpenAIConfig(
            api_key="test-openai-key",
            default_model="gpt-4",
            temperature=0.7,
            max_tokens=4096
        )

        provider = OpenAIProvider(config)

        assert provider.config == config
        assert provider.client is not None
        assert hasattr(provider.client, 'chat')
        assert hasattr(provider.client.chat, 'completions')

    def test_gemini_provider_instantiation(self):
        """Test that Gemini provider can be instantiated with valid config."""
        config = GoogleGeminiConfig(
            api_key="test-gemini-key",
            default_model="gemini-pro",
            temperature=0.7,
            max_tokens=4096
        )

        with patch('google.generativeai.configure') as mock_configure:
            provider = GeminiProvider(config)

            assert provider.config == config
            mock_configure.assert_called_once_with(api_key="test-gemini-key")

    def test_anthropic_provider_instantiation(self):
        """Test that Anthropic provider can be instantiated with valid config."""
        config = AnthropicConfig(
            api_key="test-anthropic-key",
            default_model="claude-3-sonnet-20240229",
            temperature=0.7,
            max_tokens=4096
        )

        provider = AnthropicProvider(config)

        assert provider.config == config
        assert provider.client is not None

    def test_openai_provider_invalid_config(self):
        """Test that OpenAI provider fails with invalid config."""
        with pytest.raises(ValueError):
            OpenAIConfig(api_key="")  # Empty API key should fail validation

    def test_gemini_provider_invalid_config(self):
        """Test that Gemini provider fails with invalid config."""
        with pytest.raises(ValueError):
            GoogleGeminiConfig(api_key="")  # Empty API key should fail validation

    def test_anthropic_provider_invalid_config(self):
        """Test that Anthropic provider fails with invalid config."""
        with pytest.raises(ValueError):
            AnthropicConfig(api_key="")  # Empty API key should fail validation


class TestProviderCreation:
    """Test provider creation logic."""

    def test_create_openai_provider_success(self):
        """Test creating OpenAI provider successfully."""
        config = AppConfig(
            openai=OpenAIConfig(api_key="test-key")
        )

        result = create_provider(["openai"], config)
        assert isinstance(result, Success)
        assert isinstance(result.unwrap(), OpenAIProvider)

    def test_create_gemini_provider_success(self):
        """Test creating Gemini provider successfully."""
        config = AppConfig(
            google_gemini=GoogleGeminiConfig(api_key="test-key")
        )

        with patch('google.generativeai.configure'):
            result = create_provider(["google_gemini"], config)
            assert isinstance(result, Success)
            assert isinstance(result.unwrap(), GeminiProvider)

    def test_create_anthropic_provider_success(self):
        """Test creating Anthropic provider successfully."""
        config = AppConfig(
            anthropic=AnthropicConfig(api_key="test-key")
        )

        result = create_provider(["anthropic"], config)
        assert isinstance(result, Success)
        assert isinstance(result.unwrap(), AnthropicProvider)

    def test_create_provider_missing_config(self):
        """Test creating provider when config is missing."""
        config = AppConfig()  # No provider configs

        result = create_provider(["openai"], config)
        assert isinstance(result, Failure)
        assert "not configured" in result.failure()

    def test_create_provider_invalid_name(self):
        """Test creating provider with invalid provider name."""
        config = AppConfig(
            openai=OpenAIConfig(api_key="test-key")
        )

        result = create_provider(["invalid_provider"], config)
        assert isinstance(result, Failure)
        assert "not configured" in result.failure()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
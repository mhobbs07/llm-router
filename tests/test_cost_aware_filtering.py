import os
import json
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure
from fastapi.testclient import TestClient
import datetime
import time

from src.config import (
    AppConfig, OpenAIConfig, GoogleGeminiConfig,
    load_config_from_env, get_provider_config, CostConfig
)
from src.providers import OpenAIProvider, GeminiProvider
from src.models import ChatRequest, Message, ChatResponse, ChatStreamResponse, PolicyRequest
from src.main import (
    get_provider_name, get_fallback_providers,
    create_provider, chat_completions, chat_completions_stream, app, metrics,
    is_model_allowed, filter_models_by_policy, get_available_models,
    estimate_model_cost, select_optimal_model, check_budget_limits,
    cost_tracker, response_cache
)


class TestCostAwareModelFiltering:
    """Test cost-aware model filtering functionality including dynamic discovery, policy filtering, and policy management."""

    def setup_method(self):
        """Reset global state before each test."""
        from src.main import response_cache, metrics, cost_tracker
        response_cache.clear()
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

    def test_is_model_allowed_allowed_models_whitelist(self):
        """Test model allowance with allowed models whitelist."""
        from src.main import is_model_allowed

        config = AppConfig(cost=CostConfig(
            allowed_models=["gpt-4", "gemini-pro"],
            allow_unknown_models=False
        ))

        assert is_model_allowed("gpt-4", config) == True
        assert is_model_allowed("gemini-pro", config) == True
        assert is_model_allowed("gpt-3.5-turbo", config) == False  # Not in whitelist

    def test_is_model_allowed_blocked_models(self):
        """Test model allowance with blocked models list."""
        from src.main import is_model_allowed

        config = AppConfig(cost=CostConfig(
            blocked_models=["gpt-3.5-turbo"],
            allow_unknown_models=True
        ))

        assert is_model_allowed("gpt-4", config) == True
        assert is_model_allowed("gpt-3.5-turbo", config) == False  # Blocked
        assert is_model_allowed("unknown-model", config) == True  # Unknown allowed

    def test_is_model_allowed_unknown_models_not_allowed(self):
        """Test model allowance when unknown models are not allowed."""
        from src.main import is_model_allowed

        config = AppConfig(cost=CostConfig(
            allow_unknown_models=False,
            model_costs={"gpt-4": {"input": 0.03, "output": 0.06}}
        ))

        assert is_model_allowed("gpt-4", config) == True  # Known model
        assert is_model_allowed("unknown-model", config) == False  # Unknown not allowed

    def test_is_model_allowed_combined_policies(self):
        """Test model allowance with combined whitelist and blacklist."""
        from src.main import is_model_allowed

        config = AppConfig(cost=CostConfig(
            allowed_models=["gpt-4", "gemini-pro", "claude-3-sonnet"],
            blocked_models=["gpt-3.5-turbo"],
            allow_unknown_models=False
        ))

        assert is_model_allowed("gpt-4", config) == True  # In whitelist
        assert is_model_allowed("gemini-pro", config) == True  # In whitelist
        assert is_model_allowed("gpt-3.5-turbo", config) == False  # Blocked despite being in allowed
        assert is_model_allowed("claude-3-sonnet", config) == True  # In whitelist and in model_costs
        assert is_model_allowed("unknown-model", config) == False  # Not in whitelist

    def test_filter_models_by_policy_no_filters(self):
        """Test model filtering with no policy filters."""
        from src.main import filter_models_by_policy

        config = AppConfig(cost=CostConfig())
        models = ["gpt-4", "gemini-pro", "claude-3", "unknown-model"]

        filtered = filter_models_by_policy(models, config)
        assert filtered == models  # All should pass

    def test_filter_models_by_policy_with_filters(self):
        """Test model filtering with policy filters applied."""
        from src.main import filter_models_by_policy

        config = AppConfig(cost=CostConfig(
            allowed_models=["gpt-4", "gemini-pro"],
            blocked_models=["gpt-3.5-turbo"],
            allow_unknown_models=False
        ))
        models = ["gpt-4", "gpt-3.5-turbo", "gemini-pro", "claude-3", "unknown-model"]

        filtered = filter_models_by_policy(models, config)
        assert filtered == ["gpt-4", "gemini-pro"]  # Only allowed models

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key',
        'GOOGLE_GEMINI_API_KEY': 'test-gemini-key'
    })
    @patch('src.providers.OpenAIProvider.list_models')
    @patch('src.providers.GeminiProvider.list_models')
    @pytest.mark.asyncio
    async def test_get_available_models_success(self, mock_gemini_list, mock_openai_list):
        """Test successful dynamic model discovery."""
        from src.main import get_available_models

        # Mock provider responses as async
        mock_openai_list.return_value = Success(["gpt-4", "gpt-3.5-turbo"])
        mock_gemini_list.return_value = Success(["gemini-pro", "gemini-pro-vision"])

        result = await get_available_models()

        assert "object" in result
        assert result["object"] == "list"
        assert "data" in result
        assert len(result["data"]) == 4

        # Check that models are properly formatted
        model_ids = [model["id"] for model in result["data"]]
        assert "gpt-4" in model_ids
        assert "gpt-3.5-turbo" in model_ids
        assert "gemini-pro" in model_ids
        assert "gemini-pro-vision" in model_ids

        # Check ownership
        for model in result["data"]:
            if "gpt" in model["id"]:
                assert model["owned_by"] == "openai"
            elif "gemini" in model["id"]:
                assert model["owned_by"] == "google"

    @patch.dict(os.environ, {}, clear=True)
    @pytest.mark.asyncio
    async def test_get_available_models_no_config(self):
        """Test model discovery when no providers are configured."""
        from src.main import get_available_models

        result = await get_available_models()

        assert result["object"] == "list"
        assert result["data"] == []  # Empty list when no providers configured

    @patch('src.main.load_config_from_env')
    @patch('src.providers.OpenAIProvider.list_models')
    @pytest.mark.asyncio
    async def test_get_available_models_with_policy_filtering(self, mock_openai_list, mock_load_config):
        """Test model discovery with policy filtering applied."""
        from src.main import get_available_models

        # Mock provider response
        mock_openai_list.return_value = Success(["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"])

        # Mock config with policy settings
        config = AppConfig(
            openai=OpenAIConfig(api_key="test-key"),
            cost=CostConfig(
                allowed_models=["gpt-4", "gpt-4-turbo"],
                allow_unknown_models=False,
                model_costs={
                    "gpt-4": {"input": 0.03, "output": 0.06},
                    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
                }
            )
        )
        mock_load_config.return_value = Success(config)

        result = await get_available_models()

        assert len(result["data"]) == 2
        model_ids = [model["id"] for model in result["data"]]
        assert "gpt-4" in model_ids
        assert "gpt-4-turbo" in model_ids
        assert "gpt-3.5-turbo" not in model_ids  # Filtered out

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_policy_endpoints_get_success(self):
        """Test GET /models/policy endpoint success."""

        client = TestClient(app)

        response = client.get("/models/policy")

        assert response.status_code == 200
        data = response.json()

        # Should return default policy configuration
        assert "allowed_models" in data
        assert "blocked_models" in data
        assert "cost_limits" in data
        assert "allow_unknown_models" in data
        assert "enforce_budget_limits" in data

    @patch.dict(os.environ, {
        'COST_ALLOWED_MODELS_JSON': '["gpt-4", "gemini-pro"]',
        'COST_BLOCKED_MODELS_JSON': '["gpt-3.5-turbo"]',
        'COST_ALLOW_UNKNOWN_MODELS': 'false',
        'COST_ENFORCE_BUDGET_LIMITS': 'true'
    })
    def test_policy_endpoints_get_with_custom_config(self):
        """Test GET /models/policy endpoint with custom configuration."""
        client = TestClient(app)

        response = client.get("/models/policy")

        assert response.status_code == 200
        data = response.json()

        assert data["allowed_models"] == ["gpt-4", "gemini-pro"]
        assert data["blocked_models"] == ["gpt-3.5-turbo"]
        assert data["allow_unknown_models"] == False
        assert data["enforce_budget_limits"] == True

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_policy_endpoints_post_update_allowed_models(self):
        """Test POST /models/policy endpoint updating allowed models."""

        client = TestClient(app)

        update_request = PolicyRequest(
            allowed_models=["gpt-4", "claude-3"],
            blocked_models=None,
            allow_unknown_models=False
        )

        response = client.post("/models/policy", json=update_request.dict())

        assert response.status_code == 200
        data = response.json()

        assert data["allowed_models"] == ["gpt-4", "claude-3"]
        assert data["allow_unknown_models"] == False

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_policy_endpoints_post_update_cost_limits(self):
        """Test POST /models/policy endpoint updating cost limits."""

        client = TestClient(app)

        update_request = PolicyRequest(
            cost_limits={"gpt-4": 10.0, "gemini-pro": 5.0}
        )

        response = client.post("/models/policy", json=update_request.dict())

        assert response.status_code == 200
        data = response.json()

        assert data["cost_limits"]["gpt-4"] == 10.0
        assert data["cost_limits"]["gemini-pro"] == 5.0

    def test_estimate_model_cost_known_model(self):
        """Test cost estimation for known models."""
        from src.main import estimate_model_cost

        config = AppConfig(cost=CostConfig())

        cost = estimate_model_cost("gpt-4", 1000, 500, config)
        expected_cost = (1000 * 0.03) + (500 * 0.06)  # input + output costs
        assert cost == expected_cost

    def test_estimate_model_cost_unknown_model(self):
        """Test cost estimation for unknown models."""
        from src.main import estimate_model_cost

        config = AppConfig(cost=CostConfig())

        cost = estimate_model_cost("unknown-model", 1000, 500, config)
        assert cost == 0.0  # Unknown models return 0 cost

    def test_select_optimal_model_cost_optimized(self):
        """Test model selection with cost optimization."""
        from src.main import select_optimal_model
        from src.models import ChatRequest, Message

        config = AppConfig(cost=CostConfig())

        # Short request should prefer cheaper models
        request = ChatRequest(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4"
        )

        optimal = select_optimal_model(request, config)
        # Should select from cost-optimized hierarchy
        assert optimal in ["gemini-pro", "grok-1", "gpt-3.5-turbo", "gpt-4", "claude-3-haiku", "claude-3-sonnet"]

    def test_select_optimal_model_performance_for_complex(self):
        """Test model selection prefers performance for complex requests."""
        from src.main import select_optimal_model
        from src.models import ChatRequest, Message, Tool

        config = AppConfig(cost=CostConfig())

        # Complex request with tools should prefer performance
        request = ChatRequest(
            messages=[Message(role="user", content="Analyze this complex data")] * 10,  # Long context
            model="gpt-4",
            tools=[Tool(type="function", function={"name": "analyze", "parameters": {}})]
        )

        optimal = select_optimal_model(request, config)
        # Should prefer performance models for complex requests
        assert optimal in ["gpt-4", "claude-3-sonnet"]

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_budget_limit_checking(self):
        """Test budget limit checking and alerts."""
        from src.main import check_budget_limits, cost_tracker

        config = AppConfig(cost=CostConfig(
            budget_daily_limit=10.0,
            budget_monthly_limit=100.0,
            budget_alert_threshold=0.8
        ))

        # Set high usage to trigger alert
        cost_tracker["daily_spend"] = 9.0  # 90% of daily limit
        cost_tracker["monthly_spend"] = 85.0  # 85% of monthly limit

        check_budget_limits(config)

        # Should have sent alerts (tracked in budget_alerts_sent)
        assert len(cost_tracker["budget_alerts_sent"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
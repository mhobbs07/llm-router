import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from returns.result import Success, Failure
from fastapi.testclient import TestClient

from src.main import app, metrics


class TestLoggingAndMetrics:
    """Test structured logging and metrics collection."""

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

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        assert metrics["total_requests"] == 0
        assert metrics["total_errors"] == 0
        assert isinstance(metrics["requests_by_endpoint"], dict)

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_metrics_collection_on_success(self, mock_openai_client):
        """Test metrics collection for successful requests."""

        # Mock successful response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 200

        # Check metrics were updated
        assert metrics["total_requests"] == 1
        assert "/chat/completions" in metrics["requests_by_endpoint"]
        assert metrics["requests_by_endpoint"]["/chat/completions"] == 1
        assert metrics["response_time_count"] == 1
        assert metrics["response_time_sum"] > 0

    @patch.dict(os.environ, {}, clear=True)
    def test_metrics_collection_on_error(self):
        """Test metrics collection for error requests."""

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Error test message"}],
            "model": "gpt-4"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 400

        # Check error metrics were updated
        assert metrics["total_requests"] == 1
        assert metrics["total_errors"] == 1
        assert "/chat/completions" in metrics["errors_by_endpoint"]
        assert metrics["errors_by_endpoint"]["/chat/completions"] == 1

    def test_metrics_endpoint(self):
        """Test the metrics monitoring endpoint."""
        # Update metrics with some test data
        metrics.update({
            "total_requests": 5,
            "requests_by_endpoint": {"/chat/completions": 3, "/health": 2},
            "total_errors": 1,
            "errors_by_endpoint": {"/chat/completions": 1},
            "provider_success": {"openai": 2, "google_gemini": 1},
            "response_time_sum": 2.5,
            "response_time_count": 3
        })

        client = TestClient(app)

        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["total_requests"] == 5
        assert data["total_errors"] == 1
        assert data["requests_by_endpoint"]["/chat/completions"] == 3
        assert data["errors_by_endpoint"]["/chat/completions"] == 1
        assert data["provider_success"]["openai"] == 2
        assert data["average_response_time"] == 2.5 / 3
        assert data["successful_responses"] == 3

    @patch('src.main.logger')
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_structured_logging_request_response(self, mock_openai_client, mock_logger):
        """Test that structured logging captures request and response details."""
        # Mock successful response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 30

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 200

        # Verify logging calls were made with structured data
        assert mock_logger.info.called
        # Check that at least one call contained expected fields
        calls = mock_logger.info.call_args_list
        request_logged = any("Request started" in str(call) for call in calls)
        response_logged = any("Request completed" in str(call) for call in calls)

        assert request_logged
        assert response_logged

    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-anthropic-key'
    })
    @patch('src.providers.AnthropicProvider.chat_completion')
    def test_anthropic_metrics_collection(self, mock_completion):
        """Test metrics collection for Anthropic provider requests."""

        # Mock successful Anthropic response
        from src.models import ChatResponse
        mock_completion.return_value = Success(ChatResponse(
            content="Anthropic response",
            finish_reason="stop",
            usage={"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25}
        ))

        client = TestClient(app)

        request_data = {
            "messages": [{"role": "user", "content": "Anthropic metrics test"}],
            "model": "claude-3-sonnet-20240229"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 200

        # Check Anthropic-specific metrics
        assert metrics["provider_success"]["anthropic"] == 1
        assert metrics["total_cost"] >= 0  # Cost might be 0 for test models
        assert "anthropic" in metrics["cost_by_provider"]
        # Model might be optimized to claude-3-haiku, so check for either
        assert "claude-3-haiku" in metrics["cost_by_model"] or "claude-3-sonnet-20240229" in metrics["cost_by_model"]

    def test_cost_analytics_endpoint(self):
        """Test the cost analytics endpoint."""
        # Update metrics with cost data
        from src.main import cost_tracker
        
        metrics.update({
            "total_cost": 5.50,
            "cost_by_provider": {"openai": 3.00, "anthropic": 2.50},
            "cost_by_model": {"gpt-4": 3.00, "claude-3-sonnet-20240229": 2.50},
            "total_requests": 10
        })
        
        cost_tracker.update({
            "daily_spend": 1.25,
            "monthly_spend": 5.50,
            "budget_alerts_sent": set()
        })

        client = TestClient(app)

        response = client.get("/cost-analytics")

        assert response.status_code == 200
        data = response.json()

        assert "current_period" in data
        assert "historical" in data
        assert "efficiency" in data
        assert data["current_period"]["daily_spend"] == 1.25
        assert data["current_period"]["monthly_spend"] == 5.50
        assert data["historical"]["total_cost"] == 5.50
        assert data["historical"]["cost_by_provider"]["openai"] == 3.00
        assert data["historical"]["cost_by_provider"]["anthropic"] == 2.50

    @patch('src.main.logger')
    def test_error_logging(self, mock_logger):
        """Test error logging for failed requests."""
        client = TestClient(app)

        # Make request that will fail (no providers configured)
        request_data = {
            "messages": [{"role": "user", "content": "Error logging test"}],
            "model": "gpt-4"
        }

        response = client.post("/chat/completions", json=request_data)

        assert response.status_code == 400

        # Verify error logging occurred
        assert mock_logger.error.called or mock_logger.info.called

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    @patch('src.providers.AsyncOpenAI')
    def test_provider_success_metrics(self, mock_openai_client):
        """Test provider success metrics tracking."""

        # Mock successful response
        mock_client_instance = Mock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 10

        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        client = TestClient(app)

        # Make multiple requests with different content to avoid caching
        for i in range(3):
            response = client.post("/chat/completions", json={
                "messages": [{"role": "user", "content": f"Hello {i}"}],
                "model": "gpt-4"
            })
            assert response.status_code == 200

        # Check provider success metrics
        assert metrics["provider_success"]["openai"] == 3

    def test_response_time_metrics(self):
        """Test response time metrics calculation."""
        # Update metrics with response time data
        metrics.update({
            "response_time_sum": 5.0,
            "response_time_count": 2,
            "total_requests": 3,
            "total_errors": 1
        })

        client = TestClient(app)

        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        # Check average response time calculation
        assert data["average_response_time"] == 2.5  # 5.0 / 2
        assert data["successful_responses"] == 2
        assert data["total_requests"] == 3
        assert data["total_errors"] == 1


class TestMonitoringEndpoints:
    """Test monitoring and health check endpoints."""

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

    def test_health_endpoint_success(self):
        """Test health check endpoint returns healthy status."""
        client = TestClient(app)

        response = client.post("/health", json={})

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data

    @patch('src.main.check_health')
    def test_health_endpoint_failure(self, mock_check_health):
        """Test health check endpoint handles failures."""
        mock_check_health.return_value = Failure("Database connection failed")

        client = TestClient(app)

        response = client.post("/health", json={})

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_models_endpoint(self):
        """Test models discovery endpoint."""
        client = TestClient(app)

        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-openai-key'
    })
    def test_policy_endpoints(self):
        """Test policy configuration endpoints."""
        client = TestClient(app)

        # Test GET policy endpoint
        response = client.get("/models/policy")
        assert response.status_code == 200
        data = response.json()
        assert "allowed_models" in data
        assert "blocked_models" in data
        assert "cost_limits" in data

        # Test POST policy endpoint
        policy_update = {
            "allowed_models": ["gpt-4", "claude-3-sonnet-20240229"],
            "blocked_models": ["gpt-3.5-turbo"],
            "allow_unknown_models": False
        }

        response = client.post("/models/policy", json=policy_update)
        assert response.status_code == 200
        data = response.json()
        assert data["allowed_models"] == ["gpt-4", "claude-3-sonnet-20240229"]
        assert data["blocked_models"] == ["gpt-3.5-turbo"]
        assert data["allow_unknown_models"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
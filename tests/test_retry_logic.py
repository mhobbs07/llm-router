"""
Tests for retry logic with exponential backoff.

This module tests the retry utility functions in isolation,
focusing on the retry mechanism itself rather than provider integration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.retry import (
    retry_with_backoff,
    RetryConfig,
    _is_retryable_error,
    _calculate_delay,
    AGGRESSIVE_RETRY,
    CONSERVATIVE_RETRY,
    QUICK_RETRY
)


class TestRetryableErrorDetection:
    """Test the error categorization logic."""

    def test_retryable_network_errors(self):
        """Test that network-related errors are marked as retryable."""
        # Test various network error patterns
        retryable_errors = [
            Exception("Connection timeout"),
            Exception("Network error occurred"),
            Exception("Rate limit exceeded"),
            Exception("Service unavailable"),
            Exception("Internal server error 500"),
            Exception("Bad gateway 502"),
            Exception("Gateway timeout 504"),
        ]
        
        for error in retryable_errors:
            assert _is_retryable_error(error), f"Should be retryable: {error}"

    def test_non_retryable_errors(self):
        """Test that authentication and client errors are not retryable."""
        non_retryable_errors = [
            Exception("Authentication failed"),
            Exception("Invalid API key"),
            Exception("Bad request 400"),
            Exception("Not found 404"),
            Exception("Forbidden 403"),
        ]
        
        for error in non_retryable_errors:
            assert not _is_retryable_error(error), f"Should not be retryable: {error}"


class TestDelayCalculation:
    """Test exponential backoff delay calculation."""

    def test_exponential_backoff_progression(self):
        """Test that delays increase exponentially."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        # Test first few attempts
        assert _calculate_delay(0, config) == 1.0  # 1.0 * 2^0
        assert _calculate_delay(1, config) == 2.0  # 1.0 * 2^1
        assert _calculate_delay(2, config) == 4.0  # 1.0 * 2^2
        assert _calculate_delay(3, config) == 8.0  # 1.0 * 2^3

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)
        
        # Large attempt number should be capped
        delay = _calculate_delay(10, config)
        assert delay == 5.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delays."""
        config = RetryConfig(base_delay=2.0, jitter=True, jitter_range=0.1)
        
        # Generate multiple delays and check they vary
        # For attempt 1: base_delay * exponential_base^1 = 2.0 * 2^1 = 4.0
        delays = [_calculate_delay(1, config) for _ in range(10)]
        
        # All delays should be around 4.0 but with variation (±10% jitter)
        assert all(3.6 <= delay <= 4.4 for delay in delays)
        # Should have some variation (not all identical)
        assert len(set(delays)) > 1


class TestRetryConfig:
    """Test retry configuration objects."""

    def test_default_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_predefined_configs(self):
        """Test predefined retry configurations."""
        # Aggressive retry should have more attempts
        assert AGGRESSIVE_RETRY.max_attempts == 5
        assert AGGRESSIVE_RETRY.base_delay == 0.5
        
        # Conservative retry should have longer delays
        assert CONSERVATIVE_RETRY.max_attempts == 3
        assert CONSERVATIVE_RETRY.base_delay == 2.0
        
        # Quick retry should have minimal attempts
        assert QUICK_RETRY.max_attempts == 2
        assert QUICK_RETRY.base_delay == 0.1


class TestRetryWithBackoff:
    """Test the main retry function."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test that successful functions return immediately."""
        mock_func = AsyncMock(return_value="success")
        
        result = await retry_with_backoff(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Test that retryable errors trigger retries."""
        mock_func = AsyncMock()
        # Fail twice, then succeed
        mock_func.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Connection timeout"),
            "success"
        ]
        
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # Fast for testing
        result = await retry_with_backoff(mock_func, config=config)
        
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors don't trigger retries."""
        mock_func = AsyncMock(side_effect=Exception("Authentication failed"))
        
        config = RetryConfig(max_attempts=3)
        
        with pytest.raises(Exception, match="Authentication failed"):
            await retry_with_backoff(mock_func, config=config)
        
        # Should only be called once (no retries)
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_max_attempts_respected(self):
        """Test that retry stops after max_attempts."""
        mock_func = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await retry_with_backoff(mock_func, config=config)
        
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_delay_between_retries(self):
        """Test that there's actual delay between retry attempts."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            Exception("Timeout"),
            "success"
        ]
        
        config = RetryConfig(max_attempts=2, base_delay=0.1, jitter=False)
        
        start_time = asyncio.get_event_loop().time()
        result = await retry_with_backoff(mock_func, config=config)
        end_time = asyncio.get_event_loop().time()
        
        assert result == "success"
        # Should have taken at least the base delay time
        assert end_time - start_time >= 0.1
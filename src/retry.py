"""
Retry utility with exponential backoff for LLM provider calls.

This module provides a robust retry mechanism with configurable backoff
strategies for handling transient errors in LLM API calls.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable, TypeVar, Any, Union
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    """Maximum number of retry attempts (including initial attempt)."""
    
    base_delay: float = 1.0
    """Base delay in seconds for exponential backoff."""
    
    max_delay: float = 60.0
    """Maximum delay in seconds between retries."""
    
    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""
    
    jitter: bool = True
    """Whether to add random jitter to delay times."""
    
    jitter_range: float = 0.1
    """Range for jitter as a fraction of delay (0.1 = ±10%)."""


def _is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception to check.
        
    Returns:
        True if the error should be retried, False otherwise.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Network/connection errors are retryable
    retryable_patterns = [
        'timeout',
        'connection',
        'network',
        'temporary',
        'rate limit',
        'throttle',
        'service unavailable',
        'internal server error',
        'bad gateway',
        'gateway timeout',
        '429',  # Rate limit status code
        '500',  # Internal server error
        '502',  # Bad gateway
        '503',  # Service unavailable
        '504',  # Gateway timeout
    ]
    
    # Check if error message or type contains retryable patterns
    for pattern in retryable_patterns:
        if pattern in error_str or pattern in error_type:
            return True
    
    # Specific provider error types that are retryable
    retryable_types = [
        'ratelimiterror',
        'timeouterror',
        'connectionerror',
        'httpstatuserror',
    ]
    
    if error_type in retryable_types:
        return True
    
    return False


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for exponential backoff.
    
    Args:
        attempt: Current attempt number (0-based).
        config: Retry configuration.
        
    Returns:
        Delay in seconds.
    """
    # Calculate exponential delay
    delay = config.base_delay * (config.exponential_base ** attempt)
    
    # Cap at maximum delay
    delay = min(delay, config.max_delay)
    
    # Add jitter if enabled
    if config.jitter:
        jitter_amount = delay * config.jitter_range
        jitter = random.uniform(-jitter_amount, jitter_amount)
        delay = max(0, delay + jitter)
    
    return delay


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """
    Execute a function with retry and exponential backoff.
    
    Args:
        func: The async function to execute.
        *args: Positional arguments to pass to the function.
        config: Retry configuration. Uses default if None.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        The result of the function call.
        
    Raises:
        The last exception encountered if all retries fail.
    """
    if config is None:
        config = RetryConfig()
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_attempts} for {func.__name__}")
            result = await func(*args, **kwargs)
            
            if attempt > 0:
                logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Don't retry on the last attempt
            if attempt == config.max_attempts - 1:
                logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts: {str(e)}")
                break
            
            # Check if error is retryable
            if not _is_retryable_error(e):
                logger.warning(f"Non-retryable error in {func.__name__}: {str(e)}")
                break
            
            # Calculate delay and wait
            delay = _calculate_delay(attempt, config)
            logger.warning(
                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                f"Retrying in {delay:.2f}s..."
            )
            
            await asyncio.sleep(delay)
    
    # All retries failed, raise the last exception
    raise last_exception


def with_retry(config: RetryConfig = None):
    """
    Decorator to add retry behavior to async functions.
    
    Args:
        config: Retry configuration. Uses default if None.
        
    Returns:
        Decorated function with retry behavior.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_with_backoff(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# Predefined retry configurations for common scenarios
AGGRESSIVE_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True
)

CONSERVATIVE_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)

QUICK_RETRY = RetryConfig(
    max_attempts=2,
    base_delay=0.1,
    max_delay=1.0,
    exponential_base=2.0,
    jitter=False
)
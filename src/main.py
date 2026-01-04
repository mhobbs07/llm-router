import datetime
import json
import logging
import time
import hashlib
from typing import AsyncIterable, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pythonjsonlogger import jsonlogger
from returns.result import Result, Success, Failure

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from .models import ChatRequest, ChatResponse, PolicyRequest, PolicyResponse
from .config import load_config_from_env, get_provider_config, CURATED_MODELS, DEFAULT_MODEL, FALLBACK_ORDER
from .providers import OpenAIProvider, GeminiProvider, GrokProvider, AnthropicProvider, LLMProvider

app = FastAPI()

# Metrics storage
metrics = {
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
    "last_cost_reset": datetime.datetime.now(datetime.UTC).date()
}

# Response cache
response_cache: Dict[str, Dict[str, Any]] = {}

# Cost tracking
cost_tracker = {
    "daily_spend": 0.0,
    "monthly_spend": 0.0,
    "last_daily_reset": datetime.datetime.now(datetime.UTC).date(),
    "last_monthly_reset": datetime.datetime.now(datetime.UTC).replace(day=1).date(),
    "budget_alerts_sent": set()
}

# Set up structured JSON logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    endpoint = request.url.path
    logger.info("Request started", extra={
        "method": request.method,
        "url": str(request.url),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
    })
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        logger.info("Request completed", extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration": duration,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        })
        # Collect metrics
        metrics["total_requests"] += 1
        metrics["requests_by_endpoint"].setdefault(endpoint, 0)
        metrics["requests_by_endpoint"][endpoint] += 1
        if response.status_code >= 400:
            metrics["total_errors"] += 1
            metrics["errors_by_endpoint"].setdefault(endpoint, 0)
            metrics["errors_by_endpoint"][endpoint] += 1
        else:
            metrics["response_time_sum"] += duration
            metrics["response_time_count"] += 1
        return response
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Request failed", extra={
            "method": request.method,
            "url": str(request.url),
            "error": str(e),
            "duration": duration,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        })
        # Collect error metrics
        metrics["total_requests"] += 1
        metrics["requests_by_endpoint"].setdefault(endpoint, 0)
        metrics["requests_by_endpoint"][endpoint] += 1
        metrics["total_errors"] += 1
        metrics["errors_by_endpoint"].setdefault(endpoint, 0)
        metrics["errors_by_endpoint"][endpoint] += 1
        raise

class HealthCheckRequest(BaseModel):
    pass  # Empty request model for simplicity; can be extended with fields if needed

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str

async def check_health(request: HealthCheckRequest) -> Result[HealthCheckResponse, str]:
    try:
        # Simulate async health check logic (e.g., database connectivity, external services)
        # In a real implementation, perform actual checks here
        status = "healthy"
        timestamp = datetime.datetime.now().isoformat()
        return Success(HealthCheckResponse(status=status, timestamp=timestamp))
    except Exception as e:
        return Failure(str(e))


def get_provider_name(model: str) -> Result[str, str]:
    """Determine provider name based on model string."""
    # First check curated models
    if model in CURATED_MODELS:
        provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google_gemini",
            "xai": "xai"
        }
        return Success(provider_map[CURATED_MODELS[model].provider])
    
    # Fallback to pattern matching for unknown models
    model_lower = model.lower()
    if "gpt" in model_lower or "chatgpt" in model_lower:
        return Success("openai")
    elif "claude" in model_lower:
        return Success("anthropic")
    elif "grok" in model_lower:
        return Success("xai")
    elif "gemini" in model_lower:
        return Success("google_gemini")
    else:
        # Route unknown models to Hugging Face provider
        return Success("huggingface")


def get_fallback_providers(primary_provider: str) -> list[str]:
    """Get list of providers to try, starting with primary and including cost-optimized fallbacks."""
    # Cost-optimized fallback: never fallback to more expensive models
    fallbacks = {
        "openai": ["openai", "google_gemini"],  # Keep existing for legacy models
        "google_gemini": ["google_gemini", "xai"],  # Fallback to cheaper xAI
        "anthropic": ["anthropic", "google_gemini", "xai"],  # Fallback to cheaper options
        "xai": ["xai", "google_gemini"],  # xAI is cheapest, fallback to balanced
        "huggingface": ["huggingface"]
    }
    return fallbacks.get(primary_provider, [primary_provider])


def generate_request_hash(request: ChatRequest) -> str:
    """Generate a hash for request caching based on key request parameters."""
    # Create a deterministic string representation of the request
    request_str = json.dumps({
        "model": request.model,
        "messages": [msg.model_dump() for msg in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "tools": [tool.model_dump() for tool in request.tools] if request.tools else None
    }, sort_keys=True, default=str)
    return hashlib.sha256(request_str.encode()).hexdigest()


def get_cached_response(request_hash: str, config) -> Optional[ChatResponse]:
    """Retrieve cached response if available and not expired."""
    if not config.cost.caching_enabled:
        return None

    cache_entry = response_cache.get(request_hash)
    if not cache_entry:
        return None

    # Check TTL
    cached_time = cache_entry.get("timestamp", 0)
    if time.time() - cached_time > config.cost.cache_ttl_seconds:
        del response_cache[request_hash]
        return None

    return ChatResponse(**cache_entry["response"])


def set_cached_response(request_hash: str, response: ChatResponse, config):
    """Cache a response with timestamp."""
    if not config.cost.caching_enabled:
        return

    # Check cache size limit (simple approximation)
    cache_size_mb = len(json.dumps(response_cache).encode()) / (1024 * 1024)
    if cache_size_mb > config.cost.cache_max_size_mb:
        # Simple LRU: remove oldest entries
        oldest_keys = sorted(response_cache.keys(),
                           key=lambda k: response_cache[k]["timestamp"])[:10]
        for key in oldest_keys:
            del response_cache[key]

    response_cache[request_hash] = {
        "response": response.model_dump(),
        "timestamp": time.time()
    }


def calculate_cost(model: str, usage: Dict[str, Any], config) -> float:
    """Calculate cost based on model and token usage using curated model pricing."""
    if not usage:
        return 0.0
    
    # Use curated model pricing if available
    if model in CURATED_MODELS:
        metadata = CURATED_MODELS[model]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Convert from per-1M pricing to per-token
        input_cost = (input_tokens / 1_000_000) * metadata.pricing.input_per_1m
        output_cost = (output_tokens / 1_000_000) * metadata.pricing.output_per_1m
        return input_cost + output_cost
    
    # Fallback to legacy config pricing
    if model not in config.cost.model_costs:
        return 0.0

    model_costs = config.cost.model_costs[model]
    input_cost = usage.get("prompt_tokens", 0) * model_costs.get("input", 0)
    output_cost = usage.get("completion_tokens", 0) * model_costs.get("output", 0)
    return input_cost + output_cost


def is_model_allowed(model: str, config) -> bool:
    """Check if a model is allowed based on policy configuration."""
    cost_config = config.cost

    # Check blocked models
    if cost_config.blocked_models and model in cost_config.blocked_models:
        return False

    # Check allowed models (if whitelist is set)
    if cost_config.allowed_models and model not in cost_config.allowed_models:
        return False

    # Check if model has cost configuration (unless unknown models are allowed)
    if not cost_config.allow_unknown_models and model not in cost_config.model_costs:
        return False

    return True


def estimate_model_cost(model: str, input_tokens: int, output_tokens: int, config) -> float:
    """Estimate cost for a model based on token counts using curated model pricing."""
    # Use curated model pricing if available
    if model in CURATED_MODELS:
        metadata = CURATED_MODELS[model]
        input_cost = (input_tokens / 1_000_000) * metadata.pricing.input_per_1m
        output_cost = (output_tokens / 1_000_000) * metadata.pricing.output_per_1m
        return input_cost + output_cost
    
    # Fallback to legacy config pricing
    if model not in config.cost.model_costs:
        return 0.0

    model_costs = config.cost.model_costs[model]
    input_cost = input_tokens * model_costs.get("input", 0)
    output_cost = output_tokens * model_costs.get("output", 0)
    return input_cost + output_cost


def filter_models_by_policy(models: list[str], config) -> list[str]:
    """Filter a list of models based on policy configuration."""
    return [model for model in models if is_model_allowed(model, config)]


def update_cost_metrics(cost: float, provider: str, model: str):
    """Update cost tracking metrics."""
    metrics["total_cost"] += cost
    metrics["cost_by_provider"].setdefault(provider, 0.0)
    metrics["cost_by_provider"][provider] += cost
    metrics["cost_by_model"].setdefault(model, 0.0)
    metrics["cost_by_model"][model] += cost

    # Update daily/monthly costs
    today = datetime.datetime.now(datetime.UTC).date()
    month_start = today.replace(day=1)

    if cost_tracker["last_daily_reset"] != today:
        cost_tracker["daily_spend"] = 0.0
        cost_tracker["last_daily_reset"] = today

    if cost_tracker["last_monthly_reset"] != month_start:
        cost_tracker["monthly_spend"] = 0.0
        cost_tracker["last_monthly_reset"] = month_start

    cost_tracker["daily_spend"] += cost
    cost_tracker["monthly_spend"] += cost

    metrics["daily_cost"] = cost_tracker["daily_spend"]
    metrics["monthly_cost"] = cost_tracker["monthly_spend"]


def check_budget_limits(config):
    """Check if budget limits are exceeded and log warnings."""
    daily_usage = cost_tracker["daily_spend"] / config.cost.budget_daily_limit
    monthly_usage = cost_tracker["monthly_spend"] / config.cost.budget_monthly_limit

    alerts = []
    if daily_usage >= config.cost.budget_alert_threshold:
        alert_key = f"daily_{datetime.datetime.now(datetime.UTC).date()}"
        if alert_key not in cost_tracker["budget_alerts_sent"]:
            alerts.append(f"Daily budget usage at {daily_usage:.1%} (${cost_tracker['daily_spend']:.2f})")
            cost_tracker["budget_alerts_sent"].add(alert_key)

    if monthly_usage >= config.cost.budget_alert_threshold:
        alert_key = f"monthly_{datetime.datetime.now(datetime.UTC).replace(day=1).date()}"
        if alert_key not in cost_tracker["budget_alerts_sent"]:
            alerts.append(f"Monthly budget usage at {monthly_usage:.1%} (${cost_tracker['monthly_spend']:.2f})")
            cost_tracker["budget_alerts_sent"].add(alert_key)

    if alerts:
        logger.warning("Budget alert", extra={"alerts": alerts})


def log_model_selection(model: str, estimated_cost: float) -> None:
    """Log model selection with cost awareness and warnings."""
    if model not in CURATED_MODELS:
        logger.warning("Unknown model selected",
                      extra={"model": model, "suggestion": f"Consider {DEFAULT_MODEL} for better value"})
        return
    
    metadata = CURATED_MODELS[model]
    
    if metadata.tier == "premium":
        logger.warning(
            f"Premium model selected: {model}",
            extra={
                "model": model,
                "tier": "premium",
                "estimated_cost": estimated_cost,
                "pricing": {
                    "input_per_1m": metadata.pricing.input_per_1m,
                    "output_per_1m": metadata.pricing.output_per_1m
                },
                "suggestion": f"Consider {DEFAULT_MODEL} for better value",
                "cost_difference": f"Premium model costs {metadata.pricing.input_per_1m / CURATED_MODELS[DEFAULT_MODEL].pricing.input_per_1m:.1f}x more"
            }
        )
    elif metadata.tier == "balanced":
        logger.info("Balanced model selected",
                   extra={"model": model, "tier": "balanced", "estimated_cost": estimated_cost})
    else:  # budget
        logger.info("Budget model selected",
                   extra={"model": model, "tier": "budget", "estimated_cost": estimated_cost})


def optimize_context(request: ChatRequest, config) -> ChatRequest:
    """Optimize context by trimming old messages if context is too long."""
    # Simple optimization: keep only recent messages if total tokens exceed threshold
    max_context_tokens = getattr(config.cost, 'max_context_tokens', 8000)

    # Rough token estimation (4 chars per token)
    total_chars = sum(len(msg.content) for msg in request.messages if isinstance(msg.content, str))
    estimated_tokens = total_chars // 4

    if estimated_tokens <= max_context_tokens:
        return request

    # Keep system message and last N messages
    optimized_messages = []
    system_messages = [msg for msg in request.messages if msg.role == "system"]
    other_messages = [msg for msg in request.messages if msg.role != "system"]

    optimized_messages.extend(system_messages)

    # Keep last messages that fit within token limit
    remaining_tokens = max_context_tokens - sum(len(msg.content) // 4 for msg in system_messages if isinstance(msg.content, str))

    for msg in reversed(other_messages):
        msg_tokens = len(msg.content) // 4 if isinstance(msg.content, str) else 0
        if remaining_tokens >= msg_tokens:
            optimized_messages.insert(-len(system_messages), msg)
            remaining_tokens -= msg_tokens
        else:
            break

    return ChatRequest(
        messages=optimized_messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        system=request.system,
        tools=request.tools
    )


def select_optimal_model(request: ChatRequest, config) -> str:
    """Select optimal model based on cost and performance criteria using curated models."""
    # If model is not in curated list, use default
    if request.model not in CURATED_MODELS:
        logger.info("Unknown model requested, using default",
                   extra={"requested_model": request.model, "default_model": DEFAULT_MODEL})
        return DEFAULT_MODEL
    
    # Check if request needs high performance (long context, complex reasoning)
    total_chars = sum(len(msg.content) for msg in request.messages if isinstance(msg.content, str))
    has_tools = request.tools is not None and len(request.tools) > 0
    
    # Determine if premium model is needed
    needs_premium = (
        total_chars > 10000 or  # Long context
        has_tools or  # Tool usage
        any("complex" in msg.content.lower() or "reasoning" in msg.content.lower()
            for msg in request.messages if isinstance(msg.content, str))
    )
    
    requested_metadata = CURATED_MODELS[request.model]
    
    # If premium is needed and current model isn't premium, suggest upgrade
    if needs_premium and requested_metadata.tier != "premium":
        # Find best premium model for the same provider
        premium_models = [
            model_id for model_id, metadata in CURATED_MODELS.items()
            if metadata.tier == "premium" and metadata.provider == requested_metadata.provider
        ]
        if premium_models:
            selected = premium_models[0]  # Take first premium model
            logger.info("Upgrading to premium model for complex task",
                       extra={"original_model": request.model, "selected_model": selected})
            return selected
    
    # If simple task and using expensive model, suggest downgrade
    if not needs_premium and requested_metadata.tier == "premium":
        # Find best balanced model for same provider
        balanced_models = [
            model_id for model_id, metadata in CURATED_MODELS.items()
            if metadata.tier == "balanced" and metadata.provider == requested_metadata.provider
        ]
        if balanced_models:
            selected = balanced_models[0]
            logger.info("Downgrading to balanced model for simple task",
                       extra={"original_model": request.model, "selected_model": selected})
            return selected
        
        # If no balanced model, try budget
        budget_models = [
            model_id for model_id, metadata in CURATED_MODELS.items()
            if metadata.tier == "budget" and metadata.provider == requested_metadata.provider
        ]
        if budget_models:
            selected = budget_models[0]
            logger.info("Downgrading to budget model for simple task",
                       extra={"original_model": request.model, "selected_model": selected})
            return selected
    
    return request.model  # Keep original if no optimization needed


async def get_available_models() -> dict:
    """Get available models with cost-optimized curated list and pricing metadata."""
    # Load config
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        logger.error("Failed to load config for model discovery", extra={"error": config_result.failure()})
        return {"object": "list", "data": []}

    config = config_result.unwrap()

    # Get all curated models
    available_models = []
    
    # Check which providers are configured and add their models
    provider_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google_gemini",
        "xai": "xai"
    }
    
    for model_id, metadata in CURATED_MODELS.items():
        provider_name = provider_map[metadata.provider]
        provider_config_result = get_provider_config(provider_name, config)
        
        if isinstance(provider_config_result, Success):
            logger.debug(f"Provider {provider_name} configured, adding model {model_id}",
                        extra={"provider": provider_name, "model": model_id})
            available_models.append(model_id)
        else:
            logger.debug(f"Provider {provider_name} not configured, skipping model {model_id}",
                        extra={"provider": provider_name, "model": model_id})

    # Apply cost-aware filtering
    filtered_models = filter_models_by_cost_policy(available_models, config)

    # Use a fixed created timestamp (January 2026)
    created_timestamp = 1735689600  # January 1, 2026

    # Build response in OpenAI-compatible format with enhanced metadata
    model_data = []
    for model_id in filtered_models:
        metadata = CURATED_MODELS[model_id]
        
        model_entry = {
            "id": model_id,
            "object": "model",
            "created": created_timestamp,
            "owned_by": metadata.provider,
            "tier": metadata.tier,
            "description": metadata.description,
            "pricing": {
                "input_per_1m": metadata.pricing.input_per_1m,
                "output_per_1m": metadata.pricing.output_per_1m,
                "currency": "USD"
            },
            "context_window": metadata.context_window,
            "recommended_for": metadata.recommended_for
        }
        
        model_data.append(model_entry)

    # Sort by tier (budget first, then balanced, then premium) and then by cost
    def sort_key(model):
        tier_order = {"budget": 0, "balanced": 1, "premium": 2}
        return (tier_order[model["tier"]], model["pricing"]["input_per_1m"])
    
    model_data.sort(key=sort_key)

    return {
        "object": "list",
        "data": model_data
    }


def filter_models_by_cost_policy(models: list[str], config) -> list[str]:
    """Filter models based on cost policy configuration including cost limits."""
    cost_config = config.cost
    filtered_models = []

    for model in models:
        # Apply existing policy filters
        if not is_model_allowed(model, config):
            continue

        # Apply cost limit filtering
        if cost_config.cost_limits and model in cost_config.cost_limits:
            # Check if model has a cost limit and if current usage would exceed it
            # For now, we'll allow models with cost limits since we can't check current usage here
            # In a real implementation, you might want to check against current spending
            pass

        filtered_models.append(model)

    return filtered_models


def create_provider(provider_names: list[str], config) -> Result[LLMProvider, str]:
    """Create provider instance, trying providers in order with fallback logic."""
    errors = []

    for provider_name in provider_names:
        # Try to get config for this provider
        provider_config_result = get_provider_config(provider_name, config)
        if isinstance(provider_config_result, Failure):
            errors.append(f"{provider_name}: {provider_config_result.failure()}")
            continue

        provider_config = provider_config_result.unwrap()

        # Try to create the provider instance
        try:
            if provider_name == "openai":
                return Success(OpenAIProvider(provider_config))
            elif provider_name == "google_gemini":
                return Success(GeminiProvider(provider_config))
            elif provider_name == "xai":
                return Success(GrokProvider(provider_config))
            elif provider_name == "anthropic":
                return Success(AnthropicProvider(provider_config))
            else:
                errors.append(f"{provider_name}: Provider not implemented yet")
                continue
        except Exception as e:
            errors.append(f"{provider_name}: Failed to initialize - {str(e)}")
            continue

    # If we get here, all providers failed
    return Failure(f"All providers failed: {'; '.join(errors)}")


async def chat_completions(request: ChatRequest) -> Result[ChatResponse, str]:
    """Handle chat completions request with routing logic."""
    # Basic validation
    if not request.messages:
        return Failure("Messages cannot be empty")

    # Load config
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        return config_result

    config = config_result.unwrap()

    # Check budget limits
    check_budget_limits(config)

    # Optimize context
    request = optimize_context(request, config)

    # Smart model selection with cost optimization
    original_model = request.model
    request.model = select_optimal_model(request, config)

    if request.model != original_model:
        logger.info("Model optimized", extra={"original_model": original_model, "selected_model": request.model})

    # Estimate cost and log model selection with warnings
    estimated_tokens = sum(len(msg.content) // 4 for msg in request.messages if isinstance(msg.content, str))
    estimated_cost = estimate_model_cost(request.model, estimated_tokens, estimated_tokens // 2, config)
    log_model_selection(request.model, estimated_cost)

    # Check cache
    request_hash = generate_request_hash(request)
    cached_response = get_cached_response(request_hash, config)
    if cached_response:
        logger.info("Cache hit", extra={"model": request.model, "hash": request_hash[:8]})
        return Success(cached_response)

    # Get provider name
    provider_name_result = get_provider_name(request.model)
    if isinstance(provider_name_result, Failure):
        return provider_name_result

    provider_name = provider_name_result.unwrap()

    logger.info("Provider routing decision", extra={"model": request.model, "provider": provider_name, "endpoint": "chat_completions"})

    # Try providers in fallback order, but always start with the correct provider for the model
    fallback_providers = get_fallback_providers(provider_name)
    
    # For curated models, ensure we try the correct provider first
    if request.model in CURATED_MODELS:
        current_metadata = CURATED_MODELS[request.model]
        correct_provider = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google_gemini",
            "xai": "xai"
        }[current_metadata.provider]
        
        # Ensure the correct provider is first in the list
        if correct_provider not in fallback_providers:
            fallback_providers = [correct_provider] + fallback_providers
        elif fallback_providers[0] != correct_provider:
            # Move correct provider to front
            fallback_providers.remove(correct_provider)
            fallback_providers.insert(0, correct_provider)
    
    errors = []

    for provider_name_to_try in fallback_providers:
        # Try to get config for this provider
        provider_config_result = get_provider_config(provider_name_to_try, config)
        if isinstance(provider_config_result, Failure):
            errors.append(f"{provider_name_to_try}: {provider_config_result.failure()}")
            continue

        provider_config = provider_config_result.unwrap()

        # Try to create the provider instance
        try:
            if provider_name_to_try == "openai":
                provider = OpenAIProvider(provider_config)
            elif provider_name_to_try == "google_gemini":
                provider = GeminiProvider(provider_config)
            elif provider_name_to_try == "xai":
                provider = GrokProvider(provider_config)
            elif provider_name_to_try == "anthropic":
                provider = AnthropicProvider(provider_config)
            else:
                errors.append(f"{provider_name_to_try}: Provider not implemented yet")
                continue
        except Exception as e:
            errors.append(f"{provider_name_to_try}: Failed to initialize - {str(e)}")
            continue

        # Try to execute chat completion
        try:
            result = await provider.chat_completion(request)
            if isinstance(result, Success):
                response = result.unwrap()
                logger.info("Chat completion success", extra={"model": request.model, "provider": provider_name_to_try, "status": "success"})

                # Track cost
                cost = calculate_cost(request.model, response.usage, config)
                update_cost_metrics(cost, provider_name_to_try, request.model)

                # Cache response
                set_cached_response(request_hash, response, config)

                # Collect provider usage metric
                metrics["provider_success"].setdefault(provider_name_to_try, 0)
                metrics["provider_success"][provider_name_to_try] += 1

                logger.info("Cost tracked", extra={"model": request.model, "cost": cost, "total_cost": metrics["total_cost"]})
                return result
            else:
                errors.append(f"{provider_name_to_try}: {result.failure()}")
                continue
        except Exception as e:
            errors.append(f"{provider_name_to_try}: Unexpected error - {str(e)}")
            continue

    # If we get here, all providers failed
    logger.error("Chat completion failure", extra={"model": request.model, "provider": provider_name, "status": "failure", "error": f"All providers failed: {'; '.join(errors)}"})
    return Failure(f"All providers failed: {'; '.join(errors)}")


async def chat_completions_stream(request: ChatRequest) -> Result[AsyncIterable[dict], str]:
    """Handle streaming chat completions request with routing logic."""
    # Basic validation
    if not request.messages:
        return Failure("Messages cannot be empty")

    # Load config
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        return config_result

    config = config_result.unwrap()

    # Check budget limits
    check_budget_limits(config)

    # Optimize context
    request = optimize_context(request, config)

    # Smart model selection with cost optimization
    original_model = request.model
    request.model = select_optimal_model(request, config)

    if request.model != original_model:
        logger.info("Model optimized", extra={"original_model": original_model, "selected_model": request.model})

    # Estimate cost and log model selection with warnings
    estimated_tokens = sum(len(msg.content) // 4 for msg in request.messages if isinstance(msg.content, str))
    estimated_cost = estimate_model_cost(request.model, estimated_tokens, estimated_tokens // 2, config)
    log_model_selection(request.model, estimated_cost)

    # Get provider name
    provider_name_result = get_provider_name(request.model)
    if isinstance(provider_name_result, Failure):
        return provider_name_result

    provider_name = provider_name_result.unwrap()

    logger.info("Provider routing decision", extra={"model": request.model, "provider": provider_name, "endpoint": "chat_completions_stream"})

    # Try providers in fallback order, but always start with the correct provider for the model
    fallback_providers = get_fallback_providers(provider_name)
    
    # For curated models, ensure we try the correct provider first
    if request.model in CURATED_MODELS:
        current_metadata = CURATED_MODELS[request.model]
        correct_provider = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google_gemini",
            "xai": "xai"
        }[current_metadata.provider]
        
        # Ensure the correct provider is first in the list
        if correct_provider not in fallback_providers:
            fallback_providers = [correct_provider] + fallback_providers
        elif fallback_providers[0] != correct_provider:
            # Move correct provider to front
            fallback_providers.remove(correct_provider)
            fallback_providers.insert(0, correct_provider)
    
    errors = []

    for provider_name_to_try in fallback_providers:
        # Try to get config for this provider
        provider_config_result = get_provider_config(provider_name_to_try, config)
        if isinstance(provider_config_result, Failure):
            errors.append(f"{provider_name_to_try}: {provider_config_result.failure()}")
            continue

        provider_config = provider_config_result.unwrap()

        # Try to create the provider instance
        try:
            if provider_name_to_try == "openai":
                provider = OpenAIProvider(provider_config)
            elif provider_name_to_try == "google_gemini":
                provider = GeminiProvider(provider_config)
            elif provider_name_to_try == "xai":
                provider = GrokProvider(provider_config)
            elif provider_name_to_try == "anthropic":
                provider = AnthropicProvider(provider_config)
            else:
                errors.append(f"{provider_name_to_try}: Provider not implemented yet")
                continue
        except Exception as e:
            errors.append(f"{provider_name_to_try}: Failed to initialize - {str(e)}")
            continue

        # Try to execute streaming chat completion
        try:
            stream_result = await provider.chat_completion_stream(request)
            if isinstance(stream_result, Success):
                logger.info("Chat completion stream success", extra={"model": request.model, "provider": provider_name_to_try, "status": "success"})
                # Collect provider usage metric
                metrics["provider_success"].setdefault(provider_name_to_try, 0)
                metrics["provider_success"][provider_name_to_try] += 1

                # Wrap the stream to track costs when usage becomes available
                async def cost_tracking_stream():
                    async for chunk in stream_result.unwrap():
                        # Check if this chunk has usage information (typically the last chunk)
                        if hasattr(chunk, 'usage') and chunk.usage:
                            cost = calculate_cost(request.model, chunk.usage, config)
                            update_cost_metrics(cost, provider_name_to_try, request.model)
                            logger.info("Streaming cost tracked", extra={"model": request.model, "cost": cost, "total_cost": metrics["total_cost"]})
                        yield chunk

                return Success(cost_tracking_stream())
            else:
                errors.append(f"{provider_name_to_try}: {stream_result.failure()}")
                continue
        except Exception as e:
            errors.append(f"{provider_name_to_try}: Unexpected error - {str(e)}")
            continue

    # If we get here, all providers failed
    logger.error("Chat completion stream failure", extra={"model": request.model, "provider": provider_name, "status": "failure", "error": f"All providers failed: {'; '.join(errors)}"})
    return Failure(f"All providers failed: {'; '.join(errors)}")


@app.post("/health", response_model=HealthCheckResponse)
async def health_endpoint(request: HealthCheckRequest):
    result = await check_health(request)
    if isinstance(result, Success):
        return result.unwrap()
    else:
        raise HTTPException(status_code=500, detail=result.failure())

@app.get("/metrics")
async def metrics_endpoint():
    """Return JSON-formatted metrics for monitoring."""
    avg_response_time = (
        metrics["response_time_sum"] / metrics["response_time_count"]
        if metrics["response_time_count"] > 0 else 0.0
    )
    return {
        "total_requests": metrics["total_requests"],
        "requests_by_endpoint": metrics["requests_by_endpoint"],
        "total_errors": metrics["total_errors"],
        "errors_by_endpoint": metrics["errors_by_endpoint"],
        "provider_success": metrics["provider_success"],
        "average_response_time": avg_response_time,
        "successful_responses": metrics["response_time_count"],
        "cost_metrics": {
            "total_cost": metrics["total_cost"],
            "cost_by_provider": metrics["cost_by_provider"],
            "cost_by_model": metrics["cost_by_model"],
            "daily_cost": metrics["daily_cost"],
            "monthly_cost": metrics["monthly_cost"]
        },
        "cache_info": {
            "cache_size": len(response_cache),
            "cache_enabled": True  # Will be configurable
        }
    }

@app.get("/cost-analytics")
async def cost_analytics_endpoint():
    """Return detailed cost analytics and insights."""
    config_result = load_config_from_env()
    config = config_result.unwrap() if isinstance(config_result, Success) else None

    analytics = {
        "current_period": {
            "daily_spend": cost_tracker["daily_spend"],
            "monthly_spend": cost_tracker["monthly_spend"],
            "daily_budget": config.cost.budget_daily_limit if config else 10.0,
            "monthly_budget": config.cost.budget_monthly_limit if config else 100.0,
            "daily_usage_percent": (cost_tracker["daily_spend"] / (config.cost.budget_daily_limit if config else 10.0)) * 100,
            "monthly_usage_percent": (cost_tracker["monthly_spend"] / (config.cost.budget_monthly_limit if config else 100.0)) * 100
        },
        "historical": {
            "total_cost": metrics["total_cost"],
            "cost_by_provider": metrics["cost_by_provider"],
            "cost_by_model": metrics["cost_by_model"]
        },
        "efficiency": {
            "cache_hit_ratio": 0.0,  # Would need to track cache hits vs misses
            "average_cost_per_request": metrics["total_cost"] / max(metrics["total_requests"], 1),
            "cost_savings_potential": 0.0  # Could calculate based on model selection
        },
        "alerts": {
            "budget_warnings": list(cost_tracker["budget_alerts_sent"]) if cost_tracker["budget_alerts_sent"] else []
        }
    }

    return analytics

@app.post("/chat/completions")
async def chat_completions_endpoint(request: ChatRequest):
    # Handle streaming requests
    if request.stream:
        return await chat_completions_stream_endpoint(request)
    
    result = await chat_completions(request)
    if isinstance(result, Success):
        response = result.unwrap()
        
        # Convert to OpenAI-compatible format
        import uuid
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())
        
        openai_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content
                    },
                    "finish_reason": response.finish_reason or "stop"
                }
            ],
            "usage": response.usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return openai_response
    else:
        raise HTTPException(status_code=400, detail=result.failure())


@app.post("/chat/completions/stream")
async def chat_completions_stream_endpoint(request: ChatRequest):
    # Validate that stream is True
    if not request.stream:
        raise HTTPException(status_code=400, detail="Stream must be set to true for this endpoint")

    result = await chat_completions_stream(request)
    if isinstance(result, Failure):
        raise HTTPException(status_code=400, detail=result.failure())

    stream = result.unwrap()

    # Generate unique ID and timestamp for this completion
    import uuid
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_timestamp = int(time.time())

    async def generate():
        try:
            async for chunk in stream:
                # Format as OpenAI-compatible SSE
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.delta
                            },
                            "finish_reason": chunk.finish_reason
                        }
                    ]
                }
                yield f"data: {json.dumps(data)}\n\n"
            
            # Send final chunk with finish_reason
            final_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            
            # Send final done message
            yield "data: [DONE]\n\n"
        except Exception as e:
            # Send error as SSE
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/models/policy", response_model=PolicyResponse)
async def get_policy_endpoint():
    """Return current model policy configuration."""
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        raise HTTPException(status_code=500, detail=config_result.failure())

    config = config_result.unwrap()
    return PolicyResponse(
        allowed_models=config.cost.allowed_models,
        blocked_models=config.cost.blocked_models,
        cost_limits=config.cost.cost_limits,
        allow_unknown_models=config.cost.allow_unknown_models,
        enforce_budget_limits=config.cost.enforce_budget_limits
    )


@app.post("/models/policy", response_model=PolicyResponse)
async def update_policy_endpoint(request: PolicyRequest):
    """Update model policy configuration."""
    # Update environment variables based on the request
    import os
    import json

    if request.allowed_models is not None:
        os.environ["COST_ALLOWED_MODELS_JSON"] = json.dumps(request.allowed_models)
    if request.blocked_models is not None:
        os.environ["COST_BLOCKED_MODELS_JSON"] = json.dumps(request.blocked_models)
    if request.cost_limits is not None:
        os.environ["COST_COST_LIMITS_JSON"] = json.dumps(request.cost_limits)
    if request.allow_unknown_models is not None:
        os.environ["COST_ALLOW_UNKNOWN_MODELS"] = str(request.allow_unknown_models).lower()
    if request.enforce_budget_limits is not None:
        os.environ["COST_ENFORCE_BUDGET_LIMITS"] = str(request.enforce_budget_limits).lower()

    # Return the updated configuration
    config_result = load_config_from_env()
    if isinstance(config_result, Failure):
        raise HTTPException(status_code=500, detail=config_result.failure())

    config = config_result.unwrap()
    return PolicyResponse(
        allowed_models=config.cost.allowed_models,
        blocked_models=config.cost.blocked_models,
        cost_limits=config.cost.cost_limits,
        allow_unknown_models=config.cost.allow_unknown_models,
        enforce_budget_limits=config.cost.enforce_budget_limits
    )


@app.get("/models")
async def models_endpoint():
    """Return available models in OpenAI-compatible format for auto-discovery."""
    return await get_available_models()

# Ollama API compatibility endpoints for Open WebUI
@app.get("/ollama/api/version")
async def ollama_version_endpoint():
    """Return Ollama-compatible version information for Open WebUI."""
    return {"version": "0.1.0"}

@app.get("/api/version")
async def api_version_endpoint():
    """Alternative version endpoint for compatibility."""
    return {"version": "0.1.0"}

@app.get("/ollama/api/tags")
async def ollama_tags_endpoint():
    """Return available models in Ollama-compatible format."""
    models_data = await get_available_models()
    
    # Convert to Ollama format
    ollama_models = []
    for model in models_data.get("data", []):
        ollama_models.append({
            "name": model["id"],
            "model": model["id"],
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        })
    
    return {"models": ollama_models}

@app.post("/ollama/api/generate")
async def ollama_generate_endpoint(request: dict):
    """Handle Ollama-style generate requests by converting to chat completions."""
    # Convert Ollama format to ChatRequest format
    chat_request = ChatRequest(
        model=request.get("model", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": request.get("prompt", "")}],
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 1000)
    )
    
    if chat_request.stream:
        return await chat_completions_stream_endpoint(chat_request)
    else:
        result = await chat_completions(chat_request)
        if isinstance(result, Success):
            response = result.unwrap()
            # Convert to Ollama format
            return {
                "model": chat_request.model,
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "response": response.content,
                "done": True
            }
        else:
            raise HTTPException(status_code=400, detail=result.failure())

@app.post("/ollama/api/chat")
async def ollama_chat_endpoint(request: dict):
    """Handle Ollama-style chat requests."""
    # Convert Ollama format to ChatRequest format
    messages = request.get("messages", [])
    chat_request = ChatRequest(
        model=request.get("model", "gpt-3.5-turbo"),
        messages=messages,
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 1000)
    )
    
    if chat_request.stream:
        # For streaming, we need to convert the response format
        result = await chat_completions_stream(chat_request)
        if isinstance(result, Failure):
            raise HTTPException(status_code=400, detail=result.failure())
        
        stream = result.unwrap()
        
        async def ollama_stream_format():
            async for chunk in stream:
                # Convert to Ollama streaming format
                ollama_chunk = {
                    "model": chat_request.model,
                    "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "message": {
                        "role": "assistant",
                        "content": chunk.delta
                    },
                    "done": chunk.finish_reason is not None
                }
                yield f"data: {json.dumps(ollama_chunk)}\n\n"
            
            # Send final done message
            final_chunk = {
                "model": chat_request.model,
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "done": True
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        
        return StreamingResponse(
            ollama_stream_format(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    else:
        result = await chat_completions(chat_request)
        if isinstance(result, Success):
            response = result.unwrap()
            # Convert to Ollama format
            return {
                "model": chat_request.model,
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "message": {
                    "role": "assistant",
                    "content": response.content
                },
                "done": True
            }
        else:
            raise HTTPException(status_code=400, detail=result.failure())

if __name__ == "__main__":
    import uvicorn
    import socket
    import sys
    
    def is_port_in_use(port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return False
            except OSError:
                return True
    
    def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if not is_port_in_use(port):
                return port
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")
    
    try:
        port = 8000
        if is_port_in_use(port):
            logger.warning(f"Port {port} is already in use, finding alternative port")
            port = find_available_port(8001)  # Start from 8001 if 8000 is taken
            logger.info(f"Using alternative port {port}")
        
        logger.info(f"Starting LLM Router server on http://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
        
    except RuntimeError as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error starting server: {e}")
        sys.exit(1)
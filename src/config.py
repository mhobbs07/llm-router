import json
import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from returns.result import Result, Success, Failure


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic provider."""
    api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    default_model: str = Field(default="claude-3-sonnet-20240229")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI provider."""
    api_key: str = Field(..., env="OPENAI_API_KEY")
    default_model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class XAIConfig(BaseModel):
    """Configuration for xAI/Grok provider."""
    api_key: str = Field(..., env="XAI_API_KEY")
    default_model: str = Field(default="grok-1")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class GoogleGeminiConfig(BaseModel):
    """Configuration for Google Gemini provider."""
    api_key: str = Field(..., env="GOOGLE_GEMINI_API_KEY")
    default_model: str = Field(default="gemini-pro")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class HuggingFaceConfig(BaseModel):
    """Configuration for Hugging Face provider."""
    endpoint_url: str = Field(..., env="HUGGINGFACE_ENDPOINT_URL")
    model_mapping: dict[str, str] = Field(default_factory=dict)
    api_key: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator('endpoint_url')
    @classmethod
    def validate_endpoint_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Endpoint URL cannot be empty")
        return v.strip()


class CostConfig(BaseModel):
    """Configuration for cost tracking and management."""
    model_costs: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            # OpenAI models (per 1K tokens)
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            # Google Gemini (per 1K tokens)
            "gemini-pro": {"input": 0.00025, "output": 0.0005},
            "gemini-pro-vision": {"input": 0.00025, "output": 0.0005},
            # Anthropic Claude (per 1K tokens)
            "claude-3-sonnet": {"input": 0.015, "output": 0.075},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            # xAI Grok (per 1K tokens)
            "grok-1": {"input": 0.005, "output": 0.015}
        },
        description="Model costs per token: model -> {'input': cost_per_token, 'output': cost_per_token}"
    )
    caching_enabled: bool = Field(default=True, env="COST_CACHING_ENABLED")
    cache_ttl_seconds: int = Field(default=3600, env="COST_CACHE_TTL_SECONDS")  # 1 hour default
    cache_max_size_mb: int = Field(default=100, env="COST_CACHE_MAX_SIZE_MB")
    budget_daily_limit: float = Field(default=10.0, env="COST_BUDGET_DAILY_LIMIT")  # $10 per day
    budget_monthly_limit: float = Field(default=100.0, env="COST_BUDGET_MONTHLY_LIMIT")  # $100 per month
    budget_alert_threshold: float = Field(default=0.8, env="COST_BUDGET_ALERT_THRESHOLD")  # alert at 80% usage
    max_context_tokens: int = Field(default=8000, env="COST_MAX_CONTEXT_TOKENS")  # Max context for optimization
    allowed_models: Optional[list[str]] = Field(default=None, description="List of allowed model names. If set, only these models can be used.")
    blocked_models: Optional[list[str]] = Field(default=None, description="List of blocked model names. These models cannot be used.")
    cost_limits: dict[str, float] = Field(default_factory=dict, description="Per-model cost limits: model -> max_cost")
    enforce_budget_limits: bool = Field(default=True, env="COST_ENFORCE_BUDGET_LIMITS", description="Whether to enforce budget limits")
    allow_unknown_models: bool = Field(default=True, env="COST_ALLOW_UNKNOWN_MODELS", description="Whether to allow models not in model_costs")


class ModelPricing(BaseModel):
    """Pricing information for a model."""
    model_config = ConfigDict(frozen=True)
    
    input_per_1m: float = Field(description="Cost per 1M input tokens in USD")
    output_per_1m: float = Field(description="Cost per 1M output tokens in USD")


class ModelMetadata(BaseModel):
    """Metadata for a curated model."""
    model_config = ConfigDict(frozen=True)
    
    id: str
    provider: Literal["openai", "anthropic", "google", "xai"]
    tier: Literal["budget", "balanced", "premium"]
    description: str
    pricing: ModelPricing
    context_window: int
    recommended_for: list[str]


# Cost-optimized curated model list based on January 2026 pricing
CURATED_MODELS: dict[str, ModelMetadata] = {
    # Budget Tier - High-volume, simple tasks
    "grok-4-fast": ModelMetadata(
        id="grok-4-fast",
        provider="xai",
        tier="budget",
        description="Extremely cheap for high-volume tasks",
        pricing=ModelPricing(input_per_1m=0.20, output_per_1m=0.50),
        context_window=128000,
        recommended_for=["high-volume", "simple-tasks"]
    ),
    "grok-4.1": ModelMetadata(
        id="grok-4.1",
        provider="xai",
        tier="budget",
        description="Latest version, same low price",
        pricing=ModelPricing(input_per_1m=0.20, output_per_1m=0.50),
        context_window=128000,
        recommended_for=["high-volume", "simple-tasks"]
    ),
    "claude-3-5-haiku-20241022": ModelMetadata(
        id="claude-3-5-haiku-20241022",
        provider="anthropic",
        tier="budget",
        description="Ultra-fast, reliable for simple tasks",
        pricing=ModelPricing(input_per_1m=0.25, output_per_1m=1.25),
        context_window=200000,
        recommended_for=["simple-tasks", "fast-responses"]
    ),
    "gemini-2.5-flash-lite": ModelMetadata(
        id="gemini-2.5-flash-lite",
        provider="google",
        tier="budget",
        description="Cheapest Google option",
        pricing=ModelPricing(input_per_1m=0.15, output_per_1m=1.25),
        context_window=1000000,
        recommended_for=["high-volume", "simple-tasks"]
    ),
    
    # Balanced Tier - Production quality
    "gemini-2.5-flash": ModelMetadata(
        id="gemini-2.5-flash",
        provider="google",
        tier="balanced",
        description="Best workhorse value - Fast and cheap",
        pricing=ModelPricing(input_per_1m=0.30, output_per_1m=2.50),
        context_window=1000000,
        recommended_for=["general", "coding", "analysis"]
    ),
    "claude-3-5-sonnet-20241022": ModelMetadata(
        id="claude-3-5-sonnet-20241022",
        provider="anthropic",
        tier="balanced",
        description="Highest quality balanced model",
        pricing=ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
        context_window=200000,
        recommended_for=["coding", "reasoning", "analysis"]
    ),
    "claude-3-5-sonnet-20240620": ModelMetadata(
        id="claude-3-5-sonnet-20240620",
        provider="anthropic",
        tier="balanced",
        description="Stable Sonnet model",
        pricing=ModelPricing(input_per_1m=3.00, output_per_1m=15.00),
        context_window=200000,
        recommended_for=["coding", "reasoning", "analysis"]
    ),
    
    # Premium Tier - Complex reasoning, specialized tasks
    "gemini-2.5-pro": ModelMetadata(
        id="gemini-2.5-pro",
        provider="google",
        tier="premium",
        description="Long-context tasks and complex reasoning",
        pricing=ModelPricing(input_per_1m=2.00, output_per_1m=12.00),
        context_window=2000000,
        recommended_for=["long-context", "complex-reasoning", "research"]
    ),
    "claude-opus-4-5-20251101": ModelMetadata(
        id="claude-opus-4-5-20251101",
        provider="anthropic",
        tier="premium",
        description="Best for coding and complex reasoning",
        pricing=ModelPricing(input_per_1m=5.00, output_per_1m=25.00),
        context_window=200000,
        recommended_for=["coding", "complex-reasoning", "research"]
    ),
    
    # OpenAI Models
    "gpt-4o": ModelMetadata(
        id="gpt-4o",
        provider="openai",
        tier="premium",
        description="Most capable OpenAI model",
        pricing=ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
        context_window=128000,
        recommended_for=["complex-reasoning", "coding", "analysis"]
    ),
    "gpt-4o-mini": ModelMetadata(
        id="gpt-4o-mini",
        provider="openai",
        tier="balanced",
        description="Fast and efficient OpenAI model",
        pricing=ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
        context_window=128000,
        recommended_for=["general", "coding", "analysis"]
    ),
    "gpt-3.5-turbo": ModelMetadata(
        id="gpt-3.5-turbo",
        provider="openai",
        tier="budget",
        description="Cost-effective OpenAI model",
        pricing=ModelPricing(input_per_1m=0.50, output_per_1m=1.50),
        context_window=16385,
        recommended_for=["simple-tasks", "general"]
    )
}

# Default model for best value
DEFAULT_MODEL = "gemini-2.5-flash"

# Cost-optimized fallback order (never fallback to more expensive)
FALLBACK_ORDER = [
    "gemini-2.5-flash",      # Default - best value
    "claude-3-5-haiku-20241022",  # Cheap, reliable
    "grok-4-fast",           # Cheapest
    "claude-3-5-sonnet-20241022",  # If quality needed
]


class MCPConfig(BaseModel):
    """Configuration for MCP server."""
    enabled: bool = Field(default=False, env="MCP_ENABLED")
    transport: str = Field(default="stdio", env="MCP_TRANSPORT")  # "stdio" or "http"
    host: str = Field(default="localhost", env="MCP_HOST")
    port: int = Field(default=8001, env="MCP_PORT")


class AppConfig(BaseModel):
    """Main application configuration containing all provider configs."""
    anthropic: Optional[AnthropicConfig] = None
    openai: Optional[OpenAIConfig] = None
    xai: Optional[XAIConfig] = None
    google_gemini: Optional[GoogleGeminiConfig] = None
    huggingface: Optional[HuggingFaceConfig] = None
    cost: CostConfig = Field(default_factory=CostConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


def load_config_from_env() -> Result[AppConfig, str]:
    """
    Load configuration from environment variables.

    Returns:
        Result containing AppConfig on success, or error message on failure.
    """
    try:
        config_data = {}

        # Load each provider config if env vars are present
        if os.getenv("ANTHROPIC_API_KEY"):
            config_data["anthropic"] = AnthropicConfig(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                default_model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-sonnet-20240229"),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
            )

        if os.getenv("OPENAI_API_KEY"):
            config_data["openai"] = OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY"),
                default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
            )

        if os.getenv("XAI_API_KEY"):
            config_data["xai"] = XAIConfig(
                api_key=os.getenv("XAI_API_KEY"),
                default_model=os.getenv("XAI_DEFAULT_MODEL", "grok-1"),
                temperature=float(os.getenv("XAI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("XAI_MAX_TOKENS", "4096"))
            )

        if os.getenv("GOOGLE_GEMINI_API_KEY"):
            config_data["google_gemini"] = GoogleGeminiConfig(
                api_key=os.getenv("GOOGLE_GEMINI_API_KEY"),
                default_model=os.getenv("GOOGLE_GEMINI_DEFAULT_MODEL", "gemini-pro"),
                temperature=float(os.getenv("GOOGLE_GEMINI_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("GOOGLE_GEMINI_MAX_TOKENS", "4096"))
            )

        if os.getenv("HUGGINGFACE_ENDPOINT_URL"):
            model_mapping_str = os.getenv("HUGGINGFACE_MODEL_MAPPING", "{}")
            try:
                model_mapping = json.loads(model_mapping_str)
            except json.JSONDecodeError:
                model_mapping = {}
            config_data["huggingface"] = HuggingFaceConfig(
                endpoint_url=os.getenv("HUGGINGFACE_ENDPOINT_URL"),
                model_mapping=model_mapping,
                api_key=os.getenv("HUGGINGFACE_API_KEY"),
                temperature=float(os.getenv("HUGGINGFACE_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("HUGGINGFACE_MAX_TOKENS", "4096"))
            )

        # Load cost configuration
        model_costs_str = os.getenv("COST_MODEL_COSTS_JSON", "{}")
        try:
            model_costs = json.loads(model_costs_str)
        except json.JSONDecodeError:
            model_costs = {}
        cost_limits_str = os.getenv("COST_COST_LIMITS_JSON", "{}")
        try:
            cost_limits = json.loads(cost_limits_str)
        except json.JSONDecodeError:
            cost_limits = {}
        allowed_models_str = os.getenv("COST_ALLOWED_MODELS_JSON")
        allowed_models = None
        if allowed_models_str:
            try:
                allowed_models = json.loads(allowed_models_str)
            except json.JSONDecodeError:
                allowed_models = None
        blocked_models_str = os.getenv("COST_BLOCKED_MODELS_JSON")
        blocked_models = None
        if blocked_models_str:
            try:
                blocked_models = json.loads(blocked_models_str)
            except json.JSONDecodeError:
                blocked_models = None
        config_data["cost"] = CostConfig(
            model_costs=model_costs,
            caching_enabled=os.getenv("COST_CACHING_ENABLED", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("COST_CACHE_TTL_SECONDS", "3600")),
            cache_max_size_mb=int(os.getenv("COST_CACHE_MAX_SIZE_MB", "100")),
            budget_daily_limit=float(os.getenv("COST_BUDGET_DAILY_LIMIT", "10.0")),
            budget_monthly_limit=float(os.getenv("COST_BUDGET_MONTHLY_LIMIT", "100.0")),
            budget_alert_threshold=float(os.getenv("COST_BUDGET_ALERT_THRESHOLD", "0.8")),
            max_context_tokens=int(os.getenv("COST_MAX_CONTEXT_TOKENS", "8000")),
            allowed_models=allowed_models,
            blocked_models=blocked_models,
            cost_limits=cost_limits,
            enforce_budget_limits=os.getenv("COST_ENFORCE_BUDGET_LIMITS", "true").lower() == "true",
            allow_unknown_models=os.getenv("COST_ALLOW_UNKNOWN_MODELS", "true").lower() == "true"
        )

        config = AppConfig(**config_data)
        return Success(config)

    except Exception as e:
        return Failure(f"Failed to load configuration: {str(e)}")


def get_provider_config(provider_name: str, config: AppConfig) -> Result[BaseModel, str]:
    """
    Safely retrieve provider configuration by name.

    Args:
        provider_name: Name of the provider ("anthropic", "openai", "xai", "google_gemini")
        config: The loaded AppConfig

    Returns:
        Result containing provider config on success, or error message on failure.
    """
    provider_configs = {
        "anthropic": config.anthropic,
        "openai": config.openai,
        "xai": config.xai,
        "google_gemini": config.google_gemini,
        "huggingface": config.huggingface
    }

    provider_config = provider_configs.get(provider_name)
    if provider_config is None:
        return Failure(f"Provider '{provider_name}' not configured")

    return Success(provider_config)
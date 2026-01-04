from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class Tool(BaseModel):
    """MCP tool definition model."""
    type: str = "function"  # Currently only "function" is supported
    function: Dict[str, Any]  # Function definition with name, description, parameters


class ToolCall(BaseModel):
    """MCP tool call model."""
    id: str
    type: str = "function"
    function: Dict[str, Any]  # Function call with name and arguments


class Message(BaseModel):
    """Unified message model for chat completions across providers."""
    role: str  # "user", "assistant", "system", "tool"
    content: Union[str, List[Dict[str, Any]]]  # string or multimodal content
    tool_calls: Optional[List[ToolCall]] = None  # For assistant messages with tool calls
    tool_call_id: Optional[str] = None  # For tool result messages


class ChatRequest(BaseModel):
    """Unified chat completion request model."""
    messages: List[Message]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    system: Optional[str] = None  # For providers that separate system messages
    tools: Optional[List[Tool]] = None  # MCP tool definitions


class ChatResponse(BaseModel):
    """Unified chat completion response model for non-streaming responses."""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None  # MCP tool calls from the model


class ChatStreamResponse(BaseModel):
    """Unified chat completion streaming response model."""
    delta: str
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None  # MCP tool calls from the model


class PolicyResponse(BaseModel):
    """Response model for policy configuration."""
    allowed_models: Optional[List[str]] = None
    blocked_models: Optional[List[str]] = None
    cost_limits: Dict[str, float]
    allow_unknown_models: bool
    enforce_budget_limits: bool


class PolicyRequest(BaseModel):
    """Request model for updating policy configuration."""
    allowed_models: Optional[List[str]] = None
    blocked_models: Optional[List[str]] = None
    cost_limits: Optional[Dict[str, float]] = None
    allow_unknown_models: Optional[bool] = None
    enforce_budget_limits: Optional[bool] = None
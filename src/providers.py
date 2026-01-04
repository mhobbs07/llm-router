from typing import Protocol, AsyncIterable, List, Dict, Any
from returns.result import Result, Success, Failure
import openai
from openai import AsyncOpenAI
import google.genai as genai
import xai_sdk
import anthropic

from .models import ChatRequest, ChatResponse, ChatStreamResponse
from .config import OpenAIConfig, GoogleGeminiConfig, XAIConfig, AnthropicConfig
from .retry import retry_with_backoff, RetryConfig, CONSERVATIVE_RETRY


class LLMProvider(Protocol):
    """Protocol interface that all LLM providers must implement."""

    async def chat_completion(
        self, request: ChatRequest
    ) -> Result[ChatResponse, str]:
        """
        Perform a non-streaming chat completion.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing ChatResponse on success, or error message on failure.
        """
        ...

    async def chat_completion_stream(
        self, request: ChatRequest
    ) -> Result[AsyncIterable[ChatStreamResponse], str]:
        """
        Perform a streaming chat completion.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing AsyncIterable of ChatStreamResponse on success,
            or error message on failure.
        """
        ...

    async def list_models(self) -> Result[List[str], str]:
        """
        List available models for this provider.

        Returns:
            Result containing list of model names on success, or error message on failure.
        """
        ...


class OpenAIProvider:
    """OpenAI provider implementation for LLM chat completions."""

    def __init__(self, config: OpenAIConfig):
        """
        Initialize the OpenAI provider.

        Args:
            config: OpenAI configuration containing API key and defaults.
        """
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)

    def _map_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map unified message format to OpenAI format.

        Args:
            messages: List of unified messages.

        Returns:
            List of OpenAI-formatted messages.
        """
        openai_messages = []
        for msg in messages:
            message_dict = {
                "role": msg["role"],
                "content": msg["content"]
            }
            # Add tool_calls if present (for assistant messages)
            if msg.get("tool_calls"):
                message_dict["tool_calls"] = msg["tool_calls"]
            # Add tool_call_id if present (for tool messages)
            if msg.get("tool_call_id"):
                message_dict["tool_call_id"] = msg["tool_call_id"]
            openai_messages.append(message_dict)
        return openai_messages

    def _normalize_response(self, response: Any) -> ChatResponse:
        """
        Normalize OpenAI response to unified ChatResponse format.

        Args:
            response: OpenAI API response.

        Returns:
            Normalized ChatResponse.
        """
        choice = response.choices[0]
        tool_calls = None
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            tool_calls = [tool_call.model_dump() for tool_call in choice.message.tool_calls]

        return ChatResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None,
            tool_calls=tool_calls
        )

    def _normalize_stream_response(self, chunk: Any) -> ChatStreamResponse:
        """
        Normalize OpenAI streaming chunk to unified ChatStreamResponse format.

        Args:
            chunk: OpenAI streaming chunk.

        Returns:
            Normalized ChatStreamResponse.
        """
        choice = chunk.choices[0]
        delta = choice.delta
        tool_calls = None
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            tool_calls = [tool_call.model_dump() for tool_call in delta.tool_calls]

        return ChatStreamResponse(
            delta=delta.content or "",
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls
        )

    async def chat_completion(
        self, request: ChatRequest
    ) -> Result[ChatResponse, str]:
        """
        Perform a non-streaming chat completion using OpenAI.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing ChatResponse on success, or error message on failure.
        """
        async def _make_api_call():
            # Map messages to OpenAI format
            messages = self._map_messages([msg.model_dump() for msg in request.messages])

            # Prepare API call parameters
            params = {
                "model": request.model,
                "messages": messages,
                "stream": False
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.tools:
                params["tools"] = [tool.model_dump() for tool in request.tools]

            # Make API call with retry
            return await self.client.chat.completions.create(**params)

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            
            # Normalize response
            normalized_response = self._normalize_response(response)
            return Success(normalized_response)

        except openai.APIError as e:
            return Failure(f"OpenAI API error: {str(e)}")
        except openai.AuthenticationError as e:
            return Failure(f"OpenAI authentication error: {str(e)}")
        except openai.RateLimitError as e:
            return Failure(f"OpenAI rate limit error: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error in OpenAI chat completion: {str(e)}")

    async def chat_completion_stream(
        self, request: ChatRequest
    ) -> Result[AsyncIterable[ChatStreamResponse], str]:
        """
        Perform a streaming chat completion using OpenAI.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing AsyncIterable of ChatStreamResponse on success,
            or error message on failure.
        """
        try:
            # Map messages to OpenAI format
            messages = self._map_messages([msg.model_dump() for msg in request.messages])

            # Prepare API call parameters
            params = {
                "model": request.model,
                "messages": messages,
                "stream": True
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.tools:
                params["tools"] = [tool.model_dump() for tool in request.tools]

            # Make streaming API call
            stream = await self.client.chat.completions.create(**params)

            async def response_generator():
                try:
                    async for chunk in stream:
                        normalized_chunk = self._normalize_stream_response(chunk)
                        yield normalized_chunk
                except Exception as e:
                    # In streaming context, we can't return Failure directly
                    # This would need to be handled by the caller
                    raise e

            return Success(response_generator())

        except openai.APIError as e:
            return Failure(f"OpenAI API error: {str(e)}")
        except openai.AuthenticationError as e:
            return Failure(f"OpenAI authentication error: {str(e)}")
        except openai.RateLimitError as e:
            return Failure(f"OpenAI rate limit error: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error in OpenAI streaming chat completion: {str(e)}")

    async def list_models(self) -> Result[List[str], str]:
        """
        List available models for OpenAI.

        Returns:
            Result containing list of model names on success, or error message on failure.
        """
        async def _make_api_call():
            return await self.client.models.list()

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            model_names = [model.id for model in response.data]
            return Success(model_names)
        except openai.APIError as e:
            return Failure(f"OpenAI API error: {str(e)}")
        except openai.AuthenticationError as e:
            return Failure(f"OpenAI authentication error: {str(e)}")
        except openai.RateLimitError as e:
            return Failure(f"OpenAI rate limit error: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error listing OpenAI models: {str(e)}")


class GeminiProvider:
    """Google Gemini provider implementation for LLM chat completions."""

    def __init__(self, config: GoogleGeminiConfig):
        """
        Initialize the Gemini provider.

        Args:
            config: Google Gemini configuration containing API key and defaults.
        """
        self.client = genai.Client(api_key=config.api_key)
        self.config = config

    def _map_messages_to_history(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map unified messages to Gemini chat history format.

        Args:
            messages: List of unified messages.

        Returns:
            List of Gemini history entries.
        """
        history = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                # Skip system messages for history, handle separately
                continue

            parts = [{"text": msg["content"]}] if isinstance(msg["content"], str) else msg["content"]
            history.append({
                "role": gemini_role,
                "parts": parts
            })
        return history

    def _get_system_instruction(self, request: ChatRequest) -> str | None:
        """
        Extract system instruction from request.

        Args:
            request: The chat request.

        Returns:
            System instruction string or None.
        """
        if request.system:
            return request.system
        # Check if first message is system
        if request.messages and request.messages[0].role == "system":
            return request.messages[0].content
        return None

    def _normalize_response(self, response: Any) -> ChatResponse:
        """
        Normalize Gemini response to unified ChatResponse format.

        Args:
            response: Gemini API response.

        Returns:
            Normalized ChatResponse.
        """
        content = ""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        content += part.text

        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name.lower()

        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }

        return ChatResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage
        )

    def _normalize_stream_response(self, chunk: Any) -> ChatStreamResponse:
        """
        Normalize Gemini streaming chunk to unified ChatStreamResponse format.

        Args:
            chunk: Gemini streaming chunk.

        Returns:
            Normalized ChatStreamResponse.
        """
        delta = ""
        finish_reason = None

        if chunk.candidates:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        delta += part.text
            if candidate.finish_reason and candidate.finish_reason.name:
                finish_reason = candidate.finish_reason.name.lower()

        return ChatStreamResponse(
            delta=delta,
            finish_reason=finish_reason
        )

    async def chat_completion(
        self, request: ChatRequest
    ) -> Result[ChatResponse, str]:
        """
        Perform a non-streaming chat completion using Google Gemini.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing ChatResponse on success, or error message on failure.
        """
        async def _make_api_call():
            # Get system instruction
            system_instruction = self._get_system_instruction(request)

            # Map messages to history, excluding system if used as instruction
            messages = [msg.model_dump() for msg in request.messages]
            if system_instruction and messages and messages[0]["role"] == "system":
                messages = messages[1:]  # Remove system message from history
            history = self._map_messages_to_history(messages)

            # Prepare generation config
            generation_config = {}
            if request.temperature is not None:
                generation_config["temperature"] = request.temperature
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens

            # Get the last message to send
            if not messages:
                raise ValueError("No messages provided")

            # Use the new client API with retry
            return await self.client.chats.create(
                model=request.model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                system_instruction=system_instruction,
                generation_config=generation_config if generation_config else None
            )

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            
            # Normalize response
            normalized_response = self._normalize_response(response)
            return Success(normalized_response)

        except Exception as e:
            if "StopCandidateException" in str(type(e)):
                return Failure(f"Gemini content generation stopped: {str(e)}")
            elif "BlockedPromptException" in str(type(e)):
                return Failure(f"Gemini prompt blocked: {str(e)}")
            return Failure(f"Unexpected error in Gemini chat completion: {str(e)}")

    async def chat_completion_stream(
        self, request: ChatRequest
    ) -> Result[AsyncIterable[ChatStreamResponse], str]:
        """
        Perform a streaming chat completion using Google Gemini.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing AsyncIterable of ChatStreamResponse on success,
            or error message on failure.
        """
        try:
            # Get system instruction
            system_instruction = self._get_system_instruction(request)

            # Map messages to history, excluding system if used as instruction
            messages = [msg.model_dump() for msg in request.messages]
            if system_instruction and messages and messages[0]["role"] == "system":
                messages = messages[1:]  # Remove system message from history
            history = self._map_messages_to_history(messages)

            # Prepare generation config
            generation_config = {}
            if request.temperature is not None:
                generation_config["temperature"] = request.temperature
            if request.max_tokens is not None:
                generation_config["max_output_tokens"] = request.max_tokens

            # Get the last message to send
            if not messages:
                return Failure("No messages provided")

            # Use the new client API with streaming
            response = await self.client.chats.create(
                model=request.model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                system_instruction=system_instruction,
                generation_config=generation_config if generation_config else None,
                stream=True
            )

            async def response_generator():
                try:
                    async for chunk in response:
                        normalized_chunk = self._normalize_stream_response(chunk)
                        yield normalized_chunk
                except Exception as e:
                    # In streaming context, we can't return Failure directly
                    raise e

            return Success(response_generator())

        except Exception as e:
            if "StopCandidateException" in str(type(e)):
                return Failure(f"Gemini content generation stopped: {str(e)}")
            elif "BlockedPromptException" in str(type(e)):
                return Failure(f"Gemini prompt blocked: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error in Gemini streaming chat completion: {str(e)}")

    async def list_models(self) -> Result[List[str], str]:
        """
        List available models for Gemini.

        Returns:
            Result containing list of model names on success, or error message on failure.
        """
        async def _make_api_call():
            return await self.client.models.list()

        try:
            models = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            model_names = [model.name for model in models.models]
            return Success(model_names)
        except Exception as e:
            return Failure(f"Unexpected error listing Gemini models: {str(e)}")


class GrokProvider:
    """xAI/Grok provider implementation for LLM chat completions."""

    def __init__(self, config: XAIConfig):
        """
        Initialize the Grok provider.

        Args:
            config: xAI configuration containing API key and defaults.
        """
        self.config = config
        self.client = xai_sdk.Client(api_key=config.api_key)

    def _map_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map unified message format to Grok format.

        Args:
            messages: List of unified messages.

        Returns:
            List of Grok-formatted messages.
        """
        grok_messages = []
        for msg in messages:
            # Grok expects role and content directly, similar to OpenAI
            grok_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        return grok_messages

    def _normalize_response(self, response: Any) -> ChatResponse:
        """
        Normalize Grok response to unified ChatResponse format.

        Args:
            response: Grok API response.

        Returns:
            Normalized ChatResponse.
        """
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else None
        )

    def _normalize_stream_response(self, chunk: Any) -> ChatStreamResponse:
        """
        Normalize Grok streaming chunk to unified ChatStreamResponse format.

        Args:
            chunk: Grok streaming chunk.

        Returns:
            Normalized ChatStreamResponse.
        """
        choice = chunk.choices[0]
        delta = choice.delta
        return ChatStreamResponse(
            delta=delta.content or "",
            finish_reason=choice.finish_reason
        )

    async def chat_completion(
        self, request: ChatRequest
    ) -> Result[ChatResponse, str]:
        """
        Perform a non-streaming chat completion using Grok.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing ChatResponse on success, or error message on failure.
        """
        async def _make_api_call():
            # Map messages to Grok format
            messages = self._map_messages([msg.model_dump() for msg in request.messages])

            # Prepare API call parameters (XAI doesn't support stream parameter)
            params = {
                "model": request.model,
                "messages": messages
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens

            # Make API call with retry
            return await self.client.chat.create(**params)

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            
            # Normalize response
            normalized_response = self._normalize_response(response)
            return Success(normalized_response)

        except Exception as e:
            return Failure(f"Unexpected error in Grok chat completion: {str(e)}")

    async def chat_completion_stream(
        self, request: ChatRequest
    ) -> Result[AsyncIterable[ChatStreamResponse], str]:
        """
        Perform a streaming chat completion using Grok.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing AsyncIterable of ChatStreamResponse on success,
            or error message on failure.
        """
        try:
            # Map messages to Grok format
            messages = self._map_messages([msg.model_dump() for msg in request.messages])

            # Prepare API call parameters
            params = {
                "model": request.model,
                "messages": messages,
                "stream": True
            }

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens

            # Remove stream parameter for XAI - it doesn't support it in create()
            params_no_stream = {k: v for k, v in params.items() if k != "stream"}
            
            # Make streaming API call
            stream = await self.client.chat.create(**params_no_stream)

            async def response_generator():
                try:
                    async for chunk in stream:
                        normalized_chunk = self._normalize_stream_response(chunk)
                        yield normalized_chunk
                except Exception as e:
                    # In streaming context, we can't return Failure directly
                    # This would need to be handled by the caller
                    raise e

            return Success(response_generator())

        except Exception as e:
            return Failure(f"Unexpected error in Grok streaming chat completion: {str(e)}")

    async def list_models(self) -> Result[List[str], str]:
        """
        List available models for Grok.

        Returns:
            Result containing list of model names on success, or error message on failure.
        """
        async def _make_api_call():
            return await self.client.models.list()

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            model_names = [model.id for model in response.models]
            return Success(model_names)
        except Exception as e:
            return Failure(f"Unexpected error listing Grok models: {str(e)}")


class AnthropicProvider:
    """Anthropic Claude provider implementation for LLM chat completions."""

    def __init__(self, config: AnthropicConfig):
        """
        Initialize the Anthropic provider.

        Args:
            config: Anthropic configuration containing API key and defaults.
        """
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)

    def _extract_system_message(self, request: ChatRequest) -> tuple[str | None, List[Dict[str, Any]]]:
        """
        Extract system message for Anthropic's separate system parameter.
        
        Args:
            request: The chat request.
            
        Returns:
            Tuple of (system_message, filtered_messages) where system_message is the
            system content and filtered_messages excludes system messages.
        """
        system_message = request.system
        messages = [msg.model_dump() for msg in request.messages]
        
        # If no explicit system parameter, check if first message is system
        if not system_message and messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]  # Remove system message from messages array
            
        return system_message, messages

    def _map_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Map unified message format to Anthropic format.

        Args:
            messages: List of unified messages (excluding system messages).

        Returns:
            List of Anthropic-formatted messages.
        """
        anthropic_messages = []
        for msg in messages:
            message_dict = {
                "role": msg["role"],
                "content": msg["content"]
            }
            # Add tool_calls if present (for assistant messages)
            if msg.get("tool_calls"):
                message_dict["tool_calls"] = msg["tool_calls"]
            # Add tool_call_id if present (for tool messages)
            if msg.get("tool_call_id"):
                message_dict["tool_call_id"] = msg["tool_call_id"]
            anthropic_messages.append(message_dict)
        return anthropic_messages

    def _normalize_response(self, response: Any) -> ChatResponse:
        """
        Normalize Anthropic response to unified ChatResponse format.

        Args:
            response: Anthropic API response.

        Returns:
            Normalized ChatResponse.
        """
        content = ""
        tool_calls = None
        
        # Extract content from response
        if hasattr(response, 'content') and response.content:
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    content += content_block.text
                elif hasattr(content_block, 'tool_use'):
                    # Handle tool calls
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": content_block.tool_use.id,
                        "type": "function",
                        "function": {
                            "name": content_block.tool_use.name,
                            "arguments": content_block.tool_use.input
                        }
                    })

        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

        return ChatResponse(
            content=content,
            finish_reason=response.stop_reason if hasattr(response, 'stop_reason') else None,
            usage=usage,
            tool_calls=tool_calls
        )

    def _normalize_stream_response(self, chunk: Any) -> ChatStreamResponse:
        """
        Normalize Anthropic streaming chunk to unified ChatStreamResponse format.

        Args:
            chunk: Anthropic streaming chunk.

        Returns:
            Normalized ChatStreamResponse.
        """
        delta = ""
        finish_reason = None
        tool_calls = None

        if hasattr(chunk, 'type'):
            if chunk.type == "content_block_delta":
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    delta = chunk.delta.text
            elif chunk.type == "message_stop":
                finish_reason = "stop"
            elif chunk.type == "content_block_start":
                if hasattr(chunk, 'content_block') and hasattr(chunk.content_block, 'tool_use'):
                    # Handle tool call start
                    tool_calls = [{
                        "id": chunk.content_block.tool_use.id,
                        "type": "function",
                        "function": {
                            "name": chunk.content_block.tool_use.name,
                            "arguments": ""
                        }
                    }]

        return ChatStreamResponse(
            delta=delta,
            finish_reason=finish_reason,
            tool_calls=tool_calls
        )

    async def chat_completion(
        self, request: ChatRequest
    ) -> Result[ChatResponse, str]:
        """
        Perform a non-streaming chat completion using Anthropic Claude.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing ChatResponse on success, or error message on failure.
        """
        async def _make_api_call():
            # Extract system message and filter messages
            system_message, messages = self._extract_system_message(request)
            
            # Map messages to Anthropic format
            anthropic_messages = self._map_messages(messages)

            # Prepare API call parameters
            params = {
                "model": request.model,
                "messages": anthropic_messages,
                "max_tokens": request.max_tokens or self.config.max_tokens
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
            
            # Add tools if provided
            if request.tools:
                params["tools"] = [
                    {
                        "name": tool.function["name"],
                        "description": tool.function.get("description", ""),
                        "input_schema": tool.function.get("parameters", {})
                    }
                    for tool in request.tools
                ]

            # Make API call with retry
            return await self.client.messages.create(**params)

        try:
            response = await retry_with_backoff(_make_api_call, config=CONSERVATIVE_RETRY)
            
            # Normalize response
            normalized_response = self._normalize_response(response)
            return Success(normalized_response)

        except anthropic.APIError as e:
            return Failure(f"Anthropic API error: {str(e)}")
        except anthropic.AuthenticationError as e:
            return Failure(f"Anthropic authentication error: {str(e)}")
        except anthropic.RateLimitError as e:
            return Failure(f"Anthropic rate limit error: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error in Anthropic chat completion: {str(e)}")

    async def chat_completion_stream(
        self, request: ChatRequest
    ) -> Result[AsyncIterable[ChatStreamResponse], str]:
        """
        Perform a streaming chat completion using Anthropic Claude.

        Args:
            request: The unified chat request containing messages and parameters.

        Returns:
            Result containing AsyncIterable of ChatStreamResponse on success,
            or error message on failure.
        """
        try:
            # Extract system message and filter messages
            system_message, messages = self._extract_system_message(request)
            
            # Map messages to Anthropic format
            anthropic_messages = self._map_messages(messages)

            # Prepare API call parameters
            params = {
                "model": request.model,
                "messages": anthropic_messages,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "stream": True
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Add optional parameters if provided
            if request.temperature is not None:
                params["temperature"] = request.temperature
                
            # Add tools if provided
            if request.tools:
                params["tools"] = [
                    {
                        "name": tool.function["name"],
                        "description": tool.function.get("description", ""),
                        "input_schema": tool.function.get("parameters", {})
                    }
                    for tool in request.tools
                ]

            # Make streaming API call
            stream = await self.client.messages.create(**params)

            async def response_generator():
                try:
                    async for chunk in stream:
                        normalized_chunk = self._normalize_stream_response(chunk)
                        yield normalized_chunk
                except Exception as e:
                    # In streaming context, we can't return Failure directly
                    # This would need to be handled by the caller
                    raise e

            return Success(response_generator())

        except anthropic.APIError as e:
            return Failure(f"Anthropic API error: {str(e)}")
        except anthropic.AuthenticationError as e:
            return Failure(f"Anthropic authentication error: {str(e)}")
        except anthropic.RateLimitError as e:
            return Failure(f"Anthropic rate limit error: {str(e)}")
        except Exception as e:
            return Failure(f"Unexpected error in Anthropic streaming chat completion: {str(e)}")

    async def list_models(self) -> Result[List[str], str]:
        """
        List available models for Anthropic.

        Returns:
            Result containing list of model names on success, or error message on failure.
        """
        try:
            # Anthropic doesn't have a public models API, so we return known models
            known_models = [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
            return Success(known_models)
        except Exception as e:
            return Failure(f"Unexpected error listing Anthropic models: {str(e)}")
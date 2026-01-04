import json
import asyncio
import logging
from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPToolResult(BaseModel):
    """Result of a tool execution."""
    content: List[Dict[str, Any]]
    is_error: bool = False


@dataclass
class MCPToolHandler:
    """Handler for a registered MCP tool."""
    tool: MCPTool
    handler: Callable[[Dict[str, Any]], Any]


class MCPServer:
    """MCP (Model Context Protocol) server implementation."""

    def __init__(self):
        self.tools: Dict[str, MCPToolHandler] = {}
        self.logger = logging.getLogger(__name__)

    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable[[Dict[str, Any]], Any]):
        """Register a tool with the MCP server.

        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for tool inputs
            handler: Function to call when tool is executed
        """
        tool = MCPTool(
            name=name,
            description=description,
            input_schema=input_schema
        )
        self.tools[name] = MCPToolHandler(tool=tool, handler=handler)
        self.logger.info(f"Registered MCP tool: {name}")

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response
        """
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method == "tools/list":
                return await self._handle_tools_list(request_id)
            elif method == "tools/call":
                return await self._handle_tools_call(params, request_id)
            elif method == "initialize":
                return await self._handle_initialize(request_id)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        except Exception as e:
            self.logger.error(f"Error handling MCP request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }

    async def _handle_initialize(self, request_id: Any) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "llm-router-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    async def _handle_tools_list(self, request_id: Any) -> Dict[str, Any]:
        """Handle tools/list request."""
        tools = []
        for handler in self.tools.values():
            tools.append({
                "name": handler.tool.name,
                "description": handler.tool.description,
                "inputSchema": handler.tool.input_schema
            })

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": tools
            }
        }

    async def _handle_tools_call(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if tool_name not in self.tools:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Tool not found: {tool_name}"
                }
            }

        try:
            handler = self.tools[tool_name]
            result = await self._execute_tool(handler, tool_args)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }

    async def _execute_tool(self, tool_handler: MCPToolHandler, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and format the result."""
        try:
            # Call the tool handler
            if asyncio.iscoroutinefunction(tool_handler.handler):
                result = await tool_handler.handler(args)
            else:
                result = tool_handler.handler(args)

            # Format result according to MCP spec
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result)}]
            else:
                content = [{"type": "text", "text": str(result)}]

            return content

        except Exception as e:
            # Return error result
            return [{"type": "text", "text": f"Error: {str(e)}"}]


class MCPStdioServer:
    """MCP server that communicates over stdio."""

    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server

    async def run(self):
        """Run the MCP server over stdio."""
        import sys

        # Send initialize response
        init_response = await self.mcp_server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        })
        print(json.dumps(init_response), flush=True)

        # Read from stdin and respond
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = await self.mcp_server.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                self.mcp_server.logger.error("Invalid JSON received")
            except Exception as e:
                self.mcp_server.logger.error(f"Error processing request: {e}")


class MCPHTTPServer:
    """MCP server that communicates over HTTP."""

    def __init__(self, mcp_server: MCPServer, host: str = "localhost", port: int = 8001):
        self.mcp_server = mcp_server
        self.host = host
        self.port = port

    async def run(self):
        """Run the MCP server over HTTP."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.post("/")
        async def handle_request(request: Request):
            try:
                data = await request.json()
                response = await self.mcp_server.handle_request(data)
                return JSONResponse(content=response)
            except Exception as e:
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        }
                    },
                    status_code=500
                )

        import uvicorn
        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()
#!/usr/bin/env python3
"""
MCP Server for LLM Router.

This server provides tools that can be used by MCP-compatible clients,
including LLMs that support tool calling.
"""

import asyncio
import logging
import argparse
from .mcp_server import MCPServer, MCPStdioServer, MCPHTTPServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_tools(mcp_server: MCPServer):
    """Register some sample tools for demonstration."""

    # Calculator tool
    def calculate(args):
        """Simple calculator tool."""
        try:
            expression = args.get("expression", "")
            # Note: In production, use a safe evaluation method
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    mcp_server.register_tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        handler=calculate
    )

    # Weather tool (mock)
    def get_weather(args):
        """Mock weather tool."""
        location = args.get("location", "unknown")
        return f"Weather in {location}: Sunny, 72°F"

    mcp_server.register_tool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Location to get weather for"
                }
            },
            "required": ["location"]
        },
        handler=get_weather
    )

    # File reader tool
    def read_file(args):
        """Read contents of a file."""
        try:
            filepath = args.get("filepath", "")
            with open(filepath, 'r') as f:
                content = f.read()
            return content[:1000]  # Limit output
        except Exception as e:
            return f"Error reading file: {str(e)}"

    mcp_server.register_tool(
        name="read_file",
        description="Read the contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["filepath"]
        },
        handler=read_file
    )


async def main():
    """Main entry point for MCP server."""
    parser = argparse.ArgumentParser(description="MCP Server for LLM Router")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                       help="Transport method (default: stdio)")
    parser.add_argument("--host", default="localhost", help="HTTP server host")
    parser.add_argument("--port", type=int, default=8001, help="HTTP server port")

    args = parser.parse_args()

    # Create MCP server
    mcp_server = MCPServer()

    # Register sample tools
    create_sample_tools(mcp_server)

    logger.info(f"Starting MCP server with transport: {args.transport}")

    if args.transport == "stdio":
        server = MCPStdioServer(mcp_server)
        await server.run()
    elif args.transport == "http":
        server = MCPHTTPServer(mcp_server, args.host, args.port)
        await server.run()


if __name__ == "__main__":
    asyncio.run(main())
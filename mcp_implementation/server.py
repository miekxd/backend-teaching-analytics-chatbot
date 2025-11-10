"""
Embedded MCP Server

Uses the correct MCP SDK API for tool registration.
"""
from mcp.server import Server
from mcp.types import Tool, TextContent
from typing import Any, Sequence
import json
import logging

logger = logging.getLogger(__name__)

# Global MCP server instance
_mcp_server = None

# Store tool handlers
_tool_handlers = {}


def create_mcp_server() -> Server:
    """
    Create and configure MCP server with all tools
    """
    server = Server("teaching-assistant-v2")
    
    # Import and register tools
    from mcp_implementation.tools.lesson_analysis import get_lesson_tools
    from mcp_implementation.tools.visualization import get_visualization_tools
    from mcp_implementation.tools.information import get_information_tools
    from mcp_implementation.tools.resource_tools import get_resource_tools
    from mcp_implementation.tools.lesson_planning import get_lesson_planning_tools
    
    # Collect all tools
    all_tools = []
    all_tools.extend(get_lesson_tools())
    all_tools.extend(get_visualization_tools())
    all_tools.extend(get_information_tools())
    all_tools.extend(get_resource_tools())
    all_tools.extend(get_lesson_planning_tools())
    
    # Store tools and their handlers
    for tool_def, handler in all_tools:
        _tool_handlers[tool_def.name] = handler
    
    # Register list_tools handler
    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """Return list of available tools"""
        return [tool_def for tool_def, _ in all_tools]
    
    # Register call_tool handler
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        """Execute a tool by name"""
        if name not in _tool_handlers:
            error_msg = f"Unknown tool: {name}"
            logger.error(f"❌ {error_msg}")
            return [TextContent(type="text", text=json.dumps({"error": error_msg}))]
        
        handler = _tool_handlers[name]
        return await handler(arguments)
    
    logger.info(f"✅ MCP Server initialized with {len(all_tools)} tools")
    return server


def get_mcp_server() -> Server:
    """
    Get singleton MCP server instance
    """
    global _mcp_server
    
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
    
    return _mcp_server


async def execute_tool_safe(server: Server, tool_name: str, arguments: dict) -> Any:
    """
    Safely execute an MCP tool with error handling
    """
    try:
        if tool_name not in _tool_handlers:
            return {
                "error": f"Tool {tool_name} not found",
                "tool": tool_name
            }
        
        # Execute tool handler directly
        result = await _tool_handlers[tool_name](arguments)
        
        # Parse result
        if isinstance(result, list) and len(result) > 0:
            return json.loads(result[0].text)
        elif hasattr(result, 'text'):
            return json.loads(result.text)
        else:
            return result
    
    except Exception as e:
        logger.error(f"❌ Error executing tool {tool_name}: {str(e)}", exc_info=True)
        return {
            "error": f"Tool execution failed: {str(e)}",
            "tool": tool_name
        }


def get_tool_schemas_for_openai(server: Server) -> list[dict]:
    """
    Convert MCP tool schemas to Azure OpenAI function calling format
    """
    # Get tools from handlers
    tools_list = []
    
    for tool_def, _ in [(tool_def, handler) for tool_def, handler in 
                        [(td, h) for td, h in 
                         [item for sublist in [
                             get_lesson_tools(),
                             get_visualization_tools(), 
                             get_information_tools(),
                             get_resource_tools(),
                             get_lesson_planning_tools()
                         ] for item in sublist]]]:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.inputSchema
            }
        }
        tools_list.append(openai_tool)
    
    return tools_list


# Import tool definitions (circular import safe)
def get_lesson_tools():
    from mcp_implementation.tools.lesson_analysis import get_lesson_tools
    return get_lesson_tools()

def get_visualization_tools():
    from mcp_implementation.tools.visualization import get_visualization_tools
    return get_visualization_tools()

def get_information_tools():
    from mcp_implementation.tools.information import get_information_tools
    return get_information_tools()

def get_resource_tools():
    from mcp_implementation.tools.resource_tools import get_resource_tools
    return get_resource_tools()

def get_lesson_planning_tools():
    from mcp_implementation.tools.lesson_planning import get_lesson_planning_tools
    return get_lesson_planning_tools()

def get_tool_list() -> list[str]:
    """
    Get list of registered tool names
    
    Returns:
        List of tool names
    """
    return list(_tool_handlers.keys())


def get_tool_info() -> list[dict]:
    """
    Get detailed information about all registered tools
    
    Returns:
        List of dicts with tool info
    """
    from mcp_implementation.tools.lesson_analysis import get_lesson_tools
    from mcp_implementation.tools.visualization import get_visualization_tools
    from mcp_implementation.tools.information import get_information_tools
    from mcp_implementation.tools.resource_tools import get_resource_tools
    from mcp_implementation.tools.lesson_planning import get_lesson_planning_tools

    all_tools = []
    all_tools.extend(get_lesson_tools())
    all_tools.extend(get_visualization_tools())
    all_tools.extend(get_information_tools())
    all_tools.extend(get_resource_tools())
    all_tools.extend(get_lesson_planning_tools())
    
    return [
        {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.inputSchema
        }
        for tool_def, _ in all_tools
    ]
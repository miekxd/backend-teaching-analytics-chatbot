"""
MCP Client Wrapper

Provides a convenient interface for FastAPI to interact with the embedded MCP server.
"""
from openai import AsyncAzureOpenAI
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import logging

from app.core.config import settings
from mcp_implementation.server import get_mcp_server, execute_tool_safe, get_tool_schemas_for_openai

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client wrapper for MCP server interactions
    
    Handles:
    - Azure OpenAI function calling with MCP tools
    - Tool execution
    - Streaming responses
    - Error handling
    """
    
    def __init__(self):
        self.mcp_server = get_mcp_server()
        
        # Azure OpenAI client
        self.openai_client = AsyncAzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
        
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_RAG
        
        # System prompt for orchestration
        self.system_prompt = """You are an AI teaching mentor for Singapore educators with access to powerful analysis tools.

You have access to these tools:
- analyze_lesson: Deep analysis of teaching lessons with evidence from transcripts
- get_lesson_summary: Quick summaries of lessons
- detect_visualization_needs: Determine if graphs/visualizations would help
- get_lesson_metadata: Get basic information about lessons
- find_teaching_resources: Find external teaching resources (papers, videos, articles) for skills/competencies
- generate_lesson_plan: Generate improved lesson plans based on transcript analysis 


Use tools intelligently:
1. For specific lesson questions → use analyze_lesson
2. For visualization requests → use detect_visualization_needs first, then analyze_lesson
3. For general questions → use analyze_lesson (it handles both specific and general queries)
4. For resource requests → use find_teaching_resources
5. For lesson plan generation → use generate_lesson_plan
6. You can call multiple tools if needed

Always provide helpful, actionable coaching feedback referencing the Singapore Teaching Practice framework when relevant."""
        
        logger.info("✅ MCPClient initialized")
    
    
    async def chat(
        self,
        message: str,
        file_ids: List[int],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message with LLM orchestration of MCP tools
        
        Args:
            message: User's message
            file_ids: Relevant lesson file IDs
            conversation_history: Previous conversation
        
        Returns:
            Response dict with:
                - response: Text response
                - tools_used: List of tools called
                - needs_graph: Whether visualization is needed
                - graph_types: Types of graphs if needed
        """
        try:
            logger.info(f"💬 MCPClient.chat: '{message[:50]}...', files={file_ids}")
            
            # Build messages for OpenAI
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 3 exchanges
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})
            
            # Add current message with context
            current_message = f"""User's question: {message}

Available lesson files: {file_ids}

Please use the appropriate tools to help answer this question."""
            
            messages.append({"role": "user", "content": current_message})
            
            # Get tool schemas
            tools = get_tool_schemas_for_openai(self.mcp_server)
            
            # First LLM call - tool selection
            response = await self.openai_client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            response_message = response.choices[0].message
            tools_used = []
            
            # Execute tools if LLM wants to call them
            if response_message.tool_calls:
                messages.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🔧 LLM calling tool: {tool_name}")
                    tools_used.append(tool_name)
                    
                    # Add file_ids if not in args
                    if "file_ids" not in tool_args and file_ids:
                        tool_args["file_ids"] = file_ids
                    
                    # Execute tool
                    result = await execute_tool_safe(
                        server=self.mcp_server,
                        tool_name=tool_name,
                        arguments=tool_args
                    )
                    
                    # Add tool result to messages
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result)
                    })
                
                # Second LLM call - generate final response with tool results
                final_response = await self.openai_client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=0.7
                )
                
                final_text = final_response.choices[0].message.content
            else:
                # No tools needed
                final_text = response_message.content
            
            # Check if any tool was detect_visualization_needs
            needs_graph = False
            graph_types = []
            
            for msg in messages:
                if msg.get("role") == "tool" and msg.get("name") == "detect_visualization_needs":
                    try:
                        tool_result = json.loads(msg.get("content", "{}"))
                        needs_graph = tool_result.get("needs_graph", False)
                        graph_types = tool_result.get("graph_types", [])
                    except:
                        pass
            
            result = {
                "response": final_text,
                "tools_used": tools_used,
                "needs_graph": needs_graph,
                "graph_types": graph_types,
                "lesson_filter": file_ids,
                "area_filter": []
            }
            
            logger.info(f"✅ MCPClient.chat completed: {len(tools_used)} tools used")
            return result
        
        except Exception as e:
            logger.error(f"❌ MCPClient.chat error: {str(e)}", exc_info=True)
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "tools_used": [],
                "needs_graph": False,
                "graph_types": [],
                "error": str(e)
            }
    
    
    async def chat_stream(
        self,
        message: str,
        file_ids: List[int],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming version of chat
        """
        try:
            logger.info(f"🌊 MCPClient.chat_stream: '{message[:50]}...', files={file_ids}")
            
            # Build messages
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if conversation_history:
                for msg in conversation_history[-6:]:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        messages.append({"role": role, "content": content})
            
            current_message = f"""User's question: {message}

    Available lesson files: {file_ids}

    Please use the appropriate tools to help answer this question."""
            
            messages.append({"role": "user", "content": current_message})
            
            # Get tool schemas
            tools = get_tool_schemas_for_openai(self.mcp_server)
            
            # First call - tool selection (non-streaming)
            response = await self.openai_client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )
            
            response_message = response.choices[0].message
            
            # Execute tools if needed
            if response_message.tool_calls:
                messages.append(response_message)
                
                graph_metadata_sent = False
                
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🔧 Streaming - calling tool: {tool_name}")
                    
                    if "file_ids" not in tool_args and file_ids:
                        tool_args["file_ids"] = file_ids
                    
                    # Execute tool
                    result = await execute_tool_safe(
                        server=self.mcp_server,
                        tool_name=tool_name,
                        arguments=tool_args
                    )
                    
                    # If visualization tool, send graph metadata first
                    if tool_name == "detect_visualization_needs" and not graph_metadata_sent:
                        if result.get("needs_graph"):
                            for graph_obj in result.get("graph_types", []):
                                graph_data = {
                                    'type': 'graph_code',
                                    'code': f'GRAPH:{graph_obj.get("type")}',
                                    'reason': graph_obj.get("reason"),
                                    'lesson_filter': result.get("lesson_filter", []),
                                    'area_filter': result.get("area_filter", [])
                                }
                                yield f"data: {json.dumps(graph_data)}\n\n"
                            graph_metadata_sent = True
                    
                    # Add tool result
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result)
                    })
                
                # Stream final response
                stream = await self.openai_client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in stream:
                    # ✅ FIX: Check if choices exists and has content
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield delta.content
            
            else:
                # No tools - just stream direct response
                stream = await self.openai_client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in stream:
                    # ✅ FIX: Check if choices exists and has content
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield delta.content
        
        except Exception as e:
            logger.error(f"❌ MCPClient.chat_stream error: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
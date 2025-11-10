# app/mcp/teaching_assistant_mcp.py
"""
Teaching Assistant MCP Server

This MCP server exposes the teaching assistant functionality to AI clients
(Claude Desktop, AI agents, etc.) while keeping FastAPI endpoints for the frontend.

Run this server separately from FastAPI:
    python -m app.mcp.teaching_assistant_mcp

Or:
    python app/mcp/teaching_assistant_mcp.py
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
import logging

# MCP imports
from mcp.server import Server
import mcp.types as types

# Import your EXISTING services (shared with FastAPI)
from app.services.unified import unified_assistant
from app.services.rag_specific import rag_assistant
from app.services.general import general_assistant
from app.services.intent_analyzer import intent_analyzer
from app.db.supabase import get_supabase_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# Initialize MCP Server
# ==========================================
mcp_server = Server("teaching-assistant")

logger.info("🎓 Teaching Assistant MCP Server initialized")


# ==========================================
# TOOL 1: Unified Chat (Main Entry Point)
# ==========================================
@mcp_server.tool()
async def unified_chat(
    message: str,
    file_ids: list[int],
    conversation_history: list[dict] | None = None
) -> str:
    """
    Main teaching assistant chat interface with intelligent routing.
    
    This tool provides comprehensive teaching analysis and feedback by:
    - Analyzing intent to determine if query needs RAG or general assistant
    - Routing to appropriate assistant with proper context
    - Handling class period filtering (beginning/middle/end)
    - Detecting when data visualization would be helpful
    - Filtering by specific lessons or teaching areas
    
    The assistant can handle:
    - Lesson-specific questions ("What happened in the first 10 minutes?")
    - General teaching advice ("How can I improve student engagement?")
    - Evidence-based feedback with timestamps from lesson transcripts
    - Singapore Teaching Practice framework alignment
    - Graph/visualization recommendations
    
    Args:
        message: The educator's question or request
        file_ids: List of lesson file IDs for context (required)
                 These should be valid file IDs from your lesson database
        conversation_history: Previous conversation messages for context
                            Format: [{"role": "user"|"assistant", "content": "..."}]
    
    Returns:
        JSON string containing:
        {
            "response": "The assistant's detailed response",
            "needs_graph": true|false,
            "graph_types": [
                {"type": "teaching_area_distribution", "reason": "Shows..."},
                {"type": "utterance_timeline", "reason": "Displays..."}
            ],
            "lesson_filter": [37, 38],  // Specific lessons to filter
            "area_filter": ["3.3", "3.4"],  // Teaching areas to focus on
            "agent_used": "rag_assistant"|"general_assistant",
            "class_period": "beginning"|"middle"|"end"|null
        }
    
    Examples:
        # General feedback request
        result = await unified_chat(
            message="How did my lesson go overall?",
            file_ids=[37, 38],
            conversation_history=[]
        )
        
        # Time-specific analysis
        result = await unified_chat(
            message="Show me examples from the beginning of the lesson",
            file_ids=[37],
            conversation_history=[]
        )
        
        # Graph/visualization request
        result = await unified_chat(
            message="Show me a chart of my questioning patterns over time",
            file_ids=[37, 38, 39],
            conversation_history=[]
        )
        
        # Follow-up question with context
        result = await unified_chat(
            message="Can you give me specific examples?",
            file_ids=[37],
            conversation_history=[
                {"role": "user", "content": "How did I do with questioning?"},
                {"role": "assistant", "content": "You used good questioning techniques..."}
            ]
        )
    """
    try:
        # Validate inputs
        if not message or not message.strip():
            return json.dumps({"error": "Message cannot be empty"}, indent=2)
        
        if not file_ids or len(file_ids) == 0:
            return json.dumps({"error": "file_ids are required for lesson context"}, indent=2)
        
        user_message = message.strip()
        logger.info(f"📝 Processing query: {user_message[:50]}... with {len(file_ids)} file(s)")
        
        # Step 1: Analyze intent to determine routing
        logger.info("🧠 Analyzing intent...")
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history or []
        )
        
        agent_to_use = intent_analysis.get("agent_to_use", "general_assistant")
        class_period = intent_analysis.get("class_period")
        transformed_query = intent_analysis.get("transformed_query", user_message)
        needs_graph = intent_analysis.get("needs_graph", False)
        graph_types = intent_analysis.get("graph_types", [])
        lesson_filter = intent_analysis.get("lesson_filter", [])
        area_filter = intent_analysis.get("area_filter", [])
        
        logger.info(f"🎯 Routing to: {agent_to_use}")
        if class_period:
            logger.info(f"⏰ Class period: {class_period}")
        if needs_graph:
            logger.info(f"📊 Graphs recommended: {len(graph_types)}")
        
        # Step 2: Route to appropriate assistant
        if needs_graph == 'true' or needs_graph == True:
            # For graph requests, use transformed query
            if agent_to_use == "rag_assistant":
                logger.info("🔍 Using RAG assistant with graph context...")
                response = await rag_assistant.get_response(
                    semantic_query=transformed_query,
                    user_message=transformed_query,
                    file_ids=file_ids,
                    class_period=class_period,
                    conversation_history=conversation_history or []
                )
            else:
                logger.info("💬 Using general assistant with graph context...")
                response = await general_assistant.get_response(
                    user_message=transformed_query,
                    file_ids=file_ids,
                    conversation_history=conversation_history or []
                )
        else:
            # For regular queries
            if agent_to_use == "rag_assistant":
                logger.info("🔍 Using RAG assistant...")
                response = await rag_assistant.get_response(
                    semantic_query=transformed_query,
                    user_message=user_message,
                    file_ids=file_ids,
                    class_period=class_period,
                    conversation_history=conversation_history or []
                )
            else:
                logger.info("💬 Using general assistant...")
                response = await general_assistant.get_response(
                    user_message=user_message,
                    file_ids=file_ids,
                    conversation_history=conversation_history or []
                )
        
        # Step 3: Return structured response
        result = {
            "response": response,
            "needs_graph": needs_graph,
            "graph_types": graph_types,
            "lesson_filter": lesson_filter,
            "area_filter": area_filter,
            "agent_used": agent_to_use,
            "class_period": class_period
        }
        
        logger.info("✅ Response generated successfully")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"❌ Unified chat error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Chat error: {str(e)}"}, indent=2)


# ==========================================
# TOOL 2: Unified Chat Streaming
# ==========================================
@mcp_server.tool()
async def unified_chat_streaming(
    message: str,
    file_ids: list[int],
    conversation_history: list[dict] | None = None
) -> str:
    """
    Streaming version of unified chat that collects the complete response.
    
    Note: MCP tools return complete strings, so this collects the entire
    streamed response before returning. For true streaming in your client,
    you would need to implement streaming at the MCP client level.
    
    This tool is useful when you want to ensure you get the complete response
    even if there are network interruptions during generation.
    
    Args:
        message: The educator's question
        file_ids: List of lesson file IDs
        conversation_history: Previous conversation context
    
    Returns:
        JSON string with:
        {
            "response": "Complete streamed response text",
            "graph_codes": ["GRAPH:teaching_area_distribution", ...],
            "graph_details": [
                {
                    "type": "teaching_area_distribution",
                    "reason": "Shows distribution across teaching areas",
                    "lesson_filter": [37, 38],
                    "area_filter": ["3.3"]
                }
            ],
            "agent_used": "rag_assistant"|"general_assistant",
            "class_period": "beginning"|"middle"|"end"|null,
            "needs_graph": true|false
        }
    
    Example:
        result = await unified_chat_streaming(
            message="How did my questioning improve over the three lessons?",
            file_ids=[37, 38, 39],
            conversation_history=[]
        )
    """
    try:
        if not message or not message.strip():
            return json.dumps({"error": "Message cannot be empty"}, indent=2)
        
        if not file_ids or len(file_ids) == 0:
            return json.dumps({"error": "file_ids are required for lesson context"}, indent=2)
        
        user_message = message.strip()
        logger.info(f"📝 Processing streaming query: {user_message[:50]}...")
        
        # Step 1: Analyze intent
        logger.info("🧠 Analyzing intent for streaming...")
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids,
            conversation_history=conversation_history or []
        )
        
        agent_to_use = intent_analysis.get("agent_to_use", "general_assistant")
        class_period = intent_analysis.get("class_period")
        transformed_query = intent_analysis.get("transformed_query", user_message)
        needs_graph = intent_analysis.get("needs_graph", False)
        graph_types = intent_analysis.get("graph_types", [])
        lesson_filter = intent_analysis.get("lesson_filter", [])
        area_filter = intent_analysis.get("area_filter", [])
        
        logger.info(f"🎯 Streaming from: {agent_to_use}")
        
        # Normalize needs_graph to boolean
        if isinstance(needs_graph, str):
            needs_graph_bool = needs_graph.lower() == "true"
        else:
            needs_graph_bool = bool(needs_graph)
        
        # Prepare graph information
        graph_codes = []
        graph_details = []
        if needs_graph_bool and graph_types:
            for graph_obj in graph_types:
                graph_type = graph_obj.get("type")
                graph_reason = graph_obj.get("reason")
                if graph_type:
                    graph_codes.append(f"GRAPH:{graph_type}")
                    graph_details.append({
                        "type": graph_type,
                        "reason": graph_reason,
                        "lesson_filter": lesson_filter,
                        "area_filter": area_filter
                    })
        
        # Step 2: Collect streaming response
        logger.info("📡 Collecting stream...")
        response_chunks = []
        
        if needs_graph_bool:
            if agent_to_use == "rag_assistant":
                async for chunk in rag_assistant.get_response_stream(
                    semantic_query=transformed_query,
                    user_message=transformed_query,
                    file_ids=file_ids,
                    class_period=class_period,
                    conversation_history=conversation_history or []
                ):
                    response_chunks.append(chunk)
            else:
                async for chunk in general_assistant.get_response_stream(
                    user_message=transformed_query,
                    file_ids=file_ids,
                    conversation_history=conversation_history or []
                ):
                    response_chunks.append(chunk)
        else:
            if agent_to_use == "rag_assistant":
                async for chunk in rag_assistant.get_response_stream(
                    semantic_query=transformed_query,
                    user_message=user_message,
                    file_ids=file_ids,
                    class_period=class_period,
                    conversation_history=conversation_history or []
                ):
                    response_chunks.append(chunk)
            else:
                async for chunk in general_assistant.get_response_stream(
                    user_message=user_message,
                    file_ids=file_ids,
                    conversation_history=conversation_history or []
                ):
                    response_chunks.append(chunk)
        
        # Combine all chunks
        complete_response = "".join(response_chunks)
        
        result = {
            "response": complete_response,
            "graph_codes": graph_codes,
            "graph_details": graph_details,
            "agent_used": agent_to_use,
            "class_period": class_period,
            "needs_graph": needs_graph_bool
        }
        
        logger.info(f"✅ Streaming complete ({len(response_chunks)} chunks)")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"❌ Streaming error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Streaming error: {str(e)}"}, indent=2)


# ==========================================
# TOOL 3: Intent Analysis (Debugging)
# ==========================================
@mcp_server.tool()
async def analyze_intent(
    message: str,
    file_ids: list[int] | None = None,
    conversation_history: list[dict] | None = None
) -> str:
    """
    Analyze query intent without generating a response.
    
    This is a diagnostic tool that shows how the system will process a query:
    - Which agent will be used (general_assistant or rag_assistant)
    - Class period detection (beginning/middle/end)
    - Query transformation (how the query is optimized)
    - Graph recommendations
    - Lesson and area filtering
    
    Useful for:
    - Understanding routing decisions
    - Debugging unexpected behavior
    - Testing query classification
    - Pre-flight validation before actual chat
    
    Args:
        message: The query to analyze
        file_ids: Optional lesson file IDs for context
        conversation_history: Optional previous conversation
    
    Returns:
        JSON string with complete analysis:
        {
            "intent_analysis": {
                "agent_to_use": "rag_assistant",
                "class_period": "beginning",
                "transformed_query": "Optimized version of the query",
                "needs_graph": true,
                "graph_types": [{...}],
                "lesson_filter": [37],
                "area_filter": ["3.3"]
            },
            "routing_explanation": "Detailed explanation of why this routing"
        }
    
    Example:
        analysis = await analyze_intent(
            message="What happened in the first 10 minutes?",
            file_ids=[37],
            conversation_history=[]
        )
        # Returns detailed breakdown of how query will be handled
    """
    try:
        if not message or not message.strip():
            return json.dumps({"error": "Message cannot be empty"}, indent=2)
        
        user_message = message.strip()
        logger.info(f"🔍 Analyzing intent for: {user_message[:50]}...")
        
        intent_analysis = await intent_analyzer.analyze_intent(
            user_message=user_message,
            file_ids=file_ids or [],
            conversation_history=conversation_history or []
        )
        
        # Generate routing explanation
        agent = intent_analysis.get("agent_to_use")
        class_period = intent_analysis.get("class_period")
        needs_graph = intent_analysis.get("needs_graph")
        
        explanation_parts = []
        explanation_parts.append(f"Query will be routed to: {agent}")
        
        if class_period:
            explanation_parts.append(f"Time period detected: {class_period}")
        
        if needs_graph:
            graph_types = intent_analysis.get("graph_types", [])
            if graph_types:
                graph_names = [g.get("type") for g in graph_types]
                explanation_parts.append(f"Graphs recommended: {', '.join(graph_names)}")
        
        if agent == "rag_assistant":
            explanation_parts.append("Reason: Query requires specific evidence or detailed analysis from lesson transcripts")
        else:
            explanation_parts.append("Reason: Query can be answered with general teaching expertise and lesson summaries")
        
        routing_explanation = ". ".join(explanation_parts)
        
        result = {
            "intent_analysis": intent_analysis,
            "routing_explanation": routing_explanation
        }
        
        logger.info(f"✅ Intent analyzed: {agent}")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"❌ Intent analysis error: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Intent analysis error: {str(e)}"}, indent=2)


# ==========================================
# TOOL 4: Direct Unified Assistant
# ==========================================
@mcp_server.tool()
async def ask_unified_assistant(
    user_message: str,
    file_ids: list[int],
    conversation_history: list[dict] | None = None,
    top_k: int = 5
) -> str:
    """
    Direct call to unified teaching assistant (bypasses intent routing).
    
    This tool provides direct access to the UnifiedTeachingAssistant which:
    - Always retrieves lesson context
    - Conditionally applies RAG based on built-in query relevance analysis
    - Blends lesson-specific evidence with general teaching expertise
    - Maintains conversation context
    - References Singapore Teaching Practice framework
    
    Use this when you want the assistant's built-in intelligence rather than
    explicit routing through the intent analyzer. The assistant will automatically
    determine if your query needs RAG or can be answered conversationally.
    
    Args:
        user_message: The educator's question
        file_ids: List of lesson file IDs (required)
        conversation_history: Previous conversation messages
        top_k: Number of relevant chunks to retrieve (default: 5)
                Higher values provide more context but may be slower
    
    Returns:
        Direct text response from the unified assistant (no JSON wrapper)
    
    Examples:
        # General teaching advice
        response = await ask_unified_assistant(
            user_message="How can I improve student engagement?",
            file_ids=[37, 38],
            conversation_history=[],
            top_k=5
        )
        
        # Lesson-specific query
        response = await ask_unified_assistant(
            user_message="What happened when I introduced the new concept?",
            file_ids=[37],
            conversation_history=[],
            top_k=7
        )
    """
    try:
        if not user_message or not user_message.strip():
            return json.dumps({"error": "Message cannot be empty"}, indent=2)
        
        if not file_ids or len(file_ids) == 0:
            return json.dumps({"error": "file_ids are required"}, indent=2)
        
        logger.info(f"🎓 Direct unified assistant call: {user_message[:50]}...")
        
        response = await unified_assistant.get_response(
            user_message=user_message.strip(),
            file_ids=file_ids,
            conversation_history=conversation_history or [],
            top_k=top_k
        )
        
        logger.info("✅ Unified assistant response generated")
        return response
    
    except Exception as e:
        logger.error(f"❌ Unified assistant error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


# ==========================================
# TOOL 5: Prepare Teaching Context
# ==========================================
@mcp_server.tool()
async def prepare_teaching_context(
    user_message: str,
    file_ids: list[int],
    top_k: int = 5
) -> str:
    """
    Prepare and return teaching context without generating a response.
    
    This diagnostic tool shows exactly what context would be used to answer
    a query, including:
    - Query classification (type, relevance, response mode)
    - Lesson summaries retrieved
    - Relevant chunks identified
    - Teaching areas detected
    - Semantic search results
    
    Perfect for:
    - Debugging RAG retrieval
    - Understanding why certain chunks are selected
    - Validating chunk relevance to queries
    - Previewing context before generating full responses
    - Tuning top_k parameter for optimal results
    
    Args:
        user_message: The query to analyze
        file_ids: Lesson file IDs to retrieve context from
        top_k: Number of chunks to retrieve (default: 5)
    
    Returns:
        JSON string with detailed context information:
        {
            "query_analysis": {
                "query_type": "lesson_specific"|"time_based"|"general",
                "lesson_relevance": "HIGH"|"MEDIUM"|"LOW",
                "response_mode": "lesson_analysis"|"blended"|"conversational"
            },
            "lesson_summaries": "Full text of lesson summaries",
            "context_chunks": "Formatted chunks for LLM context",
            "retrieved_chunks": [
                {
                    "file_id": 37,
                    "start_time": "00:05",
                    "end_time": "00:10",
                    "teaching_areas": ["3.3", "3.4"],
                    "text_preview": "First 200 chars of chunk..."
                }
            ],
            "num_chunks_retrieved": 3
        }
    
    Example:
        context = await prepare_teaching_context(
            user_message="Show me my questioning techniques",
            file_ids=[37, 38],
            top_k=5
        )
        # Examine retrieved chunks before generating response
    """
    try:
        if not user_message or not user_message.strip():
            return json.dumps({"error": "Message cannot be empty"}, indent=2)
        
        if not file_ids or len(file_ids) == 0:
            return json.dumps({"error": "file_ids are required"}, indent=2)
        
        logger.info(f"🔧 Preparing context for: {user_message[:50]}...")
        
        # Analyze query
        query_analysis = unified_assistant._analyze_query(user_message.strip())
        logger.info(f"Query type: {query_analysis['query_type']}, Relevance: {query_analysis['lesson_relevance']}")
        
        # Get lesson summaries
        lesson_summaries = unified_assistant._get_file_summaries(file_ids)
        
        # Get and process chunks
        all_chunks = []
        for fid in file_ids:
            chunks = unified_assistant._get_chunks_from_supabase(fid)
            all_chunks.extend(chunks)
        
        logger.info(f"Retrieved {len(all_chunks)} total chunks")
        
        context_chunks_text = "No chunks available"
        retrieved_chunks_info = []
        
        if all_chunks and query_analysis["lesson_relevance"] in ["HIGH", "MEDIUM"]:
            # Apply time filtering
            if query_analysis["query_type"] in ["lesson_specific", "time_based"]:
                filtered_chunks = unified_assistant._apply_time_filters(user_message, all_chunks)
                logger.info(f"After time filtering: {len(filtered_chunks)} chunks")
            else:
                filtered_chunks = all_chunks
            
            if filtered_chunks:
                search_k = top_k if query_analysis["lesson_relevance"] == "HIGH" else min(3, top_k)
                top_chunks = unified_assistant._semantic_search(user_message, filtered_chunks, top_k=search_k)
                logger.info(f"After semantic search: {len(top_chunks)} chunks")
                
                if top_chunks:
                    context_chunks_text = unified_assistant._format_chunks_for_context(
                        top_chunks,
                        query_analysis["lesson_relevance"]
                    )
                    
                    # Extract chunk metadata
                    for chunk in top_chunks:
                        retrieved_chunks_info.append({
                            "file_id": chunk.get("file_id"),
                            "start_time": chunk.get("start_time"),
                            "end_time": chunk.get("end_time"),
                            "teaching_areas": chunk.get("teaching_areas", []),
                            "text_preview": unified_assistant._get_chunk_text(chunk)[:200]
                        })
        
        result = {
            "query_analysis": query_analysis,
            "lesson_summaries": lesson_summaries,
            "context_chunks": context_chunks_text,
            "retrieved_chunks": retrieved_chunks_info,
            "num_chunks_retrieved": len(retrieved_chunks_info)
        }
        
        logger.info(f"✅ Context prepared: {len(retrieved_chunks_info)} chunks retrieved")
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"❌ Context preparation error: {str(e)}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2)


# ==========================================
# TOOL 6: Get Lesson Information
# ==========================================
@mcp_server.tool()
async def get_lesson_info(
    file_ids: list[int]
) -> str:
    """
    Get detailed information about specific lesson files.
    
    Retrieves metadata and summaries for the requested lesson files,
    including filename, upload date, summary, and any other available metadata.
    
    Args:
        file_ids: List of lesson file IDs to retrieve information for
    
    Returns:
        JSON string with lesson information:
        {
            "lessons": [
                {
                    "file_id": 37,
                    "stored_filename": "Science_Lesson1(09-07-2024).xlsx",
                    "data_summary": "This lesson focused on...",
                    "upload_date": "2024-07-09",
                    ...other metadata
                }
            ]
        }
    
    Example:
        info = await get_lesson_info(file_ids=[37, 38, 39])
    """
    try:
        if not file_ids or len(file_ids) == 0:
            return json.dumps({"error": "file_ids are required"}, indent=2)
        
        logger.info(f"📚 Retrieving info for {len(file_ids)} lesson(s)...")
        
        supabase = get_supabase_client()
        result = supabase.table("files").select("*").in_("file_id", file_ids).execute()
        
        if result.data:
            lessons_info = {
                "lessons": result.data,
                "count": len(result.data)
            }
            logger.info(f"✅ Retrieved info for {len(result.data)} lesson(s)")
            return json.dumps(lessons_info, indent=2)
        else:
            return json.dumps({"error": "No lessons found", "lessons": []}, indent=2)
    
    except Exception as e:
        logger.error(f"❌ Error retrieving lesson info: {str(e)}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2)


# ==========================================
# RESOURCES: Framework Information
# ==========================================
@mcp_server.resource("teaching://framework/description")
async def get_framework_description() -> str:
    """
    Get complete Singapore Teaching Practice framework information.
    
    Returns detailed description of all teaching areas used in analysis,
    including area codes, names, and descriptions.
    """
    logger.info("📖 Retrieving framework description")
    
    framework = {
        "name": "Singapore Teaching Practice Framework",
        "description": "Framework for analyzing and improving teaching practices in Singapore schools",
        "version": "1.0",
        "teaching_areas": {
            "1.1": {
                "code": "1.1",
                "name": "Establishing Interaction and Rapport",
                "description": "Building positive relationships and connections with students",
                "focus": "Teacher-student relationships, classroom climate, emotional connection"
            },
            "1.2": {
                "code": "1.2",
                "name": "Setting and Maintaining Rules and Routine",
                "description": "Clear expectations and classroom management",
                "focus": "Behavior management, procedures, expectations, transitions"
            },
            "3.1": {
                "code": "3.1",
                "name": "Activating Prior Knowledge",
                "description": "Connecting to existing knowledge and experiences",
                "focus": "Prior learning, connections, schema activation, relevance"
            },
            "3.2": {
                "code": "3.2",
                "name": "Motivating Learners for Learning Engagement",
                "description": "Inspiring active participation and engagement",
                "focus": "Student motivation, interest, active participation, enthusiasm"
            },
            "3.3": {
                "code": "3.3",
                "name": "Using Questions to Deepen Learning",
                "description": "Strategic questioning for critical thinking",
                "focus": "Question types, wait time, follow-up questions, higher-order thinking"
            },
            "3.4": {
                "code": "3.4",
                "name": "Facilitating Collaborative Learning",
                "description": "Effective student-to-student interactions",
                "focus": "Group work, peer learning, collaboration, discussion"
            },
            "3.5": {
                "code": "3.5",
                "name": "Concluding the Lesson",
                "description": "Summarizing and providing closure",
                "focus": "Lesson wrap-up, key points, reflection, preview of next lesson"
            },
            "4.1": {
                "code": "4.1",
                "name": "Checking for Understanding and Providing Feedback",
                "description": "Assessment and constructive feedback",
                "focus": "Formative assessment, feedback quality, checking comprehension"
            }
        }
    }
    
    return json.dumps(framework, indent=2)


# ==========================================
# PROMPTS: Common Teaching Query Templates
# ==========================================
@mcp_server.prompt()
async def comprehensive_lesson_analysis() -> list[types.PromptMessage]:
    """
    Pre-configured prompt for comprehensive lesson analysis.
    
    Generates a detailed analysis covering strengths, areas for improvement,
    key teaching moments, and framework alignment.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text="Please provide a comprehensive analysis of my lesson, covering: 1) Overall strengths and what worked well, 2) Areas for improvement with specific, actionable suggestions, 3) Key teaching moments with timestamps and detailed analysis, 4) Alignment with Singapore Teaching Practice framework areas, 5) Recommendations for future lessons"
            )
        )
    ]


@mcp_server.prompt()
async def questioning_analysis() -> list[types.PromptMessage]:
    """
    Pre-configured prompt for analyzing questioning techniques.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text="Analyze my questioning techniques throughout the lesson. Show specific examples with timestamps of: 1) Effective questions that deepened student thinking, 2) Missed opportunities for better questions, 3) Balance of question types (open/closed, higher/lower order thinking), 4) Wait time patterns and follow-up questioning, 5) Student responses to different question types"
            )
        )
    ]


@mcp_server.prompt()
async def engagement_analysis() -> list[types.PromptMessage]:
    """
    Pre-configured prompt for student engagement analysis.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text="Analyze student engagement throughout my lesson. Identify: 1) Moments of high engagement with timestamps and what made them effective, 2) Moments where engagement dropped and possible reasons why, 3) Specific strategies I used to maintain motivation (3.2), 4) Connection between engagement and collaborative learning (3.4), 5) Concrete suggestions for improving engagement in similar lessons"
            )
        )
    ]


@mcp_server.prompt()
async def time_management_analysis() -> list[types.PromptMessage]:
    """
    Pre-configured prompt for lesson time management analysis.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text="Analyze my lesson time management and pacing. Break down: 1) Time allocation for each section of the lesson with timestamps, 2) Pacing analysis - where I moved too quickly or slowly and why, 3) Effectiveness of the beginning (first 10 minutes), middle sections, and conclusion (3.5), 4) How well transitions were managed, 5) Specific suggestions for better time management in future lessons"
            )
        )
    ]


@mcp_server.prompt()
async def framework_alignment() -> list[types.PromptMessage]:
    """
    Pre-configured prompt for Singapore Teaching Practice framework alignment.
    """
    return [
        types.PromptMessage(
            role="user",
            content=types.TextContent(
                type="text",
                text="Analyze how well my lesson aligns with the Singapore Teaching Practice framework. For each relevant area (1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1): 1) Provide specific examples with timestamps, 2) Rate the strength of implementation, 3) Identify what was done well, 4) Suggest specific improvements, 5) Show connections between different framework areas in my teaching"
            )
        )
    ]


# ==========================================
# Main Entry Point
# ==========================================
async def main():
    """
    Main entry point for running the MCP server.
    """
    logger.info("=" * 50)
    logger.info("🎓 Teaching Assistant MCP Server")
    logger.info("=" * 50)
    logger.info("Server: teaching-assistant")
    logger.info("Tools available: 6")
    logger.info("  - unified_chat (main interface)")
    logger.info("  - unified_chat_streaming")
    logger.info("  - analyze_intent")
    logger.info("  - ask_unified_assistant")
    logger.info("  - prepare_teaching_context")
    logger.info("  - get_lesson_info")
    logger.info("Resources: 1 (framework description)")
    logger.info("Prompts: 5 (analysis templates)")
    logger.info("=" * 50)
    logger.info("🚀 Starting server...")
    
    try:
        await mcp_server.run()
    except KeyboardInterrupt:
        logger.info("\n👋 Server shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
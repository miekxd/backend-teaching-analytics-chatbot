from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v2 import unified_chat_mcp
from app.core.config import settings

from contextlib import asynccontextmanager
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    try:
        from mcp_implementation.server import get_mcp_server, _tool_handlers
        logger.info("🚀 Initializing MCP server...")
        mcp_server = get_mcp_server()
        
        # Get tool count from our handlers dict
        tool_count = len(_tool_handlers)
        logger.info(f"✅ MCP server ready with {tool_count} tools:")
        for tool_name in _tool_handlers.keys():
            logger.info(f"   - {tool_name}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize MCP server: {e}", exc_info=True)
        # Don't raise - let app continue even if MCP fails
        logger.warning("⚠️  Continuing without MCP server")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("👋 Shutting down...")


# Create FastAPI app with lifespan
application  = FastAPI(
    title="NIE Backend API",
    description="AI Teaching Assistant with MCP Tool Orchestration",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
application.include_router(
    unified_chat_mcp.router,
    prefix="/api/v2",
    tags=["unified_chat_mcp"]
)


@application.get("/")
async def root():
    return {
        "message": "NIE Backend API is running",
        "mcp_enabled": True,
        "version": "2.0.0"
    }


@application.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_enabled": True
    }
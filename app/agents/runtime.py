"""
Containerized Agent Runtime

Replaces Claude Code CLI execution with direct Claude API integration.
Runs inside Docker containers instead of tmux sessions.
"""

import asyncio
import json
import os
import sys
import signal
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

import structlog
from anthropic import AsyncAnthropic
import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.config import settings
from core.redis import get_message_broker
from core.database import get_session

logger = structlog.get_logger()


class ContainerizedAgent:
    """
    Claude agent running in Docker container.
    
    Replaces tmux + Claude Code CLI with containerized execution:
    - Direct Claude API integration (no CLI dependency)
    - Container-native lifecycle management
    - Health checks and monitoring
    - Resource isolation and security
    """
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.agent_id = os.getenv("HOSTNAME", f"{agent_type}-{os.getpid()}")
        self.workspace = os.getenv("AGENT_WORKSPACE", "/app/workspace")
        
        # Initialize Claude API client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")
        
        self.anthropic_client = AsyncAnthropic(api_key=api_key)
        
        # Redis connection for task queue
        self.redis = None
        self.db_session = None
        
        # Runtime state
        self.running = False
        self.current_task = None
        self.health_status = "starting"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal", signal=signum, agent_id=self.agent_id)
        self.running = False
    
    async def initialize(self):
        """Initialize agent connections and resources."""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis = aioredis.from_url(redis_url, decode_responses=True)
            
            # Test Redis connection
            await self.redis.ping()
            logger.info("Connected to Redis", agent_id=self.agent_id)
            
            # Initialize database session (for result storage)
            # Note: Using sync session as async DB ops are handled by orchestrator
            
            self.health_status = "ready"
            logger.info("Agent initialized successfully", agent_id=self.agent_id, agent_type=self.agent_type)
            
        except Exception as e:
            self.health_status = "failed"
            logger.error("Failed to initialize agent", agent_id=self.agent_id, error=str(e))
            raise
    
    async def run(self):
        """
        Main agent execution loop.
        
        Replaces tmux session + Claude Code CLI workflow with:
        - Redis task polling
        - Direct Claude API calls
        - Container-native execution
        """
        self.running = True
        logger.info("Agent started successfully", agent_id=self.agent_id, agent_type=self.agent_type)
        
        # Start health check server in background
        health_task = asyncio.create_task(self._health_check_server())
        
        try:
            while self.running:
                try:
                    # Get task from Redis queue (same as tmux approach)
                    task = await self._get_next_task()
                    
                    if task:
                        await self._process_task(task)
                    else:
                        # No tasks available, wait briefly
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(
                        "Error in agent main loop",
                        agent_id=self.agent_id,
                        error=str(e),
                        traceback=traceback.format_exc()
                    )
                    await asyncio.sleep(5)  # Back off on errors
                    
        except KeyboardInterrupt:
            logger.info("Agent interrupted", agent_id=self.agent_id)
        finally:
            # Cleanup
            health_task.cancel()
            await self._cleanup()
    
    async def _get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from Redis queue."""
        try:
            # Listen on agent-specific queue and general queue
            queues = [
                f"agent_messages:{self.agent_id}",
                f"agent_messages:{self.agent_type}",
                "agent_messages:general"
            ]
            
            for queue in queues:
                task_data = await self.redis.brpop(queue, timeout=1)
                if task_data:
                    _, task_json = task_data
                    return json.loads(task_json)
            
            return None
            
        except Exception as e:
            logger.error("Failed to get task from queue", error=str(e))
            return None
    
    async def _process_task(self, task: Dict[str, Any]):
        """
        Process task using Claude API.
        
        Replaces subprocess call to Claude Code CLI.
        """
        task_id = task.get("id", "unknown")
        prompt = task.get("prompt", "")
        
        self.current_task = task_id
        self.health_status = "processing"
        
        logger.info(
            "Processing task",
            agent_id=self.agent_id,
            task_id=task_id,
            prompt_length=len(prompt)
        )
        
        start_time = datetime.now()
        
        try:
            # Execute task via Claude API (replaces claude CLI call)
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.1,
                system=self._get_system_prompt(),
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract response content
            result_content = ""
            for content_block in response.content:
                if content_block.type == "text":
                    result_content += content_block.text
            
            # Process any tool use (file operations, etc.)
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_results = await self._process_tool_calls(response.tool_calls)
                result_content += f"\n\nTool Results:\n{tool_results}"
            
            # Store task result in Redis (for orchestrator to pick up)
            result = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "completed",
                "result": result_content,
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
            await self.redis.lpush("task_results", json.dumps(result))
            
            logger.info(
                "Task completed successfully",
                agent_id=self.agent_id,
                task_id=task_id,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            # Store error result
            error_result = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "status": "failed",
                "error": str(e),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
            await self.redis.lpush("task_results", json.dumps(error_result))
            
            logger.error(
                "Task failed",
                agent_id=self.agent_id,
                task_id=task_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
        
        finally:
            self.current_task = None
            self.health_status = "ready"
    
    def _get_system_prompt(self) -> str:
        """Get agent-specific system prompt."""
        base_prompt = """You are a specialized AI agent in the LeanVibe Agent Hive autonomous development system. 
        You work collaboratively with other agents to develop, test, and deploy software."""
        
        agent_prompts = {
            "architect": base_prompt + """
            
            You are the Architecture Agent responsible for:
            - System design and architecture decisions
            - Technical planning and roadmaps
            - Code organization and structure
            - Integration patterns and best practices
            
            Always provide detailed technical specifications and consider scalability, maintainability, and performance.
            """,
            
            "developer": base_prompt + """
            
            You are the Development Agent responsible for:
            - Feature implementation
            - Bug fixes and debugging
            - Code quality and best practices
            - Unit testing and documentation
            
            Write clean, maintainable code following established patterns and conventions.
            """,
            
            "qa": base_prompt + """
            
            You are the QA Agent responsible for:
            - Test creation and execution
            - Quality assurance and validation
            - Bug detection and reporting
            - Performance and security testing
            
            Ensure comprehensive test coverage and maintain high quality standards.
            """,
            
            "meta": base_prompt + """
            
            You are the Meta Agent responsible for:
            - System improvement and optimization
            - Agent coordination and communication
            - Process enhancement
            - Self-modification and evolution
            
            Focus on system-wide improvements and agent collaboration optimization.
            """
        }
        
        return agent_prompts.get(self.agent_type, base_prompt)
    
    async def _process_tool_calls(self, tool_calls) -> str:
        """Process any tool calls from Claude response."""
        # Implementation for tool processing (file operations, etc.)
        # This replaces the file system access that Claude Code CLI provided
        results = []
        
        for tool_call in tool_calls:
            # Handle different tool types
            if tool_call.name == "write_file":
                result = await self._write_file(tool_call.arguments)
                results.append(result)
            elif tool_call.name == "read_file":
                result = await self._read_file(tool_call.arguments)
                results.append(result)
            # Add more tool handlers as needed
        
        return "\n".join(results)
    
    async def _write_file(self, args: Dict[str, Any]) -> str:
        """Handle file write operations."""
        try:
            file_path = args.get("file_path", "")
            content = args.get("content", "")
            
            # Ensure file path is within workspace for security
            if not file_path.startswith(self.workspace):
                file_path = os.path.join(self.workspace, file_path.lstrip("/"))
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            return f"Successfully wrote file: {file_path}"
            
        except Exception as e:
            return f"Failed to write file: {str(e)}"
    
    async def _read_file(self, args: Dict[str, Any]) -> str:
        """Handle file read operations."""
        try:
            file_path = args.get("file_path", "")
            
            # Ensure file path is within workspace for security
            if not file_path.startswith(self.workspace):
                file_path = os.path.join(self.workspace, file_path.lstrip("/"))
            
            if not os.path.exists(file_path):
                return f"File not found: {file_path}"
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            return f"File content:\n{content}"
            
        except Exception as e:
            return f"Failed to read file: {str(e)}"
    
    async def _health_check_server(self):
        """Simple HTTP health check server."""
        from aiohttp import web
        
        async def health(request):
            return web.json_response({
                "status": self.health_status,
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "current_task": self.current_task,
                "timestamp": datetime.now().isoformat()
            })
        
        app = web.Application()
        app.router.add_get('/health', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        
        try:
            await site.start()
            logger.info("Health check server started", agent_id=self.agent_id, port=8080)
            
            # Keep running until cancelled
            while self.running:
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Health check server stopping", agent_id=self.agent_id)
        finally:
            await runner.cleanup()
    
    async def _cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up agent resources", agent_id=self.agent_id)
        
        if self.redis:
            await self.redis.aclose()
        
        self.health_status = "shutdown"


async def main():
    """Main entry point for containerized agent."""
    agent_type = os.getenv("AGENT_TYPE", "developer")
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger.info("Starting containerized agent", agent_type=agent_type)
    
    agent = ContainerizedAgent(agent_type)
    
    try:
        await agent.initialize()
        await agent.run()
    except Exception as e:
        logger.error("Agent startup failed", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)
    
    logger.info("Agent shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
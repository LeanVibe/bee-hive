"""
Enhanced Agent Launcher for LeanVibe Agent Hive 2.0 with Tmux Integration

Spawns different types of CLI coding agents (Claude Code, Cursor Agent, Open Code)
in isolated tmux sessions with proper Redis stream integration and environment setup.

Features:
- Multi-agent type support (Claude Code, Cursor, Open Code, custom agents)
- Tmux session isolation with unique workspace directories
- Redis stream communication for orchestrator coordination
- Environment setup and configuration management
- Working directory and git repository management
- Health monitoring and automatic recovery
- Session persistence across system restarts
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import structlog
import libtmux
from redis.asyncio import Redis

from .config import settings
from .short_id_generator import ShortIDGenerator
from .tmux_session_manager import TmuxSessionManager, SessionInfo, SessionStatus
from .enhanced_redis_streams_manager import EnhancedRedisStreamsManager, ConsumerGroupType
from ..models.agent import Agent, AgentStatus, AgentType

logger = structlog.get_logger()


class AgentLauncherType(Enum):
    """Supported CLI agent types for launching."""
    CLAUDE_CODE = "claude-code"
    CURSOR_AGENT = "cursor-agent"  
    OPEN_CODE = "open-code"
    AIDER = "aider"
    CONTINUE = "continue"
    CUSTOM = "custom"


@dataclass
class AgentLaunchConfig:
    """Configuration for launching an agent."""
    agent_type: AgentLauncherType
    task_id: Optional[str] = None
    workspace_name: Optional[str] = None
    git_branch: Optional[str] = None
    working_directory: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None
    agent_config: Optional[Dict[str, Any]] = None
    redis_stream: str = "agent_tasks"
    consumer_group: ConsumerGroupType = ConsumerGroupType.GENERAL_AGENTS
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["agent_type"] = self.agent_type.value
        result["consumer_group"] = self.consumer_group.value
        return result


@dataclass
class AgentLaunchResult:
    """Result of agent launch operation."""
    success: bool
    agent_id: str
    session_id: str
    session_name: str
    workspace_path: str
    error_message: Optional[str] = None
    launch_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnhancedAgentLauncher:
    """
    Enhanced agent launcher that spawns CLI coding agents in tmux sessions
    with Redis stream integration for orchestrator communication.
    """
    
    def __init__(
        self,
        tmux_manager: TmuxSessionManager,
        redis_manager: EnhancedRedisStreamsManager,
        short_id_generator: ShortIDGenerator
    ):
        self.tmux_manager = tmux_manager
        self.redis_manager = redis_manager
        self.short_id_generator = short_id_generator
        
        # Agent configuration templates
        self.agent_configs = {
            AgentLauncherType.CLAUDE_CODE: {
                "command": "claude",
                "default_args": ["--session", "{session_name}"],
                "env_vars": {
                    "CLAUDE_CODE_MODE": "agent",
                    "CLAUDE_CODE_REDIS_STREAM": "{redis_stream}",
                    "CLAUDE_CODE_CONSUMER_GROUP": "{consumer_group}"
                },
                "health_check_command": "claude --version",
                "setup_commands": [
                    "echo 'Initializing Claude Code agent...'",
                    "claude --help > /dev/null 2>&1 || echo 'Warning: Claude Code not available'"
                ]
            },
            AgentLauncherType.CURSOR_AGENT: {
                "command": "cursor",
                "default_args": ["--agent-mode"],
                "env_vars": {
                    "CURSOR_AGENT_MODE": "true",
                    "CURSOR_REDIS_STREAM": "{redis_stream}",
                    "CURSOR_CONSUMER_GROUP": "{consumer_group}"
                },
                "health_check_command": "cursor --version",
                "setup_commands": [
                    "echo 'Initializing Cursor Agent...'",
                    "cursor --help > /dev/null 2>&1 || echo 'Warning: Cursor not available'"
                ]
            },
            AgentLauncherType.OPEN_CODE: {
                "command": "opencode",
                "default_args": ["--agent"],
                "env_vars": {
                    "OPENCODE_AGENT_MODE": "true", 
                    "OPENCODE_REDIS_STREAM": "{redis_stream}",
                    "OPENCODE_CONSUMER_GROUP": "{consumer_group}"
                },
                "health_check_command": "opencode --version",
                "setup_commands": [
                    "echo 'Initializing Open Code agent...'",
                    "opencode --help > /dev/null 2>&1 || echo 'Warning: Open Code not available'"
                ]
            },
            AgentLauncherType.AIDER: {
                "command": "aider",
                "default_args": ["--auto-commits", "--stream"],
                "env_vars": {
                    "AIDER_AGENT_MODE": "true",
                    "AIDER_REDIS_STREAM": "{redis_stream}",
                    "AIDER_CONSUMER_GROUP": "{consumer_group}"
                },
                "health_check_command": "aider --version",
                "setup_commands": [
                    "echo 'Initializing Aider agent...'",
                    "aider --help > /dev/null 2>&1 || echo 'Warning: Aider not available'"
                ]
            },
            AgentLauncherType.CONTINUE: {
                "command": "continue",
                "default_args": ["--agent-mode"],
                "env_vars": {
                    "CONTINUE_AGENT_MODE": "true",
                    "CONTINUE_REDIS_STREAM": "{redis_stream}",
                    "CONTINUE_CONSUMER_GROUP": "{consumer_group}"
                },
                "health_check_command": "continue --version",
                "setup_commands": [
                    "echo 'Initializing Continue agent...'",
                    "continue --help > /dev/null 2>&1 || echo 'Warning: Continue not available'"
                ]
            }
        }
        
        # Performance metrics
        self.metrics = {
            "agents_launched": 0,
            "launch_failures": 0,
            "average_launch_time": 0.0,
            "active_sessions": 0
        }
    
    async def launch_agent(
        self,
        config: AgentLaunchConfig,
        agent_name: Optional[str] = None
    ) -> AgentLaunchResult:
        """
        Launch a new agent in a tmux session with Redis stream integration.
        
        Args:
            config: Agent launch configuration
            agent_name: Optional human-readable agent name
            
        Returns:
            AgentLaunchResult with launch details and status
        """
        start_time = asyncio.get_event_loop().time()
        
        # Generate unique identifiers
        agent_id = str(uuid.uuid4())
        short_agent_id = self.short_id_generator.generate_short_id("AGT")
        
        if agent_name is None:
            agent_name = f"{config.agent_type.value}-{short_agent_id}"
        
        logger.info(
            "🚀 Launching agent in tmux session",
            agent_id=agent_id,
            short_id=short_agent_id,
            agent_type=config.agent_type.value,
            agent_name=agent_name,
            task_id=config.task_id
        )
        
        try:
            # Validate agent type availability
            if not await self._validate_agent_availability(config.agent_type):
                return AgentLaunchResult(
                    success=False,
                    agent_id=agent_id,
                    session_id="",
                    session_name="",
                    workspace_path="",
                    error_message=f"Agent type {config.agent_type.value} is not available or not installed"
                )
            
            # Create tmux session for the agent
            session_info = await self._create_agent_session(
                agent_id=agent_id,
                short_id=short_agent_id,
                agent_name=agent_name,
                config=config
            )
            
            # Set up Redis stream integration
            await self._setup_redis_integration(session_info, config)
            
            # Launch the actual agent process
            await self._launch_agent_process(session_info, config)
            
            # Wait for agent to be ready and perform health check
            await self._wait_for_agent_ready(session_info, config)
            
            # Record metrics
            launch_time = asyncio.get_event_loop().time() - start_time
            self.metrics["agents_launched"] += 1
            self.metrics["active_sessions"] += 1
            self._update_average_launch_time(launch_time)
            
            result = AgentLaunchResult(
                success=True,
                agent_id=agent_id,
                session_id=session_info.session_id,
                session_name=session_info.session_name,
                workspace_path=session_info.workspace_path,
                launch_time_seconds=launch_time
            )
            
            logger.info(
                "✅ Agent launched successfully",
                agent_id=agent_id,
                short_id=short_agent_id,
                session_name=session_info.session_name,
                launch_time=launch_time,
                workspace_path=session_info.workspace_path
            )
            
            return result
            
        except Exception as e:
            self.metrics["launch_failures"] += 1
            
            logger.error(
                "❌ Failed to launch agent",
                agent_id=agent_id,
                short_id=short_agent_id,
                agent_type=config.agent_type.value,
                error=str(e)
            )
            
            return AgentLaunchResult(
                success=False,
                agent_id=agent_id,
                session_id="",
                session_name="",
                workspace_path="",
                error_message=str(e)
            )
    
    async def terminate_agent(self, agent_id: str, cleanup_workspace: bool = True) -> bool:
        """
        Terminate an agent and clean up its tmux session.
        
        Args:
            agent_id: Agent identifier
            cleanup_workspace: Whether to clean up workspace directory
            
        Returns:
            True if termination was successful
        """
        logger.info("🛑 Terminating agent", agent_id=agent_id)
        
        try:
            # Find session by agent ID
            session_info = self.tmux_manager.get_agent_session(agent_id)
            if not session_info:
                logger.warning("Session not found for agent", agent_id=agent_id)
                return False
            
            # Send shutdown signal to agent process
            await self._send_shutdown_signal(session_info)
            
            # Wait for graceful shutdown
            await asyncio.sleep(2)
            
            # Terminate tmux session
            success = await self.tmux_manager.terminate_session(
                session_info.session_id, 
                cleanup_workspace=cleanup_workspace
            )
            
            if success:
                self.metrics["active_sessions"] = max(0, self.metrics["active_sessions"] - 1)
                
                logger.info(
                    "✅ Agent terminated successfully",
                    agent_id=agent_id,
                    session_name=session_info.session_name
                )
            
            return success
            
        except Exception as e:
            logger.error(
                "❌ Failed to terminate agent",
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        session_info = self.tmux_manager.get_agent_session(agent_id)
        if not session_info:
            return None
        
        # Check if agent process is still running
        is_running = await self._check_agent_process_running(session_info)
        
        # Get recent logs
        logs = await self._get_agent_logs(session_info, lines=20)
        
        return {
            "agent_id": agent_id,
            "session_info": session_info.to_dict(),
            "is_running": is_running,
            "recent_logs": logs,
            "metrics": self._get_agent_metrics(session_info)
        }
    
    async def list_active_agents(self) -> List[Dict[str, Any]]:
        """List all active agents with their status."""
        sessions = self.tmux_manager.list_sessions()
        active_agents = []
        
        for session in sessions:
            if session.status != SessionStatus.TERMINATED:
                agent_status = await self.get_agent_status(session.agent_id)
                if agent_status:
                    active_agents.append(agent_status)
        
        return active_agents
    
    async def get_launcher_metrics(self) -> Dict[str, Any]:
        """Get comprehensive launcher metrics."""
        sessions = self.tmux_manager.list_sessions()
        session_metrics = await self.tmux_manager.get_session_metrics()
        
        return {
            "launcher_metrics": self.metrics,
            "session_metrics": session_metrics,
            "agent_type_distribution": self._get_agent_type_distribution(sessions),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Private helper methods
    
    async def _validate_agent_availability(self, agent_type: AgentLauncherType) -> bool:
        """Check if the specified agent type is available."""
        if agent_type not in self.agent_configs:
            return False
        
        config = self.agent_configs[agent_type]
        health_check = config.get("health_check_command")
        
        if not health_check:
            return True  # Assume available if no health check defined
        
        try:
            process = await asyncio.create_subprocess_shell(
                health_check,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        except Exception:
            return False
    
    async def _create_agent_session(
        self,
        agent_id: str,
        short_id: str,
        agent_name: str,
        config: AgentLaunchConfig
    ) -> SessionInfo:
        """Create tmux session for the agent."""
        workspace_name = config.workspace_name or f"agent-{short_id}"
        git_branch = config.git_branch or f"agent/{short_id}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Prepare environment variables
        env_vars = {
            "LEANVIBE_AGENT_ID": agent_id,
            "LEANVIBE_AGENT_SHORT_ID": short_id,
            "LEANVIBE_AGENT_TYPE": config.agent_type.value,
            "LEANVIBE_TASK_ID": config.task_id or "",
            "LEANVIBE_REDIS_STREAM": config.redis_stream,
            "LEANVIBE_CONSUMER_GROUP": config.consumer_group.value
        }
        
        # Add agent-specific environment variables
        agent_config = self.agent_configs.get(config.agent_type, {})
        agent_env_vars = agent_config.get("env_vars", {})
        
        for key, value in agent_env_vars.items():
            formatted_value = value.format(
                session_name=f"agent-{short_id}",
                redis_stream=config.redis_stream,
                consumer_group=config.consumer_group.value,
                agent_id=agent_id
            )
            env_vars[key] = formatted_value
        
        # Add user-provided environment overrides
        if config.environment_vars:
            env_vars.update(config.environment_vars)
        
        return await self.tmux_manager.create_agent_session(
            agent_id=agent_id,
            agent_name=agent_name,
            workspace_name=workspace_name,
            git_branch=git_branch,
            environment_overrides=env_vars
        )
    
    async def _setup_redis_integration(
        self,
        session_info: SessionInfo,
        config: AgentLaunchConfig
    ) -> None:
        """Set up Redis stream integration for the agent."""
        # Create Redis configuration file in the workspace
        redis_config = {
            "stream_name": config.redis_stream,
            "consumer_group": config.consumer_group.value,
            "consumer_id": f"{session_info.agent_id}",
            "redis_url": getattr(settings, 'REDIS_URL', 'redis://localhost:6379'),
            "batch_size": 10,
            "timeout_ms": 5000
        }
        
        redis_config_path = Path(session_info.workspace_path) / ".leanvibe" / "redis_config.json"
        redis_config_path.parent.mkdir(exist_ok=True)
        
        with open(redis_config_path, 'w') as f:
            json.dump(redis_config, f, indent=2)
        
        logger.debug(
            "Redis configuration created",
            agent_id=session_info.agent_id,
            config_path=str(redis_config_path)
        )
    
    async def _launch_agent_process(
        self,
        session_info: SessionInfo,
        config: AgentLaunchConfig
    ) -> None:
        """Launch the actual agent process in the tmux session."""
        agent_config = self.agent_configs.get(config.agent_type)
        if not agent_config:
            raise ValueError(f"No configuration found for agent type: {config.agent_type.value}")
        
        # Run setup commands first
        setup_commands = agent_config.get("setup_commands", [])
        for setup_cmd in setup_commands:
            await self.tmux_manager.execute_command(
                session_info.session_id,
                setup_cmd,
                capture_output=False
            )
            await asyncio.sleep(0.5)  # Brief pause between setup commands
        
        # Prepare the main agent launch command
        command = agent_config["command"]
        args = agent_config.get("default_args", [])
        
        # Format args with session-specific values
        formatted_args = []
        for arg in args:
            if isinstance(arg, str) and "{" in arg:
                formatted_arg = arg.format(
                    session_name=session_info.session_name,
                    agent_id=session_info.agent_id,
                    workspace_path=session_info.workspace_path
                )
                formatted_args.append(formatted_arg)
            else:
                formatted_args.append(arg)
        
        # Add working directory if specified
        if config.working_directory:
            change_dir_cmd = f"cd {config.working_directory}"
            await self.tmux_manager.execute_command(
                session_info.session_id,
                change_dir_cmd,
                capture_output=False
            )
        
        # Launch the agent
        launch_command = f"{command} {' '.join(formatted_args)}"
        
        logger.debug(
            "Launching agent process",
            agent_id=session_info.agent_id,
            command=launch_command
        )
        
        await self.tmux_manager.execute_command(
            session_info.session_id,
            launch_command,
            capture_output=False
        )
    
    async def _wait_for_agent_ready(
        self,
        session_info: SessionInfo,
        config: AgentLaunchConfig,
        timeout_seconds: int = 30
    ) -> None:
        """Wait for agent to be ready and perform health check."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            # Check if agent process is running
            if await self._check_agent_process_running(session_info):
                logger.debug(
                    "Agent process is running",
                    agent_id=session_info.agent_id
                )
                return
            
            await asyncio.sleep(1)
        
        # If we get here, agent didn't start within timeout
        logs = await self._get_agent_logs(session_info, lines=50)
        logger.warning(
            "Agent did not start within timeout",
            agent_id=session_info.agent_id,
            timeout_seconds=timeout_seconds,
            recent_logs=logs
        )
    
    async def _check_agent_process_running(self, session_info: SessionInfo) -> bool:
        """Check if the agent process is still running in the session."""
        try:
            # Get process list for the session
            result = await self.tmux_manager.execute_command(
                session_info.session_id,
                "ps aux | grep -v grep | grep -E '(claude|cursor|opencode|aider|continue)'",
                capture_output=True
            )
            
            return result.get("success", False) and bool(result.get("output", "").strip())
            
        except Exception:
            return False
    
    async def _get_agent_logs(self, session_info: SessionInfo, lines: int = 100) -> List[str]:
        """Get recent logs from the agent session."""
        try:
            # Capture session output
            result = await self.tmux_manager.execute_command(
                session_info.session_id,
                f"tail -n {lines} ~/.leanvibe/agent.log 2>/dev/null || echo 'No log file found'",
                capture_output=True
            )
            
            if result.get("success") and result.get("output"):
                return result["output"].split('\n')
            else:
                # Fallback: capture tmux pane content
                result = await self.tmux_manager.execute_command(
                    session_info.session_id,
                    f"tmux capture-pane -t {session_info.session_name} -p",
                    capture_output=True
                )
                if result.get("success") and result.get("output"):
                    return result["output"].split('\n')[-lines:]
            
        except Exception as e:
            logger.debug(f"Failed to get agent logs: {e}")
        
        return []
    
    async def _send_shutdown_signal(self, session_info: SessionInfo) -> None:
        """Send shutdown signal to the agent process."""
        try:
            # Try to send graceful shutdown signal
            await self.tmux_manager.execute_command(
                session_info.session_id,
                "pkill -TERM -f 'claude|cursor|opencode|aider|continue'",
                capture_output=False
            )
            
            # Give it a moment to shut down gracefully
            await asyncio.sleep(2)
            
            # Force kill if still running
            await self.tmux_manager.execute_command(
                session_info.session_id,
                "pkill -KILL -f 'claude|cursor|opencode|aider|continue'",
                capture_output=False
            )
            
        except Exception as e:
            logger.debug(f"Error sending shutdown signal: {e}")
    
    def _get_agent_metrics(self, session_info: SessionInfo) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        return {
            "uptime_seconds": (datetime.utcnow() - session_info.created_at).total_seconds(),
            "last_activity": session_info.last_activity.isoformat(),
            "status": session_info.status.value,
            "performance_metrics": session_info.performance_metrics
        }
    
    def _get_agent_type_distribution(self, sessions: List[SessionInfo]) -> Dict[str, int]:
        """Get distribution of agent types across sessions."""
        distribution = {}
        
        for session in sessions:
            # Extract agent type from environment variables
            agent_type = session.environment_vars.get("LEANVIBE_AGENT_TYPE", "unknown")
            distribution[agent_type] = distribution.get(agent_type, 0) + 1
        
        return distribution
    
    def _update_average_launch_time(self, launch_time: float) -> None:
        """Update average launch time metric."""
        current_avg = self.metrics["average_launch_time"]
        launch_count = self.metrics["agents_launched"]
        
        if launch_count > 1:
            new_avg = ((current_avg * (launch_count - 1)) + launch_time) / launch_count
            self.metrics["average_launch_time"] = new_avg
        else:
            self.metrics["average_launch_time"] = launch_time


# Factory function for dependency injection
async def create_enhanced_agent_launcher(
    tmux_manager: Optional[TmuxSessionManager] = None,
    redis_manager: Optional[EnhancedRedisStreamsManager] = None,
    short_id_generator: Optional[ShortIDGenerator] = None
) -> EnhancedAgentLauncher:
    """
    Factory function to create EnhancedAgentLauncher with proper dependencies.
    """
    if tmux_manager is None:
        tmux_manager = TmuxSessionManager()
        await tmux_manager.initialize()
    
    if redis_manager is None:
        redis_manager = EnhancedRedisStreamsManager()
        await redis_manager.initialize()
    
    if short_id_generator is None:
        short_id_generator = ShortIDGenerator()
    
    return EnhancedAgentLauncher(
        tmux_manager=tmux_manager,
        redis_manager=redis_manager,
        short_id_generator=short_id_generator
    )
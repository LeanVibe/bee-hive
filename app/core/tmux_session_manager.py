"""
Tmux Session Manager for LeanVibe Agent Hive 2.0

Manages isolated tmux sessions for individual agents, providing:
- Session creation and isolation
- Git repository management per session
- Environment setup and configuration
- Session monitoring and cleanup
- Cross-session communication
- Performance monitoring

Performance targets:
- Session creation: <5 seconds
- Git checkout: <3 seconds
- Environment setup: <2 seconds
- Session isolation validation
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog
import libtmux

from .config import settings
from .database import get_session
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class SessionStatus(Enum):
    """Tmux session status states."""
    CREATING = "creating"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    SLEEPING = "sleeping"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class SessionInfo:
    """Information about a tmux session."""
    session_id: str
    agent_id: str
    session_name: str
    status: SessionStatus
    workspace_path: str
    git_branch: str
    created_at: datetime
    last_activity: datetime
    environment_vars: Dict[str, str]
    performance_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "session_name": self.session_name,
            "status": self.status.value,
            "workspace_path": self.workspace_path,
            "git_branch": self.git_branch,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "environment_vars": self.environment_vars,
            "performance_metrics": self.performance_metrics
        }


class TmuxSessionManager:
    """
    Manages tmux sessions for agent isolation and workspace management.
    
    Each agent gets its own tmux session with:
    - Isolated workspace directory
    - Dedicated git branch
    - Custom environment variables
    - Performance monitoring
    - Automated cleanup
    """
    
    def __init__(self):
        self.tmux_server = None
        self.sessions: Dict[str, SessionInfo] = {}
        self.base_workspace_dir = Path(settings.WORKSPACES_DIR if hasattr(settings, 'WORKSPACES_DIR') else './workspaces')
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.metrics = {
            'sessions_created': 0,
            'sessions_terminated': 0,
            'average_creation_time': 0.0,
            'git_operations_count': 0,
            'workspace_cleanups': 0
        }
    
    async def initialize(self) -> None:
        """Initialize the tmux session manager."""
        logger.info("🖥️ Initializing Tmux Session Manager...")
        
        try:
            # Initialize libtmux server
            self.tmux_server = libtmux.Server()
            
            # Ensure base workspace directory exists
            self.base_workspace_dir.mkdir(parents=True, exist_ok=True)
            
            # Start monitoring tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Clean up any stale sessions from previous runs
            await self._cleanup_stale_sessions()
            
            logger.info("✅ Tmux Session Manager initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize Tmux Session Manager", error=str(e))
            raise
    
    async def create_agent_session(
        self,
        agent_id: str,
        agent_name: str,
        workspace_name: Optional[str] = None,
        git_branch: Optional[str] = None,
        environment_overrides: Optional[Dict[str, str]] = None
    ) -> SessionInfo:
        """
        Create a new tmux session for an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            workspace_name: Optional workspace directory name
            git_branch: Git branch to checkout (defaults to agent-specific branch)
            environment_overrides: Additional environment variables
            
        Returns:
            SessionInfo with session details
        """
        start_time = asyncio.get_event_loop().time()
        
        # Generate session identifiers
        session_id = str(uuid.uuid4())
        session_name = f"agent-{agent_id[:8]}"
        
        if workspace_name is None:
            workspace_name = f"workspace-{agent_id[:8]}"
        
        if git_branch is None:
            git_branch = f"agent/{agent_id[:8]}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(
            "🖥️ Creating tmux session for agent",
            agent_id=agent_id,
            session_name=session_name,
            workspace_name=workspace_name,
            git_branch=git_branch
        )
        
        try:
            # Create workspace directory
            workspace_path = self.base_workspace_dir / workspace_name
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Set up git repository
            await self._setup_git_workspace(workspace_path, git_branch)
            
            # Prepare environment variables
            env_vars = self._prepare_environment_variables(
                agent_id, workspace_path, environment_overrides
            )
            
            # Create tmux session
            tmux_session = await self._create_tmux_session(
                session_name, workspace_path, env_vars
            )
            
            # Initialize session info
            session_info = SessionInfo(
                session_id=session_id,
                agent_id=agent_id,
                session_name=session_name,
                status=SessionStatus.ACTIVE,
                workspace_path=str(workspace_path),
                git_branch=git_branch,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                environment_vars=env_vars,
                performance_metrics={}
            )
            
            # Store session info
            self.sessions[session_id] = session_info
            
            # Update agent in database with session info
            await self._update_agent_session_info(agent_id, session_id, session_name)
            
            # Record performance metrics
            creation_time = asyncio.get_event_loop().time() - start_time
            session_info.performance_metrics['creation_time'] = creation_time
            self.metrics['sessions_created'] += 1
            self._update_average_creation_time(creation_time)
            
            logger.info(
                "✅ Tmux session created successfully",
                agent_id=agent_id,
                session_id=session_id,
                session_name=session_name,
                creation_time=creation_time,
                workspace_path=str(workspace_path)
            )
            
            return session_info
            
        except Exception as e:
            logger.error(
                "❌ Failed to create tmux session",
                agent_id=agent_id,
                session_name=session_name,
                error=str(e)
            )
            raise
    
    async def terminate_session(self, session_id: str, cleanup_workspace: bool = True) -> bool:
        """
        Terminate a tmux session and clean up resources.
        
        Args:
            session_id: Session identifier
            cleanup_workspace: Whether to clean up workspace directory
            
        Returns:
            True if session was terminated successfully
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return False
        
        session_info = self.sessions[session_id]
        
        logger.info(
            "🛑 Terminating tmux session",
            session_id=session_id,
            session_name=session_info.session_name,
            agent_id=session_info.agent_id
        )
        
        try:
            # Update session status
            session_info.status = SessionStatus.TERMINATED
            
            # Kill tmux session
            await self._kill_tmux_session(session_info.session_name)
            
            # Clean up workspace if requested
            if cleanup_workspace:
                await self._cleanup_workspace(Path(session_info.workspace_path))
                self.metrics['workspace_cleanups'] += 1
            
            # Update agent in database
            await self._clear_agent_session_info(session_info.agent_id)
            
            # Remove from active sessions
            del self.sessions[session_id]
            
            self.metrics['sessions_terminated'] += 1
            
            logger.info(
                "✅ Tmux session terminated successfully",
                session_id=session_id,
                session_name=session_info.session_name
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "❌ Failed to terminate tmux session",
                session_id=session_id,
                error=str(e)
            )
            return False
    
    async def execute_command(
        self,
        session_id: str,
        command: str,
        window_name: Optional[str] = None,
        capture_output: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a command in a tmux session.
        
        Args:
            session_id: Session identifier
            command: Command to execute
            window_name: Optional window name (creates new window if not exists)
            capture_output: Whether to capture command output
            
        Returns:
            Dictionary with execution results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_info = self.sessions[session_id]
        
        logger.debug(
            "💻 Executing command in tmux session",
            session_id=session_id,
            command=command[:100],  # Truncate long commands
            window_name=window_name
        )
        
        try:
            # Update last activity
            session_info.last_activity = datetime.utcnow()
            session_info.status = SessionStatus.BUSY
            
            # Get tmux session
            tmux_session = self.tmux_server.find_where({"session_name": session_info.session_name})
            if not tmux_session:
                raise RuntimeError(f"Tmux session {session_info.session_name} not found")
            
            # Get or create window
            if window_name:
                window = tmux_session.find_where({"window_name": window_name})
                if not window:
                    window = tmux_session.new_window(window_name=window_name)
            else:
                window = tmux_session.attached_window
            
            # Get the pane
            pane = window.attached_pane
            
            # Execute command
            pane.send_keys(command)
            
            # Capture output if requested
            output = None
            if capture_output:
                await asyncio.sleep(1)  # Wait for command to complete
                output = pane.capture_pane()
            
            # Update session status
            session_info.status = SessionStatus.ACTIVE
            
            result = {
                "success": True,
                "command": command,
                "window_name": window_name or window.window_name,
                "output": output,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.debug(
                "✅ Command executed successfully",
                session_id=session_id,
                command=command[:50]
            )
            
            return result
            
        except Exception as e:
            session_info.status = SessionStatus.ERROR
            
            logger.error(
                "❌ Failed to execute command",
                session_id=session_id,
                command=command[:50],
                error=str(e)
            )
            
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def create_git_checkpoint(
        self,
        session_id: str,
        checkpoint_message: str,
        files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a git checkpoint in the session workspace.
        
        Args:
            session_id: Session identifier
            checkpoint_message: Commit message
            files: Optional list of specific files to commit
            
        Returns:
            Dictionary with checkpoint results
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_info = self.sessions[session_id]
        workspace_path = Path(session_info.workspace_path)
        
        logger.info(
            "📝 Creating git checkpoint",
            session_id=session_id,
            checkpoint_message=checkpoint_message
        )
        
        try:
            # Add files to git
            if files:
                for file in files:
                    await self._run_git_command(workspace_path, ["add", file])
            else:
                await self._run_git_command(workspace_path, ["add", "."])
            
            # Create commit
            commit_result = await self._run_git_command(
                workspace_path, 
                ["commit", "-m", checkpoint_message]
            )
            
            # Get commit hash
            hash_result = await self._run_git_command(
                workspace_path, 
                ["rev-parse", "HEAD"]
            )
            
            commit_hash = hash_result.get("output", "").strip()
            
            self.metrics['git_operations_count'] += 1
            
            result = {
                "success": True,
                "commit_hash": commit_hash,
                "message": checkpoint_message,
                "branch": session_info.git_branch,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace_path": str(workspace_path)
            }
            
            logger.info(
                "✅ Git checkpoint created successfully",
                session_id=session_id,
                commit_hash=commit_hash[:8]
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "❌ Failed to create git checkpoint",
                session_id=session_id,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information by ID."""
        return self.sessions.get(session_id)
    
    def get_agent_session(self, agent_id: str) -> Optional[SessionInfo]:
        """Get session information by agent ID."""
        for session in self.sessions.values():
            if session.agent_id == agent_id:
                return session
        return None
    
    def list_sessions(self) -> List[SessionInfo]:
        """List all active sessions."""
        return list(self.sessions.values())
    
    async def get_session_metrics(self) -> Dict[str, Any]:
        """Get comprehensive session metrics."""
        active_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.ACTIVE])
        busy_sessions = len([s for s in self.sessions.values() if s.status == SessionStatus.BUSY])
        
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "busy_sessions": busy_sessions,
            "metrics": self.metrics,
            "session_details": [session.to_dict() for session in self.sessions.values()]
        }
    
    async def _setup_git_workspace(self, workspace_path: Path, git_branch: str) -> None:
        """Set up git repository in workspace."""
        logger.debug(f"Setting up git workspace at {workspace_path}")
        
        try:
            # Initialize git repository if needed
            if not (workspace_path / ".git").exists():
                await self._run_git_command(workspace_path, ["init"])
                
                # Set up initial configuration
                await self._run_git_command(workspace_path, ["config", "user.email", "agent@leanvibe.dev"])
                await self._run_git_command(workspace_path, ["config", "user.name", "LeanVibe Agent"])
                
                # Create initial commit
                (workspace_path / "README.md").write_text("# Agent Workspace\n\nInitialized by LeanVibe Agent Hive")
                await self._run_git_command(workspace_path, ["add", "README.md"])
                await self._run_git_command(workspace_path, ["commit", "-m", "Initial workspace setup"])
            
            # Create and checkout agent branch
            await self._run_git_command(workspace_path, ["checkout", "-b", git_branch], allow_failure=True)
            
        except Exception as e:
            logger.error(f"Failed to setup git workspace: {e}")
            raise
    
    def _prepare_environment_variables(
        self,
        agent_id: str,
        workspace_path: Path,
        overrides: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """Prepare environment variables for the session."""
        env_vars = {
            "AGENT_ID": agent_id,
            "WORKSPACE_PATH": str(workspace_path),
            "PYTHONPATH": f"{workspace_path}:{os.environ.get('PYTHONPATH', '')}",
            "PATH": os.environ.get("PATH", ""),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "LEANVIBE_AGENT_MODE": "true",
        }
        
        if overrides:
            env_vars.update(overrides)
        
        return env_vars
    
    async def _create_tmux_session(
        self,
        session_name: str,
        workspace_path: Path,
        env_vars: Dict[str, str]
    ) -> Any:
        """Create a new tmux session."""
        try:
            # Kill existing session with same name if it exists
            existing_session = self.tmux_server.find_where({"session_name": session_name})
            if existing_session:
                existing_session.kill_session()
            
            # Create new session
            session = self.tmux_server.new_session(
                session_name=session_name,
                start_directory=str(workspace_path),
                attach=False
            )
            
            # Set environment variables
            for key, value in env_vars.items():
                session.set_environment(key, value)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to create tmux session {session_name}: {e}")
            raise
    
    async def _kill_tmux_session(self, session_name: str) -> None:
        """Kill a tmux session."""
        try:
            session = self.tmux_server.find_where({"session_name": session_name})
            if session:
                session.kill_session()
        except Exception as e:
            logger.error(f"Failed to kill tmux session {session_name}: {e}")
    
    async def _cleanup_workspace(self, workspace_path: Path) -> None:
        """Clean up workspace directory."""
        try:
            if workspace_path.exists():
                import shutil
                shutil.rmtree(workspace_path)
                logger.debug(f"Cleaned up workspace: {workspace_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup workspace {workspace_path}: {e}")
    
    async def _run_git_command(
        self,
        workspace_path: Path,
        command: List[str],
        allow_failure: bool = False
    ) -> Dict[str, Any]:
        """Run a git command in the workspace."""
        full_command = ["git"] + command
        
        try:
            process = await asyncio.create_subprocess_exec(
                *full_command,
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            result = {
                "command": " ".join(full_command),
                "returncode": process.returncode,
                "output": stdout.decode().strip(),
                "error": stderr.decode().strip()
            }
            
            if process.returncode != 0 and not allow_failure:
                raise RuntimeError(f"Git command failed: {result['error']}")
            
            return result
            
        except Exception as e:
            if not allow_failure:
                raise
            return {"command": " ".join(full_command), "error": str(e)}
    
    async def _update_agent_session_info(
        self,
        agent_id: str,
        session_id: str,
        session_name: str
    ) -> None:
        """Update agent with session information in database."""
        try:
            async with get_session() as db_session:
                from sqlalchemy import update
                await db_session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(tmux_session=session_name)
                )
                await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update agent session info: {e}")
    
    async def _clear_agent_session_info(self, agent_id: str) -> None:
        """Clear agent session information in database."""
        try:
            async with get_session() as db_session:
                from sqlalchemy import update
                await db_session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(tmux_session=None)
                )
                await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to clear agent session info: {e}")
    
    def _update_average_creation_time(self, creation_time: float) -> None:
        """Update average session creation time."""
        current_avg = self.metrics['average_creation_time']
        session_count = self.metrics['sessions_created']
        
        if session_count > 1:
            new_avg = ((current_avg * (session_count - 1)) + creation_time) / session_count
            self.metrics['average_creation_time'] = new_avg
        else:
            self.metrics['average_creation_time'] = creation_time
    
    async def _monitoring_loop(self) -> None:
        """Background task to monitor session health."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for session_id, session_info in list(self.sessions.items()):
                    # Check if session is still alive
                    tmux_session = self.tmux_server.find_where({"session_name": session_info.session_name})
                    
                    if not tmux_session:
                        logger.warning(f"Tmux session {session_info.session_name} not found, marking as terminated")
                        session_info.status = SessionStatus.TERMINATED
                        continue
                    
                    # Check for idle sessions
                    idle_time = (current_time - session_info.last_activity).total_seconds()
                    if idle_time > 300 and session_info.status == SessionStatus.ACTIVE:  # 5 minutes
                        session_info.status = SessionStatus.IDLE
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in session monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_stale_sessions()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_stale_sessions(self) -> None:
        """Clean up stale sessions that are no longer needed."""
        logger.debug("Running stale session cleanup")
        
        stale_sessions = []
        current_time = datetime.utcnow()
        
        for session_id, session_info in self.sessions.items():
            # Mark sessions as stale if terminated or idle for too long
            if (session_info.status == SessionStatus.TERMINATED or
                (session_info.status == SessionStatus.IDLE and 
                 (current_time - session_info.last_activity).total_seconds() > 3600)):  # 1 hour
                stale_sessions.append(session_id)
        
        for session_id in stale_sessions:
            await self.terminate_session(session_id, cleanup_workspace=True)
        
        if stale_sessions:
            logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
    
    async def shutdown(self) -> None:
        """Shutdown the tmux session manager."""
        logger.info("🛑 Shutting down Tmux Session Manager...")
        
        # Cancel monitoring tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Terminate all active sessions
        active_sessions = list(self.sessions.keys())
        for session_id in active_sessions:
            await self.terminate_session(session_id, cleanup_workspace=False)
        
        logger.info("✅ Tmux Session Manager shutdown complete")
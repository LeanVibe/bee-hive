"""
Real tmux Workspace Management for LeanVibe Agent Hive 2.0

This system provides agents with real development environments where they can:
- Execute code safely in isolated workspaces
- Manage development servers and processes
- Handle multiple projects simultaneously
- Monitor resource usage and performance

Each agent gets its own tmux session with proper isolation and resource limits.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import libtmux
import psutil
import structlog

from .configuration_service import get_config
from .database import get_session
from ..models.agent import Agent
from ..models.session import Session


class TmuxSessionManager:
    """
    Enhanced tmux session management for agent isolation and lifecycle management.
    
    Provides:
    - Isolated tmux sessions per agent with resource monitoring
    - Session recovery and restoration from checkpoints
    - Template-based session configuration
    - Resource allocation and limits per session
    - Health monitoring and automatic recovery
    """
    
    def __init__(self):
        self.tmux_server = libtmux.Server()
        self.session_registry: Dict[str, Dict[str, Any]] = {}
        self.session_templates: Dict[str, Dict[str, Any]] = {}
        self.resource_limits: Dict[str, Dict[str, Any]] = {}
        
        # Session health monitoring
        self.health_check_interval = 30  # seconds
        self.unhealthy_threshold = 3  # failed health checks before recovery
        self.session_timeouts = {}
        
        # Initialize default templates
        self._initialize_session_templates()
    
    def _initialize_session_templates(self):
        """Initialize default session templates for different agent types."""
        self.session_templates = {
            "development": {
                "windows": [
                    {"name": "main", "commands": ["cd {workspace_path}"]},
                    {"name": "code", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "tests", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "server", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "monitor", "commands": ["htop"], "split_panes": [
                        {"command": "tail -f logs/*.log 2>/dev/null || echo 'No logs yet'"},
                        {"command": "watch -n 5 'ps aux | grep {agent_id}'"}
                    ]}
                ],
                "environment": {
                    "AGENT_ID": "{agent_id}",
                    "WORKSPACE_PATH": "{workspace_path}",
                    "PYTHONPATH": "{workspace_path}:$PYTHONPATH"
                }
            },
            "ai_agent": {
                "windows": [
                    {"name": "main", "commands": ["cd {workspace_path}"]},
                    {"name": "context", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "tasks", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "memory", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "tools", "commands": ["cd {workspace_path}", "source venv/bin/activate"]},
                    {"name": "logs", "commands": ["tail -f logs/agent_{agent_id}.log || touch logs/agent_{agent_id}.log && tail -f logs/agent_{agent_id}.log"]}
                ],
                "environment": {
                    "AGENT_ID": "{agent_id}",
                    "WORKSPACE_PATH": "{workspace_path}",
                    "CLAUDE_API_ENABLED": "1",
                    "CONTEXT_CACHE_SIZE": "1000"
                }
            },
            "minimal": {
                "windows": [
                    {"name": "main", "commands": ["cd {workspace_path}"]},
                    {"name": "work", "commands": ["cd {workspace_path}"]}
                ],
                "environment": {
                    "AGENT_ID": "{agent_id}",
                    "WORKSPACE_PATH": "{workspace_path}"
                }
            }
        }
    
    async def create_agent_session(
        self,
        agent_id: str,
        workspace_path: Path,
        template: str = "ai_agent",
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Optional[libtmux.Session]:
        """Create a new tmux session for an agent with specified template."""
        try:
            session_name = f"agent-{agent_id}"
            
            # Kill existing session if it exists
            existing_session = self.tmux_server.find_where({"session_name": session_name})
            if existing_session:
                logger.warning(f"Terminating existing session for agent {agent_id}")
                existing_session.kill()
                await asyncio.sleep(1)  # Allow cleanup
            
            # Get template configuration
            template_config = self.session_templates.get(template, self.session_templates["minimal"])
            if custom_config:
                template_config = self._merge_config(template_config, custom_config)
            
            # Create session
            session = self.tmux_server.new_session(
                session_name=session_name,
                window_name="main",
                start_directory=str(workspace_path),
                detach=True
            )
            
            # Set up environment variables
            env_vars = template_config.get("environment", {})
            for key, value in env_vars.items():
                formatted_value = value.format(
                    agent_id=agent_id,
                    workspace_path=str(workspace_path)
                )
                session.set_environment(key, formatted_value)
            
            # Create additional windows based on template
            windows_config = template_config.get("windows", [])
            for i, window_config in enumerate(windows_config):
                if i == 0:
                    # Use the main window that was already created
                    window = session.windows[0]
                    window.rename_window(window_config["name"])
                else:
                    # Create new window
                    window = session.new_window(
                        window_name=window_config["name"],
                        start_directory=str(workspace_path)
                    )
                
                # Execute initial commands
                pane = window.panes[0]
                commands = window_config.get("commands", [])
                for command in commands:
                    formatted_command = command.format(
                        agent_id=agent_id,
                        workspace_path=str(workspace_path)
                    )
                    pane.send_keys(formatted_command)
                    await asyncio.sleep(0.2)  # Small delay between commands
                
                # Handle pane splits if configured
                split_panes = window_config.get("split_panes", [])
                for split_config in split_panes:
                    split_pane = window.split_window(
                        vertical=split_config.get("vertical", True)
                    )
                    split_command = split_config.get("command", "")
                    if split_command:
                        formatted_command = split_command.format(
                            agent_id=agent_id,
                            workspace_path=str(workspace_path)
                        )
                        split_pane.send_keys(formatted_command)
            
            # Register session in our tracking
            self.session_registry[agent_id] = {
                "session": session,
                "session_name": session_name,
                "created_at": datetime.utcnow(),
                "template": template,
                "workspace_path": str(workspace_path),
                "status": "active",
                "health_failures": 0,
                "last_health_check": datetime.utcnow()
            }
            
            # Set resource limits
            await self._apply_resource_limits(agent_id, session)
            
            # Select main window
            session.select_window("main")
            
            logger.info(f"Created tmux session for agent {agent_id} using template '{template}'")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create tmux session for agent {agent_id}: {e}")
            return None
    
    async def _apply_resource_limits(self, agent_id: str, session: libtmux.Session):
        """Apply resource limits to a tmux session."""
        try:
            # Get session PID for process monitoring
            session_pid = None
            try:
                # This is a simplified approach - in production you'd want more robust PID tracking
                result = subprocess.run(
                    ["tmux", "list-sessions", "-F", f"#{session.id}:#{session_pid}"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.startswith(session.id):
                            session_pid = int(line.split(':')[1])
                            break
            except Exception as e:
                logger.debug(f"Could not get session PID for {agent_id}: {e}")
            
            if session_pid:
                self.resource_limits[agent_id] = {
                    "session_pid": session_pid,
                    "max_memory_mb": 2048,
                    "max_cpu_percent": 50.0,
                    "max_processes": 20
                }
                logger.debug(f"Applied resource limits to session {agent_id} (PID: {session_pid})")
            
        except Exception as e:
            logger.warning(f"Failed to apply resource limits to session {agent_id}: {e}")
    
    def _merge_config(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom configuration with base template configuration."""
        merged = base_config.copy()
        
        for key, value in custom_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    async def restore_session_from_checkpoint(
        self,
        agent_id: str,
        workspace_path: Path,
        checkpoint_data: Dict[str, Any]
    ) -> Optional[libtmux.Session]:
        """Restore a tmux session from checkpoint data."""
        try:
            logger.info(f"Restoring tmux session for agent {agent_id} from checkpoint")
            
            # Extract session configuration from checkpoint
            session_config = checkpoint_data.get("tmux_session", {})
            template = session_config.get("template", "ai_agent")
            
            # Create session with template
            session = await self.create_agent_session(agent_id, workspace_path, template)
            
            if not session:
                return None
            
            # Restore window states
            windows_state = session_config.get("windows", {})
            for window_name, window_state in windows_state.items():
                window = session.find_where({"window_name": window_name})
                if window:
                    # Restore pane commands/states
                    panes_state = window_state.get("panes", [])
                    for i, pane_state in enumerate(panes_state):
                        if i < len(window.panes):
                            pane = window.panes[i]
                            # Restore pane history or working directory
                            working_dir = pane_state.get("working_directory")
                            if working_dir:
                                pane.send_keys(f"cd {working_dir}")
                            
                            # Restore environment variables
                            env_vars = pane_state.get("environment", {})
                            for key, value in env_vars.items():
                                pane.send_keys(f"export {key}='{value}'")
            
            # Update session registry
            if agent_id in self.session_registry:
                self.session_registry[agent_id]["status"] = "restored"
                self.session_registry[agent_id]["restored_at"] = datetime.utcnow()
            
            logger.info(f"Successfully restored tmux session for agent {agent_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to restore tmux session for agent {agent_id}: {e}")
            return None
    
    async def capture_session_state(self, agent_id: str) -> Dict[str, Any]:
        """Capture current state of a tmux session for checkpointing."""
        try:
            if agent_id not in self.session_registry:
                return {}
            
            session_info = self.session_registry[agent_id]
            session = session_info["session"]
            
            state = {
                "session_name": session_info["session_name"],
                "template": session_info["template"],
                "workspace_path": session_info["workspace_path"],
                "created_at": session_info["created_at"].isoformat(),
                "windows": {}
            }
            
            # Capture window and pane states
            for window in session.windows:
                window_state = {
                    "name": window.name,
                    "active": window.id == session.attached_window.id,
                    "panes": []
                }
                
                for pane in window.panes:
                    try:
                        pane_state = {
                            "active": pane.id == window.active_pane.id,
                            "working_directory": pane.current_path,
                            "environment": {},  # Could capture env vars if needed
                            "last_command": ""  # Could capture command history
                        }
                        window_state["panes"].append(pane_state)
                    except Exception as e:
                        logger.debug(f"Could not capture pane state: {e}")
                
                state["windows"][window.name] = window_state
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to capture session state for agent {agent_id}: {e}")
            return {}
    
    async def health_check_session(self, agent_id: str) -> bool:
        """Perform health check on a tmux session."""
        try:
            if agent_id not in self.session_registry:
                return False
            
            session_info = self.session_registry[agent_id]
            session = session_info["session"]
            
            # Check if session is still alive
            if not session.server.has_session(session.id):
                logger.warning(f"Session {agent_id} is no longer alive")
                session_info["status"] = "dead"
                return False
            
            # Check resource usage if limits are set
            if agent_id in self.resource_limits:
                limits = self.resource_limits[agent_id]
                session_pid = limits.get("session_pid")
                
                if session_pid:
                    try:
                        process = psutil.Process(session_pid)
                        
                        # Check memory usage
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        if memory_mb > limits["max_memory_mb"]:
                            logger.warning(f"Session {agent_id} exceeds memory limit: {memory_mb}MB")
                            return False
                        
                        # Check CPU usage
                        cpu_percent = process.cpu_percent()
                        if cpu_percent > limits["max_cpu_percent"]:
                            logger.warning(f"Session {agent_id} exceeds CPU limit: {cpu_percent}%")
                            return False
                        
                        # Check process count
                        children = process.children(recursive=True)
                        if len(children) > limits["max_processes"]:
                            logger.warning(f"Session {agent_id} exceeds process limit: {len(children)}")
                            return False
                        
                    except psutil.NoSuchProcess:
                        logger.warning(f"Session process {session_pid} for agent {agent_id} not found")
                        return False
            
            # Update health check timestamp
            session_info["last_health_check"] = datetime.utcnow()
            session_info["health_failures"] = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for session {agent_id}: {e}")
            return False
    
    async def terminate_session(self, agent_id: str) -> bool:
        """Terminate a tmux session for an agent."""
        try:
            if agent_id not in self.session_registry:
                logger.warning(f"No session found for agent {agent_id}")
                return True
            
            session_info = self.session_registry[agent_id]
            session = session_info["session"]
            
            # Capture final state before termination
            final_state = await self.capture_session_state(agent_id)
            
            # Terminate session
            session.kill()
            
            # Update registry
            session_info["status"] = "terminated"
            session_info["terminated_at"] = datetime.utcnow()
            session_info["final_state"] = final_state
            
            # Clean up resource limits
            if agent_id in self.resource_limits:
                del self.resource_limits[agent_id]
            
            logger.info(f"Terminated tmux session for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate session for agent {agent_id}: {e}")
            return False
    
    async def get_session_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a tmux session."""
        try:
            if agent_id not in self.session_registry:
                return {}
            
            session_info = self.session_registry[agent_id]
            metrics = {
                "agent_id": agent_id,
                "status": session_info["status"],
                "uptime_seconds": (datetime.utcnow() - session_info["created_at"]).total_seconds(),
                "last_health_check": session_info["last_health_check"].isoformat(),
                "health_failures": session_info["health_failures"],
                "resource_usage": {}
            }
            
            # Get resource usage if available
            if agent_id in self.resource_limits:
                limits = self.resource_limits[agent_id]
                session_pid = limits.get("session_pid")
                
                if session_pid:
                    try:
                        process = psutil.Process(session_pid)
                        metrics["resource_usage"] = {
                            "memory_mb": process.memory_info().rss / 1024 / 1024,
                            "cpu_percent": process.cpu_percent(),
                            "num_processes": len(process.children(recursive=True)),
                            "limits": {
                                "max_memory_mb": limits["max_memory_mb"],
                                "max_cpu_percent": limits["max_cpu_percent"],
                                "max_processes": limits["max_processes"]
                            }
                        }
                    except psutil.NoSuchProcess:
                        metrics["resource_usage"]["error"] = "Process not found"
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get session metrics for agent {agent_id}: {e}")
            return {}
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active tmux sessions."""
        sessions = []
        for agent_id, session_info in self.session_registry.items():
            if session_info["status"] in ["active", "restored"]:
                metrics = await self.get_session_metrics(agent_id)
                sessions.append({
                    "agent_id": agent_id,
                    "session_name": session_info["session_name"],
                    "template": session_info["template"],
                    "status": session_info["status"],
                    "metrics": metrics
                })
        
        return sessions


logger = structlog.get_logger()


class WorkspaceStatus(Enum):
    """Status of agent workspaces."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"
    TERMINATED = "terminated"


class ProcessType(Enum):
    """Types of processes running in workspaces."""
    DEVELOPMENT_SERVER = "development_server"
    TEST_RUNNER = "test_runner"
    BUILD_PROCESS = "build_process"
    DATABASE = "database"
    BACKGROUND_TASK = "background_task"
    CODE_EXECUTION = "code_execution"


@dataclass
class WorkspaceConfig:
    """Configuration for agent workspace."""
    agent_id: str
    workspace_name: str
    project_path: Path
    
    # Resource limits
    max_memory_mb: int = 2048
    max_cpu_percent: float = 50.0
    max_disk_mb: int = 5120
    max_processes: int = 20
    
    # Environment settings
    python_version: str = "3.11"
    node_version: str = "18"
    additional_packages: List[str] = None
    environment_variables: Dict[str, str] = None
    
    # Security settings
    network_access: bool = True
    file_system_access: List[str] = None  # Allowed paths
    sudo_access: bool = False
    
    def __post_init__(self):
        if self.additional_packages is None:
            self.additional_packages = []
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.file_system_access is None:
            self.file_system_access = [str(self.project_path)]


@dataclass
class RunningProcess:
    """Information about a running process in workspace."""
    pid: int
    name: str
    process_type: ProcessType
    command: str
    started_at: datetime
    cpu_percent: float
    memory_mb: float
    status: str


@dataclass
class WorkspaceMetrics:
    """Real-time metrics for agent workspace."""
    agent_id: str
    workspace_name: str
    status: WorkspaceStatus
    
    # Resource usage
    total_memory_mb: float
    total_cpu_percent: float
    disk_usage_mb: float
    
    # Process information
    active_processes: List[RunningProcess]
    total_processes: int
    
    # Activity metrics
    commands_executed: int
    files_modified: int
    network_requests: int
    
    # Performance
    uptime_seconds: int
    last_activity: datetime


class AgentWorkspace:
    """
    Individual agent workspace with enhanced tmux session and resource management.
    
    Provides a sandboxed environment where agents can:
    - Execute code and run development servers
    - Manage files and projects
    - Run tests and builds
    - Monitor their own resource usage
    - Restore from checkpoints with tmux session recovery
    """
    
    def __init__(self, config: WorkspaceConfig, tmux_manager: Optional[TmuxSessionManager] = None):
        self.config = config
        self.tmux_session: Optional[libtmux.Session] = None
        self.tmux_manager = tmux_manager or TmuxSessionManager()
        
        self.status = WorkspaceStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Resource monitoring
        self.process_monitor: Dict[int, psutil.Process] = {}
        self.metrics_history: List[WorkspaceMetrics] = []
        
        # Activity tracking
        self.commands_executed = 0
        self.files_modified = 0
        self.network_requests = 0
        
        # Checkpoint integration
        self.checkpoint_state: Optional[Dict[str, Any]] = None
    
    async def initialize(self) -> bool:
        """Initialize the workspace environment using enhanced tmux session management."""
        try:
            logger.info(
                "Initializing agent workspace",
                agent_id=self.config.agent_id,
                workspace_name=self.config.workspace_name
            )
            
            # Create project directory
            self.config.project_path.mkdir(parents=True, exist_ok=True)
            
            # Determine tmux template based on workspace configuration
            template = "ai_agent"  # Default for agent workspaces
            if hasattr(self.config, 'workspace_type'):
                template = getattr(self.config, 'workspace_type', 'ai_agent')
            
            # Create tmux session using the enhanced session manager
            self.tmux_session = await self.tmux_manager.create_agent_session(
                agent_id=self.config.agent_id,
                workspace_path=self.config.project_path,
                template=template,
                custom_config={
                    "environment": {
                        "WORKSPACE_NAME": self.config.workspace_name,
                        "MAX_MEMORY_MB": str(self.config.max_memory_mb),
                        "MAX_CPU_PERCENT": str(self.config.max_cpu_percent)
                    }
                }
            )
            
            if not self.tmux_session:
                raise Exception("Failed to create tmux session")
            
            # Setup development environment if needed
            await self._setup_additional_environment()
            
            self.status = WorkspaceStatus.ACTIVE
            
            logger.info(
                "Agent workspace initialized successfully",
                agent_id=self.config.agent_id,
                session_name=f"agent-{self.config.agent_id}",
                template=template
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to initialize workspace",
                agent_id=self.config.agent_id,
                error=str(e)
            )
            self.status = WorkspaceStatus.ERROR
            return False
    
    async def restore_from_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Restore workspace from checkpoint data."""
        try:
            logger.info(f"Restoring workspace for agent {self.config.agent_id} from checkpoint")
            
            # Create project directory
            self.config.project_path.mkdir(parents=True, exist_ok=True)
            
            # Restore tmux session using session manager
            self.tmux_session = await self.tmux_manager.restore_session_from_checkpoint(
                agent_id=self.config.agent_id,
                workspace_path=self.config.project_path,
                checkpoint_data=checkpoint_data
            )
            
            if not self.tmux_session:
                logger.warning(f"Failed to restore tmux session, creating new one")
                return await self.initialize()
            
            # Restore workspace-specific state
            workspace_state = checkpoint_data.get("workspace_state", {})
            self.commands_executed = workspace_state.get("commands_executed", 0)
            self.files_modified = workspace_state.get("files_modified", 0)
            self.network_requests = workspace_state.get("network_requests", 0)
            
            # Store checkpoint state for future reference
            self.checkpoint_state = checkpoint_data
            
            self.status = WorkspaceStatus.ACTIVE
            self.last_activity = datetime.utcnow()
            
            logger.info(f"Successfully restored workspace for agent {self.config.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore workspace from checkpoint: {e}")
            self.status = WorkspaceStatus.ERROR
            return False
    
    async def capture_checkpoint_state(self) -> Dict[str, Any]:
        """Capture current workspace state for checkpointing."""
        try:
            # Capture tmux session state
            tmux_state = await self.tmux_manager.capture_session_state(self.config.agent_id)
            
            # Capture workspace-specific state
            workspace_state = {
                "commands_executed": self.commands_executed,
                "files_modified": self.files_modified,
                "network_requests": self.network_requests,
                "status": self.status.value,
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat()
            }
            
            # Capture current metrics
            current_metrics = await self.get_metrics()
            
            return {
                "tmux_session": tmux_state,
                "workspace_state": workspace_state,
                "current_metrics": asdict(current_metrics),
                "config": {
                    "agent_id": self.config.agent_id,
                    "workspace_name": self.config.workspace_name,
                    "project_path": str(self.config.project_path),
                    "max_memory_mb": self.config.max_memory_mb,
                    "max_cpu_percent": self.config.max_cpu_percent
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to capture checkpoint state: {e}")
            return {}
    
    async def _setup_additional_environment(self) -> None:
        """Setup additional development environment customizations."""
        
        if not self.tmux_session:
            return
        
        main_pane = self.tmux_session.windows[0].panes[0]
        
        # Set environment variables
        for key, value in self.config.environment_variables.items():
            main_pane.send_keys(f"export {key}='{value}'")
        
        # Setup Python environment
        setup_commands = [
            "# Setting up development environment",
            f"cd {self.config.project_path}",
            "pwd",
            "# Python environment setup",
            f"python{self.config.python_version} -m venv venv || python3 -m venv venv",
            "source venv/bin/activate",
            "python --version",
            "pip install --upgrade pip",
        ]
        
        # Install additional packages
        if self.config.additional_packages:
            packages = " ".join(self.config.additional_packages)
            setup_commands.append(f"pip install {packages}")
        
        # Execute setup commands
        for command in setup_commands:
            main_pane.send_keys(command)
            await asyncio.sleep(0.5)  # Allow command to execute
        
        # Create workspace structure
        workspace_dirs = ["src", "tests", "docs", "scripts", "data", "logs"]
        for dir_name in workspace_dirs:
            (self.config.project_path / dir_name).mkdir(exist_ok=True)
    
    def _create_workspace_windows(self) -> None:
        """Create specialized tmux windows for different purposes."""
        
        if not self.tmux_session:
            return
        
        # Window 1: Code execution and development
        code_window = self.tmux_session.new_window(
            window_name="code",
            start_directory=str(self.config.project_path)
        )
        code_window.panes[0].send_keys("source venv/bin/activate")
        
        # Window 2: Testing
        test_window = self.tmux_session.new_window(
            window_name="tests",
            start_directory=str(self.config.project_path)
        )
        test_window.panes[0].send_keys("source venv/bin/activate")
        
        # Window 3: Development server
        server_window = self.tmux_session.new_window(
            window_name="server",
            start_directory=str(self.config.project_path)
        )
        server_window.panes[0].send_keys("source venv/bin/activate")
        
        # Window 4: Monitoring and logs
        monitor_window = self.tmux_session.new_window(
            window_name="monitor",
            start_directory=str(self.config.project_path)
        )
        
        # Split monitor window for different monitoring tasks
        log_pane = monitor_window.split_window(vertical=True)
        log_pane.send_keys("tail -f logs/*.log 2>/dev/null || echo 'No logs yet'")
        
        resource_pane = monitor_window.split_window(vertical=False)
        resource_pane.send_keys("htop")
        
        # Select main window
        self.tmux_session.select_window("main")
    
    async def execute_command(
        self,
        command: str,
        window: str = "code",
        capture_output: bool = True,
        timeout_seconds: int = 300
    ) -> Tuple[bool, str, str]:
        """Execute a command in the workspace."""
        
        if not self.tmux_session or self.status != WorkspaceStatus.ACTIVE:
            return False, "", "Workspace not active"
        
        try:
            # Find the target window
            target_window = None
            for window_obj in self.tmux_session.windows:
                if window_obj.name == window:
                    target_window = window_obj
                    break
            
            if not target_window:
                return False, "", f"Window '{window}' not found"
            
            pane = target_window.panes[0]
            
            # Execute command
            pane.send_keys(command)
            self.commands_executed += 1
            self.last_activity = datetime.utcnow()
            
            if capture_output:
                # Wait for command to complete and capture output
                await asyncio.sleep(1)  # Initial wait
                
                # Capture pane content
                output = pane.capture_pane()
                
                # Simple success detection (would need more sophisticated logic)
                success = "error" not in output.lower() and "failed" not in output.lower()
                
                logger.info(
                    "Command executed in workspace",
                    agent_id=self.config.agent_id,
                    command=command[:100],  # Truncate long commands
                    window=window,
                    success=success
                )
                
                return success, output, ""
            else:
                return True, "", ""
                
        except Exception as e:
            logger.error(
                "Failed to execute command in workspace",
                agent_id=self.config.agent_id,
                command=command,
                error=str(e)
            )
            return False, "", str(e)
    
    async def start_development_server(
        self,
        command: str,
        port: int,
        process_name: str = "dev_server"
    ) -> bool:
        """Start a development server in the workspace."""
        
        try:
            # Execute server command in server window
            success, output, error = await self.execute_command(
                command,
                window="server",
                capture_output=False
            )
            
            if success:
                logger.info(
                    "Development server started",
                    agent_id=self.config.agent_id,
                    process_name=process_name,
                    port=port
                )
            
            return success
            
        except Exception as e:
            logger.error(
                "Failed to start development server",
                agent_id=self.config.agent_id,
                error=str(e)
            )
            return False
    
    async def run_tests(
        self,
        test_command: str = "python -m pytest",
        test_path: str = "tests/"
    ) -> Tuple[bool, str]:
        """Run tests in the workspace."""
        
        full_command = f"{test_command} {test_path} -v"
        success, output, error = await self.execute_command(
            full_command,
            window="tests",
            capture_output=True
        )
        
        # Parse test results
        test_results = self._parse_test_output(output)
        
        logger.info(
            "Tests executed",
            agent_id=self.config.agent_id,
            success=success,
            results=test_results
        )
        
        return success, output
    
    def _parse_test_output(self, output: str) -> Dict[str, Any]:
        """Parse test output to extract results."""
        # Simple parsing - would be more sophisticated in production
        lines = output.split('\n')
        
        passed = 0
        failed = 0
        errors = 0
        
        for line in lines:
            if "passed" in line.lower():
                # Extract number of passed tests
                words = line.split()
                for i, word in enumerate(words):
                    if word.isdigit() and i + 1 < len(words) and "passed" in words[i + 1]:
                        passed = int(word)
                        break
            elif "failed" in line.lower():
                # Extract number of failed tests
                words = line.split()
                for i, word in enumerate(words):
                    if word.isdigit() and i + 1 < len(words) and "failed" in words[i + 1]:
                        failed = int(word)
                        break
        
        return {
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total": passed + failed + errors
        }
    
    async def get_metrics(self) -> WorkspaceMetrics:
        """Get current workspace metrics."""
        
        # Get resource usage
        total_memory = 0.0
        total_cpu = 0.0
        processes = []
        
        try:
            # Get all processes in the workspace
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status', 'cmdline']):
                try:
                    pinfo = proc.info
                    
                    # Check if process belongs to this workspace
                    if self._is_workspace_process(proc):
                        memory_mb = pinfo['memory_info'].rss / 1024 / 1024
                        cpu_percent = pinfo['cpu_percent'] or 0.0
                        
                        total_memory += memory_mb
                        total_cpu += cpu_percent
                        
                        running_proc = RunningProcess(
                            pid=pinfo['pid'],
                            name=pinfo['name'],
                            process_type=self._detect_process_type(pinfo),
                            command=' '.join(pinfo['cmdline'][:3]) if pinfo['cmdline'] else pinfo['name'],
                            started_at=datetime.fromtimestamp(proc.create_time()),
                            cpu_percent=cpu_percent,
                            memory_mb=memory_mb,
                            status=pinfo['status']
                        )
                        processes.append(running_proc)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        
        except Exception as e:
            logger.error("Error collecting workspace metrics", error=str(e))
        
        # Calculate disk usage
        try:
            disk_usage = shutil.disk_usage(self.config.project_path)
            disk_used_mb = (disk_usage.used) / 1024 / 1024
        except Exception:
            disk_used_mb = 0.0
        
        # Calculate uptime
        uptime_seconds = int((datetime.utcnow() - self.created_at).total_seconds())
        
        metrics = WorkspaceMetrics(
            agent_id=self.config.agent_id,
            workspace_name=self.config.workspace_name,
            status=self.status,
            total_memory_mb=total_memory,
            total_cpu_percent=total_cpu,
            disk_usage_mb=disk_used_mb,
            active_processes=processes,
            total_processes=len(processes),
            commands_executed=self.commands_executed,
            files_modified=self.files_modified,
            network_requests=self.network_requests,
            uptime_seconds=uptime_seconds,
            last_activity=self.last_activity
        )
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        # Keep only last 100 metrics entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return metrics
    
    def _is_workspace_process(self, process: psutil.Process) -> bool:
        """Check if a process belongs to this workspace."""
        try:
            # Check if process is running in workspace directory
            cwd = process.cwd()
            return str(self.config.project_path) in cwd
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def _detect_process_type(self, pinfo: Dict[str, Any]) -> ProcessType:
        """Detect the type of process based on its command."""
        
        name = pinfo['name'].lower()
        cmdline = ' '.join(pinfo['cmdline']).lower() if pinfo['cmdline'] else ''
        
        if any(keyword in cmdline for keyword in ['pytest', 'test', 'unittest']):
            return ProcessType.TEST_RUNNER
        elif any(keyword in cmdline for keyword in ['uvicorn', 'gunicorn', 'serve', 'server']):
            return ProcessType.DEVELOPMENT_SERVER
        elif any(keyword in cmdline for keyword in ['build', 'compile', 'webpack']):
            return ProcessType.BUILD_PROCESS
        elif any(keyword in cmdline for keyword in ['postgres', 'mysql', 'redis', 'mongo']):
            return ProcessType.DATABASE
        elif any(keyword in cmdline for keyword in ['python', 'node', 'npm', 'yarn']):
            return ProcessType.CODE_EXECUTION
        else:
            return ProcessType.BACKGROUND_TASK
    
    async def suspend(self) -> bool:
        """Suspend the workspace to save resources."""
        try:
            if self.tmux_session:
                # Don't kill the session, just mark as suspended
                self.status = WorkspaceStatus.SUSPENDED
                
                logger.info(
                    "Workspace suspended",
                    agent_id=self.config.agent_id
                )
                return True
        except Exception as e:
            logger.error("Failed to suspend workspace", error=str(e))
            return False
    
    async def resume(self) -> bool:
        """Resume a suspended workspace."""
        try:
            if self.status == WorkspaceStatus.SUSPENDED:
                self.status = WorkspaceStatus.ACTIVE
                self.last_activity = datetime.utcnow()
                
                logger.info(
                    "Workspace resumed",
                    agent_id=self.config.agent_id
                )
                return True
        except Exception as e:
            logger.error("Failed to resume workspace", error=str(e))
            return False
    
    async def terminate(self) -> bool:
        """Terminate the workspace and clean up resources."""
        try:
            # Capture final state before termination
            final_checkpoint = await self.capture_checkpoint_state()
            
            # Terminate tmux session using session manager
            success = await self.tmux_manager.terminate_session(self.config.agent_id)
            
            if success:
                self.tmux_session = None
                self.status = WorkspaceStatus.TERMINATED
                
                logger.info(
                    "Workspace terminated",
                    agent_id=self.config.agent_id,
                    final_state_captured=bool(final_checkpoint)
                )
                return True
            else:
                logger.warning(f"Tmux session termination failed for agent {self.config.agent_id}")
                return False
            
        except Exception as e:
            logger.error("Failed to terminate workspace", error=str(e))
            return False


class WorkspaceManager:
    """
    Manages all agent workspaces in the system with enhanced tmux session management.
    
    Handles workspace creation, monitoring, resource allocation,
    checkpoint restoration, and cleanup across all active agents.
    """
    
    def __init__(self):
        self.workspaces: Dict[str, AgentWorkspace] = {}
        self.base_workspace_path = Path(get_config().workspace_dir)
        self.base_workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Enhanced tmux session management
        self.tmux_session_manager = TmuxSessionManager()
        
        # Resource limits
        self.total_memory_limit_mb = 8192  # 8GB total
        self.total_cpu_limit = 80.0        # 80% CPU
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        self.health_check_interval = 60  # seconds
    
    async def create_workspace(
        self,
        agent_id: str,
        project_name: str = None,
        config_overrides: Dict[str, Any] = None
    ) -> Optional[AgentWorkspace]:
        """Create a new workspace for an agent."""
        
        if agent_id in self.workspaces:
            logger.warning(f"Workspace already exists for agent {agent_id}")
            return self.workspaces[agent_id]
        
        try:
            # Generate workspace name and path
            project_name = project_name or f"project-{agent_id[:8]}"
            workspace_name = f"workspace-{agent_id}"
            project_path = self.base_workspace_path / workspace_name / project_name
            
            # Create workspace config
            config = WorkspaceConfig(
                agent_id=agent_id,
                workspace_name=workspace_name,
                project_path=project_path
            )
            
            # Apply config overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Check resource availability
            if not await self._check_resource_availability(config):
                logger.error("Insufficient resources for new workspace")
                return None
            
            # Create workspace with shared tmux session manager
            workspace = AgentWorkspace(config, self.tmux_session_manager)
            success = await workspace.initialize()
            
            if success:
                self.workspaces[agent_id] = workspace
                
                # Start monitoring if not already started
                if not self.is_monitoring:
                    await self.start_monitoring()
                
                logger.info(
                    "Workspace created successfully",
                    agent_id=agent_id,
                    workspace_name=workspace_name,
                    project_path=str(project_path)
                )
                
                return workspace
            else:
                logger.error(f"Failed to initialize workspace for agent {agent_id}")
                return None
                
        except Exception as e:
            logger.error(
                "Failed to create workspace",
                agent_id=agent_id,
                error=str(e)
            )
            return None
    
    async def restore_workspace_from_checkpoint(
        self,
        agent_id: str,
        checkpoint_data: Dict[str, Any],
        config_overrides: Dict[str, Any] = None
    ) -> Optional[AgentWorkspace]:
        """Restore a workspace from checkpoint data."""
        try:
            logger.info(f"Restoring workspace for agent {agent_id} from checkpoint")
            
            # Extract workspace configuration from checkpoint or use defaults
            checkpoint_config = checkpoint_data.get("config", {})
            
            # Generate workspace configuration
            project_name = checkpoint_config.get("workspace_name", f"project-{agent_id[:8]}")
            workspace_name = f"workspace-{agent_id}"
            project_path = Path(checkpoint_config.get("project_path", 
                                                    str(self.base_workspace_path / workspace_name / project_name)))
            
            # Create workspace config
            config = WorkspaceConfig(
                agent_id=agent_id,
                workspace_name=workspace_name,
                project_path=project_path,
                max_memory_mb=checkpoint_config.get("max_memory_mb", 2048),
                max_cpu_percent=checkpoint_config.get("max_cpu_percent", 50.0)
            )
            
            # Apply config overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Check resource availability
            if not await self._check_resource_availability(config):
                logger.error("Insufficient resources for workspace restoration")
                return None
            
            # Create and restore workspace
            workspace = AgentWorkspace(config, self.tmux_session_manager)
            success = await workspace.restore_from_checkpoint(checkpoint_data)
            
            if success:
                self.workspaces[agent_id] = workspace
                
                # Start monitoring if not already started
                if not self.is_monitoring:
                    await self.start_monitoring()
                
                logger.info(
                    "Workspace restored successfully from checkpoint",
                    agent_id=agent_id,
                    workspace_name=workspace_name,
                    project_path=str(project_path)
                )
                
                return workspace
            else:
                logger.error(f"Failed to restore workspace for agent {agent_id}")
                return None
                
        except Exception as e:
            logger.error(
                "Failed to restore workspace from checkpoint",
                agent_id=agent_id,
                error=str(e)
            )
            return None
    
    async def capture_all_workspace_states(self) -> Dict[str, Dict[str, Any]]:
        """Capture checkpoint states for all active workspaces."""
        states = {}
        
        for agent_id, workspace in self.workspaces.items():
            if workspace.status == WorkspaceStatus.ACTIVE:
                try:
                    state = await workspace.capture_checkpoint_state()
                    if state:
                        states[agent_id] = state
                except Exception as e:
                    logger.error(f"Failed to capture state for workspace {agent_id}: {e}")
        
        return states
    
    async def _check_resource_availability(self, config: WorkspaceConfig) -> bool:
        """Check if there are sufficient resources for a new workspace."""
        
        # Calculate current resource usage
        current_memory = 0.0
        current_cpu = 0.0
        
        for workspace in self.workspaces.values():
            if workspace.status == WorkspaceStatus.ACTIVE:
                metrics = await workspace.get_metrics()
                current_memory += metrics.total_memory_mb
                current_cpu += metrics.total_cpu_percent
        
        # Check if new workspace would exceed limits
        projected_memory = current_memory + config.max_memory_mb
        projected_cpu = current_cpu + config.max_cpu_percent
        
        if projected_memory > self.total_memory_limit_mb:
            logger.warning(
                "Memory limit would be exceeded",
                current=current_memory,
                requested=config.max_memory_mb,
                limit=self.total_memory_limit_mb
            )
            return False
        
        if projected_cpu > self.total_cpu_limit:
            logger.warning(
                "CPU limit would be exceeded",
                current=current_cpu,
                requested=config.max_cpu_percent,
                limit=self.total_cpu_limit
            )
            return False
        
        return True
    
    async def get_workspace(self, agent_id: str) -> Optional[AgentWorkspace]:
        """Get workspace for an agent."""
        return self.workspaces.get(agent_id)
    
    async def terminate_workspace(self, agent_id: str) -> bool:
        """Terminate workspace for an agent."""
        
        workspace = self.workspaces.get(agent_id)
        if not workspace:
            return False
        
        success = await workspace.terminate()
        if success:
            del self.workspaces[agent_id]
        
        return success
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide workspace metrics."""
        
        total_workspaces = len(self.workspaces)
        active_workspaces = len([w for w in self.workspaces.values() if w.status == WorkspaceStatus.ACTIVE])
        
        total_memory = 0.0
        total_cpu = 0.0
        total_processes = 0
        
        workspace_metrics = {}
        
        for agent_id, workspace in self.workspaces.items():
            if workspace.status == WorkspaceStatus.ACTIVE:
                metrics = await workspace.get_metrics()
                total_memory += metrics.total_memory_mb
                total_cpu += metrics.total_cpu_percent
                total_processes += metrics.total_processes
                
                workspace_metrics[agent_id] = asdict(metrics)
        
        return {
            "total_workspaces": total_workspaces,
            "active_workspaces": active_workspaces,
            "total_memory_mb": total_memory,
            "total_cpu_percent": total_cpu,
            "total_processes": total_processes,
            "memory_limit_mb": self.total_memory_limit_mb,
            "cpu_limit_percent": self.total_cpu_limit,
            "memory_utilization": total_memory / self.total_memory_limit_mb,
            "cpu_utilization": total_cpu / self.total_cpu_limit,
            "workspaces": workspace_metrics
        }
    
    async def start_monitoring(self) -> None:
        """Start background monitoring of all workspaces."""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Started workspace monitoring and health checks")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped workspace monitoring and health checks")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        
        while self.is_monitoring:
            try:
                # Check each workspace
                for agent_id, workspace in list(self.workspaces.items()):
                    if workspace.status == WorkspaceStatus.ACTIVE:
                        metrics = await workspace.get_metrics()
                        
                        # Check for resource violations
                        if metrics.total_memory_mb > workspace.config.max_memory_mb * 1.2:
                            logger.warning(
                                "Workspace memory usage high",
                                agent_id=agent_id,
                                memory_mb=metrics.total_memory_mb,
                                limit_mb=workspace.config.max_memory_mb
                            )
                        
                        if metrics.total_cpu_percent > workspace.config.max_cpu_percent * 1.2:
                            logger.warning(
                                "Workspace CPU usage high",
                                agent_id=agent_id,
                                cpu_percent=metrics.total_cpu_percent,
                                limit_percent=workspace.config.max_cpu_percent
                            )
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("Error in workspace monitoring", error=str(e))
                await asyncio.sleep(5)
    
    async def _health_check_loop(self) -> None:
        """Background health check loop for tmux sessions."""
        
        while self.is_monitoring:
            try:
                # Health check all active workspaces
                for agent_id, workspace in list(self.workspaces.items()):
                    if workspace.status == WorkspaceStatus.ACTIVE:
                        try:
                            # Check tmux session health
                            session_healthy = await self.tmux_session_manager.health_check_session(agent_id)
                            
                            if not session_healthy:
                                logger.warning(f"Unhealthy session detected for agent {agent_id}")
                                
                                # Attempt session recovery
                                success = await self._recover_unhealthy_session(agent_id, workspace)
                                if success:
                                    logger.info(f"Successfully recovered session for agent {agent_id}")
                                else:
                                    logger.error(f"Failed to recover session for agent {agent_id}")
                                    workspace.status = WorkspaceStatus.ERROR
                            
                        except Exception as e:
                            logger.error(f"Health check failed for agent {agent_id}: {e}")
                
                # Sleep before next health check
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _recover_unhealthy_session(self, agent_id: str, workspace: AgentWorkspace) -> bool:
        """Attempt to recover an unhealthy tmux session."""
        try:
            logger.info(f"Attempting to recover unhealthy session for agent {agent_id}")
            
            # Capture current state before recovery
            checkpoint_state = await workspace.capture_checkpoint_state()
            
            # Try to recreate the tmux session
            new_session = await self.tmux_session_manager.create_agent_session(
                agent_id=agent_id,
                workspace_path=workspace.config.project_path,
                template="ai_agent"  # Use default template for recovery
            )
            
            if new_session:
                workspace.tmux_session = new_session
                workspace.status = WorkspaceStatus.ACTIVE
                workspace.last_activity = datetime.utcnow()
                
                logger.info(f"Successfully recovered session for agent {agent_id}")
                return True
            else:
                logger.error(f"Failed to create new session during recovery for agent {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Session recovery failed for agent {agent_id}: {e}")
            return False
    
    async def get_enhanced_system_metrics(self) -> Dict[str, Any]:
        """Get enhanced system metrics including tmux session status."""
        try:
            # Get base metrics
            base_metrics = await self.get_system_metrics()
            
            # Add tmux session metrics
            tmux_sessions = await self.tmux_session_manager.list_active_sessions()
            
            # Enhanced metrics
            enhanced_metrics = {
                **base_metrics,
                "tmux_sessions": {
                    "total_sessions": len(tmux_sessions),
                    "active_sessions": len([s for s in tmux_sessions if s["status"] == "active"]),
                    "restored_sessions": len([s for s in tmux_sessions if s["status"] == "restored"]),
                    "sessions": tmux_sessions
                },
                "health_status": {
                    "monitoring_enabled": self.is_monitoring,
                    "health_check_interval": self.health_check_interval,
                    "last_health_check": datetime.utcnow().isoformat()
                }
            }
            
            return enhanced_metrics
            
        except Exception as e:
            logger.error(f"Error getting enhanced system metrics: {e}")
            return await self.get_system_metrics()  # Fallback to base metrics


# Global workspace manager instance
workspace_manager = WorkspaceManager()
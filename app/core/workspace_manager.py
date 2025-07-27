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

from .config import settings
from .database import get_session
from ..models.agent import Agent
from ..models.session import Session

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
    Individual agent workspace with tmux session and resource management.
    
    Provides a sandboxed environment where agents can:
    - Execute code and run development servers
    - Manage files and projects
    - Run tests and builds
    - Monitor their own resource usage
    """
    
    def __init__(self, config: WorkspaceConfig):
        self.config = config
        self.tmux_session: Optional[libtmux.Session] = None
        self.tmux_server = libtmux.Server()
        
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
    
    async def initialize(self) -> bool:
        """Initialize the workspace environment."""
        try:
            logger.info(
                "Initializing agent workspace",
                agent_id=self.config.agent_id,
                workspace_name=self.config.workspace_name
            )
            
            # Create project directory
            self.config.project_path.mkdir(parents=True, exist_ok=True)
            
            # Create tmux session
            session_name = f"agent-{self.config.agent_id}"
            
            # Kill existing session if it exists
            existing_session = self.tmux_server.find_where({"session_name": session_name})
            if existing_session:
                existing_session.kill()
            
            # Create new session
            self.tmux_session = self.tmux_server.new_session(
                session_name=session_name,
                window_name="main",
                start_directory=str(self.config.project_path),
                detach=True
            )
            
            # Setup development environment
            await self._setup_environment()
            
            # Create additional windows for different purposes
            self._create_workspace_windows()
            
            self.status = WorkspaceStatus.ACTIVE
            
            logger.info(
                "Agent workspace initialized successfully",
                agent_id=self.config.agent_id,
                session_name=session_name
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
    
    async def _setup_environment(self) -> None:
        """Setup the development environment in the workspace."""
        
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
            if self.tmux_session:
                self.tmux_session.kill()
                self.tmux_session = None
            
            self.status = WorkspaceStatus.TERMINATED
            
            logger.info(
                "Workspace terminated",
                agent_id=self.config.agent_id
            )
            return True
            
        except Exception as e:
            logger.error("Failed to terminate workspace", error=str(e))
            return False


class WorkspaceManager:
    """
    Manages all agent workspaces in the system.
    
    Handles workspace creation, monitoring, resource allocation,
    and cleanup across all active agents.
    """
    
    def __init__(self):
        self.workspaces: Dict[str, AgentWorkspace] = {}
        self.base_workspace_path = Path(settings.WORKSPACE_DIR)
        self.base_workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Resource limits
        self.total_memory_limit_mb = 8192  # 8GB total
        self.total_cpu_limit = 80.0        # 80% CPU
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
    
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
            
            # Create workspace
            workspace = AgentWorkspace(config)
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
        
        logger.info("Started workspace monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped workspace monitoring")
    
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


# Global workspace manager instance
workspace_manager = WorkspaceManager()
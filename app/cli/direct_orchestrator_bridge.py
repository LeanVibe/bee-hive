"""
Direct Orchestrator Bridge for CLI Commands

Provides direct access to SimpleOrchestrator functionality from CLI commands
without requiring the FastAPI server to be running. This bridges the gap between
CLI tmux session management and SimpleOrchestrator agent spawning.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import structlog
from rich.console import Console

# Lazy imports - only import when needed for performance optimization
# from ..core.simple_orchestrator import SimpleOrchestrator, AgentRole
# from ..core.configuration_service import ConfigurationService  
# from ..core.enhanced_agent_launcher import AgentLauncherType, AgentLaunchConfig, Priority

logger = structlog.get_logger(__name__)

class DirectOrchestratorBridge:
    """
    Direct bridge between CLI commands and SimpleOrchestrator.
    
    Allows CLI commands to work without requiring the FastAPI server,
    providing seamless integration between tmux session management
    and agent orchestration.
    """
    
    def __init__(self):
        self.orchestrator: Optional[Any] = None  # SimpleOrchestrator - lazy loaded
        self.config_service: Optional[Any] = None  # ConfigurationService - lazy loaded
        self._initialized = False
        self.console = Console()
        self.logger = logger.bind(component="DirectOrchestratorBridge")
    
    async def ensure_initialized(self) -> bool:
        """
        Ensure the orchestrator is initialized.
        
        Returns:
            Whether initialization was successful
        """
        if self._initialized:
            return True
        
        try:
            # Lazy imports for performance optimization - only import when actually needed
            from ..core.simple_orchestrator import SimpleOrchestrator, AgentRole
            from ..core.enhanced_agent_launcher import AgentLauncherType, AgentLaunchConfig
            from .performance_cache import get_cached_config
            
            # Store classes for later use
            self.SimpleOrchestrator = SimpleOrchestrator
            self.AgentRole = AgentRole
            self.AgentLauncherType = AgentLauncherType
            self.AgentLaunchConfig = AgentLaunchConfig
            
            # Use cached configuration service for performance
            self.config_service = get_cached_config()
            
            # Initialize SimpleOrchestrator
            self.orchestrator = SimpleOrchestrator(self.config_service)
            
            self._initialized = True
            self.logger.info("Direct orchestrator bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize orchestrator bridge", error=str(e))
            return False
    
    async def spawn_agent(self, 
                         agent_type: str,
                         task_id: Optional[str] = None,
                         workspace_name: Optional[str] = None,
                         git_branch: Optional[str] = None,
                         working_directory: Optional[str] = None,
                         environment_vars: Optional[Dict[str, str]] = None,
                         agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Spawn an agent through direct orchestrator access.
        
        Args:
            agent_type: Type of agent to spawn
            task_id: Optional task ID to assign
            workspace_name: Workspace name for the agent
            git_branch: Git branch for the workspace
            working_directory: Working directory
            environment_vars: Environment variables
            agent_name: Custom agent name
            
        Returns:
            Spawn result with agent details
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Convert agent type string to enum
            agent_launcher_type = self._convert_agent_type(agent_type)
            agent_role = self._convert_to_agent_role(agent_type)
            
            # Generate agent ID if not provided
            agent_id = str(uuid.uuid4())
            
            # Spawn agent through orchestrator  
            launch_result = await self.orchestrator.spawn_agent(
                role=agent_role,
                agent_id=agent_id
            )
            
            if launch_result:
                # spawn_agent returns the spawned agent ID as a string
                spawned_agent_id = launch_result if isinstance(launch_result, str) else str(launch_result)
                
                # Get additional session info if available
                session_info = {}
                try:
                    session_info = await self.orchestrator.get_agent_session_info(spawned_agent_id)
                    if session_info is None:
                        session_info = {}
                except Exception:
                    session_info = {}
                
                return {
                    "success": True,
                    "agent_id": spawned_agent_id,
                    "session_name": session_info.get("session_name", f"agent-{spawned_agent_id[:8]}"),
                    "workspace_path": session_info.get("workspace_path", f"workspaces/agent-{spawned_agent_id[:8]}"),
                    "launch_result": launch_result
                }
            else:
                return {
                    "success": False,
                    "error": "Agent spawning failed"
                }
                
        except Exception as e:
            self.logger.error("Failed to spawn agent", agent_type=agent_type, error=str(e))
            return {
                "success": False,
                "error": f"Agent spawn failed: {str(e)}"
            }
    
    async def list_agents(self) -> Dict[str, Any]:
        """
        List all active agents.
        
        Returns:
            Dictionary with agent list and metadata
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get agent list from orchestrator
            agents = await self.orchestrator.list_agent_sessions()
            
            # If orchestrator has no agents, try to read from tmux sessions directly
            if not agents:
                self.logger.info("No agents found in orchestrator, reading from tmux sessions...")
                agents = await self._read_tmux_agents_directly()
            
            # Get detailed session info for each agent
            detailed_agents = []
            for agent_info in agents:
                agent_id = agent_info.get("agent_id")
                if agent_id:
                    session_info = await self.orchestrator.get_agent_session_info(agent_id)
                    enhanced_info = {
                        **agent_info,
                        "session_info": session_info
                    }
                    detailed_agents.append(enhanced_info)
                else:
                    detailed_agents.append(agent_info)
            
            return {
                "success": True,
                "agents": detailed_agents,
                "total_count": len(detailed_agents),
                "active_count": len([a for a in detailed_agents if a.get("is_running", False)])
            }
            
        except Exception as e:
            self.logger.error("Failed to list agents", error=str(e))
            return {
                "success": False,
                "error": f"Failed to list agents: {str(e)}"
            }
    
    async def _read_tmux_agents_directly(self) -> List[Dict[str, Any]]:
        """
        Read agent sessions directly from tmux when orchestrator has no data.
        
        Returns:
            List of agent session information
        """
        import subprocess
        
        try:
            # Get all tmux sessions that look like agents
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}:#{session_created}:#{session_windows}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("Failed to read tmux sessions", error=result.stderr)
                return []
            
            agent_sessions = []
            for line in result.stdout.strip().split('\n'):
                if not line or not line.startswith('agent-'):
                    continue
                    
                parts = line.split(':')
                if len(parts) >= 3:
                    session_name = parts[0]
                    created_time = parts[1]
                    windows = parts[2]
                    
                    # Extract agent ID from session name (format: agent-{uuid})
                    agent_id = session_name.replace('agent-', '')
                    
                    # Check if there's a corresponding workspace
                    workspace_path = f"workspaces/{self._find_workspace_for_session(session_name)}"
                    
                    agent_info = {
                        "agent_id": agent_id,
                        "session_name": session_name,
                        "created_time": created_time,
                        "windows": int(windows) if windows.isdigit() else 1,
                        "is_running": True,
                        "status": "active",
                        "workspace_path": workspace_path,
                        "source": "tmux_direct_read"
                    }
                    
                    agent_sessions.append(agent_info)
            
            self.logger.info(f"Found {len(agent_sessions)} active tmux agent sessions")
            return agent_sessions
            
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout reading tmux sessions")
            return []
        except Exception as e:
            self.logger.error("Failed to read tmux sessions directly", error=str(e))
            return []
    
    def _find_workspace_for_session(self, session_name: str) -> str:
        """Find workspace directory for a tmux session."""
        import os
        
        # Try to find workspace by looking for agent-* directories
        workspace_base = Path("workspaces")
        if not workspace_base.exists():
            return "unknown"
        
        # Look for directories that might match this session
        for workspace_dir in workspace_base.iterdir():
            if workspace_dir.is_dir() and workspace_dir.name.startswith('agent-'):
                # For now, return the first matching workspace pattern
                # In production, we could enhance this with better mapping logic
                return workspace_dir.name
        
        return "unknown"
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed status for a specific agent.
        
        Args:
            agent_id: Agent ID to check
            
        Returns:
            Agent status information
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get agent status from orchestrator
            status = await self.orchestrator.get_agent_status(agent_id)
            session_info = await self.orchestrator.get_agent_session_info(agent_id)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "status": status,
                "session_info": session_info,
                "is_running": status.get("is_running", False) if status else False
            }
            
        except Exception as e:
            self.logger.error("Failed to get agent status", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": f"Failed to get agent status: {str(e)}"
            }
    
    async def shutdown_agent(self, agent_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Shutdown a specific agent.
        
        Args:
            agent_id: Agent ID to shutdown
            force: Whether to force shutdown
            
        Returns:
            Shutdown result
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Shutdown agent through orchestrator
            result = await self.orchestrator.shutdown_agent(agent_id, force=force)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "shutdown_result": result
            }
            
        except Exception as e:
            self.logger.error("Failed to shutdown agent", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": f"Failed to shutdown agent: {str(e)}"
            }
    
    async def execute_command_in_session(self, 
                                       agent_id: str, 
                                       command: str) -> Dict[str, Any]:
        """
        Execute a command in an agent's tmux session.
        
        Args:
            agent_id: Agent ID
            command: Command to execute
            
        Returns:
            Execution result
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get session info
            session_info = await self.orchestrator.get_agent_session_info(agent_id)
            if not session_info:
                return {"success": False, "error": "Agent session not found"}
            
            session_name = session_info.get("session_name")
            if not session_name:
                return {"success": False, "error": "Session name not available"}
            
            # Execute command via tmux manager
            tmux_manager = self.orchestrator._agent_launcher.tmux_manager
            result = await tmux_manager.execute_command_in_session(session_name, command)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "command": command,
                "execution_result": result
            }
            
        except Exception as e:
            self.logger.error("Failed to execute command", 
                            agent_id=agent_id, command=command, error=str(e))
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}"
            }
    
    async def attach_to_session(self, agent_id: str) -> Dict[str, Any]:
        """
        Get tmux attach command for an agent session.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Attach information
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get session info
            session_info = await self.orchestrator.get_agent_session_info(agent_id)
            if not session_info:
                return {"success": False, "error": "Agent session not found"}
            
            session_name = session_info.get("session_name")
            if not session_name:
                return {"success": False, "error": "Session name not available"}
            
            return {
                "success": True,
                "agent_id": agent_id,
                "session_name": session_name,
                "attach_command": f"tmux attach-session -t {session_name}",
                "workspace_path": session_info.get("workspace_path")
            }
            
        except Exception as e:
            self.logger.error("Failed to get attach info", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": f"Failed to get attach info: {str(e)}"
            }
    
    async def get_session_logs(self, 
                             agent_id: str, 
                             lines: int = 100) -> Dict[str, Any]:
        """
        Get logs from an agent's tmux session.
        
        Args:
            agent_id: Agent ID
            lines: Number of lines to retrieve
            
        Returns:
            Session logs
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get session info
            session_info = await self.orchestrator.get_agent_session_info(agent_id)
            if not session_info:
                return {"success": False, "error": "Agent session not found"}
            
            session_name = session_info.get("session_name")
            if not session_name:
                return {"success": False, "error": "Session name not available"}
            
            # Get logs via tmux manager
            tmux_manager = self.orchestrator._agent_launcher.tmux_manager
            logs = await tmux_manager.get_session_logs(session_name, lines)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "session_name": session_name,
                "logs": logs,
                "line_count": len(logs) if logs else 0
            }
            
        except Exception as e:
            self.logger.error("Failed to get session logs", agent_id=agent_id, error=str(e))
            return {
                "success": False,
                "error": f"Failed to get logs: {str(e)}"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status including orchestrator health.
        
        Returns:
            System status information
        """
        if not await self.ensure_initialized():
            return {"success": False, "error": "Failed to initialize orchestrator"}
        
        try:
            # Get system status from orchestrator
            status = await self.orchestrator.get_system_status()
            
            # Add bridge-specific information
            status.update({
                "bridge_initialized": self._initialized,
                "direct_access": True,
                "api_required": False
            })
            
            return {
                "success": True,
                "system_status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to get system status", error=str(e))
            return {
                "success": False,
                "error": f"Failed to get system status: {str(e)}"
            }
    
    def _convert_agent_type(self, agent_type: str):
        """Convert string agent type to enum."""
        if not self._initialized:
            return None
            
        type_mapping = {
            "claude-code": self.AgentLauncherType.CLAUDE_CODE,
            "tmux-session": self.AgentLauncherType.CLAUDE_CODE,  # Map tmux-session to CLAUDE_CODE
            "claude_code": self.AgentLauncherType.CLAUDE_CODE,
            "tmux_session": self.AgentLauncherType.CLAUDE_CODE,
            "cursor-agent": self.AgentLauncherType.CURSOR_AGENT,
            "open-code": self.AgentLauncherType.OPEN_CODE,
            "aider": self.AgentLauncherType.AIDER,
            "continue": self.AgentLauncherType.CONTINUE,
            "custom": self.AgentLauncherType.CUSTOM
        }
        
        return type_mapping.get(agent_type.lower(), self.AgentLauncherType.CLAUDE_CODE)
    
    def _convert_to_agent_role(self, agent_type: str):
        """Convert agent type to appropriate role."""
        if not self._initialized:
            return None
            
        # Map agent types to roles based on common usage
        role_mapping = {
            "claude-code": self.AgentRole.BACKEND_DEVELOPER,
            "tmux-session": self.AgentRole.BACKEND_DEVELOPER,
            "backend-developer": self.AgentRole.BACKEND_DEVELOPER,
            "frontend-developer": self.AgentRole.FRONTEND_DEVELOPER,
            "qa-engineer": self.AgentRole.QA_ENGINEER,
            "devops-engineer": self.AgentRole.DEVOPS_ENGINEER,
            "meta-agent": self.AgentRole.META_AGENT
        }
        
        return role_mapping.get(agent_type.lower(), self.AgentRole.BACKEND_DEVELOPER)

# Global bridge instance
_bridge_instance: Optional[DirectOrchestratorBridge] = None

def get_bridge() -> DirectOrchestratorBridge:
    """Get the global bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = DirectOrchestratorBridge()
    return _bridge_instance
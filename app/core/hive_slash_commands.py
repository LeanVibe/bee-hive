"""
Hive Slash Commands System for LeanVibe Agent Hive 2.0

Custom slash commands with `hive:` prefix for meta-agent operations
and advanced platform control. This system provides Claude Code-style
custom commands specifically for autonomous development orchestration.

Usage:
    /hive:start              # Start multi-agent platform
    /hive:spawn <role>       # Spawn specific agent
    /hive:status             # Get platform status
    /hive:develop <project>  # Start autonomous development
    /hive:oversight          # Open remote oversight dashboard
    /hive:stop               # Stop all agents and services
"""

import asyncio
import json
import subprocess
import webbrowser
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import structlog

from .agent_spawner import get_agent_manager, spawn_development_team, get_active_agents_status
from .orchestrator import AgentRole
from .config import settings

logger = structlog.get_logger()


class HiveSlashCommand:
    """Base class for hive slash commands."""
    
    def __init__(self, name: str, description: str, usage: str):
        self.name = name
        self.description = description
        self.usage = usage
        self.created_at = datetime.utcnow()
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the command. Override in subclasses."""
        raise NotImplementedError
    
    def validate_args(self, args: List[str]) -> bool:
        """Validate command arguments. Override in subclasses."""
        return True


class HiveStartCommand(HiveSlashCommand):
    """Start the multi-agent platform with all services."""
    
    def __init__(self):
        super().__init__(
            name="start",
            description="Start multi-agent platform with all services and spawn development team",
            usage="/hive:start [--quick] [--team-size=5]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start the platform."""
        try:
            logger.info("🚀 Executing /hive:start command")
            
            # Parse arguments
            quick_mode = "--quick" in (args or [])
            team_size = 5
            
            for arg in (args or []):
                if arg.startswith("--team-size="):
                    team_size = int(arg.split("=")[1])
            
            # Check if agents already exist
            try:
                agent_status = await get_active_agents_status()
                if len(agent_status) >= team_size:
                    return {
                        "success": True,
                        "message": f"Platform already running with {len(agent_status)} agents",
                        "agents": agent_status,
                        "quick_start": True
                    }
            except:
                pass  # Platform not ready yet
            
            # Spawn development team
            team_composition = await spawn_development_team()
            
            # Get current status
            active_agents = await get_active_agents_status()
            
            return {
                "success": True,
                "message": f"Platform started successfully with {len(active_agents)} agents",
                "team_composition": team_composition,
                "active_agents": active_agents,
                "ready_for_development": len(active_agents) >= 3
            }
            
        except Exception as e:
            logger.error("Failed to execute /hive:start", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start platform"
            }


class HiveSpawnCommand(HiveSlashCommand):
    """Spawn a specific agent with given role."""
    
    def __init__(self):
        super().__init__(
            name="spawn",
            description="Spawn a specific agent with the given role",
            usage="/hive:spawn <role> [--capabilities=cap1,cap2]"
        )
    
    def validate_args(self, args: List[str]) -> bool:
        """Validate that a role is provided."""
        if not args or len(args) < 1:
            return False
        
        # Check if role is valid
        try:
            AgentRole(args[0].lower())
            return True
        except ValueError:
            return False
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Spawn the specified agent."""
        try:
            if not self.validate_args(args):
                valid_roles = [role.value for role in AgentRole]
                return {
                    "success": False,
                    "error": f"Invalid role. Valid roles: {', '.join(valid_roles)}",
                    "usage": self.usage
                }
            
            role_name = args[0].lower()
            agent_role = AgentRole(role_name)
            
            logger.info("🤖 Executing /hive:spawn command", role=role_name)
            
            # Import spawn config here to avoid circular imports
            from .agent_spawner import AgentSpawnConfig
            
            # Define role-specific capabilities
            role_capabilities = {
                AgentRole.PRODUCT_MANAGER: ["requirements_analysis", "project_planning", "documentation"],
                AgentRole.ARCHITECT: ["system_design", "architecture_planning", "technology_selection"],
                AgentRole.BACKEND_DEVELOPER: ["api_development", "database_design", "server_logic"],
                AgentRole.FRONTEND_DEVELOPER: ["ui_development", "react", "typescript"],
                AgentRole.QA_ENGINEER: ["test_creation", "quality_assurance", "validation"],
                AgentRole.DEVOPS_ENGINEER: ["deployment", "infrastructure", "monitoring"],
                AgentRole.META_AGENT: ["orchestration", "meta_coordination", "system_oversight"]
            }
            
            # Parse custom capabilities if provided
            custom_capabilities = None
            for arg in (args[1:] if len(args) > 1 else []):
                if arg.startswith("--capabilities="):
                    custom_capabilities = arg.split("=")[1].split(",")
            
            config = AgentSpawnConfig(
                role=agent_role,
                capabilities=custom_capabilities or role_capabilities.get(agent_role, ["general_development"])
            )
            
            manager = await get_agent_manager()
            agent_id = await manager.spawn_agent(config)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "role": agent_role.value,
                "capabilities": config.capabilities,
                "message": f"Successfully spawned {agent_role.value} agent"
            }
            
        except Exception as e:
            logger.error("Failed to execute /hive:spawn", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to spawn {args[0] if args else 'unknown'} agent"
            }


class HiveStatusCommand(HiveSlashCommand):
    """Get comprehensive platform status."""
    
    def __init__(self):
        super().__init__(
            name="status",
            description="Get comprehensive status of the multi-agent platform",
            usage="/hive:status [--detailed] [--agents-only]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get platform status."""
        try:
            logger.info("📊 Executing /hive:status command")
            
            detailed = "--detailed" in (args or [])
            agents_only = "--agents-only" in (args or [])
            
            # Get agent status with hybrid orchestrator integration
            spawner_agents = await get_active_agents_status()
            spawner_count = len(spawner_agents) if spawner_agents else 0
            
            # Get orchestrator agents (if available)
            orchestrator_count = 0
            orchestrator_agents = {}
            try:
                from ..main import app
                if hasattr(app.state, 'orchestrator'):
                    orchestrator = app.state.orchestrator
                    system_status = await orchestrator.get_system_status()
                    orchestrator_count = system_status.get("orchestrator_agents", 0)
                    orchestrator_agents = system_status.get("agents", {})
            except Exception as e:
                logger.debug(f"Could not get orchestrator status: {e}")
            
            total_agents = spawner_count + orchestrator_count
            
            status_result = {
                "success": True,
                "platform_active": total_agents > 0,
                "agent_count": total_agents,
                "spawner_agents": spawner_count,
                "orchestrator_agents": orchestrator_count,
                "system_ready": total_agents >= 3,
                "timestamp": datetime.utcnow().isoformat(),
                "hybrid_integration": True
            }
            
            if not agents_only:
                # Add system health info
                try:
                    import requests
                    health_response = requests.get("http://localhost:8000/health", timeout=3)
                    if health_response.status_code == 200:
                        health_data = health_response.json()
                        status_result["system_health"] = health_data.get("status", "unknown")
                        status_result["components"] = health_data.get("components", {})
                except:
                    status_result["system_health"] = "unknown"
            
            if detailed or agents_only:
                status_result["spawner_agents_detail"] = spawner_agents or {}
                status_result["orchestrator_agents_detail"] = orchestrator_agents
                
                # Add capability summary
                capabilities_summary = {}
                all_agents = (spawner_agents or {}).copy()
                all_agents.update(orchestrator_agents)
                
                for agent_data in all_agents.values():
                    role = agent_data.get("role", "unknown")
                    capabilities = agent_data.get("capabilities", [])
                    
                    if role not in capabilities_summary:
                        capabilities_summary[role] = {
                            "count": 0,
                            "capabilities": set()
                        }
                    
                    capabilities_summary[role]["count"] += 1
                    capabilities_summary[role]["capabilities"].update(capabilities)
                
                # Convert sets to lists for JSON serialization
                for role_info in capabilities_summary.values():
                    role_info["capabilities"] = list(role_info["capabilities"])
                
                status_result["team_composition"] = capabilities_summary
            
            return status_result
            
        except Exception as e:
            logger.error("Failed to execute /hive:status", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get platform status"
            }


class HiveDevelopCommand(HiveSlashCommand):
    """Start autonomous development with multi-agent coordination."""
    
    def __init__(self):
        super().__init__(
            name="develop",
            description="Start autonomous development with multi-agent coordination",
            usage="/hive:develop <project_description> [--dashboard] [--timeout=300]"
        )
    
    def validate_args(self, args: List[str]) -> bool:
        """Validate that project description is provided."""
        return args and len(args) >= 1
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start autonomous development."""
        try:
            if not self.validate_args(args):
                return {
                    "success": False,
                    "error": "Project description required",
                    "usage": self.usage
                }
            
            # Extract project description and options
            project_description = " ".join(arg for arg in args if not arg.startswith("--"))
            
            dashboard_mode = "--dashboard" in args
            timeout = 300  # 5 minutes default
            
            for arg in args:
                if arg.startswith("--timeout="):
                    timeout = int(arg.split("=")[1])
            
            logger.info("🤖 Executing /hive:develop command", 
                       project=project_description, 
                       dashboard=dashboard_mode)
            
            # Ensure agents are available
            active_agents = await get_active_agents_status()
            if len(active_agents) < 3:
                # Auto-spawn team if needed
                await spawn_development_team()
                active_agents = await get_active_agents_status()
            
            # Open dashboard if requested
            if dashboard_mode:
                try:
                    webbrowser.open("http://localhost:8000/dashboard/")
                except:
                    pass  # Dashboard opening failed, but continue
            
            # Start autonomous development
            project_root = Path(__file__).parent.parent.parent
            demo_script = project_root / "scripts" / "demos" / "autonomous_development_demo.py"
            
            if demo_script.exists():
                # Run autonomous development demo
                process = await asyncio.create_subprocess_exec(
                    "python", str(demo_script), project_description,
                    cwd=project_root,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=timeout
                    )
                    
                    success = process.returncode == 0
                    
                    return {
                        "success": success,
                        "project_description": project_description,
                        "agents_involved": len(active_agents),
                        "output": stdout.decode() if stdout else "",
                        "error": stderr.decode() if stderr and not success else None,
                        "dashboard_opened": dashboard_mode,
                        "execution_time": timeout if not success else "completed"
                    }
                    
                except asyncio.TimeoutError:
                    process.kill()
                    return {
                        "success": False,
                        "error": f"Development timed out after {timeout} seconds",
                        "project_description": project_description,
                        "message": "Development may be continuing in background"
                    }
            else:
                return {
                    "success": False,
                    "error": "Autonomous development engine not found",
                    "message": "Development demo script missing"
                }
                
        except Exception as e:
            logger.error("Failed to execute /hive:develop", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to start development: {project_description}"
            }


class HiveOversightCommand(HiveSlashCommand):
    """Open remote oversight dashboard."""
    
    def __init__(self):
        super().__init__(
            name="oversight",
            description="Open remote oversight dashboard for multi-agent monitoring",
            usage="/hive:oversight [--mobile-info]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Open oversight dashboard."""
        try:
            logger.info("🎛️ Executing /hive:oversight command")
            
            mobile_info = "--mobile-info" in (args or [])
            dashboard_url = "http://localhost:8000/dashboard/"
            
            # Open dashboard
            try:
                webbrowser.open(dashboard_url)
                dashboard_opened = True
            except Exception as e:
                dashboard_opened = False
                open_error = str(e)
            
            result = {
                "success": True,
                "dashboard_url": dashboard_url,
                "dashboard_opened": dashboard_opened,
                "message": "Remote oversight dashboard ready"
            }
            
            if not dashboard_opened:
                result["error"] = f"Could not auto-open browser: {open_error}"
                result["manual_instruction"] = f"Please manually open: {dashboard_url}"
            
            if mobile_info:
                # Try to get local IP for mobile access
                try:
                    import socket
                    hostname = socket.gethostname()
                    local_ip = socket.gethostbyname(hostname)
                    mobile_url = f"http://{local_ip}:8000/dashboard/"
                    
                    result["mobile_access"] = {
                        "url": mobile_url,
                        "instructions": "Open this URL on your mobile device for remote oversight",
                        "features": [
                            "Real-time agent status monitoring",
                            "Live task progress tracking",
                            "Mobile-optimized responsive interface",
                            "WebSocket live updates"
                        ]
                    }
                except:
                    result["mobile_access"] = {
                        "url": dashboard_url,
                        "note": "Use localhost URL or configure network access for mobile"
                    }
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute /hive:oversight", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to open oversight dashboard"
            }


class HiveStopCommand(HiveSlashCommand):
    """Stop all agents and services."""
    
    def __init__(self):
        super().__init__(
            name="stop",
            description="Stop all agents and platform services",
            usage="/hive:stop [--force] [--agents-only]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stop the platform."""
        try:
            logger.info("🛑 Executing /hive:stop command")
            
            force = "--force" in (args or [])
            agents_only = "--agents-only" in (args or [])
            
            # Get current status before stopping
            try:
                active_agents = await get_active_agents_status()
                agent_count = len(active_agents)
            except:
                agent_count = 0
            
            # Stop agent manager
            try:
                manager = await get_agent_manager()
                await manager.stop()
                agents_stopped = True
            except Exception as e:
                agents_stopped = False
                stop_error = str(e)
            
            result = {
                "success": agents_stopped,
                "agents_stopped": agent_count if agents_stopped else 0,
                "message": f"Stopped {agent_count} agents" if agents_stopped else "Failed to stop agents"
            }
            
            if not agents_stopped:
                result["error"] = f"Agent stop failed: {stop_error}"
            
            # Stop platform services if not agents-only
            if not agents_only:
                try:
                    project_root = Path(__file__).parent.parent.parent
                    subprocess.run(["make", "stop"], cwd=project_root, check=True)
                    result["platform_stopped"] = True
                    result["message"] += " and platform services"
                except Exception as e:
                    result["platform_stopped"] = False
                    result["platform_error"] = str(e)
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute /hive:stop", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to stop platform"
            }


class HiveSlashCommandRegistry:
    """Registry for all hive slash commands."""
    
    def __init__(self):
        self.commands: Dict[str, HiveSlashCommand] = {}
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register all default hive commands."""
        commands = [
            HiveStartCommand(),
            HiveSpawnCommand(), 
            HiveStatusCommand(),
            HiveDevelopCommand(),
            HiveOversightCommand(),
            HiveStopCommand()
        ]
        
        for cmd in commands:
            self.commands[cmd.name] = cmd
    
    def register_command(self, command: HiveSlashCommand):
        """Register a custom command."""
        self.commands[command.name] = command
    
    def get_command(self, name: str) -> Optional[HiveSlashCommand]:
        """Get a command by name."""
        return self.commands.get(name)
    
    def list_commands(self) -> Dict[str, str]:
        """List all available commands with descriptions."""
        return {
            name: cmd.description 
            for name, cmd in self.commands.items()
        }
    
    async def execute_command(self, command_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a hive slash command."""
        try:
            # Parse command
            if not command_text.startswith("/hive:"):
                return {
                    "success": False,
                    "error": "Invalid hive command format. Use /hive:<command>",
                    "usage": "Available commands: " + ", ".join(self.commands.keys())
                }
            
            # Extract command and arguments
            command_part = command_text[6:]  # Remove "/hive:" prefix
            parts = command_part.split()
            
            if not parts:
                return {
                    "success": False,
                    "error": "No command specified",
                    "available_commands": self.list_commands()
                }
            
            command_name = parts[0]
            args = parts[1:] if len(parts) > 1 else []
            
            # Get and execute command
            command = self.get_command(command_name)
            if not command:
                return {
                    "success": False,
                    "error": f"Unknown command: {command_name}",
                    "available_commands": self.list_commands()
                }
            
            # Execute the command
            return await command.execute(args, context)
            
        except Exception as e:
            logger.error("Failed to execute hive command", 
                        command=command_text, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Command execution failed"
            }


# Global registry instance
_command_registry: Optional[HiveSlashCommandRegistry] = None


def get_hive_command_registry() -> HiveSlashCommandRegistry:
    """Get the global hive command registry."""
    global _command_registry
    if _command_registry is None:
        _command_registry = HiveSlashCommandRegistry()
    return _command_registry


async def execute_hive_command(command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute a hive slash command."""
    registry = get_hive_command_registry()
    return await registry.execute_command(command, context)
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
    """Get comprehensive platform status with intelligent filtering."""
    
    def __init__(self):
        super().__init__(
            name="status",
            description="Get intelligent status overview with priority alerts and mobile optimization",
            usage="/hive:status [--detailed] [--agents-only] [--mobile] [--alerts-only] [--priority=critical|high|medium]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get intelligent platform status with priority filtering."""
        try:
            logger.info("📊 Executing enhanced /hive:status command")
            
            # Parse arguments
            detailed = "--detailed" in (args or [])
            agents_only = "--agents-only" in (args or [])
            mobile_mode = "--mobile" in (args or [])
            alerts_only = "--alerts-only" in (args or [])
            
            priority_filter = None
            for arg in (args or []):
                if arg.startswith("--priority="):
                    priority_filter = arg.split("=")[1].lower()
            
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
            
            # Generate intelligent alerts and priority information
            alerts = await self._generate_intelligent_alerts(spawner_agents, orchestrator_agents, total_agents)
            filtered_alerts = self._filter_alerts_by_priority(alerts, priority_filter) if priority_filter else alerts
            
            # Base status structure
            status_result = {
                "success": True,
                "platform_active": total_agents > 0,
                "agent_count": total_agents,
                "spawner_agents": spawner_count,
                "orchestrator_agents": orchestrator_count,
                "system_ready": total_agents >= 3,
                "timestamp": datetime.utcnow().isoformat(),
                "hybrid_integration": True,
                "priority_alerts": filtered_alerts,
                "total_alerts": len(alerts),
                "filtered_alerts": len(filtered_alerts)
            }
            
            # Mobile-optimized response
            if mobile_mode:
                return await self._generate_mobile_optimized_status(status_result, filtered_alerts)
            
            # Alerts-only response
            if alerts_only:
                return {
                    "success": True,
                    "alerts": filtered_alerts,
                    "alert_summary": self._generate_alert_summary(filtered_alerts),
                    "requires_action": any(alert["priority"] in ["critical", "high"] for alert in filtered_alerts),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            if not agents_only:
                # Add enhanced system health info
                system_health = await self._get_enhanced_system_health()
                status_result.update(system_health)
            
            if detailed or agents_only:
                status_result["spawner_agents_detail"] = spawner_agents or {}
                status_result["orchestrator_agents_detail"] = orchestrator_agents
                
                # Add intelligent capability analysis
                capabilities_analysis = await self._analyze_team_capabilities(spawner_agents, orchestrator_agents)
                status_result["team_analysis"] = capabilities_analysis
            
            return status_result
            
        except Exception as e:
            logger.error("Failed to execute /hive:status", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to get platform status"
            }
    
    async def _generate_intelligent_alerts(self, spawner_agents: dict, orchestrator_agents: dict, total_agents: int) -> List[Dict[str, Any]]:
        """Generate intelligent alerts based on system state."""
        alerts = []
        
        # Critical alerts
        if total_agents == 0:
            alerts.append({
                "priority": "critical",
                "type": "system_down",
                "message": "No agents active - platform offline",
                "action": "Execute /hive:start to initialize platform",
                "timestamp": datetime.utcnow().isoformat()
            })
        elif total_agents < 3:
            alerts.append({
                "priority": "high",
                "type": "insufficient_agents",
                "message": f"Only {total_agents} agents active - minimum 3 recommended",
                "action": "Consider running /hive:start to spawn more agents",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Agent health analysis
        all_agents = (spawner_agents or {}).copy()
        all_agents.update(orchestrator_agents or {})
        
        stuck_agents = []
        for agent_id, agent_data in all_agents.items():
            # Check for stuck agents (placeholder logic)
            last_activity = agent_data.get("last_activity")
            if last_activity:
                # If agent hasn't been active recently, flag as potentially stuck
                pass  # Would implement actual stuck detection logic
        
        # Performance alerts
        try:
            import requests
            health_response = requests.get("http://localhost:8000/health", timeout=2)
            if health_response.status_code != 200:
                alerts.append({
                    "priority": "high",
                    "type": "api_health",
                    "message": "API health check failed",
                    "action": "Check system logs and restart services if needed",
                    "timestamp": datetime.utcnow().isoformat()
                })
        except:
            alerts.append({
                "priority": "medium",
                "type": "api_unreachable",
                "message": "API health check unreachable",
                "action": "Verify system is running with make start",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Success alerts
        if total_agents >= 5:
            alerts.append({
                "priority": "info",
                "type": "system_optimal",
                "message": f"Platform running optimally with {total_agents} agents",
                "action": "Ready for autonomous development tasks",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return alerts
    
    def _filter_alerts_by_priority(self, alerts: List[Dict[str, Any]], priority: str) -> List[Dict[str, Any]]:
        """Filter alerts by priority level."""
        priority_levels = {
            "critical": ["critical"],
            "high": ["critical", "high"],
            "medium": ["critical", "high", "medium"],
            "all": ["critical", "high", "medium", "info"]
        }
        
        allowed_priorities = priority_levels.get(priority, ["critical", "high", "medium", "info"])
        return [alert for alert in alerts if alert["priority"] in allowed_priorities]
    
    async def _generate_mobile_optimized_status(self, base_status: Dict[str, Any], alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate mobile-optimized status response."""
        # Filter to only critical decision points for mobile
        critical_alerts = [alert for alert in alerts if alert["priority"] in ["critical", "high"]]
        
        mobile_status = {
            "success": True,
            "mobile_optimized": True,
            "system_state": "operational" if base_status["system_ready"] else "degraded",
            "agent_count": base_status["agent_count"],
            "requires_attention": len(critical_alerts) > 0,
            "critical_alerts": critical_alerts,
            "quick_actions": [],
            "timestamp": base_status["timestamp"]
        }
        
        # Generate contextual quick actions
        if not base_status["platform_active"]:
            mobile_status["quick_actions"].append({
                "action": "start_platform",
                "command": "/hive:start",
                "description": "Start multi-agent platform"
            })
        elif base_status["agent_count"] < 3:
            mobile_status["quick_actions"].append({
                "action": "spawn_agents",
                "command": "/hive:spawn backend_developer",
                "description": "Add more agents to team"
            })
        else:
            mobile_status["quick_actions"].append({
                "action": "start_development",
                "command": "/hive:develop",
                "description": "Begin autonomous development"
            })
        
        return mobile_status
    
    async def _get_enhanced_system_health(self) -> Dict[str, Any]:
        """Get enhanced system health information."""
        health_info = {}
        
        try:
            import requests
            health_response = requests.get("http://localhost:8000/health", timeout=3)
            if health_response.status_code == 200:
                health_data = health_response.json()
                health_info["system_health"] = health_data.get("status", "unknown")
                health_info["components"] = health_data.get("components", {})
                health_info["response_time_ms"] = health_response.elapsed.total_seconds() * 1000
            else:
                health_info["system_health"] = "degraded"
                health_info["health_status_code"] = health_response.status_code
        except Exception as e:
            health_info["system_health"] = "unknown"
            health_info["health_error"] = str(e)
        
        return health_info
    
    async def _analyze_team_capabilities(self, spawner_agents: dict, orchestrator_agents: dict) -> Dict[str, Any]:
        """Analyze team capabilities and provide recommendations."""
        all_agents = (spawner_agents or {}).copy()
        all_agents.update(orchestrator_agents or {})
        
        role_distribution = {}
        total_capabilities = set()
        
        for agent_data in all_agents.values():
            role = agent_data.get("role", "unknown")
            capabilities = agent_data.get("capabilities", [])
            
            if role not in role_distribution:
                role_distribution[role] = {
                    "count": 0,
                    "capabilities": set()
                }
            
            role_distribution[role]["count"] += 1
            role_distribution[role]["capabilities"].update(capabilities)
            total_capabilities.update(capabilities)
        
        # Convert sets to lists for JSON serialization
        for role_info in role_distribution.values():
            role_info["capabilities"] = list(role_info["capabilities"])
        
        # Generate recommendations
        recommendations = []
        essential_roles = ["product_manager", "backend_developer", "frontend_developer"]
        missing_roles = [role for role in essential_roles if role not in role_distribution]
        
        if missing_roles:
            recommendations.append({
                "type": "missing_roles",
                "message": f"Consider adding: {', '.join(missing_roles)}",
                "action": f"Run /hive:spawn {missing_roles[0]} to add missing capability"
            })
        
        return {
            "role_distribution": role_distribution,
            "total_capabilities": list(total_capabilities),
            "team_size": len(all_agents),
            "recommendations": recommendations,
            "team_readiness": len(all_agents) >= 3 and len(missing_roles) == 0
        }
    
    def _generate_alert_summary(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate summary of alerts by priority."""
        summary = {"critical": 0, "high": 0, "medium": 0, "info": 0}
        for alert in alerts:
            priority = alert.get("priority", "info")
            summary[priority] = summary.get(priority, 0) + 1
        return summary


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


class HiveFocusCommand(HiveSlashCommand):
    """Context-aware command that provides intelligent recommendations based on current system state."""
    
    def __init__(self):
        super().__init__(
            name="focus",
            description="Get context-aware recommendations and intelligent next steps",
            usage="/hive:focus [area] [--mobile] [--priority=critical|high]"
        )
    
    async def execute(self, args: List[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Provide intelligent, context-aware recommendations."""
        try:
            logger.info("🎯 Executing /hive:focus command")
            
            focus_area = args[0] if args and not args[0].startswith("--") else None
            mobile_mode = "--mobile" in (args or [])
            
            priority_filter = None
            for arg in (args or []):
                if arg.startswith("--priority="):
                    priority_filter = arg.split("=")[1].lower()
            
            # Get current system state
            status_command = HiveStatusCommand()
            current_status = await status_command.execute(["--alerts-only"])
            
            # Generate context-aware recommendations
            recommendations = await self._generate_contextual_recommendations(
                current_status, focus_area, priority_filter
            )
            
            result = {
                "success": True,
                "focus_area": focus_area or "general",
                "recommendations": recommendations,
                "context": {
                    "system_state": current_status.get("alert_summary", {}),
                    "requires_immediate_action": current_status.get("requires_action", False)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if mobile_mode:
                result = await self._optimize_for_mobile(result)
            
            return result
            
        except Exception as e:
            logger.error("Failed to execute /hive:focus", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate contextual recommendations"
            }
    
    async def _generate_contextual_recommendations(self, status: Dict[str, Any], focus_area: str, priority_filter: str) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations based on system state."""
        recommendations = []
        alerts = status.get("alerts", [])
        
        # System state analysis
        if status.get("requires_action", False):
            critical_alerts = [a for a in alerts if a["priority"] == "critical"]
            if critical_alerts:
                recommendations.extend([{
                    "priority": "critical",
                    "category": "system_health",
                    "title": "Critical System Issues Detected",
                    "description": alert["message"],
                    "action": alert["action"],
                    "command": alert.get("command", ""),
                    "estimated_time": "2-5 minutes"
                } for alert in critical_alerts[:2]])  # Limit to top 2
        
        # Focus area specific recommendations
        if focus_area:
            area_recommendations = await self._get_area_specific_recommendations(focus_area)
            recommendations.extend(area_recommendations)
        else:
            # General workflow recommendations
            recommendations.extend(await self._get_general_workflow_recommendations())
        
        # Filter by priority if specified
        if priority_filter:
            priority_levels = {"critical": ["critical"], "high": ["critical", "high"]}
            allowed = priority_levels.get(priority_filter, ["critical", "high", "medium", "info"])
            recommendations = [r for r in recommendations if r["priority"] in allowed]
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    async def _get_area_specific_recommendations(self, area: str) -> List[Dict[str, Any]]:
        """Get recommendations specific to a focus area."""
        area_recommendations = {
            "development": [
                {
                    "priority": "high",
                    "category": "development",
                    "title": "Start Autonomous Development",
                    "description": "Begin AI-powered development with multi-agent coordination",
                    "action": "Use /hive:develop with your project description",
                    "command": "/hive:develop \"Your project description here\"",
                    "estimated_time": "5-30 minutes"
                }
            ],
            "monitoring": [
                {
                    "priority": "medium",
                    "category": "monitoring", 
                    "title": "Open Remote Oversight Dashboard",
                    "description": "Monitor agent activities and system health in real-time",
                    "action": "Launch the oversight dashboard",
                    "command": "/hive:oversight --mobile-info",
                    "estimated_time": "< 1 minute"
                }
            ],
            "performance": [
                {
                    "priority": "medium",
                    "category": "performance",
                    "title": "System Performance Analysis",
                    "description": "Get detailed performance metrics and optimization suggestions",
                    "action": "Run detailed status check",
                    "command": "/hive:status --detailed",
                    "estimated_time": "1-2 minutes"
                }
            ]
        }
        
        return area_recommendations.get(area.lower(), [])
    
    async def _get_general_workflow_recommendations(self) -> List[Dict[str, Any]]:
        """Get general workflow recommendations."""
        return [
            {
                "priority": "high",
                "category": "getting_started",
                "title": "Quick Platform Status Check",
                "description": "Verify system health and agent readiness",
                "action": "Check current platform status",
                "command": "/hive:status --mobile",
                "estimated_time": "< 30 seconds"
            },
            {
                "priority": "medium", 
                "category": "productivity",
                "title": "Start Development Session",
                "description": "Begin autonomous development with full team coordination",
                "action": "Describe what you want to build",
                "command": "/hive:develop \"Describe your project\"",
                "estimated_time": "5-60 minutes"
            }
        ]
    
    async def _optimize_for_mobile(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize recommendations for mobile interface."""
        recommendations = result.get("recommendations", [])
        
        # Mobile-optimized format
        mobile_result = {
            "success": True,
            "mobile_optimized": True,
            "summary": {
                "total_recommendations": len(recommendations),
                "critical_actions": len([r for r in recommendations if r["priority"] == "critical"]),
                "estimated_total_time": "5-10 minutes"
            },
            "quick_actions": [],
            "detailed_recommendations": recommendations[:3],  # Limit for mobile
            "timestamp": result["timestamp"]
        }
        
        # Convert to quick actions for mobile
        for rec in recommendations[:3]:
            mobile_result["quick_actions"].append({
                "title": rec["title"],
                "priority": rec["priority"],
                "command": rec.get("command", ""),
                "time": rec.get("estimated_time", "< 5 min")
            })
        
        return mobile_result


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
            HiveFocusCommand(),
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
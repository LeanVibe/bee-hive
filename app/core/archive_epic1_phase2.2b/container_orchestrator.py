"""
Container-based Agent Orchestrator

⚠️ DEPRECATED: This module has been consolidated into SpecializedOrchestratorPlugin.
   New code should use: from .specialized_orchestrator_plugin import ContainerManagementModule
   This file will be removed in a future release after migration completion.

Replaces tmux-based agent management with Docker containers.
Provides scalable, production-ready agent lifecycle management.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
import structlog

# Use TYPE_CHECKING to avoid runtime import issues
if TYPE_CHECKING:
    import docker
    from docker.models.containers import Container
else:
    try:
        import docker
        from docker.models.containers import Container
    except ImportError:
        # Graceful fallback if docker is not available
        docker = None
        Container = None

from .config import settings
from .redis import get_message_broker
from ..models.agent import Agent, AgentStatus, AgentType

logger = structlog.get_logger()


@dataclass
class ContainerAgentSpec:
    """Specification for containerized agent deployment."""
    agent_type: str
    image_name: str
    resource_limits: Dict[str, str]
    environment_vars: Dict[str, str]
    volumes: Dict[str, Dict[str, str]]
    health_check_path: str = "/health"
    restart_policy: str = "unless-stopped"


class ContainerAgentOrchestrator:
    """
    Container-based orchestrator for managing Claude agents.
    
    Replaces tmux session management with Docker containers for:
    - Production scalability (50+ concurrent agents)
    - Security isolation 
    - Kubernetes deployment capability
    - Auto-scaling and resource management
    """
    
    def __init__(self):
        if docker is None:
            raise ImportError(
                "Docker package not available. Install with: pip install docker"
            )
        self.docker_client = docker.from_env()
        self.running_agents: Dict[str, Any] = {}  # Use Any instead of docker.models.containers.Container
        self.agent_specs = self._initialize_agent_specs()
        self.network_name = "leanvibe_network"
        
    def _initialize_agent_specs(self) -> Dict[str, ContainerAgentSpec]:
        """Initialize container specifications for each agent type."""
        base_env = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            "DATABASE_URL": settings.database_url,
            "REDIS_URL": settings.redis_url,
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development")
        }
        
        base_volumes = {
            "agent_workspace": {"bind": "/app/workspace", "mode": "rw"},
            ".": {"bind": "/app/src", "mode": "ro"}  # Source code read-only
        }
        
        return {
            "architect": ContainerAgentSpec(
                agent_type="architect",
                image_name="leanvibe/agent-architect:latest",
                resource_limits={"memory": "1g", "cpu_period": 100000, "cpu_quota": 50000},
                environment_vars={**base_env, "AGENT_TYPE": "architect"},
                volumes=base_volumes
            ),
            "developer": ContainerAgentSpec(
                agent_type="developer", 
                image_name="leanvibe/agent-developer:latest",
                resource_limits={"memory": "1g", "cpu_period": 100000, "cpu_quota": 75000},
                environment_vars={**base_env, "AGENT_TYPE": "developer"},
                volumes=base_volumes
            ),
            "qa": ContainerAgentSpec(
                agent_type="qa",
                image_name="leanvibe/agent-qa:latest", 
                resource_limits={"memory": "512m", "cpu_period": 100000, "cpu_quota": 50000},
                environment_vars={**base_env, "AGENT_TYPE": "qa"},
                volumes=base_volumes
            ),
            "meta": ContainerAgentSpec(
                agent_type="meta",
                image_name="leanvibe/agent-meta:latest",
                resource_limits={"memory": "1g", "cpu_period": 100000, "cpu_quota": 50000},
                environment_vars={**base_env, "AGENT_TYPE": "meta"},
                volumes=base_volumes
            )
        }
    
    async def spawn_agent(self, agent_type: str, agent_id: Optional[str] = None) -> str:
        """
        Spawn new agent in Docker container.
        
        Replaces tmux new-session with docker container creation.
        Target: <10 second spawn time (PRD requirement).
        """
        start_time = datetime.now()
        
        if agent_type not in self.agent_specs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if agent_id is None:
            agent_id = f"{agent_type}-{uuid.uuid4().hex[:8]}"
        
        spec = self.agent_specs[agent_type]
        
        try:
            # Create container with proper configuration
            container = await asyncio.get_event_loop().run_in_executor(
                None, self._create_container, agent_id, spec
            )
            
            # Store container reference
            self.running_agents[agent_id] = container
            
            # Wait for container to be ready (health check)
            await self._wait_for_agent_ready(agent_id, timeout=30)
            
            spawn_time = (datetime.now() - start_time).total_seconds()
            logger.info(
                "Agent spawned successfully",
                agent_id=agent_id,
                agent_type=agent_type,
                spawn_time_seconds=spawn_time,
                container_id=container.short_id
            )
            
            # Validate spawn time requirement
            if spawn_time > 10:
                logger.warning(
                    "Agent spawn time exceeded target",
                    agent_id=agent_id,
                    spawn_time_seconds=spawn_time,
                    target_seconds=10
                )
            
            return agent_id
            
        except Exception as e:
            logger.error(
                "Failed to spawn agent",
                agent_id=agent_id,
                agent_type=agent_type,
                error=str(e)
            )
            # Cleanup on failure
            if agent_id in self.running_agents:
                await self.stop_agent(agent_id)
            raise
    
    def _create_container(self, agent_id: str, spec: ContainerAgentSpec) -> Any:
        """Create Docker container with specified configuration."""
        return self.docker_client.containers.run(
            image=spec.image_name,
            name=agent_id,
            environment=spec.environment_vars,
            volumes=spec.volumes,
            network=self.network_name,
            detach=True,
            auto_remove=False,  # Keep for debugging and logs
            restart_policy={"Name": spec.restart_policy},
            mem_limit=spec.resource_limits.get("memory", "512m"),
            cpu_period=spec.resource_limits.get("cpu_period", 100000),
            cpu_quota=spec.resource_limits.get("cpu_quota", 50000),
            labels={
                "leanvibe.agent.id": agent_id,
                "leanvibe.agent.type": spec.agent_type,
                "leanvibe.component": "agent"
            }
        )
    
    async def _wait_for_agent_ready(self, agent_id: str, timeout: int = 30):
        """Wait for agent container to be ready (replaces tmux session check)."""
        container = self.running_agents.get(agent_id)
        if not container:
            raise ValueError(f"Agent {agent_id} not found")
        
        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            try:
                container.reload()
                if container.status == "running":
                    # Check if agent is responsive via health check
                    # This replaces checking if tmux session is active
                    logs = container.logs(tail=10).decode('utf-8')
                    if "Agent started successfully" in logs or "Ready to process tasks" in logs:
                        return
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(
                    "Health check failed during agent startup",
                    agent_id=agent_id,
                    error=str(e)
                )
                await asyncio.sleep(1)
        
        raise TimeoutError(f"Agent {agent_id} failed to become ready within {timeout} seconds")
    
    async def stop_agent(self, agent_id: str, graceful_timeout: int = 10):
        """
        Gracefully stop agent container.
        
        Replaces tmux kill-session with container stop.
        """
        if agent_id not in self.running_agents:
            logger.warning("Agent not found for stopping", agent_id=agent_id)
            return
        
        container = self.running_agents[agent_id]
        
        try:
            # Send graceful shutdown signal
            await asyncio.get_event_loop().run_in_executor(
                None, container.stop, graceful_timeout
            )
            
            # Remove container
            await asyncio.get_event_loop().run_in_executor(
                None, container.remove
            )
            
            logger.info("Agent stopped successfully", agent_id=agent_id)
            
        except Exception as e:
            logger.error(
                "Failed to stop agent gracefully, forcing removal",
                agent_id=agent_id,
                error=str(e)
            )
            # Force removal if graceful stop fails
            try:
                container.remove(force=True)
            except:
                pass
        
        finally:
            # Remove from tracking
            del self.running_agents[agent_id]
    
    async def get_agent_logs(self, agent_id: str, tail: int = 100) -> str:
        """
        Get agent container logs.
        
        Replaces tmux capture-pane for log access.
        """
        if agent_id not in self.running_agents:
            return f"Agent {agent_id} not found"
        
        container = self.running_agents[agent_id]
        try:
            logs = await asyncio.get_event_loop().run_in_executor(
                None, lambda: container.logs(tail=tail).decode('utf-8')
            )
            return logs
        except Exception as e:
            return f"Failed to get logs for {agent_id}: {str(e)}"
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get comprehensive agent status.
        
        Replaces tmux list-sessions for agent monitoring.
        """
        if agent_id not in self.running_agents:
            return {"status": "not_found", "error": f"Agent {agent_id} not found"}
        
        container = self.running_agents[agent_id]
        
        try:
            container.reload()
            stats = container.stats(stream=False)
            
            # Calculate resource usage
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            return {
                "status": container.status,
                "container_id": container.short_id,
                "created": container.attrs['Created'],
                "started_at": container.attrs['State'].get('StartedAt'),
                "resource_usage": {
                    "cpu_percent": cpu_usage,
                    "memory_usage_mb": memory_usage / (1024 * 1024),
                    "memory_percent": memory_percent
                },
                "health": self._get_health_status(container)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_count = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
                return (cpu_delta / system_delta) * cpu_count * 100.0
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
    
    def _get_health_status(self, container) -> str:
        """Get container health status."""
        try:
            health = container.attrs['State'].get('Health', {})
            return health.get('Status', 'unknown')
        except:
            return 'unknown'
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all running agents.
        
        Replaces tmux list-sessions for agent inventory.
        """
        agents = []
        for agent_id in list(self.running_agents.keys()):  # Copy keys to avoid modification during iteration
            status = await self.get_agent_status(agent_id)
            agents.append({
                "agent_id": agent_id,
                "agent_type": agent_id.split('-')[0],
                **status
            })
        return agents
    
    async def scale_agents(self, agent_type: str, target_count: int) -> List[str]:
        """
        Scale agents of specified type to target count.
        
        Supports production scaling requirements (50+ concurrent agents).
        """
        current_agents = [
            agent_id for agent_id in self.running_agents.keys()
            if agent_id.startswith(agent_type)
        ]
        
        current_count = len(current_agents)
        
        if target_count > current_count:
            # Scale up
            new_agents = []
            for _ in range(target_count - current_count):
                agent_id = await self.spawn_agent(agent_type)
                new_agents.append(agent_id)
            
            logger.info(
                "Scaled up agents",
                agent_type=agent_type,
                from_count=current_count,
                to_count=target_count,
                new_agents=new_agents
            )
            return new_agents
            
        elif target_count < current_count:
            # Scale down
            agents_to_stop = current_agents[target_count:]
            for agent_id in agents_to_stop:
                await self.stop_agent(agent_id)
            
            logger.info(
                "Scaled down agents",
                agent_type=agent_type,
                from_count=current_count,
                to_count=target_count,
                stopped_agents=agents_to_stop
            )
            return []
        
        return current_agents
    
    async def cleanup_failed_agents(self):
        """Clean up failed or unhealthy agent containers."""
        failed_agents = []
        
        for agent_id in list(self.running_agents.keys()):
            container = self.running_agents[agent_id]
            try:
                container.reload()
                if container.status in ['exited', 'dead', 'oom']:
                    failed_agents.append(agent_id)
                    await self.stop_agent(agent_id)
            except docker.errors.NotFound:
                # Container was removed externally
                del self.running_agents[agent_id]
                failed_agents.append(agent_id)
            except Exception as e:
                logger.error(
                    "Error checking agent health",
                    agent_id=agent_id,
                    error=str(e)
                )
        
        if failed_agents:
            logger.info(
                "Cleaned up failed agents",
                failed_agents=failed_agents,
                count=len(failed_agents)
            )
        
        return failed_agents
    
    async def shutdown_all(self):
        """Gracefully shutdown all agents."""
        logger.info("Shutting down all agents", count=len(self.running_agents))
        
        # Stop all agents concurrently
        shutdown_tasks = [
            self.stop_agent(agent_id)
            for agent_id in list(self.running_agents.keys())
        ]
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.running_agents.clear()
        logger.info("All agents shut down successfully")


# Global instance for use by FastAPI app
_container_orchestrator: Optional[ContainerAgentOrchestrator] = None


def get_container_orchestrator() -> ContainerAgentOrchestrator:
    """Get global container orchestrator instance."""
    global _container_orchestrator
    if _container_orchestrator is None:
        _container_orchestrator = ContainerAgentOrchestrator()
    return _container_orchestrator
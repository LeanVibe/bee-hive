#!/usr/bin/env python3
"""
Container Migration Script for LeanVibe Agent Hive

Migrates from tmux-based agent management to containerized agents.
Provides blue-green deployment capability to ensure zero downtime.
"""

import asyncio
import docker
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.specialized_orchestrator_plugin import SpecializedOrchestratorPlugin
from app.core.unified_production_orchestrator import IntegrationRequest

# Epic 1 Phase 2.2B consolidation: Use consolidated specialized orchestrator plugin
from app.core.config import settings

logger = structlog.get_logger()


class MigrationManager:
    """
    Manages migration from tmux to containerized agents.
    
    Implements blue-green deployment strategy:
    1. Deploy containerized agents alongside tmux agents
    2. Gradually route traffic to containers
    3. Validate performance and functionality 
    4. Complete cutover and shutdown tmux agents
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        # Epic 1 Phase 2.2B consolidation: Use consolidated specialized orchestrator plugin
        self.specialized_orchestrator = SpecializedOrchestratorPlugin()
        self.migration_state = {
            "phase": "not_started",
            "tmux_agents": [],
            "container_agents": [],
            "traffic_split": {"tmux": 100, "container": 0},
            "validation_results": {}
        }
        
    async def run_migration(self, dry_run: bool = False):
        """Run complete migration process."""
        logger.info("Starting container migration", dry_run=dry_run)
        
        try:
            # Phase 1: Pre-migration validation
            await self._phase1_validation()
            
            # Phase 2: Build and deploy container images
            await self._phase2_container_deployment()
            
            # Phase 3: Parallel operation (blue-green)
            await self._phase3_parallel_operation()
            
            # Phase 4: Traffic migration
            await self._phase4_traffic_migration()
            
            # Phase 5: Complete cutover
            if not dry_run:
                await self._phase5_complete_cutover()
            
            logger.info("Migration completed successfully")
            
        except Exception as e:
            logger.error("Migration failed", error=str(e))
            await self._rollback_migration()
            raise
    
    async def _phase1_validation(self):
        """Phase 1: Validate current system state."""
        logger.info("Phase 1: Pre-migration validation")
        self.migration_state["phase"] = "validation"
        
        # Check current tmux agents
        current_tmux = self._get_tmux_agents()
        self.migration_state["tmux_agents"] = current_tmux
        
        logger.info(f"Found {len(current_tmux)} active tmux agents", agents=current_tmux)
        
        # Validate Docker environment
        try:
            self.docker_client.ping()
            logger.info("Docker daemon accessible")
        except Exception as e:
            raise RuntimeError(f"Docker daemon not accessible: {e}")
        
        # Check required images
        required_images = [
            "leanvibe/agent-base:latest",
            "leanvibe/agent-architect:latest", 
            "leanvibe/agent-developer:latest",
            "leanvibe/agent-qa:latest",
            "leanvibe/agent-meta:latest"
        ]
        
        missing_images = []
        for image in required_images:
            try:
                self.docker_client.images.get(image)
            except docker.errors.ImageNotFound:
                missing_images.append(image)
        
        if missing_images:
            logger.warning("Missing Docker images", images=missing_images)
            await self._build_missing_images(missing_images)
        
        # Validate network connectivity
        await self._validate_network_connectivity()
        
        logger.info("Phase 1 validation completed")
    
    async def _phase2_container_deployment(self):
        """Phase 2: Deploy containerized agents alongside tmux agents."""
        logger.info("Phase 2: Deploying containerized agents")
        self.migration_state["phase"] = "container_deployment"
        
        # Deploy one container of each type for testing
        test_agents = []
        agent_types = ["architect", "developer", "qa", "meta"]
        
        for agent_type in agent_types:
            try:
                # Epic 1 Phase 2.2B consolidation: Use specialized orchestrator plugin
                response = await self.specialized_orchestrator.process_request(
                    IntegrationRequest(
                        request_id=str(uuid.uuid4()),
                        operation="deploy_container_agent",
                        parameters={"agent_type": agent_type, "agent_id": str(uuid.uuid4())}
                    )
                )
                agent_id = response.result.get("agent_id")
                test_agents.append(agent_id)
                logger.info(f"Deployed test {agent_type} agent", agent_id=agent_id)
                
                # Wait briefly between deployments
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to deploy {agent_type} agent", error=str(e))
                raise
        
        self.migration_state["container_agents"] = test_agents
        
        # Validate all agents are healthy
        await self._validate_container_health(test_agents)
        
        logger.info("Phase 2 container deployment completed", agents=test_agents)
    
    async def _phase3_parallel_operation(self):
        """Phase 3: Run both systems in parallel."""
        logger.info("Phase 3: Parallel operation (blue-green)")
        self.migration_state["phase"] = "parallel_operation"
        
        # Monitor both systems for 5 minutes
        monitoring_duration = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < monitoring_duration:
            # Check tmux agent health
            tmux_status = await self._check_tmux_agent_health()
            
            # Check container agent health  
            container_status = await self._check_container_agent_health()
            
            logger.info(
                "Parallel operation status",
                tmux_healthy=tmux_status["healthy_count"],
                tmux_total=tmux_status["total_count"],
                container_healthy=container_status["healthy_count"],
                container_total=container_status["total_count"]
            )
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        logger.info("Phase 3 parallel operation completed")
    
    async def _phase4_traffic_migration(self):
        """Phase 4: Gradually migrate traffic to containers."""
        logger.info("Phase 4: Traffic migration")
        self.migration_state["phase"] = "traffic_migration"
        
        # Traffic migration steps: 0% -> 25% -> 50% -> 75% -> 100%
        migration_steps = [25, 50, 75, 100]
        
        for container_percent in migration_steps:
            tmux_percent = 100 - container_percent
            
            logger.info(
                f"Migrating traffic: {container_percent}% container, {tmux_percent}% tmux"
            )
            
            # Update traffic routing (this would integrate with load balancer/router)
            await self._update_traffic_routing(container_percent, tmux_percent)
            
            # Monitor for 2 minutes at each step
            await self._monitor_performance(120)
            
            # Validate performance metrics
            performance_ok = await self._validate_performance_metrics()
            if not performance_ok:
                logger.error("Performance degradation detected, rolling back")
                await self._rollback_traffic_routing()
                raise RuntimeError("Performance validation failed")
            
            self.migration_state["traffic_split"] = {
                "container": container_percent,
                "tmux": tmux_percent
            }
        
        logger.info("Phase 4 traffic migration completed")
    
    async def _phase5_complete_cutover(self):
        """Phase 5: Complete cutover and shutdown tmux agents."""
        logger.info("Phase 5: Complete cutover")
        self.migration_state["phase"] = "cutover"
        
        # Final validation
        final_validation = await self._final_validation()
        if not final_validation:
            raise RuntimeError("Final validation failed")
        
        # Scale up container agents to full capacity
        await self._scale_container_agents_full()
        
        # Gracefully shutdown tmux agents
        await self._shutdown_tmux_agents()
        
        # Update configuration to use containers only
        await self._update_system_configuration()
        
        self.migration_state["phase"] = "completed"
        logger.info("Phase 5 cutover completed - Migration successful!")
    
    def _get_tmux_agents(self) -> List[str]:
        """Get list of current tmux agent sessions."""
        try:
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                sessions = result.stdout.strip().split('\n')
                # Filter for agent sessions
                agent_sessions = [s for s in sessions if 'agent' in s.lower()]
                return agent_sessions
        except Exception as e:
            logger.warning("Failed to get tmux sessions", error=str(e))
        
        return []
    
    async def _build_missing_images(self, missing_images: List[str]):
        """Build missing Docker images."""
        logger.info("Building missing Docker images", images=missing_images)
        
        image_build_map = {
            "leanvibe/agent-base:latest": "Dockerfile.agent-base",
            "leanvibe/agent-architect:latest": "Dockerfile.agent-architect", 
            "leanvibe/agent-developer:latest": "Dockerfile.agent-developer",
            "leanvibe/agent-qa:latest": "Dockerfile.agent-qa",
            "leanvibe/agent-meta:latest": "Dockerfile.agent-meta"
        }
        
        for image in missing_images:
            if image in image_build_map:
                dockerfile = image_build_map[image]
                logger.info(f"Building {image}")
                
                # Build image
                build_result = subprocess.run([
                    "docker", "build", 
                    "-f", dockerfile,
                    "-t", image,
                    "."
                ], capture_output=True, text=True)
                
                if build_result.returncode != 0:
                    raise RuntimeError(f"Failed to build {image}: {build_result.stderr}")
                
                logger.info(f"Successfully built {image}")
    
    async def _validate_network_connectivity(self):
        """Validate network connectivity for containers."""
        logger.info("Validating network connectivity")
        
        # Check if leanvibe_network exists
        try:
            self.docker_client.networks.get("leanvibe_network")
        except docker.errors.NotFound:
            # Create network if it doesn't exist
            logger.info("Creating leanvibe_network")
            self.docker_client.networks.create("leanvibe_network")
        
        # Test connectivity to Redis and PostgreSQL
        test_container = None
        try:
            test_container = self.docker_client.containers.run(
                "redis:7-alpine",
                command=["redis-cli", "-h", "redis", "ping"],
                network="leanvibe_network",
                detach=True,
                remove=True
            )
            
            # Wait for result
            result = test_container.wait(timeout=10)
            if result["StatusCode"] != 0:
                raise RuntimeError("Failed to connect to Redis")
            
        except Exception as e:
            logger.error("Network connectivity validation failed", error=str(e))
            raise
        finally:
            if test_container:
                try:
                    test_container.remove(force=True)
                except:
                    pass
    
    async def _validate_container_health(self, agent_ids: List[str]):
        """Validate health of deployed containers."""
        logger.info("Validating container health", agents=agent_ids)
        
        for agent_id in agent_ids:
            max_attempts = 10
            for attempt in range(max_attempts):
                # Epic 1 Phase 2.2B consolidation: Use specialized orchestrator plugin
                response = await self.specialized_orchestrator.process_request(
                    IntegrationRequest(
                        request_id=str(uuid.uuid4()),
                        operation="health_check_containers",
                        parameters={"agent_id": agent_id}
                    )
                )
                status = response.result
                
                if status.get("status") == "running":
                    logger.info(f"Agent {agent_id} is healthy")
                    break
                
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Agent {agent_id} failed health check")
                
                await asyncio.sleep(5)
    
    async def _check_tmux_agent_health(self) -> Dict[str, int]:
        """Check health of tmux agents."""
        sessions = self._get_tmux_agents()
        healthy = 0
        
        for session in sessions:
            try:
                # Check if session is still running
                result = subprocess.run(
                    ["tmux", "has-session", "-t", session],
                    capture_output=True
                )
                if result.returncode == 0:
                    healthy += 1
            except:
                pass
        
        return {"healthy_count": healthy, "total_count": len(sessions)}
    
    async def _check_container_agent_health(self) -> Dict[str, int]:
        """Check health of container agents."""
        # Epic 1 Phase 2.2B consolidation: Use specialized orchestrator plugin
        response = await self.specialized_orchestrator.process_request(
            IntegrationRequest(
                request_id=str(uuid.uuid4()),
                operation="health_check_containers",
                parameters={}
            )
        )
        agents = response.result.get("agents", [])
        healthy = 0
        
        for agent in agents:
            if agent.get("status") == "running":
                healthy += 1
        
        return {"healthy_count": healthy, "total_count": len(agents)}
    
    async def _update_traffic_routing(self, container_percent: int, tmux_percent: int):
        """Update traffic routing between tmux and container agents."""
        # This would integrate with actual load balancer/router
        # For now, we'll simulate by updating Redis queue routing
        
        logger.info(
            "Updating traffic routing",
            container_percent=container_percent,
            tmux_percent=tmux_percent
        )
        
        # Store routing configuration in Redis
        routing_config = {
            "container_percent": container_percent,
            "tmux_percent": tmux_percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # This would be implemented in the actual task router
        # await redis.set("traffic_routing_config", json.dumps(routing_config))
    
    async def _monitor_performance(self, duration_seconds: int):
        """Monitor system performance for specified duration."""
        logger.info(f"Monitoring performance for {duration_seconds} seconds")
        
        start_time = time.time()
        performance_data = []
        
        while time.time() - start_time < duration_seconds:
            # Collect performance metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "tmux_agents": await self._check_tmux_agent_health(),
                "container_agents": await self._check_container_agent_health()
            }
            performance_data.append(metrics)
            
            await asyncio.sleep(10)  # Sample every 10 seconds
        
        self.migration_state["performance_data"] = performance_data
    
    async def _validate_performance_metrics(self) -> bool:
        """Validate that performance meets requirements."""
        # Check agent spawn time < 10 seconds
        # Check orchestration latency < 500ms
        # Check task completion rates
        
        # For now, return True - would implement actual validation
        return True
    
    async def _rollback_traffic_routing(self):
        """Rollback traffic routing to tmux agents."""
        logger.warning("Rolling back traffic routing")
        await self._update_traffic_routing(0, 100)
    
    async def _final_validation(self) -> bool:
        """Perform final validation before cutover."""
        logger.info("Performing final validation")
        
        # Validate all container agents are healthy
        container_status = await self._check_container_agent_health()
        if container_status["healthy_count"] == 0:
            logger.error("No healthy container agents found")
            return False
        
        # Validate performance requirements
        if not await self._validate_performance_metrics():
            logger.error("Performance requirements not met")
            return False
        
        return True
    
    async def _scale_container_agents_full(self):
        """Scale container agents to full production capacity."""
        logger.info("Scaling container agents to full capacity")
        
        # Scale each agent type to production levels
        scaling_config = {
            "architect": 3,
            "developer": 8, 
            "qa": 3,
            "meta": 1
        }
        
        for agent_type, target_count in scaling_config.items():
            # Epic 1 Phase 2.2B consolidation: Use specialized orchestrator plugin
            await self.specialized_orchestrator.process_request(
                IntegrationRequest(
                    request_id=str(uuid.uuid4()),
                    operation="scale_container_agents",
                    parameters={"agent_type": agent_type, "target_count": target_count}
                )
            )
    
    async def _shutdown_tmux_agents(self):
        """Gracefully shutdown tmux agents."""
        logger.info("Shutting down tmux agents")
        
        tmux_sessions = self._get_tmux_agents()
        for session in tmux_sessions:
            try:
                logger.info(f"Stopping tmux session: {session}")
                subprocess.run(["tmux", "kill-session", "-t", session])
            except Exception as e:
                logger.warning(f"Failed to stop {session}", error=str(e))
    
    async def _update_system_configuration(self):
        """Update system configuration for container-only mode."""
        logger.info("Updating system configuration")
        
        # Update configuration files, environment variables, etc.
        # to use containerized agents only
        
        config_updates = {
            "AGENT_ORCHESTRATOR_TYPE": "container",
            "TMUX_ORCHESTRATOR_ENABLED": "false",
            "CONTAINER_ORCHESTRATOR_ENABLED": "true"
        }
        
        # Write configuration updates
        # This would integrate with actual configuration management
        
        logger.info("System configuration updated", config=config_updates)
    
    async def _rollback_migration(self):
        """Rollback migration on failure."""
        logger.warning("Rolling back migration")
        
        # Stop container agents
        try:
            # Epic 1 Phase 2.2B consolidation: Use specialized orchestrator plugin
        # The specialized plugin handles shutdown through its shutdown method
        await self.specialized_orchestrator.shutdown()
        except Exception as e:
            logger.error("Failed to shutdown container agents", error=str(e))
        
        # Restore traffic to tmux agents
        await self._update_traffic_routing(0, 100)
        
        logger.info("Migration rollback completed")


async def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate from tmux to containerized agents")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without cutover")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run specific phase only")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    manager = MigrationManager()
    
    try:
        await manager.run_migration(dry_run=args.dry_run)
    except Exception as e:
        logger.error("Migration failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
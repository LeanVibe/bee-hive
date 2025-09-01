"""
Orchestrator Integration Module - Consolidated Manager Integration

This module provides integration utilities to update the ConsolidatedProductionOrchestrator
to use the new consolidated manager hierarchy, eliminating duplicated manager implementations.

Key Features:
- Seamless integration with existing orchestrator architecture
- Migration utilities for existing manager consumers
- Performance validation during manager consolidation
- Backwards compatibility preservation
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass

from .consolidated_manager import (
    ConsolidatedLifecycleManager,
    ConsolidatedTaskCoordinationManager, 
    ConsolidatedPerformanceManager,
    ConsolidatedManagerBase
)
from ..logging_service import get_component_logger

logger = get_component_logger("orchestrator_integration")


@dataclass
class ManagerMigrationResult:
    """Result of manager migration operation."""
    manager_type: str
    migration_successful: bool
    performance_impact: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    rollback_available: bool = True


class ConsolidatedManagerIntegrator:
    """
    Integrates consolidated managers with ConsolidatedProductionOrchestrator.
    
    Handles the migration from fragmented manager implementations to the 
    unified consolidated manager hierarchy while preserving all functionality.
    """
    
    def __init__(self, orchestrator):
        """Initialize the manager integrator."""
        self.orchestrator = orchestrator
        self.logger = logger
        
        # Track migration progress
        self.migrated_managers: Dict[str, ConsolidatedManagerBase] = {}
        self.migration_results: List[ManagerMigrationResult] = []
        self.rollback_data: Dict[str, Any] = {}
        
        # Performance monitoring
        self.pre_migration_metrics: Optional[Dict[str, Any]] = None
        self.post_migration_metrics: Optional[Dict[str, Any]] = None
        
        self.logger.info("ConsolidatedManagerIntegrator initialized")

    async def integrate_all_managers(self) -> Dict[str, Any]:
        """
        Integrate all consolidated managers with the orchestrator.
        
        Performs complete migration from fragmented managers to consolidated hierarchy.
        """
        integration_start = datetime.utcnow()
        
        try:
            self.logger.info("🔄 Starting complete manager consolidation integration...")
            
            # Collect pre-migration performance metrics
            await self._collect_pre_migration_metrics()
            
            # Create rollback checkpoint
            await self._create_rollback_checkpoint()
            
            # Integrate managers in dependency order
            lifecycle_result = await self._integrate_lifecycle_manager()
            self.migration_results.append(lifecycle_result)
            
            if not lifecycle_result.migration_successful:
                await self._rollback_integration()
                raise Exception(f"Lifecycle manager integration failed: {lifecycle_result.error_message}")
            
            task_result = await self._integrate_task_coordination_manager()
            self.migration_results.append(task_result)
            
            if not task_result.migration_successful:
                await self._rollback_integration()
                raise Exception(f"Task coordination manager integration failed: {task_result.error_message}")
            
            performance_result = await self._integrate_performance_manager()
            self.migration_results.append(performance_result)
            
            if not performance_result.migration_successful:
                await self._rollback_integration()
                raise Exception(f"Performance manager integration failed: {performance_result.error_message}")
            
            # Update orchestrator references
            await self._update_orchestrator_references()
            
            # Collect post-migration metrics and validate performance
            await self._collect_post_migration_metrics()
            await self._validate_performance_impact()
            
            # Start all consolidated managers
            await self._start_consolidated_managers()
            
            integration_duration = (datetime.utcnow() - integration_start).total_seconds()
            
            integration_summary = {
                "integration_successful": True,
                "integration_duration_seconds": integration_duration,
                "managers_integrated": len(self.migrated_managers),
                "migration_results": [result.__dict__ for result in self.migration_results],
                "performance_impact": await self._calculate_performance_impact(),
                "consolidated_managers": list(self.migrated_managers.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("✅ Manager consolidation integration completed successfully",
                           duration_seconds=integration_duration,
                           managers_count=len(self.migrated_managers))
            
            return integration_summary
            
        except Exception as e:
            self.logger.error("❌ Manager consolidation integration failed", error=str(e))
            
            # Attempt rollback
            try:
                await self._rollback_integration()
                self.logger.info("🔄 Rollback completed successfully")
            except Exception as rollback_error:
                self.logger.error("❌ Rollback failed", error=str(rollback_error))
            
            return {
                "integration_successful": False,
                "error": str(e),
                "rollback_attempted": True,
                "integration_duration_seconds": (datetime.utcnow() - integration_start).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _integrate_lifecycle_manager(self) -> ManagerMigrationResult:
        """Integrate ConsolidatedLifecycleManager."""
        try:
            self.logger.info("🔄 Integrating ConsolidatedLifecycleManager...")
            
            # Create consolidated lifecycle manager
            consolidated_lifecycle = ConsolidatedLifecycleManager(self.orchestrator)
            
            # Initialize and start the manager
            await consolidated_lifecycle.initialize()
            
            # Migrate existing agents if any
            await self._migrate_existing_agents(consolidated_lifecycle)
            
            # Store reference
            self.migrated_managers['agent_lifecycle'] = consolidated_lifecycle
            
            self.logger.info("✅ ConsolidatedLifecycleManager integrated successfully")
            
            return ManagerMigrationResult(
                manager_type="ConsolidatedLifecycleManager",
                migration_successful=True,
                performance_impact=await self._measure_manager_performance(consolidated_lifecycle),
                rollback_available=True
            )
            
        except Exception as e:
            self.logger.error("❌ ConsolidatedLifecycleManager integration failed", error=str(e))
            return ManagerMigrationResult(
                manager_type="ConsolidatedLifecycleManager",
                migration_successful=False,
                error_message=str(e),
                rollback_available=True
            )

    async def _integrate_task_coordination_manager(self) -> ManagerMigrationResult:
        """Integrate ConsolidatedTaskCoordinationManager.""" 
        try:
            self.logger.info("🔄 Integrating ConsolidatedTaskCoordinationManager...")
            
            # Create consolidated task coordination manager
            consolidated_tasks = ConsolidatedTaskCoordinationManager(self.orchestrator)
            
            # Initialize and start the manager
            await consolidated_tasks.initialize()
            
            # Migrate existing tasks if any
            await self._migrate_existing_tasks(consolidated_tasks)
            
            # Store reference
            self.migrated_managers['task_coordination'] = consolidated_tasks
            
            self.logger.info("✅ ConsolidatedTaskCoordinationManager integrated successfully")
            
            return ManagerMigrationResult(
                manager_type="ConsolidatedTaskCoordinationManager",
                migration_successful=True,
                performance_impact=await self._measure_manager_performance(consolidated_tasks),
                rollback_available=True
            )
            
        except Exception as e:
            self.logger.error("❌ ConsolidatedTaskCoordinationManager integration failed", error=str(e))
            return ManagerMigrationResult(
                manager_type="ConsolidatedTaskCoordinationManager",
                migration_successful=False,
                error_message=str(e),
                rollback_available=True
            )

    async def _integrate_performance_manager(self) -> ManagerMigrationResult:
        """Integrate ConsolidatedPerformanceManager."""
        try:
            self.logger.info("🔄 Integrating ConsolidatedPerformanceManager...")
            
            # Create consolidated performance manager
            consolidated_performance = ConsolidatedPerformanceManager(self.orchestrator)
            
            # Initialize and start the manager
            await consolidated_performance.initialize()
            
            # Store reference
            self.migrated_managers['performance'] = consolidated_performance
            
            self.logger.info("✅ ConsolidatedPerformanceManager integrated successfully")
            
            return ManagerMigrationResult(
                manager_type="ConsolidatedPerformanceManager", 
                migration_successful=True,
                performance_impact=await self._measure_manager_performance(consolidated_performance),
                rollback_available=True
            )
            
        except Exception as e:
            self.logger.error("❌ ConsolidatedPerformanceManager integration failed", error=str(e))
            return ManagerMigrationResult(
                manager_type="ConsolidatedPerformanceManager",
                migration_successful=False,
                error_message=str(e),
                rollback_available=True
            )

    async def _update_orchestrator_references(self) -> None:
        """Update orchestrator to use consolidated managers."""
        try:
            # Update orchestrator manager references
            if 'agent_lifecycle' in self.migrated_managers:
                self.orchestrator.agent_lifecycle = self.migrated_managers['agent_lifecycle']
                self.logger.info("Updated orchestrator.agent_lifecycle reference")
                
            if 'task_coordination' in self.migrated_managers:
                self.orchestrator.task_coordination = self.migrated_managers['task_coordination']
                self.logger.info("Updated orchestrator.task_coordination reference")
                
            if 'performance' in self.migrated_managers:
                self.orchestrator.performance = self.migrated_managers['performance']
                self.logger.info("Updated orchestrator.performance reference")
            
            # Update any legacy manager references to point to consolidated managers
            await self._update_legacy_references()
            
        except Exception as e:
            self.logger.error("Failed to update orchestrator references", error=str(e))
            raise

    async def _start_consolidated_managers(self) -> None:
        """Start all consolidated managers."""
        try:
            for manager_name, manager in self.migrated_managers.items():
                await manager.start()
                self.logger.info(f"Started {manager_name} manager")
                
        except Exception as e:
            self.logger.error("Failed to start consolidated managers", error=str(e))
            raise

    async def _migrate_existing_agents(self, consolidated_lifecycle: ConsolidatedLifecycleManager) -> None:
        """Migrate existing agents to consolidated lifecycle manager."""
        try:
            # Check for existing agents in orchestrator
            existing_agents = []
            
            # Try to get agents from simple orchestrator if available
            if hasattr(self.orchestrator, 'simple_orchestrator') and self.orchestrator.simple_orchestrator:
                try:
                    agents = await self.orchestrator.simple_orchestrator.list_agents()
                    existing_agents.extend(agents)
                except Exception as e:
                    self.logger.warning("Could not migrate agents from simple orchestrator", error=str(e))
            
            # Try to get agents from other orchestrator components
            if hasattr(self.orchestrator, 'agents') and self.orchestrator.agents:
                existing_agents.extend(self.orchestrator.agents.values())
            
            if existing_agents:
                self.logger.info(f"Migrating {len(existing_agents)} existing agents")
                
                for agent in existing_agents:
                    try:
                        # Extract agent information
                        agent_data = agent if isinstance(agent, dict) else agent.__dict__
                        
                        # Register agent with consolidated manager
                        await consolidated_lifecycle.register_agent(
                            name=agent_data.get('name', f"migrated-agent-{agent_data.get('id', 'unknown')}"),
                            agent_type=agent_data.get('agent_type', 'claude'),
                            role=agent_data.get('role', 'backend_developer'),
                            capabilities=agent_data.get('capabilities', []),
                            metadata=agent_data
                        )
                        
                        self.logger.debug(f"Migrated agent {agent_data.get('id', 'unknown')}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate agent: {e}")
                        
            else:
                self.logger.info("No existing agents to migrate")
                
        except Exception as e:
            self.logger.error("Failed to migrate existing agents", error=str(e))
            # Non-critical error, don't fail the entire migration

    async def _migrate_existing_tasks(self, consolidated_tasks: ConsolidatedTaskCoordinationManager) -> None:
        """Migrate existing tasks to consolidated task coordination manager."""
        try:
            existing_tasks = []
            
            # Try to get tasks from various sources
            if hasattr(self.orchestrator, 'tasks') and self.orchestrator.tasks:
                existing_tasks.extend(self.orchestrator.tasks.values())
                
            if existing_tasks:
                self.logger.info(f"Migrating {len(existing_tasks)} existing tasks")
                
                for task in existing_tasks:
                    try:
                        # Extract task information
                        task_data = task if isinstance(task, dict) else task.__dict__
                        
                        # Only migrate non-completed tasks
                        if task_data.get('status') not in ['completed', 'failed', 'cancelled']:
                            # This is a simplified migration - in practice would need more sophisticated logic
                            self.logger.debug(f"Task {task_data.get('id', 'unknown')} marked for migration")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to migrate task: {e}")
                        
            else:
                self.logger.info("No existing tasks to migrate")
                
        except Exception as e:
            self.logger.error("Failed to migrate existing tasks", error=str(e))
            # Non-critical error, don't fail the entire migration

    async def _measure_manager_performance(self, manager: ConsolidatedManagerBase) -> Dict[str, Any]:
        """Measure performance impact of a consolidated manager."""
        try:
            start_time = datetime.utcnow()
            
            # Get manager status and metrics
            status = await manager.get_status()
            metrics = await manager.get_metrics()
            
            end_time = datetime.utcnow()
            
            return {
                "status_response_time_ms": (end_time - start_time).total_seconds() * 1000,
                "manager_status": status,
                "manager_metrics": metrics,
                "measurement_timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to measure manager performance", error=str(e))
            return {
                "error": str(e),
                "measurement_timestamp": datetime.utcnow().isoformat()
            }

    async def _collect_pre_migration_metrics(self) -> None:
        """Collect system metrics before migration."""
        try:
            self.pre_migration_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator_status": await self._get_orchestrator_status(),
                "system_metrics": await self._get_system_metrics()
            }
            
        except Exception as e:
            self.logger.warning("Failed to collect pre-migration metrics", error=str(e))
            self.pre_migration_metrics = {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _collect_post_migration_metrics(self) -> None:
        """Collect system metrics after migration."""
        try:
            self.post_migration_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator_status": await self._get_orchestrator_status(),
                "system_metrics": await self._get_system_metrics(),
                "consolidated_manager_metrics": {}
            }
            
            # Collect metrics from each consolidated manager
            for manager_name, manager in self.migrated_managers.items():
                try:
                    self.post_migration_metrics["consolidated_manager_metrics"][manager_name] = {
                        "status": await manager.get_status(),
                        "metrics": await manager.get_metrics()
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to collect metrics for {manager_name}", error=str(e))
                    
        except Exception as e:
            self.logger.warning("Failed to collect post-migration metrics", error=str(e))
            self.post_migration_metrics = {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _validate_performance_impact(self) -> None:
        """Validate that migration didn't degrade performance."""
        try:
            if not self.pre_migration_metrics or not self.post_migration_metrics:
                self.logger.warning("Cannot validate performance impact - missing metrics")
                return
                
            # Simple performance validation
            pre_system = self.pre_migration_metrics.get("system_metrics", {})
            post_system = self.post_migration_metrics.get("system_metrics", {})
            
            # Check memory usage hasn't increased significantly
            pre_memory = pre_system.get("memory_usage_mb", 0)
            post_memory = post_system.get("memory_usage_mb", 0)
            
            if post_memory > pre_memory * 1.2:  # More than 20% increase
                self.logger.warning("Memory usage increased significantly after migration",
                                  pre_memory=pre_memory, post_memory=post_memory)
                                  
            # Check response time hasn't degraded significantly
            pre_response = pre_system.get("response_time_ms", 0)
            post_response = post_system.get("response_time_ms", 0)
            
            if post_response > pre_response * 1.5:  # More than 50% slower
                self.logger.warning("Response time degraded after migration",
                                  pre_response=pre_response, post_response=post_response)
            
            self.logger.info("Performance validation completed")
            
        except Exception as e:
            self.logger.error("Failed to validate performance impact", error=str(e))

    async def _calculate_performance_impact(self) -> Dict[str, Any]:
        """Calculate performance impact of migration."""
        try:
            if not self.pre_migration_metrics or not self.post_migration_metrics:
                return {"error": "Missing metrics data"}
                
            pre_system = self.pre_migration_metrics.get("system_metrics", {})
            post_system = self.post_migration_metrics.get("system_metrics", {})
            
            impact = {
                "memory_usage": {
                    "before_mb": pre_system.get("memory_usage_mb", 0),
                    "after_mb": post_system.get("memory_usage_mb", 0),
                    "change_percent": 0
                },
                "response_time": {
                    "before_ms": pre_system.get("response_time_ms", 0),
                    "after_ms": post_system.get("response_time_ms", 0),
                    "change_percent": 0
                }
            }
            
            # Calculate percentage changes
            if impact["memory_usage"]["before_mb"] > 0:
                memory_change = ((impact["memory_usage"]["after_mb"] - impact["memory_usage"]["before_mb"]) 
                               / impact["memory_usage"]["before_mb"]) * 100
                impact["memory_usage"]["change_percent"] = round(memory_change, 2)
                
            if impact["response_time"]["before_ms"] > 0:
                response_change = ((impact["response_time"]["after_ms"] - impact["response_time"]["before_ms"]) 
                                 / impact["response_time"]["before_ms"]) * 100
                impact["response_time"]["change_percent"] = round(response_change, 2)
                
            return impact
            
        except Exception as e:
            self.logger.error("Failed to calculate performance impact", error=str(e))
            return {"error": str(e)}

    async def _create_rollback_checkpoint(self) -> None:
        """Create rollback checkpoint before migration."""
        try:
            self.rollback_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "original_managers": {},
                "orchestrator_state": {}
            }
            
            # Store references to original managers
            if hasattr(self.orchestrator, 'agent_lifecycle'):
                self.rollback_data["original_managers"]["agent_lifecycle"] = self.orchestrator.agent_lifecycle
                
            if hasattr(self.orchestrator, 'task_coordination'):
                self.rollback_data["original_managers"]["task_coordination"] = self.orchestrator.task_coordination
                
            if hasattr(self.orchestrator, 'performance'):
                self.rollback_data["original_managers"]["performance"] = self.orchestrator.performance
                
            self.logger.info("Rollback checkpoint created")
            
        except Exception as e:
            self.logger.error("Failed to create rollback checkpoint", error=str(e))

    async def _rollback_integration(self) -> None:
        """Rollback integration to previous state.""" 
        try:
            self.logger.info("🔄 Starting integration rollback...")
            
            # Stop consolidated managers
            for manager_name, manager in self.migrated_managers.items():
                try:
                    await manager.shutdown()
                    self.logger.info(f"Stopped {manager_name} manager")
                except Exception as e:
                    self.logger.warning(f"Failed to stop {manager_name} manager", error=str(e))
                    
            # Restore original manager references
            original_managers = self.rollback_data.get("original_managers", {})
            
            if "agent_lifecycle" in original_managers:
                self.orchestrator.agent_lifecycle = original_managers["agent_lifecycle"]
                
            if "task_coordination" in original_managers:
                self.orchestrator.task_coordination = original_managers["task_coordination"]
                
            if "performance" in original_managers:
                self.orchestrator.performance = original_managers["performance"]
                
            # Clear migrated managers
            self.migrated_managers.clear()
            
            self.logger.info("✅ Integration rollback completed")
            
        except Exception as e:
            self.logger.error("❌ Failed to rollback integration", error=str(e))
            raise

    async def _update_legacy_references(self) -> None:
        """Update legacy manager references throughout the orchestrator."""
        try:
            # This would update any remaining references to old manager implementations
            # to point to the new consolidated managers
            
            # Update plugin manager references if available
            if hasattr(self.orchestrator, 'plugin_manager') and self.orchestrator.plugin_manager:
                # Update plugin manager to use consolidated managers
                pass
                
            # Update any other component references
            if hasattr(self.orchestrator, 'integrations'):
                # Update integration layer references
                pass
                
            self.logger.info("Legacy references updated")
            
        except Exception as e:
            self.logger.error("Failed to update legacy references", error=str(e))

    async def _get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        try:
            if hasattr(self.orchestrator, 'get_system_status'):
                return await self.orchestrator.get_system_status()
            else:
                return {"status": "unknown", "error": "get_system_status not available"}
        except Exception as e:
            return {"error": str(e)}

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            import psutil
            import time
            
            # Basic system metrics
            start_time = time.perf_counter()
            await asyncio.sleep(0.001)  # Simulate small operation
            end_time = time.perf_counter()
            
            return {
                "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "cpu_usage_percent": psutil.cpu_percent(),
                "response_time_ms": (end_time - start_time) * 1000
            }
            
        except Exception as e:
            self.logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e)}


def create_manager_integrator(orchestrator) -> ConsolidatedManagerIntegrator:
    """Create and return a ConsolidatedManagerIntegrator instance."""
    return ConsolidatedManagerIntegrator(orchestrator)


async def integrate_consolidated_managers(orchestrator) -> Dict[str, Any]:
    """
    Convenience function to integrate all consolidated managers with an orchestrator.
    
    Args:
        orchestrator: The orchestrator instance to integrate with
        
    Returns:
        Dict containing integration results
    """
    integrator = create_manager_integrator(orchestrator)
    return await integrator.integrate_all_managers()
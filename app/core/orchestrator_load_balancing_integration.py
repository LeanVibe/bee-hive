"""
Load Balancing Integration for Enhanced Agent Orchestrator

Extends the existing orchestrator with advanced load balancing, capacity management,
and health monitoring capabilities for production-grade multi-agent orchestration.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import uuid

import structlog

from .agent_load_balancer import AgentLoadBalancer, LoadBalancingStrategy, LoadBalancingDecision
from .capacity_manager import CapacityManager, CapacityTier
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType
from .adaptive_scaler import AdaptiveScaler
from .resource_optimizer import ResourceOptimizer
from .health_monitor import HealthMonitor, HealthStatus
from .redis import get_message_broker, get_session_cache
from .database import get_session
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskPriority
from ..models.agent_performance import WorkloadSnapshot

logger = structlog.get_logger()


class LoadBalancingOrchestrator:
    """
    Enhanced orchestrator with integrated load balancing and capacity management.
    
    This class extends the base orchestrator functionality with:
    - Intelligent load balancing for task distribution
    - Dynamic capacity management and auto-scaling
    - Comprehensive performance monitoring
    - Health monitoring with automated recovery
    - Resource optimization for efficiency
    """
    
    def __init__(self, base_orchestrator):
        """Initialize with existing orchestrator instance."""
        self.base_orchestrator = base_orchestrator
        
        # Initialize load balancing components
        self._initialize_load_balancing_components()
        
        # Integration state
        self.load_balancing_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.config = {
            "load_balancing_enabled": True,
            "auto_scaling_enabled": True,
            "health_monitoring_enabled": True,
            "resource_optimization_enabled": True,
            "performance_collection_enabled": True,
            "integration_check_interval": 60,  # seconds
            "workload_snapshot_interval": 30,  # seconds
        }
        
        logger.info("LoadBalancingOrchestrator initialized")
    
    def _initialize_load_balancing_components(self) -> None:
        """Initialize all load balancing components."""
        
        # Performance metrics collector (foundation for other components)
        self.metrics_collector = PerformanceMetricsCollector(
            collection_interval=5.0
        )
        
        # Agent load balancer
        self.load_balancer = AgentLoadBalancer(
            default_strategy=LoadBalancingStrategy.ADAPTIVE_HYBRID
        )
        
        # Capacity manager
        self.capacity_manager = CapacityManager(
            load_balancer=self.load_balancer
        )
        
        # Adaptive scaler
        self.adaptive_scaler = AdaptiveScaler(
            load_balancer=self.load_balancer,
            capacity_manager=self.capacity_manager,
            metrics_collector=self.metrics_collector
        )
        
        # Resource optimizer
        self.resource_optimizer = ResourceOptimizer(
            metrics_collector=self.metrics_collector
        )
        
        # Health monitor
        self.health_monitor = HealthMonitor(
            metrics_collector=self.metrics_collector
        )
        
        logger.info("Load balancing components initialized")
    
    async def start_load_balancing(self) -> None:
        """Start all load balancing and monitoring services."""
        if self.load_balancing_active:
            logger.warning("Load balancing already active")
            return
        
        try:
            self.load_balancing_active = True
            
            # Start performance metrics collection
            if self.config["performance_collection_enabled"]:
                await self.metrics_collector.start_collection()
            
            # Start adaptive scaling
            if self.config["auto_scaling_enabled"]:
                await self.adaptive_scaler.start_auto_scaling()
            
            # Start resource optimization
            if self.config["resource_optimization_enabled"]:
                await self.resource_optimizer.start_optimization()
            
            # Start health monitoring
            if self.config["health_monitoring_enabled"]:
                await self.health_monitor.start_monitoring()
            
            # Start integration monitoring task
            integration_task = asyncio.create_task(self._integration_monitoring_loop())
            self.monitoring_tasks.append(integration_task)
            
            # Start workload snapshot task
            workload_task = asyncio.create_task(self._workload_snapshot_loop())
            self.monitoring_tasks.append(workload_task)
            
            logger.info("Load balancing services started")
            
        except Exception as e:
            logger.error("Failed to start load balancing services", error=str(e))
            await self.stop_load_balancing()
            raise
    
    async def stop_load_balancing(self) -> None:
        """Stop all load balancing and monitoring services."""
        if not self.load_balancing_active:
            return
        
        try:
            self.load_balancing_active = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            self.monitoring_tasks.clear()
            
            # Stop all components
            await self.metrics_collector.stop_collection()
            await self.adaptive_scaler.stop_auto_scaling()
            await self.resource_optimizer.stop_optimization()
            await self.health_monitor.stop_monitoring()
            
            logger.info("Load balancing services stopped")
            
        except Exception as e:
            logger.error("Error stopping load balancing services", error=str(e))
    
    async def _integration_monitoring_loop(self) -> None:
        """Monitor and coordinate between load balancing components."""
        while self.load_balancing_active:
            try:
                # Update load balancer with current agent states
                await self._sync_agent_states()
                
                # Check for scaling recommendations
                scaling_recommendations = await self.adaptive_scaler.get_scaling_recommendations()
                
                # Apply automatic optimizations if needed
                await self._apply_automatic_optimizations()
                
                # Update orchestrator metrics
                await self._update_orchestrator_metrics()
                
                await asyncio.sleep(self.config["integration_check_interval"])
                
            except Exception as e:
                logger.error("Error in integration monitoring loop", error=str(e))
                await asyncio.sleep(self.config["integration_check_interval"])
    
    async def _workload_snapshot_loop(self) -> None:
        """Periodically create workload snapshots for analysis."""
        while self.load_balancing_active:
            try:
                await self._create_workload_snapshots()
                await asyncio.sleep(self.config["workload_snapshot_interval"])
                
            except Exception as e:
                logger.error("Error in workload snapshot loop", error=str(e))
                await asyncio.sleep(self.config["workload_snapshot_interval"])
    
    async def _sync_agent_states(self) -> None:
        """Sync agent states between orchestrator and load balancer."""
        try:
            # Get current agents from base orchestrator
            current_agents = list(self.base_orchestrator.agents.keys())
            
            # Update load balancer with current agent loads
            for agent_id in current_agents:
                agent_instance = self.base_orchestrator.agents.get(agent_id)
                if agent_instance:
                    # Update load balancer with agent state
                    await self.load_balancer.update_agent_load_state(
                        agent_id=agent_id,
                        active_tasks=len([t for t in getattr(agent_instance, 'current_tasks', [])]),
                        context_usage_percent=agent_instance.context_window_usage * 100,
                        last_updated=datetime.utcnow()
                    )
        
        except Exception as e:
            logger.error("Error syncing agent states", error=str(e))
    
    async def _create_workload_snapshots(self) -> None:
        """Create workload snapshots for all active agents."""
        try:
            async with get_session() as session:
                for agent_id, agent_instance in self.base_orchestrator.agents.items():
                    if agent_instance.status == AgentStatus.ACTIVE:
                        # Create workload snapshot
                        snapshot = WorkloadSnapshot(
                            agent_id=uuid.UUID(agent_id) if self._is_valid_uuid(agent_id) else uuid.uuid4(),
                            active_tasks=len(getattr(agent_instance, 'current_tasks', [])),
                            pending_tasks=0,  # Would need to get from task queue
                            context_usage_percent=agent_instance.context_window_usage * 100,
                            estimated_capacity=1.0,
                            utilization_ratio=agent_instance.context_window_usage,
                            snapshot_time=datetime.utcnow()
                        )
                        
                        session.add(snapshot)
                
                await session.commit()
        
        except Exception as e:
            logger.error("Error creating workload snapshots", error=str(e))
    
    def _is_valid_uuid(self, uuid_string: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False
    
    async def _apply_automatic_optimizations(self) -> None:
        """Apply automatic optimizations based on current state."""
        try:
            # Get resource optimization recommendations
            resource_recommendations = await self.resource_optimizer.get_optimization_recommendations()
            
            # Apply high-priority optimizations automatically
            for recommendation in resource_recommendations.get("recommendations", []):
                if recommendation.get("severity") == "high":
                    optimization_type = recommendation.get("type")
                    
                    if optimization_type == "memory":
                        from .resource_optimizer import OptimizationType
                        await self.resource_optimizer.force_optimization(OptimizationType.MEMORY_OPTIMIZATION)
                    elif optimization_type == "cpu":
                        from .resource_optimizer import OptimizationType
                        await self.resource_optimizer.force_optimization(OptimizationType.TASK_QUEUE_OPTIMIZATION)
            
            # Check for unhealthy agents and take action
            health_summary = await self.health_monitor.get_health_summary()
            
            critical_agents = health_summary.get("system_health", {}).get("status_distribution", {}).get("critical", 0)
            failed_agents = health_summary.get("system_health", {}).get("status_distribution", {}).get("failed", 0)
            
            if critical_agents > 0 or failed_agents > 0:
                logger.warning("Unhealthy agents detected",
                              critical_agents=critical_agents,
                              failed_agents=failed_agents)
                
                # Could trigger automatic recovery actions here
                # For now, just log the issue
        
        except Exception as e:
            logger.error("Error applying automatic optimizations", error=str(e))
    
    async def _update_orchestrator_metrics(self) -> None:
        """Update orchestrator metrics with load balancing data."""
        try:
            # Get metrics from load balancing components
            load_metrics = await self.load_balancer.get_load_balancing_metrics()
            capacity_metrics = await self.capacity_manager.get_capacity_metrics()
            health_summary = await self.health_monitor.get_health_summary()
            
            # Update base orchestrator metrics
            if hasattr(self.base_orchestrator, 'metrics'):
                self.base_orchestrator.metrics.update({
                    'load_balancing_decisions': load_metrics.get("decision_metrics", {}).get("total_decisions", 0),
                    'avg_load_balancing_time_ms': load_metrics.get("decision_metrics", {}).get("average_decision_time_ms", 0),
                    'total_agents_managed': capacity_metrics.get("total_agents", 0),
                    'healthy_agents': health_summary.get("system_health", {}).get("status_distribution", {}).get("healthy", 0),
                    'scaling_actions_performed': len(getattr(self.adaptive_scaler, 'scaling_history', [])),
                })
        
        except Exception as e:
            logger.error("Error updating orchestrator metrics", error=str(e))
    
    async def assign_task_with_load_balancing(
        self,
        task: Task,
        required_capabilities: Optional[List[str]] = None,
        preferred_strategy: Optional[LoadBalancingStrategy] = None
    ) -> Optional[str]:
        """
        Assign task to agent using intelligent load balancing.
        
        This method integrates with the base orchestrator to provide
        intelligent task assignment with load balancing.
        """
        try:
            if not self.config["load_balancing_enabled"]:
                # Fallback to base orchestrator assignment
                return await self._fallback_task_assignment(task)
            
            # Get available agents
            available_agents = [
                agent_id for agent_id, agent in self.base_orchestrator.agents.items()
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
            ]
            
            if not available_agents:
                logger.warning("No available agents for task assignment")
                return None
            
            # Use load balancer to select optimal agent
            decision = await self.load_balancer.select_agent_for_task(
                task=task,
                available_agents=available_agents,
                strategy=preferred_strategy,
                required_capabilities=required_capabilities
            )
            
            selected_agent_id = decision.selected_agent_id
            
            # Verify agent is still available
            if (selected_agent_id in self.base_orchestrator.agents and
                self.base_orchestrator.agents[selected_agent_id].status in [AgentStatus.ACTIVE, AgentStatus.IDLE]):
                
                # Record metrics
                await self.metrics_collector.record_custom_metric(
                    "system", "task_assignments", 1, MetricType.COUNTER
                )
                await self.metrics_collector.record_custom_metric(
                    "system", "load_balancing_decision_time_ms", decision.decision_time_ms, MetricType.HISTOGRAM
                )
                
                logger.info("Task assigned with load balancing",
                           task_id=str(task.id),
                           selected_agent=selected_agent_id,
                           strategy=decision.strategy_used.value,
                           confidence=decision.decision_confidence,
                           decision_time_ms=decision.decision_time_ms)
                
                return selected_agent_id
            
            else:
                logger.warning("Selected agent no longer available, falling back",
                              selected_agent=selected_agent_id)
                return await self._fallback_task_assignment(task)
        
        except Exception as e:
            logger.error("Error in load balanced task assignment", error=str(e))
            return await self._fallback_task_assignment(task)
    
    async def _fallback_task_assignment(self, task: Task) -> Optional[str]:
        """Fallback task assignment using base orchestrator logic."""
        try:
            # Simple round-robin or first available assignment
            available_agents = [
                agent_id for agent_id, agent in self.base_orchestrator.agents.items()
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
            ]
            
            if available_agents:
                # Select first available agent (simple fallback)
                selected_agent = available_agents[0]
                
                logger.info("Task assigned with fallback method",
                           task_id=str(task.id),
                           selected_agent=selected_agent)
                
                return selected_agent
            
            return None
        
        except Exception as e:
            logger.error("Error in fallback task assignment", error=str(e))
            return None
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including load balancing metrics."""
        try:
            # Base orchestrator status
            base_status = {
                "agents_count": len(self.base_orchestrator.agents),
                "active_sessions": len(self.base_orchestrator.active_sessions),
                "is_running": self.base_orchestrator.is_running,
                "metrics": getattr(self.base_orchestrator, 'metrics', {})
            }
            
            # Load balancing status
            load_balancing_status = {
                "load_balancing_active": self.load_balancing_active,
                "components_status": {
                    "load_balancer": "active" if self.load_balancer else "inactive",
                    "capacity_manager": "active" if self.capacity_manager else "inactive",
                    "adaptive_scaler": "active" if self.adaptive_scaler.auto_scaling_enabled else "inactive",
                    "health_monitor": "active" if self.health_monitor.monitoring_active else "inactive",
                    "resource_optimizer": "active" if self.resource_optimizer.optimization_active else "inactive",
                    "metrics_collector": "active" if self.metrics_collector.collection_active else "inactive"
                }
            }
            
            # Component metrics
            component_metrics = {}
            
            if self.load_balancing_active:
                component_metrics.update({
                    "load_balancer": await self.load_balancer.get_load_balancing_metrics(),
                    "capacity_manager": await self.capacity_manager.get_capacity_metrics(),
                    "health_monitor": await self.health_monitor.get_health_summary(),
                    "adaptive_scaler": await self.adaptive_scaler.get_scaling_metrics(),
                    "resource_optimizer": await self.resource_optimizer.get_resource_metrics()
                })
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "base_orchestrator": base_status,
                "load_balancing": load_balancing_status,
                "component_metrics": component_metrics,
                "configuration": self.config
            }
        
        except Exception as e:
            logger.error("Error getting system status", error=str(e))
            return {"error": str(e)}
    
    async def trigger_manual_scaling(self, action: str, **kwargs) -> Dict[str, Any]:
        """Trigger manual scaling action."""
        try:
            if action == "scale_up":
                # Trigger scale up
                scaling_decisions = await self.adaptive_scaler.force_scaling_evaluation()
                return {"action": action, "decisions": len(scaling_decisions)}
            
            elif action == "scale_down":
                # Trigger scale down evaluation
                scaling_decisions = await self.adaptive_scaler.force_scaling_evaluation()
                return {"action": action, "decisions": len(scaling_decisions)}
            
            elif action == "rebalance":
                # Trigger load rebalancing
                # This would redistribute tasks among agents
                return {"action": action, "status": "triggered"}
            
            elif action == "optimize_resources":
                # Trigger resource optimization
                optimization_type = kwargs.get("optimization_type")
                if optimization_type:
                    from .resource_optimizer import OptimizationType
                    result = await self.resource_optimizer.force_optimization(
                        OptimizationType(optimization_type)
                    )
                    return {"action": action, "result": result.to_dict()}
                else:
                    return {"error": "optimization_type required"}
            
            else:
                return {"error": f"Unknown action: {action}"}
        
        except Exception as e:
            logger.error("Error triggering manual scaling", error=str(e))
            return {"error": str(e)}
    
    async def get_agent_health_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for specific agent or all agents."""
        try:
            if agent_id:
                return await self.health_monitor.get_agent_health_details(agent_id)
            else:
                return await self.health_monitor.get_health_summary()
        
        except Exception as e:
            logger.error("Error getting agent health status", error=str(e))
            return {"error": str(e)}
    
    async def configure_load_balancing(self, **config_updates) -> Dict[str, Any]:
        """Update load balancing configuration."""
        try:
            # Update configuration
            self.config.update(config_updates)
            
            # Apply configuration changes to components
            if "auto_scaling_enabled" in config_updates:
                if config_updates["auto_scaling_enabled"] and not self.adaptive_scaler.auto_scaling_enabled:
                    await self.adaptive_scaler.start_auto_scaling()
                elif not config_updates["auto_scaling_enabled"] and self.adaptive_scaler.auto_scaling_enabled:
                    await self.adaptive_scaler.stop_auto_scaling()
            
            # Similar logic for other components...
            
            logger.info("Load balancing configuration updated", updates=config_updates)
            
            return {"status": "success", "updated_config": self.config}
        
        except Exception as e:
            logger.error("Error configuring load balancing", error=str(e))
            return {"error": str(e)}
    
    # Integration methods for base orchestrator
    
    def integrate_with_orchestrator(self, orchestrator) -> None:
        """
        Integrate load balancing components with existing orchestrator.
        
        This method should be called to integrate the load balancing
        functionality with an existing orchestrator instance.
        """
        # Store reference to base orchestrator
        self.base_orchestrator = orchestrator
        
        # Monkey patch the task assignment method
        original_assign_task = getattr(orchestrator, 'assign_task_to_agent', None)
        
        async def enhanced_assign_task(task, **kwargs):
            # Try load balanced assignment first
            result = await self.assign_task_with_load_balancing(task, **kwargs)
            
            # Fallback to original method if needed
            if result is None and original_assign_task:
                result = await original_assign_task(task, **kwargs)
            
            return result
        
        # Replace the method
        orchestrator.assign_task_to_agent = enhanced_assign_task
        
        # Add load balancing methods to orchestrator
        orchestrator.get_load_balancing_status = self.get_system_status
        orchestrator.trigger_scaling_action = self.trigger_manual_scaling
        orchestrator.get_agent_health = self.get_agent_health_status
        orchestrator.configure_load_balancing = self.configure_load_balancing
        
        logger.info("Load balancing integration completed")


# Factory function for easy integration
async def create_load_balanced_orchestrator(base_orchestrator) -> LoadBalancingOrchestrator:
    """
    Factory function to create a load-balanced orchestrator.
    
    Args:
        base_orchestrator: Existing orchestrator instance
        
    Returns:
        LoadBalancingOrchestrator with all components initialized
    """
    lb_orchestrator = LoadBalancingOrchestrator(base_orchestrator)
    
    # Integrate with base orchestrator
    lb_orchestrator.integrate_with_orchestrator(base_orchestrator)
    
    # Start load balancing services
    await lb_orchestrator.start_load_balancing()
    
    return lb_orchestrator
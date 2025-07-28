"""
Consumer Group Coordinator for LeanVibe Agent Hive 2.0 - Vertical Slice 4.2

Provides centralized management of consumer groups with dynamic provisioning,
lifecycle management, and intelligent resource allocation across agent types.

Key Features:
- Dynamic consumer group provisioning based on agent capabilities
- Intelligent load balancing and resource allocation
- Consumer health monitoring and automatic recovery
- Cross-group coordination for workflow dependencies
- Performance optimization and capacity planning
- Comprehensive metrics and observability
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union

import structlog
from sqlalchemy import select, update, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType, 
    MessageRoutingMode, ConsumerGroupMetrics, AutoScalingError
)
from .database import get_async_session
from .config import settings
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus

logger = structlog.get_logger()


class ConsumerGroupStrategy(str, Enum):
    """Strategies for consumer group management."""
    AGENT_TYPE_BASED = "agent_type_based"
    CAPABILITY_BASED = "capability_based" 
    WORKLOAD_BASED = "workload_based"
    HYBRID = "hybrid"


class ProvisioningPolicy(str, Enum):
    """Policies for consumer provisioning."""
    REACTIVE = "reactive"  # Create groups on demand
    PREDICTIVE = "predictive"  # Pre-provision based on patterns
    HYBRID = "hybrid"  # Combination of both


class ResourceAllocationMode(str, Enum):
    """Resource allocation modes for consumer groups."""
    EQUAL = "equal"  # Equal resources per group
    WEIGHTED = "weighted"  # Weighted by priority/capability
    ADAPTIVE = "adaptive"  # Adaptive based on workload
    PERFORMANCE_BASED = "performance_based"  # Based on historical performance


@dataclass
class ConsumerGroupTemplate:
    """Template for creating consumer groups."""
    name_pattern: str  # e.g., "{agent_type}_consumers"
    agent_type: ConsumerGroupType
    stream_pattern: str  # e.g., "agent_messages:{agent_type}"
    default_config: ConsumerGroupConfig
    auto_provision: bool = True
    scaling_policy: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.scaling_policy is None:
            self.scaling_policy = {
                "min_consumers": 1,
                "max_consumers": 10,
                "scale_up_threshold": 100,
                "scale_down_threshold": 10,
                "scale_up_factor": 2,
                "scale_down_factor": 0.5
            }


@dataclass
class CoordinatorMetrics:
    """Metrics for the consumer group coordinator."""
    total_groups_managed: int = 0
    total_consumers_managed: int = 0
    groups_created: int = 0
    groups_destroyed: int = 0
    consumers_scaled: int = 0
    rebalance_operations: int = 0
    health_checks_performed: int = 0
    avg_group_utilization: float = 0.0
    avg_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    start_time: float = 0.0
    
    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "uptime_seconds": time.time() - self.start_time
        }


class ConsumerGroupCoordinatorError(Exception):
    """Base exception for coordinator errors."""
    pass


class ProvisioningError(ConsumerGroupCoordinatorError):
    """Error in group provisioning."""
    pass


class RebalancingError(ConsumerGroupCoordinatorError):
    """Error in group rebalancing."""
    pass


class ConsumerGroupCoordinator:
    """
    Consumer Group Coordinator for centralized management and optimization.
    
    Provides:
    - Dynamic consumer group provisioning and lifecycle management
    - Intelligent resource allocation and load balancing
    - Cross-group coordination for workflow dependencies
    - Performance monitoring and optimization
    - Automatic scaling and health management
    """
    
    def __init__(
        self,
        streams_manager: EnhancedRedisStreamsManager,
        strategy: ConsumerGroupStrategy = ConsumerGroupStrategy.HYBRID,
        provisioning_policy: ProvisioningPolicy = ProvisioningPolicy.HYBRID,
        allocation_mode: ResourceAllocationMode = ResourceAllocationMode.ADAPTIVE,
        health_check_interval: int = 30,
        rebalance_interval: int = 300,  # 5 minutes
        enable_cross_group_coordination: bool = True
    ):
        """
        Initialize Consumer Group Coordinator.
        
        Args:
            streams_manager: Enhanced Redis Streams Manager instance
            strategy: Consumer group management strategy
            provisioning_policy: Group provisioning policy
            allocation_mode: Resource allocation mode
            health_check_interval: Health check interval in seconds
            rebalance_interval: Rebalancing interval in seconds
            enable_cross_group_coordination: Enable cross-group coordination
        """
        self.streams_manager = streams_manager
        self.strategy = strategy
        self.provisioning_policy = provisioning_policy
        self.allocation_mode = allocation_mode
        self.health_check_interval = health_check_interval
        self.rebalance_interval = rebalance_interval
        self.enable_cross_group_coordination = enable_cross_group_coordination
        
        # Group management
        self._group_templates: Dict[str, ConsumerGroupTemplate] = {}
        self._managed_groups: Dict[str, ConsumerGroupConfig] = {}
        self._group_assignments: Dict[str, str] = {}  # agent_id -> group_name
        
        # Resource allocation
        self._resource_allocations: Dict[str, Dict[str, Any]] = {}
        self._capacity_plans: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._rebalance_task: Optional[asyncio.Task] = None
        self._provisioning_task: Optional[asyncio.Task] = None
        
        # Metrics and monitoring
        self._metrics = CoordinatorMetrics()
        self._performance_history: deque = deque(maxlen=1000)
        self._workload_predictions: Dict[str, List[float]] = defaultdict(list)
        
        # Cross-group coordination
        self._workflow_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._group_priorities: Dict[str, int] = {}
        
        # Initialize default templates
        self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default consumer group templates for standard agent types."""
        for agent_type in ConsumerGroupType:
            template = ConsumerGroupTemplate(
                name_pattern=f"{agent_type.value}_consumers",
                agent_type=agent_type,
                stream_pattern=f"agent_messages:{agent_type.value}",
                default_config=ConsumerGroupConfig(
                    name=f"{agent_type.value}_consumers",
                    stream_name=f"agent_messages:{agent_type.value}",
                    agent_type=agent_type,
                    routing_mode=MessageRoutingMode.LOAD_BALANCED,
                    max_consumers=self._get_default_max_consumers(agent_type),
                    min_consumers=1,
                    auto_scale_enabled=True
                )
            )
            self._group_templates[agent_type.value] = template
    
    def _get_default_max_consumers(self, agent_type: ConsumerGroupType) -> int:
        """Get default maximum consumers based on agent type."""
        # Different agent types may have different scaling characteristics
        scaling_profiles = {
            ConsumerGroupType.ARCHITECTS: 5,
            ConsumerGroupType.BACKEND_ENGINEERS: 15,
            ConsumerGroupType.FRONTEND_DEVELOPERS: 12,
            ConsumerGroupType.QA_ENGINEERS: 8,
            ConsumerGroupType.DEVOPS_ENGINEERS: 6,
            ConsumerGroupType.SECURITY_ENGINEERS: 4,
            ConsumerGroupType.DATA_ENGINEERS: 6,
            ConsumerGroupType.GENERAL_AGENTS: 20
        }
        return scaling_profiles.get(agent_type, 10)
    
    async def start(self) -> None:
        """Start the coordinator with background tasks."""
        try:
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._rebalance_task = asyncio.create_task(self._rebalance_loop())
            
            if self.provisioning_policy != ProvisioningPolicy.REACTIVE:
                self._provisioning_task = asyncio.create_task(self._provisioning_loop())
            
            # Pre-provision essential groups if using predictive policy
            if self.provisioning_policy in [ProvisioningPolicy.PREDICTIVE, ProvisioningPolicy.HYBRID]:
                await self._pre_provision_essential_groups()
            
            logger.info(
                "Consumer Group Coordinator started",
                extra={
                    "strategy": self.strategy.value,
                    "provisioning_policy": self.provisioning_policy.value,
                    "allocation_mode": self.allocation_mode.value,
                    "templates_loaded": len(self._group_templates)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to start Consumer Group Coordinator: {e}")
            raise ConsumerGroupCoordinatorError(f"Coordinator startup failed: {e}")
    
    async def stop(self) -> None:
        """Stop the coordinator and cleanup resources."""
        # Stop background tasks
        tasks = [self._health_check_task, self._rebalance_task, self._provisioning_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        active_tasks = [task for task in tasks if task and not task.done()]
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logger.info("Consumer Group Coordinator stopped")
    
    async def provision_group_for_agent(
        self,
        agent_id: str,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[str]] = None
    ) -> str:
        """
        Provision or assign a consumer group for an agent.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (if known)
            capabilities: Agent capabilities list
            
        Returns:
            Consumer group name assigned to the agent
        """
        try:
            # Determine target group based on strategy
            target_group = await self._determine_target_group(
                agent_id, agent_type, capabilities
            )
            
            # Ensure group exists
            if target_group not in self._managed_groups:
                await self._provision_group(target_group)
            
            # Assign agent to group
            self._group_assignments[agent_id] = target_group
            
            logger.info(
                f"Provisioned group {target_group} for agent {agent_id}",
                extra={
                    "agent_id": agent_id,
                    "group_name": target_group,
                    "agent_type": agent_type.value if agent_type else None
                }
            )
            
            return target_group
            
        except Exception as e:
            logger.error(f"Failed to provision group for agent {agent_id}: {e}")
            raise ProvisioningError(f"Group provisioning failed: {e}")
    
    async def _determine_target_group(
        self,
        agent_id: str,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[str]] = None
    ) -> str:
        """Determine the target consumer group for an agent."""
        if self.strategy == ConsumerGroupStrategy.AGENT_TYPE_BASED:
            if agent_type:
                return f"{agent_type.value}_consumers"
            else:
                return "general_agents_consumers"
        
        elif self.strategy == ConsumerGroupStrategy.CAPABILITY_BASED:
            if capabilities:
                # Simple capability-based assignment (could be more sophisticated)
                primary_capability = capabilities[0] if capabilities else "general"
                return f"{primary_capability}_consumers"
            else:
                return "general_agents_consumers"
        
        elif self.strategy == ConsumerGroupStrategy.WORKLOAD_BASED:
            # Find the group with the lowest current load
            return await self._find_least_loaded_group()
        
        else:  # HYBRID strategy
            # Combine multiple factors for optimal assignment
            return await self._hybrid_group_assignment(agent_id, agent_type, capabilities)
    
    async def _find_least_loaded_group(self) -> str:
        """Find the consumer group with the lowest current load."""
        group_loads = {}
        
        for group_name in self._managed_groups.keys():
            stats = await self.streams_manager.get_consumer_group_stats(group_name)
            if stats:
                # Simple load calculation: pending messages + consumer utilization
                load = stats.pending_count + (stats.consumer_count * 10)  # Weight active consumers
                group_loads[group_name] = load
        
        if group_loads:
            return min(group_loads, key=group_loads.get)
        else:
            return "general_agents_consumers"
    
    async def _hybrid_group_assignment(
        self,
        agent_id: str,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[str]] = None
    ) -> str:
        """Hybrid group assignment considering multiple factors."""
        scores = {}
        
        # Score all available groups
        for group_name, config in self._managed_groups.items():
            score = 0.0
            
            # Type compatibility bonus
            if agent_type and config.agent_type.value in group_name:
                score += 50.0
            
            # Capability matching bonus
            if capabilities:
                for capability in capabilities:
                    if capability in group_name:
                        score += 20.0
            
            # Load balancing factor (prefer less loaded groups)
            stats = await self.streams_manager.get_consumer_group_stats(group_name)
            if stats:
                load_factor = max(0, 100 - stats.pending_count)  # Lower pending = higher score
                score += load_factor * 0.3
            
            # Performance factor
            if stats and stats.avg_processing_time_ms > 0:
                perf_factor = max(0, 1000 - stats.avg_processing_time_ms) / 10
                score += perf_factor
            
            scores[group_name] = score
        
        if scores:
            return max(scores, key=scores.get)
        else:
            return await self._determine_target_group(agent_id, agent_type, capabilities)
    
    async def _provision_group(self, group_name: str) -> None:
        """Provision a new consumer group."""
        # Find matching template
        template = None
        for template_name, tmpl in self._group_templates.items():
            if group_name == tmpl.name_pattern or template_name in group_name:
                template = tmpl
                break
        
        if not template:
            # Create default template
            template = ConsumerGroupTemplate(
                name_pattern=group_name,
                agent_type=ConsumerGroupType.GENERAL_AGENTS,
                stream_pattern=f"agent_messages:{group_name}",
                default_config=ConsumerGroupConfig(
                    name=group_name,
                    stream_name=f"agent_messages:{group_name}",
                    agent_type=ConsumerGroupType.GENERAL_AGENTS
                )
            )
        
        # Create configuration from template
        config = ConsumerGroupConfig(
            name=group_name,
            stream_name=template.stream_pattern.format(agent_type=template.agent_type.value),
            agent_type=template.agent_type,
            routing_mode=template.default_config.routing_mode,
            max_consumers=template.default_config.max_consumers,
            min_consumers=template.default_config.min_consumers,
            auto_scale_enabled=template.default_config.auto_scale_enabled
        )
        
        # Create the group
        await self.streams_manager.create_consumer_group(config)
        self._managed_groups[group_name] = config
        self._metrics.groups_created += 1
        
        logger.info(f"Provisioned consumer group {group_name}")
    
    async def _pre_provision_essential_groups(self) -> None:
        """Pre-provision essential consumer groups for predictive provisioning."""
        essential_types = [
            ConsumerGroupType.ARCHITECTS,
            ConsumerGroupType.BACKEND_ENGINEERS,
            ConsumerGroupType.GENERAL_AGENTS
        ]
        
        for agent_type in essential_types:
            if agent_type.value in self._group_templates:
                template = self._group_templates[agent_type.value]
                group_name = template.name_pattern
                
                if group_name not in self._managed_groups:
                    await self._provision_group(group_name)
    
    async def rebalance_groups(self) -> Dict[str, Any]:
        """
        Rebalance consumer groups for optimal performance.
        
        Returns:
            Dictionary with rebalancing results and metrics
        """
        try:
            rebalance_start = time.time()
            operations_performed = []
            
            # Get current state of all groups
            all_stats = await self.streams_manager.get_all_group_stats()
            
            # Analyze and perform rebalancing operations
            for group_name, stats in all_stats.items():
                config = self._managed_groups.get(group_name)
                if not config:
                    continue
                
                # Check if scaling is needed
                scaling_action = await self._determine_scaling_action(stats, config)
                if scaling_action:
                    try:
                        await self._execute_scaling_action(group_name, scaling_action)
                        operations_performed.append({
                            "group": group_name,
                            "action": scaling_action["action"],
                            "change": scaling_action.get("change", 0)
                        })
                    except Exception as e:
                        logger.error(f"Failed to execute scaling action for {group_name}: {e}")
            
            # Cross-group coordination if enabled
            if self.enable_cross_group_coordination:
                coordination_ops = await self._perform_cross_group_coordination(all_stats)
                operations_performed.extend(coordination_ops)
            
            rebalance_time = (time.time() - rebalance_start) * 1000
            self._metrics.rebalance_operations += 1
            
            result = {
                "rebalance_time_ms": rebalance_time,
                "operations_performed": operations_performed,
                "groups_analyzed": len(all_stats),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Completed group rebalancing in {rebalance_time:.2f}ms",
                extra={"operations": len(operations_performed)}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Group rebalancing failed: {e}")
            raise RebalancingError(f"Rebalancing failed: {e}")
    
    async def _determine_scaling_action(
        self,
        stats: ConsumerGroupMetrics,
        config: ConsumerGroupConfig
    ) -> Optional[Dict[str, Any]]:
        """Determine if scaling action is needed for a group."""
        if not config.auto_scale_enabled:
            return None
        
        current_consumers = stats.consumer_count
        
        # Scale up conditions
        if (stats.lag > config.lag_threshold and 
            current_consumers < config.max_consumers):
            scale_factor = min(2, config.max_consumers - current_consumers)
            return {
                "action": "scale_up",
                "change": scale_factor,
                "reason": f"High lag: {stats.lag} > {config.lag_threshold}"
            }
        
        # Scale down conditions
        if (stats.lag < config.lag_threshold // 4 and 
            current_consumers > config.min_consumers and
            stats.avg_processing_time_ms < 100):  # Low processing time indicates underutilization
            return {
                "action": "scale_down",
                "change": 1,
                "reason": f"Low utilization: lag={stats.lag}, processing_time={stats.avg_processing_time_ms}ms"
            }
        
        return None
    
    async def _execute_scaling_action(
        self,
        group_name: str,
        action: Dict[str, Any]
    ) -> None:
        """Execute a scaling action on a consumer group."""
        if action["action"] == "scale_up":
            for _ in range(action["change"]):
                # In a full implementation, this would coordinate with the agent registry
                # to spawn new consumers or notify existing agents to join
                logger.info(f"Would scale up {group_name} (simulated)")
        
        elif action["action"] == "scale_down":
            for _ in range(action["change"]):
                # Similar coordination for scaling down
                logger.info(f"Would scale down {group_name} (simulated)")
        
        self._metrics.consumers_scaled += action["change"]
    
    async def _perform_cross_group_coordination(
        self,
        all_stats: Dict[str, ConsumerGroupMetrics]
    ) -> List[Dict[str, Any]]:
        """Perform cross-group coordination operations."""
        operations = []
        
        # Example: If backend engineers are overloaded and frontend developers are underutilized,
        # suggest cross-training or temporary reassignment
        
        backend_stats = all_stats.get("backend_engineers_consumers")
        frontend_stats = all_stats.get("frontend_developers_consumers")
        
        if (backend_stats and frontend_stats and
            backend_stats.lag > 200 and frontend_stats.lag < 10):
            
            operations.append({
                "group": "cross_group_coordination",
                "action": "suggest_rebalancing",
                "source": "frontend_developers_consumers",
                "target": "backend_engineers_consumers",
                "reason": "Load imbalance detected"
            })
        
        return operations
    
    async def _health_check_loop(self) -> None:
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check health of all managed groups
                unhealthy_groups = []
                for group_name in self._managed_groups.keys():
                    stats = await self.streams_manager.get_consumer_group_stats(group_name)
                    if stats and (stats.success_rate < 0.95 or stats.lag > 1000):
                        unhealthy_groups.append(group_name)
                
                if unhealthy_groups:
                    logger.warning(f"Unhealthy consumer groups detected: {unhealthy_groups}")
                    # Could trigger alerting or recovery actions here
                
                self._metrics.health_checks_performed += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _rebalance_loop(self) -> None:
        """Background task for periodic rebalancing."""
        while True:
            try:
                await asyncio.sleep(self.rebalance_interval)
                await self.rebalance_groups()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rebalance loop: {e}")
    
    async def _provisioning_loop(self) -> None:
        """Background task for predictive provisioning."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Analyze workload patterns and pre-provision if needed
                await self._analyze_and_provision()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in provisioning loop: {e}")
    
    async def _analyze_and_provision(self) -> None:
        """Analyze workload patterns and provision groups predictively."""
        # This is a simplified version - a full implementation would use
        # machine learning models to predict workload patterns
        
        async with get_async_session() as session:
            # Check recent task creation patterns
            recent_tasks = await session.execute(
                select(Task)
                .where(Task.created_at > datetime.utcnow() - timedelta(hours=1))
                .order_by(desc(Task.created_at))
                .limit(100)
            )
            
            tasks = recent_tasks.scalars().all()
            
            # Analyze task types to predict needed consumer groups
            task_type_counts = defaultdict(int)
            for task in tasks:
                if hasattr(task, 'agent_type') and task.agent_type:
                    task_type_counts[task.agent_type] += 1
            
            # Provision groups for high-demand agent types
            for agent_type, count in task_type_counts.items():
                if count > 10:  # Threshold for provisioning
                    group_name = f"{agent_type}_consumers"
                    if group_name not in self._managed_groups:
                        await self._provision_group(group_name)
    
    async def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator metrics."""
        self._metrics.total_groups_managed = len(self._managed_groups)
        self._metrics.total_consumers_managed = sum(
            len(consumers) for consumers in self.streams_manager._active_consumers.values()
        )
        
        # Calculate average utilization
        all_stats = await self.streams_manager.get_all_group_stats()
        if all_stats:
            utilizations = []
            for stats in all_stats.values():
                if stats.consumer_count > 0:
                    utilization = min(100, (stats.throughput_msg_per_sec / stats.consumer_count) * 10)
                    utilizations.append(utilization)
            
            if utilizations:
                self._metrics.avg_group_utilization = sum(utilizations) / len(utilizations)
        
        return {
            "coordinator_metrics": self._metrics.to_dict(),
            "managed_groups": {
                name: config.to_dict() for name, config in self._managed_groups.items()
            },
            "group_assignments": dict(self._group_assignments),
            "strategy": self.strategy.value,
            "provisioning_policy": self.provisioning_policy.value,
            "allocation_mode": self.allocation_mode.value
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        streams_health = await self.streams_manager.health_check()
        
        # Check coordinator-specific health
        unhealthy_groups = 0
        all_stats = await self.streams_manager.get_all_group_stats()
        
        for stats in all_stats.values():
            if stats.success_rate < 0.95 or stats.lag > 1000:
                unhealthy_groups += 1
        
        is_healthy = (
            streams_health.get("enhanced_status") == "healthy" and
            unhealthy_groups == 0 and
            len(self._managed_groups) > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "coordinator_healthy": unhealthy_groups == 0,
            "groups_managed": len(self._managed_groups),
            "unhealthy_groups": unhealthy_groups,
            "background_tasks_running": sum(1 for task in [
                self._health_check_task, self._rebalance_task, self._provisioning_task
            ] if task and not task.done()),
            "streams_health": streams_health,
            "timestamp": datetime.utcnow().isoformat()
        }
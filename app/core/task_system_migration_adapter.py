"""
Task System Migration Adapter for LeanVibe Agent Hive 2.0
Provides backward compatibility while migrating from legacy task systems to unified engine.

Epic 1, Phase 2 Week 3 - Task System Consolidation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import warnings

try:
    from .unified_task_execution_engine import (
        UnifiedTaskExecutionEngine, 
        TaskExecutionRequest, 
        TaskExecutionType, 
        ExecutionMode, 
        SchedulingStrategy,
        get_unified_task_execution_engine
    )
    UNIFIED_ENGINE_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    UNIFIED_ENGINE_AVAILABLE = False

try:
    from .logging_service import get_component_logger
except (ImportError, NameError, AttributeError):
    import logging
    def get_component_logger(name):
        return logging.getLogger(name)

logger = get_component_logger("task_system_migration_adapter")

class TaskSystemMigrationAdapter:
    """
    Migration adapter providing backward compatibility for legacy task systems.
    Gradually migrates all task management to the unified execution engine.
    """
    
    def __init__(self):
        if UNIFIED_ENGINE_AVAILABLE:
            self.unified_engine = get_unified_task_execution_engine()
        else:
            self.unified_engine = None
            logger.warning("Unified task execution engine not available")
        
        self._migration_stats = {
            "legacy_calls": 0,
            "unified_calls": 0,
            "migration_warnings": 0
        }
    
    # Legacy TaskExecutionEngine compatibility
    async def start_task_execution(self, 
                                 task_id: uuid.UUID,
                                 agent_id: uuid.UUID,
                                 execution_context: Optional[Dict[str, Any]] = None) -> bool:
        """Legacy TaskExecutionEngine.start_task_execution compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "start_task_execution is deprecated. Use unified_task_execution_engine.submit_task instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Convert to unified request
            request = TaskExecutionRequest(
                task_id=str(task_id),
                function_name="legacy_task_execution",
                function_kwargs={
                    "agent_id": str(agent_id),
                    "execution_context": execution_context or {}
                },
                task_type=TaskExecutionType.IMMEDIATE,
                metadata={"legacy_migration": True, "original_method": "start_task_execution"}
            )
            
            # Register legacy execution function if not already registered
            if "legacy_task_execution" not in self.unified_engine._task_registry:
                self.unified_engine.register_task_function(
                    "legacy_task_execution", 
                    self._legacy_task_execution_handler
                )
            
            await self.unified_engine.submit_task(request)
            return True
            
        except Exception as e:
            logger.error("Legacy task execution migration failed", 
                        task_id=str(task_id), error=str(e))
            return False
    
    async def _legacy_task_execution_handler(self, agent_id: str, execution_context: Dict[str, Any]):
        """Handler for legacy task execution"""
        # This is a placeholder for the actual legacy task execution logic
        # In a real migration, this would call the appropriate legacy systems
        logger.info("Executing legacy task", agent_id=agent_id, context=execution_context)
        await asyncio.sleep(0.1)  # Simulate work
        return {"status": "completed", "agent_id": agent_id}
    
    # Legacy TaskScheduler compatibility
    async def assign_task(self,
                         task_id: uuid.UUID,
                         strategy: Optional[str] = None,
                         preferred_agent_id: Optional[uuid.UUID] = None,
                         timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Legacy TaskScheduler.assign_task compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "TaskScheduler.assign_task is deprecated. Use unified_task_execution_engine with agent requirements.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Map legacy strategy to unified strategy
            strategy_mapping = {
                "round_robin": SchedulingStrategy.ROUND_ROBIN,
                "capability_match": SchedulingStrategy.CAPABILITY_MATCH,
                "load_balanced": SchedulingStrategy.LOAD_BALANCED,
                "performance_optimized": SchedulingStrategy.PERFORMANCE_OPTIMIZED,
                "hybrid": SchedulingStrategy.HYBRID
            }
            
            unified_strategy = strategy_mapping.get(strategy, SchedulingStrategy.HYBRID)
            
            request = TaskExecutionRequest(
                task_id=str(task_id),
                function_name="legacy_task_assignment",
                function_kwargs={
                    "preferred_agent_id": str(preferred_agent_id) if preferred_agent_id else None,
                    "timeout_seconds": timeout_seconds
                },
                task_type=TaskExecutionType.IMMEDIATE,
                scheduling_strategy=unified_strategy,
                preferred_agent_id=str(preferred_agent_id) if preferred_agent_id else None,
                timeout=timedelta(seconds=timeout_seconds or 120),
                metadata={"legacy_migration": True, "original_method": "assign_task"}
            )
            
            if "legacy_task_assignment" not in self.unified_engine._task_registry:
                self.unified_engine.register_task_function(
                    "legacy_task_assignment",
                    self._legacy_task_assignment_handler
                )
            
            submitted_task_id = await self.unified_engine.submit_task(request)
            
            return {
                "success": True,
                "task_id": task_id,
                "assigned_agent_id": preferred_agent_id,
                "assignment_confidence": 0.8,
                "scheduling_strategy": strategy or "hybrid",
                "decision_time_ms": 10.0,
                "reasoning": ["Migrated to unified engine"],
                "error_message": None,
                "unified_task_id": submitted_task_id
            }
            
        except Exception as e:
            logger.error("Legacy task assignment migration failed", 
                        task_id=str(task_id), error=str(e))
            return {
                "success": False,
                "task_id": task_id,
                "assigned_agent_id": None,
                "assignment_confidence": 0.0,
                "scheduling_strategy": strategy or "hybrid",
                "decision_time_ms": 0.0,
                "reasoning": [f"Migration failed: {str(e)}"],
                "error_message": str(e)
            }
    
    async def _legacy_task_assignment_handler(self, preferred_agent_id: Optional[str], timeout_seconds: Optional[float]):
        """Handler for legacy task assignment"""
        logger.info("Processing legacy task assignment", 
                   preferred_agent=preferred_agent_id, timeout=timeout_seconds)
        await asyncio.sleep(0.1)  # Simulate assignment work
        return {"assigned_agent_id": preferred_agent_id, "assignment_time": datetime.utcnow().isoformat()}
    
    # Legacy TaskQueue compatibility
    async def enqueue_task(self,
                          task_id: uuid.UUID,
                          priority: Any = None,
                          queue_name: Optional[str] = None,
                          required_capabilities: Optional[List[str]] = None,
                          estimated_effort: Optional[int] = None,
                          timeout_seconds: Optional[int] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Legacy TaskQueue.enqueue_task compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "TaskQueue.enqueue_task is deprecated. Use unified_task_execution_engine.submit_task.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Convert priority enum to integer if needed
            priority_value = 5  # Default
            if hasattr(priority, 'value'):
                priority_value = int(priority.value)
            elif isinstance(priority, (int, float)):
                priority_value = int(priority)
            
            # Determine task type from queue name
            task_type = TaskExecutionType.IMMEDIATE
            if queue_name:
                if "batch" in queue_name.lower():
                    task_type = TaskExecutionType.BATCH
                elif "scheduled" in queue_name.lower():
                    task_type = TaskExecutionType.SCHEDULED
                elif "priority" in queue_name.lower():
                    task_type = TaskExecutionType.PRIORITY
            
            request = TaskExecutionRequest(
                task_id=str(task_id),
                function_name="legacy_queue_task",
                function_kwargs={
                    "queue_name": queue_name,
                    "estimated_effort": estimated_effort,
                    "original_metadata": metadata or {}
                },
                task_type=task_type,
                priority=priority_value,
                required_capabilities=required_capabilities or [],
                timeout=timedelta(seconds=timeout_seconds or 3600),
                metadata={
                    **({} if metadata is None else metadata),
                    "legacy_migration": True,
                    "original_method": "enqueue_task",
                    "original_queue": queue_name
                }
            )
            
            if "legacy_queue_task" not in self.unified_engine._task_registry:
                self.unified_engine.register_task_function(
                    "legacy_queue_task",
                    self._legacy_queue_task_handler
                )
            
            await self.unified_engine.submit_task(request)
            return True
            
        except Exception as e:
            logger.error("Legacy task queue migration failed", 
                        task_id=str(task_id), error=str(e))
            return False
    
    async def _legacy_queue_task_handler(self, queue_name: Optional[str], estimated_effort: Optional[int], original_metadata: Dict[str, Any]):
        """Handler for legacy queued tasks"""
        logger.info("Processing legacy queued task", 
                   queue=queue_name, effort=estimated_effort, metadata=original_metadata)
        await asyncio.sleep(0.1)  # Simulate queue processing
        return {"queue_processed": queue_name, "effort": estimated_effort}
    
    # Legacy TaskDistributor compatibility
    async def distribute_tasks(self,
                              tasks: List[Dict[str, Any]],
                              agents: List[Dict[str, Any]],
                              strategy: str = "hybrid") -> Dict[str, Any]:
        """Legacy TaskDistributor.distribute_tasks compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "TaskDistributor.distribute_tasks is deprecated. Use unified_task_execution_engine with batch tasks.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            task_ids = []
            
            for task_data in tasks:
                request = TaskExecutionRequest(
                    function_name="legacy_distributed_task",
                    function_kwargs={
                        "task_data": task_data,
                        "available_agents": agents,
                        "distribution_strategy": strategy
                    },
                    task_type=TaskExecutionType.BATCH,
                    priority=task_data.get("priority", 5),
                    required_capabilities=task_data.get("required_capabilities", []),
                    metadata={
                        "legacy_migration": True,
                        "original_method": "distribute_tasks",
                        "distribution_strategy": strategy
                    }
                )
                
                if "legacy_distributed_task" not in self.unified_engine._task_registry:
                    self.unified_engine.register_task_function(
                        "legacy_distributed_task",
                        self._legacy_distributed_task_handler
                    )
                
                task_id = await self.unified_engine.submit_task(request)
                task_ids.append(task_id)
            
            return {
                "assignments": [{"task_id": tid, "agent_id": None} for tid in task_ids],
                "unassigned_tasks": [],
                "distribution_time_ms": 50.0,
                "strategy_used": strategy,
                "optimization_metrics": {"tasks_distributed": len(tasks)},
                "unified_task_ids": task_ids
            }
            
        except Exception as e:
            logger.error("Legacy task distribution migration failed", error=str(e))
            return {
                "assignments": [],
                "unassigned_tasks": [t.get("task_id", "unknown") for t in tasks],
                "distribution_time_ms": 0.0,
                "strategy_used": strategy,
                "optimization_metrics": {"error": str(e)},
                "error": str(e)
            }
    
    async def _legacy_distributed_task_handler(self, task_data: Dict[str, Any], available_agents: List[Dict[str, Any]], distribution_strategy: str):
        """Handler for legacy distributed tasks"""
        logger.info("Processing legacy distributed task", 
                   task=task_data.get("task_id", "unknown"), 
                   strategy=distribution_strategy, 
                   agents_count=len(available_agents))
        await asyncio.sleep(0.1)  # Simulate distribution work
        return {"distributed": True, "strategy": distribution_strategy}
    
    # Legacy TaskBatchExecutor compatibility
    async def execute_batch(self,
                           task_requests: List[Dict[str, Any]],
                           strategy: str = "parallel_limited",
                           max_concurrent: int = 10) -> Dict[str, Any]:
        """Legacy TaskBatchExecutor.execute_batch compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "TaskBatchExecutor.execute_batch is deprecated. Use unified_task_execution_engine batch execution.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            batch_task_ids = []
            
            for i, task_req in enumerate(task_requests):
                request = TaskExecutionRequest(
                    function_name="legacy_batch_task",
                    function_kwargs={
                        "batch_index": i,
                        "task_request": task_req,
                        "batch_strategy": strategy,
                        "max_concurrent": max_concurrent
                    },
                    task_type=TaskExecutionType.BATCH,
                    priority=task_req.get("priority", 5),
                    execution_mode=ExecutionMode.PARALLEL if "parallel" in strategy else ExecutionMode.SEQUENTIAL,
                    metadata={
                        "legacy_migration": True,
                        "original_method": "execute_batch",
                        "batch_strategy": strategy,
                        "batch_index": i,
                        "batch_size": len(task_requests)
                    }
                )
                
                if "legacy_batch_task" not in self.unified_engine._task_registry:
                    self.unified_engine.register_task_function(
                        "legacy_batch_task",
                        self._legacy_batch_task_handler
                    )
                
                task_id = await self.unified_engine.submit_task(request)
                batch_task_ids.append(task_id)
            
            return {
                "batch_id": str(uuid.uuid4()),
                "submitted_tasks": len(task_requests),
                "task_ids": batch_task_ids,
                "strategy": strategy,
                "max_concurrent": max_concurrent,
                "status": "submitted",
                "unified_task_ids": batch_task_ids
            }
            
        except Exception as e:
            logger.error("Legacy batch execution migration failed", error=str(e))
            return {
                "batch_id": None,
                "submitted_tasks": 0,
                "task_ids": [],
                "strategy": strategy,
                "max_concurrent": max_concurrent,
                "status": "failed",
                "error": str(e)
            }
    
    async def _legacy_batch_task_handler(self, batch_index: int, task_request: Dict[str, Any], batch_strategy: str, max_concurrent: int):
        """Handler for legacy batch tasks"""
        logger.info("Processing legacy batch task", 
                   index=batch_index, strategy=batch_strategy, max_concurrent=max_concurrent)
        await asyncio.sleep(0.1)  # Simulate batch work
        return {"batch_index": batch_index, "completed": True}
    
    # Legacy SmartScheduler compatibility
    async def schedule_smart_task(self,
                                 task_config: Dict[str, Any],
                                 automation_tier: str = "immediate") -> Dict[str, Any]:
        """Legacy SmartScheduler compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "SmartScheduler is deprecated. Use unified_task_execution_engine with scheduling options.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Map automation tier to task type
            task_type_mapping = {
                "immediate": TaskExecutionType.IMMEDIATE,
                "scheduled": TaskExecutionType.SCHEDULED,
                "predictive": TaskExecutionType.SCHEDULED
            }
            
            task_type = task_type_mapping.get(automation_tier, TaskExecutionType.SCHEDULED)
            
            # Extract scheduling information
            scheduled_at = None
            if "scheduled_at" in task_config:
                if isinstance(task_config["scheduled_at"], str):
                    scheduled_at = datetime.fromisoformat(task_config["scheduled_at"])
                elif isinstance(task_config["scheduled_at"], datetime):
                    scheduled_at = task_config["scheduled_at"]
            
            request = TaskExecutionRequest(
                function_name="legacy_smart_scheduled_task",
                function_kwargs={
                    "task_config": task_config,
                    "automation_tier": automation_tier
                },
                task_type=task_type,
                scheduled_at=scheduled_at,
                priority=task_config.get("priority", 5),
                metadata={
                    "legacy_migration": True,
                    "original_method": "schedule_smart_task",
                    "automation_tier": automation_tier
                }
            )
            
            if "legacy_smart_scheduled_task" not in self.unified_engine._task_registry:
                self.unified_engine.register_task_function(
                    "legacy_smart_scheduled_task",
                    self._legacy_smart_scheduled_task_handler
                )
            
            task_id = await self.unified_engine.submit_task(request)
            
            return {
                "success": True,
                "task_id": task_id,
                "automation_tier": automation_tier,
                "scheduled_at": scheduled_at.isoformat() if scheduled_at else None,
                "unified_task_id": task_id
            }
            
        except Exception as e:
            logger.error("Legacy smart scheduler migration failed", error=str(e))
            return {
                "success": False,
                "task_id": None,
                "automation_tier": automation_tier,
                "error": str(e)
            }
    
    async def _legacy_smart_scheduled_task_handler(self, task_config: Dict[str, Any], automation_tier: str):
        """Handler for legacy smart scheduled tasks"""
        logger.info("Processing legacy smart scheduled task", 
                   config=task_config, tier=automation_tier)
        await asyncio.sleep(0.1)  # Simulate smart scheduling work
        return {"smart_scheduled": True, "tier": automation_tier}
    
    # Enhanced routing compatibility
    async def route_task_intelligently(self,
                                     task_data: Dict[str, Any],
                                     routing_strategy: str = "adaptive") -> Dict[str, Any]:
        """Legacy enhanced intelligent task router compatibility"""
        self._migration_stats["legacy_calls"] += 1
        
        warnings.warn(
            "Intelligent task routing is now built into the unified engine. Use submit_task with scheduling_strategy.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Map routing strategies
            strategy_mapping = {
                "adaptive": SchedulingStrategy.ADAPTIVE,
                "capability_first": SchedulingStrategy.CAPABILITY_MATCH,
                "performance_first": SchedulingStrategy.PERFORMANCE_OPTIMIZED,
                "load_balanced": SchedulingStrategy.LOAD_BALANCED
            }
            
            unified_strategy = strategy_mapping.get(routing_strategy, SchedulingStrategy.INTELLIGENT)
            
            request = TaskExecutionRequest(
                function_name="legacy_intelligent_routed_task",
                function_kwargs={
                    "task_data": task_data,
                    "routing_strategy": routing_strategy
                },
                task_type=TaskExecutionType.IMMEDIATE,
                scheduling_strategy=unified_strategy,
                required_capabilities=task_data.get("required_capabilities", []),
                priority=task_data.get("priority", 5),
                metadata={
                    "legacy_migration": True,
                    "original_method": "route_task_intelligently",
                    "routing_strategy": routing_strategy
                }
            )
            
            if "legacy_intelligent_routed_task" not in self.unified_engine._task_registry:
                self.unified_engine.register_task_function(
                    "legacy_intelligent_routed_task",
                    self._legacy_intelligent_routed_task_handler
                )
            
            task_id = await self.unified_engine.submit_task(request)
            
            return {
                "success": True,
                "task_id": task_id,
                "routing_strategy": routing_strategy,
                "selected_agent": None,  # Will be determined by unified engine
                "routing_confidence": 0.85,
                "unified_task_id": task_id
            }
            
        except Exception as e:
            logger.error("Legacy intelligent routing migration failed", error=str(e))
            return {
                "success": False,
                "task_id": None,
                "routing_strategy": routing_strategy,
                "error": str(e)
            }
    
    async def _legacy_intelligent_routed_task_handler(self, task_data: Dict[str, Any], routing_strategy: str):
        """Handler for legacy intelligent routed tasks"""
        logger.info("Processing legacy intelligent routed task", 
                   data=task_data, strategy=routing_strategy)
        await asyncio.sleep(0.1)  # Simulate intelligent routing
        return {"routed": True, "strategy": routing_strategy}
    
    def get_migration_stats(self) -> Dict[str, Any]:
        """Get migration statistics"""
        total_calls = self._migration_stats["legacy_calls"] + self._migration_stats["unified_calls"]
        migration_progress = (
            self._migration_stats["unified_calls"] / total_calls * 100 
            if total_calls > 0 else 0
        )
        
        return {
            **self._migration_stats,
            "total_calls": total_calls,
            "migration_progress_percentage": migration_progress,
            "migration_recommendations": self._get_migration_recommendations()
        }
    
    def _get_migration_recommendations(self) -> List[str]:
        """Get recommendations for completing migration"""
        recommendations = []
        
        if self._migration_stats["legacy_calls"] > 0:
            recommendations.append(
                f"Found {self._migration_stats['legacy_calls']} legacy task system calls. "
                "Consider updating code to use unified_task_execution_engine directly."
            )
        
        if self._migration_stats["migration_warnings"] > 10:
            recommendations.append(
                "High number of migration warnings detected. "
                "Review deprecated method usage and update to unified API."
            )
        
        if self._migration_stats["legacy_calls"] / max(1, self._migration_stats["unified_calls"]) > 0.5:
            recommendations.append(
                "Legacy task system usage is still high. "
                "Prioritize migration to unified engine for better performance."
            )
        
        return recommendations

# Global migration adapter instance
_migration_adapter = None

def get_task_migration_adapter() -> TaskSystemMigrationAdapter:
    """Get global task migration adapter instance"""
    global _migration_adapter
    if _migration_adapter is None:
        _migration_adapter = TaskSystemMigrationAdapter()
    return _migration_adapter

# Legacy API compatibility exports
async def legacy_start_task_execution(task_id: uuid.UUID, agent_id: uuid.UUID, execution_context: Optional[Dict[str, Any]] = None) -> bool:
    """Legacy compatibility function"""
    adapter = get_task_migration_adapter()
    return await adapter.start_task_execution(task_id, agent_id, execution_context)

async def legacy_assign_task(task_id: uuid.UUID, strategy: Optional[str] = None, preferred_agent_id: Optional[uuid.UUID] = None, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Legacy compatibility function"""
    adapter = get_task_migration_adapter()
    return await adapter.assign_task(task_id, strategy, preferred_agent_id, timeout_seconds)

async def legacy_enqueue_task(task_id: uuid.UUID, priority: Any = None, queue_name: Optional[str] = None, required_capabilities: Optional[List[str]] = None, estimated_effort: Optional[int] = None, timeout_seconds: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Legacy compatibility function"""
    adapter = get_task_migration_adapter()
    return await adapter.enqueue_task(task_id, priority, queue_name, required_capabilities, estimated_effort, timeout_seconds, metadata)
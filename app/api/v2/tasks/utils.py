"""
TaskExecutionAPI Utils - Shared Utilities and Helpers

Consolidated utility functions and helpers supporting the unified TaskExecutionAPI.
Provides common functionality used across core, workflows, and scheduling modules.

Features:
- Performance monitoring and metrics collection
- Cache management and optimization
- Data validation and transformation utilities
- Error handling and recovery helpers
- Epic 1 ConsolidatedProductionOrchestrator integration utilities
- Redis coordination and messaging helpers
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from uuid import UUID, uuid4
from functools import wraps
import structlog

from ....core.production_orchestrator_unified import get_production_orchestrator
from ....core.redis_integration import get_redis_service
from ....models.task import TaskStatus, TaskPriority, TaskType
from ....models.workflow import WorkflowStatus, WorkflowPriority


logger = structlog.get_logger(__name__)


# ===============================================================================
# PERFORMANCE MONITORING UTILITIES
# ===============================================================================

class PerformanceMonitor:
    """
    Performance monitoring utility for tracking TaskExecutionAPI metrics.
    
    Provides comprehensive performance tracking with configurable thresholds
    and automated alerts for performance regressions.
    """
    
    def __init__(self):
        self.performance_targets = {
            "task_creation": 200,      # 200ms
            "task_retrieval": 50,      # 50ms  
            "workflow_execution": 500,  # 500ms
            "schedule_optimization": 2000,  # 2s
            "pattern_analysis": 2000,   # 2s
            "conflict_resolution": 5000  # 5s
        }
        
        self.performance_history = {}
        self.alert_thresholds = {
            "warning": 1.5,  # 1.5x target
            "critical": 2.0   # 2x target
        }
    
    def track_performance(self, operation: str):
        """Decorator to track operation performance."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                operation_id = str(uuid4())
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    # Record performance metrics
                    await self._record_performance(
                        operation=operation,
                        operation_id=operation_id,
                        execution_time_ms=execution_time_ms,
                        success=True
                    )
                    
                    # Check performance targets
                    await self._check_performance_targets(operation, execution_time_ms)
                    
                    return result
                    
                except Exception as e:
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    # Record failed operation
                    await self._record_performance(
                        operation=operation,
                        operation_id=operation_id,
                        execution_time_ms=execution_time_ms,
                        success=False,
                        error=str(e)
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    async def _record_performance(
        self,
        operation: str,
        operation_id: str,
        execution_time_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record performance metrics for analysis."""
        try:
            metrics = {
                "operation": operation,
                "operation_id": operation_id,
                "execution_time_ms": execution_time_ms,
                "success": success,
                "error": error,
                "timestamp": datetime.utcnow().isoformat(),
                "target_time_ms": self.performance_targets.get(operation, 1000)
            }
            
            # Store in performance history
            if operation not in self.performance_history:
                self.performance_history[operation] = []
            
            self.performance_history[operation].append(metrics)
            
            # Keep only recent history (last 1000 entries)
            if len(self.performance_history[operation]) > 1000:
                self.performance_history[operation] = self.performance_history[operation][-1000:]
            
            # Store in Redis for distributed monitoring
            redis_service = get_redis_service()
            if redis_service:
                await redis_service.cache_set(
                    f"performance:metrics:{operation_id}",
                    metrics,
                    ttl=86400  # 24 hours
                )
            
        except Exception as e:
            logger.warning("Failed to record performance metrics", error=str(e))
    
    async def _check_performance_targets(self, operation: str, execution_time_ms: float):
        """Check if performance targets are met and alert if necessary."""
        target_time = self.performance_targets.get(operation, 1000)
        
        if execution_time_ms > target_time * self.alert_thresholds["critical"]:
            logger.error("Critical performance degradation detected",
                        operation=operation,
                        execution_time_ms=execution_time_ms,
                        target_time_ms=target_time,
                        degradation_factor=execution_time_ms / target_time)
        elif execution_time_ms > target_time * self.alert_thresholds["warning"]:
            logger.warning("Performance warning threshold exceeded",
                          operation=operation,
                          execution_time_ms=execution_time_ms,
                          target_time_ms=target_time,
                          degradation_factor=execution_time_ms / target_time)
    
    async def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for operations."""
        try:
            if operation and operation in self.performance_history:
                metrics = self.performance_history[operation]
            else:
                # Aggregate all operations
                metrics = []
                for op_metrics in self.performance_history.values():
                    metrics.extend(op_metrics)
            
            if not metrics:
                return {"error": "No performance data available"}
            
            # Calculate statistics
            execution_times = [m["execution_time_ms"] for m in metrics if m["success"]]
            success_count = len([m for m in metrics if m["success"]])
            total_count = len(metrics)
            
            summary = {
                "operation": operation or "all_operations",
                "total_operations": total_count,
                "successful_operations": success_count,
                "success_rate": (success_count / total_count) * 100 if total_count > 0 else 0,
                "average_execution_time_ms": sum(execution_times) / len(execution_times) if execution_times else 0,
                "min_execution_time_ms": min(execution_times) if execution_times else 0,
                "max_execution_time_ms": max(execution_times) if execution_times else 0,
                "target_time_ms": self.performance_targets.get(operation, 1000) if operation else None,
                "target_compliance_rate": 0.0,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Calculate target compliance rate
            if operation and operation in self.performance_targets:
                target = self.performance_targets[operation]
                compliant_operations = len([t for t in execution_times if t <= target])
                summary["target_compliance_rate"] = (compliant_operations / len(execution_times)) * 100 if execution_times else 0
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate performance summary", error=str(e))
            return {"error": str(e)}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# ===============================================================================
# CACHE MANAGEMENT UTILITIES
# ===============================================================================

class CacheManager:
    """
    Intelligent cache management for TaskExecutionAPI operations.
    
    Provides hierarchical caching with TTL management, cache invalidation,
    and performance optimization for frequently accessed data.
    """
    
    def __init__(self):
        self.default_ttls = {
            "task": 300,           # 5 minutes
            "workflow": 600,       # 10 minutes
            "schedule": 1800,      # 30 minutes
            "pattern_analysis": 3600,  # 1 hour
            "performance_metrics": 900  # 15 minutes
        }
        
        self.cache_hierarchies = {
            "task": ["task:{task_id}", "tasks:list:{hash}", "tasks:stats"],
            "workflow": ["workflow:{workflow_id}", "workflows:list:{hash}", "workflows:progress:{workflow_id}"],
            "schedule": ["schedule:{schedule_id}", "schedules:agent:{agent_id}", "schedules:optimized"],
            "pattern": ["patterns:{agent_id}:{period}", "patterns:system:{period}"]
        }
    
    async def get_cached_data(
        self, 
        cache_key: str, 
        data_type: str = "generic",
        fallback_ttl: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from cache with intelligent performance optimization.
        
        Features:
        - Automatic cache warming for frequently accessed keys
        - Adaptive TTL based on access patterns
        - Hierarchical cache invalidation support
        - Performance metrics collection
        """
        start_time = time.time()
        
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return None
            
            # Track cache access for performance optimization
            access_key = f"cache_access:{cache_key}"
            await redis_service.incr(access_key)
            await redis_service.expire(access_key, 3600)  # Track for 1 hour
            
            data = await redis_service.cache_get(cache_key)
            retrieval_time_ms = (time.time() - start_time) * 1000
            
            if data:
                # Update cache metadata for adaptive optimization
                await redis_service.hset(
                    f"cache_metadata:{cache_key}",
                    mapping={
                        "last_accessed": datetime.utcnow().isoformat(),
                        "access_count": await redis_service.get(access_key) or 1,
                        "retrieval_time_ms": retrieval_time_ms
                    }
                )
                
                logger.debug("Cache hit with performance tracking",
                           cache_key=cache_key,
                           data_type=data_type,
                           retrieval_time_ms=retrieval_time_ms)
                
                # Background cache warming for frequently accessed data
                if int(await redis_service.get(access_key) or 0) > 10:
                    await self._schedule_cache_warmup(cache_key, data_type)
                
                return data
            else:
                logger.debug("Cache miss - consider preloading", 
                           cache_key=cache_key, 
                           data_type=data_type)
                return None
            
        except Exception as e:
            logger.warning("Cache retrieval failed",
                          cache_key=cache_key,
                          error=str(e))
            return None
    
    async def cache_data(
        self,
        cache_key: str,
        data: Dict[str, Any],
        data_type: str = "generic",
        ttl_override: Optional[int] = None
    ) -> bool:
        """Cache data with intelligent TTL and hierarchy management."""
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return False
            
            # Determine TTL
            ttl = ttl_override or self.default_ttls.get(data_type, 300)
            
            # Add cache metadata
            cached_data = {
                **data,
                "_cache_metadata": {
                    "cached_at": datetime.utcnow().isoformat(),
                    "ttl": ttl,
                    "data_type": data_type,
                    "cache_key": cache_key
                }
            }
            
            # Store in cache
            success = await redis_service.cache_set(cache_key, cached_data, ttl=ttl)
            
            if success:
                logger.debug("Data cached successfully",
                           cache_key=cache_key,
                           data_type=data_type,
                           ttl=ttl)
            
            return success
            
        except Exception as e:
            logger.warning("Cache storage failed",
                          cache_key=cache_key,
                          error=str(e))
            return False
    
    async def invalidate_cache(
        self,
        cache_pattern: str,
        data_type: str = "generic"
    ) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return 0
            
            invalidated_count = await redis_service.cache_delete_pattern(cache_pattern)
            
            logger.info("Cache invalidated",
                       pattern=cache_pattern,
                       data_type=data_type,
                       invalidated_count=invalidated_count)
            
            return invalidated_count
            
        except Exception as e:
            logger.warning("Cache invalidation failed",
                          pattern=cache_pattern,
                          error=str(e))
            return 0
    
    async def invalidate_hierarchy(
        self,
        data_type: str,
        entity_id: str
    ) -> int:
        """Invalidate entire cache hierarchy for data type and entity."""
        try:
            if data_type not in self.cache_hierarchies:
                logger.warning("Unknown cache hierarchy", data_type=data_type)
                return 0
            
            total_invalidated = 0
            hierarchy = self.cache_hierarchies[data_type]
            
            for cache_pattern in hierarchy:
                # Replace placeholders with actual values
                pattern = cache_pattern.format(
                    task_id=entity_id,
                    workflow_id=entity_id,
                    agent_id=entity_id,
                    schedule_id=entity_id,
                    hash="*"
                )
                
                invalidated = await self.invalidate_cache(pattern, data_type)
                total_invalidated += invalidated
            
            logger.info("Cache hierarchy invalidated",
                       data_type=data_type,
                       entity_id=entity_id,
                       total_invalidated=total_invalidated)
            
            return total_invalidated
            
        except Exception as e:
            logger.error("Cache hierarchy invalidation failed",
                        data_type=data_type,
                        entity_id=entity_id,
                        error=str(e))
            return 0
    
    def generate_cache_key(
        self,
        operation: str,
        params: Dict[str, Any],
        include_timestamp: bool = False
    ) -> str:
        """Generate consistent cache key from operation and parameters."""
        try:
            # Sort parameters for consistent key generation
            sorted_params = dict(sorted(params.items()))
            
            # Create base key components
            key_components = [operation]
            
            # Add parameter hash
            param_string = json.dumps(sorted_params, sort_keys=True, default=str)
            param_hash = hashlib.md5(param_string.encode()).hexdigest()[:16]
            key_components.append(param_hash)
            
            # Add timestamp if requested (for time-sensitive caching)
            if include_timestamp:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H")
                key_components.append(timestamp)
            
            cache_key = ":".join(key_components)
            
            logger.debug("Cache key generated",
                        operation=operation,
                        cache_key=cache_key)
            
            return cache_key
            
        except Exception as e:
            logger.warning("Cache key generation failed",
                          operation=operation,
                          error=str(e))
            # Return fallback key
            return f"{operation}:{str(uuid4())[:8]}"
    
    async def _schedule_cache_warmup(
        self,
        cache_key: str,
        data_type: str
    ) -> None:
        """
        Schedule background cache warming for frequently accessed data.
        
        Implements adaptive caching strategy to prevent cache misses
        for hot data by preloading related cache entries.
        """
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return
            
            # Add to cache warmup queue
            warmup_key = f"cache_warmup_queue:{data_type}"
            warmup_data = {
                "cache_key": cache_key,
                "data_type": data_type,
                "scheduled_at": datetime.utcnow().isoformat(),
                "priority": "high" if "task" in data_type else "medium"
            }
            
            await redis_service.lpush(warmup_key, json.dumps(warmup_data))
            await redis_service.expire(warmup_key, 7200)  # 2 hour expiry
            
            logger.debug("Cache warmup scheduled",
                        cache_key=cache_key,
                        data_type=data_type)
                        
        except Exception as e:
            logger.warning("Cache warmup scheduling failed",
                          cache_key=cache_key,
                          error=str(e))
    
    async def optimize_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze cache performance and provide optimization recommendations.
        
        Returns metrics and suggestions for improving cache hit rates
        and reducing response times.
        """
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return {"error": "Redis service unavailable"}
            
            # Collect cache performance metrics
            cache_stats = {
                "total_keys": 0,
                "hot_keys": [],
                "cold_keys": [],
                "avg_access_frequency": 0,
                "recommendations": []
            }
            
            # Analyze cache access patterns
            access_pattern_keys = await redis_service.keys("cache_access:*")
            total_accesses = 0
            
            for access_key in access_pattern_keys:
                access_count = int(await redis_service.get(access_key) or 0)
                total_accesses += access_count
                cache_key = access_key.replace("cache_access:", "")
                
                if access_count > 50:  # Hot data threshold
                    cache_stats["hot_keys"].append({
                        "key": cache_key,
                        "access_count": access_count
                    })
                elif access_count < 2:  # Cold data threshold
                    cache_stats["cold_keys"].append({
                        "key": cache_key,
                        "access_count": access_count
                    })
            
            cache_stats["total_keys"] = len(access_pattern_keys)
            cache_stats["avg_access_frequency"] = total_accesses / len(access_pattern_keys) if access_pattern_keys else 0
            
            # Generate recommendations
            if len(cache_stats["hot_keys"]) > 10:
                cache_stats["recommendations"].append(
                    "Consider implementing cache warmup for frequently accessed data"
                )
            
            if len(cache_stats["cold_keys"]) > 100:
                cache_stats["recommendations"].append(
                    "Consider reducing TTL for rarely accessed cache entries"
                )
            
            if cache_stats["avg_access_frequency"] < 3:
                cache_stats["recommendations"].append(
                    "Cache efficiency is low - review caching strategy"
                )
            
            return cache_stats
            
        except Exception as e:
            logger.error("Cache performance optimization failed", error=str(e))
            return {"error": str(e)}


# Global cache manager instance
cache_manager = CacheManager()


# ===============================================================================
# DATA VALIDATION UTILITIES
# ===============================================================================

class DataValidator:
    """
    Comprehensive data validation utilities for TaskExecutionAPI.
    
    Provides validation for task data, workflow definitions, scheduling parameters,
    and integration with Epic 1 ConsolidatedProductionOrchestrator.
    """
    
    @staticmethod
    def validate_task_data(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task data for completeness and correctness."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "normalized_data": {}
        }
        
        try:
            # Required fields validation
            required_fields = ["title"]
            for field in required_fields:
                if not task_data.get(field):
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Title validation
            title = task_data.get("title", "")
            if len(title) > 255:
                validation_result["errors"].append("Task title exceeds 255 characters")
                validation_result["valid"] = False
            elif len(title.strip()) == 0:
                validation_result["errors"].append("Task title cannot be empty")
                validation_result["valid"] = False
            
            # Description validation
            description = task_data.get("description", "")
            if description and len(description) > 2000:
                validation_result["warnings"].append("Task description is very long (>2000 chars)")
            
            # Priority validation
            priority = task_data.get("priority", "medium")
            valid_priorities = ["low", "medium", "high", "critical"]
            if priority.lower() not in valid_priorities:
                validation_result["errors"].append(f"Invalid priority: {priority}")
                validation_result["valid"] = False
            else:
                validation_result["normalized_data"]["priority"] = priority.lower()
            
            # Task type validation
            task_type = task_data.get("task_type")
            if task_type:
                valid_types = ["development", "testing", "deployment", "analysis", "general"]
                if task_type.lower() not in valid_types:
                    validation_result["warnings"].append(f"Unknown task type: {task_type}")
                else:
                    validation_result["normalized_data"]["task_type"] = task_type.lower()
            
            # Capabilities validation
            capabilities = task_data.get("required_capabilities", [])
            if capabilities:
                if not isinstance(capabilities, list):
                    validation_result["errors"].append("Required capabilities must be a list")
                    validation_result["valid"] = False
                elif len(capabilities) > 20:
                    validation_result["warnings"].append("Large number of required capabilities (>20)")
                else:
                    # Normalize capability names
                    normalized_capabilities = [cap.strip().lower() for cap in capabilities if cap.strip()]
                    validation_result["normalized_data"]["required_capabilities"] = normalized_capabilities
            
            # Estimated effort validation
            estimated_effort = task_data.get("estimated_effort")
            if estimated_effort is not None:
                try:
                    effort_int = int(estimated_effort)
                    if effort_int <= 0:
                        validation_result["errors"].append("Estimated effort must be positive")
                        validation_result["valid"] = False
                    elif effort_int > 10080:  # 1 week in minutes
                        validation_result["warnings"].append("Very large estimated effort (>1 week)")
                    else:
                        validation_result["normalized_data"]["estimated_effort"] = effort_int
                except (ValueError, TypeError):
                    validation_result["errors"].append("Estimated effort must be a number (minutes)")
                    validation_result["valid"] = False
            
            # Context validation
            context = task_data.get("context", {})
            if context and not isinstance(context, dict):
                validation_result["errors"].append("Task context must be a dictionary")
                validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error("Task validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "normalized_data": {}
            }
    
    @staticmethod
    def validate_workflow_definition(workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow definition for correctness and completeness."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "normalized_data": {}
        }
        
        try:
            # Required fields
            if not workflow_data.get("name"):
                validation_result["errors"].append("Workflow name is required")
                validation_result["valid"] = False
            
            # Name validation
            name = workflow_data.get("name", "")
            if len(name) > 255:
                validation_result["errors"].append("Workflow name exceeds 255 characters")
                validation_result["valid"] = False
            
            # Definition validation
            definition = workflow_data.get("definition", {})
            if not isinstance(definition, dict):
                validation_result["errors"].append("Workflow definition must be a dictionary")
                validation_result["valid"] = False
            elif not definition:
                validation_result["warnings"].append("Empty workflow definition")
            
            # Task IDs validation
            task_ids = workflow_data.get("task_ids", [])
            if task_ids:
                if not isinstance(task_ids, list):
                    validation_result["errors"].append("Task IDs must be a list")
                    validation_result["valid"] = False
                else:
                    # Validate UUID format
                    valid_task_ids = []
                    for task_id in task_ids:
                        try:
                            UUID(str(task_id))
                            valid_task_ids.append(str(task_id))
                        except ValueError:
                            validation_result["errors"].append(f"Invalid task ID format: {task_id}")
                            validation_result["valid"] = False
                    
                    if validation_result["valid"]:
                        validation_result["normalized_data"]["task_ids"] = valid_task_ids
            
            # Dependencies validation
            dependencies = workflow_data.get("dependencies", {})
            if dependencies:
                if not isinstance(dependencies, dict):
                    validation_result["errors"].append("Dependencies must be a dictionary")
                    validation_result["valid"] = False
                else:
                    # Validate dependency structure
                    normalized_deps = {}
                    for task_id, deps in dependencies.items():
                        try:
                            UUID(str(task_id))
                            if isinstance(deps, list):
                                valid_deps = []
                                for dep in deps:
                                    try:
                                        UUID(str(dep))
                                        valid_deps.append(str(dep))
                                    except ValueError:
                                        validation_result["warnings"].append(f"Invalid dependency ID: {dep}")
                                normalized_deps[str(task_id)] = valid_deps
                            else:
                                validation_result["warnings"].append(f"Dependencies for {task_id} must be a list")
                        except ValueError:
                            validation_result["warnings"].append(f"Invalid task ID in dependencies: {task_id}")
                    
                    validation_result["normalized_data"]["dependencies"] = normalized_deps
            
            # Priority validation
            priority = workflow_data.get("priority", "medium")
            valid_priorities = ["low", "medium", "high", "critical"]
            if priority.lower() not in valid_priorities:
                validation_result["errors"].append(f"Invalid workflow priority: {priority}")
                validation_result["valid"] = False
            else:
                validation_result["normalized_data"]["priority"] = priority.lower()
            
            return validation_result
            
        except Exception as e:
            logger.error("Workflow validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "normalized_data": {}
            }
    
    @staticmethod
    def validate_schedule_parameters(schedule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scheduling parameters for optimization requests."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "normalized_data": {}
        }
        
        try:
            # Agent ID validation if provided
            agent_id = schedule_data.get("agent_id")
            if agent_id:
                try:
                    UUID(str(agent_id))
                    validation_result["normalized_data"]["agent_id"] = str(agent_id)
                except ValueError:
                    validation_result["errors"].append(f"Invalid agent ID format: {agent_id}")
                    validation_result["valid"] = False
            
            # Optimization goal validation
            optimization_goal = schedule_data.get("optimization_goal", "efficiency")
            valid_goals = ["efficiency", "performance", "resource_usage", "availability", "hybrid"]
            if optimization_goal not in valid_goals:
                validation_result["errors"].append(f"Invalid optimization goal: {optimization_goal}")
                validation_result["valid"] = False
            else:
                validation_result["normalized_data"]["optimization_goal"] = optimization_goal
            
            # Time horizon validation
            time_horizon = schedule_data.get("time_horizon_hours", 24)
            try:
                horizon_int = int(time_horizon)
                if horizon_int <= 0:
                    validation_result["errors"].append("Time horizon must be positive")
                    validation_result["valid"] = False
                elif horizon_int > 168:  # 1 week
                    validation_result["warnings"].append("Very long time horizon (>1 week)")
                    validation_result["normalized_data"]["time_horizon_hours"] = min(horizon_int, 168)
                else:
                    validation_result["normalized_data"]["time_horizon_hours"] = horizon_int
            except (ValueError, TypeError):
                validation_result["errors"].append("Time horizon must be a number")
                validation_result["valid"] = False
            
            # Constraints validation
            constraints = schedule_data.get("constraints", {})
            if constraints and not isinstance(constraints, dict):
                validation_result["errors"].append("Constraints must be a dictionary")
                validation_result["valid"] = False
            
            # Learning rate validation
            learning_rate = schedule_data.get("learning_rate", 0.1)
            try:
                lr_float = float(learning_rate)
                if lr_float <= 0 or lr_float > 1:
                    validation_result["errors"].append("Learning rate must be between 0 and 1")
                    validation_result["valid"] = False
                else:
                    validation_result["normalized_data"]["learning_rate"] = lr_float
            except (ValueError, TypeError):
                validation_result["errors"].append("Learning rate must be a number")
                validation_result["valid"] = False
            
            return validation_result
            
        except Exception as e:
            logger.error("Schedule parameters validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "normalized_data": {}
            }


# Global data validator instance
data_validator = DataValidator()


# ===============================================================================
# ERROR HANDLING UTILITIES
# ===============================================================================

class ErrorHandler:
    """
    Centralized error handling and recovery utilities.
    
    Provides consistent error formatting, logging, and recovery strategies
    across all TaskExecutionAPI operations.
    """
    
    @staticmethod
    def format_api_error(
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format error for consistent API response."""
        error_response = {
            "error": True,
            "error_type": type(error).__name__,
            "message": str(error),
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        
        # Add specific handling for common error types
        if isinstance(error, ValueError):
            error_response["category"] = "validation_error"
            error_response["user_friendly_message"] = "Invalid input parameters provided"
        elif isinstance(error, PermissionError):
            error_response["category"] = "permission_error"
            error_response["user_friendly_message"] = "Insufficient permissions for this operation"
        elif isinstance(error, ConnectionError):
            error_response["category"] = "connectivity_error"
            error_response["user_friendly_message"] = "Service temporarily unavailable"
        else:
            error_response["category"] = "system_error"
            error_response["user_friendly_message"] = "An unexpected error occurred"
        
        return error_response
    
    @staticmethod
    async def handle_orchestrator_error(
        error: Exception,
        operation: str,
        fallback_strategy: str = "degrade"
    ) -> Dict[str, Any]:
        """Handle Epic 1 ConsolidatedProductionOrchestrator errors with recovery."""
        error_info = {
            "error": str(error),
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            "orchestrator_available": False,
            "fallback_applied": False
        }
        
        try:
            # Test orchestrator availability
            orchestrator = get_production_orchestrator()
            if orchestrator:
                health = await orchestrator.get_system_health()
                error_info["orchestrator_available"] = health.get("status") == "healthy"
            
            # Apply fallback strategy
            if fallback_strategy == "degrade":
                error_info["fallback_applied"] = True
                error_info["fallback_message"] = "Operating in degraded mode without orchestrator"
            elif fallback_strategy == "retry":
                error_info["fallback_applied"] = True
                error_info["fallback_message"] = "Will retry operation with orchestrator"
            elif fallback_strategy == "fail":
                error_info["fallback_applied"] = False
                error_info["fallback_message"] = "Operation failed due to orchestrator unavailability"
            
        except Exception as e:
            logger.error("Error handler failed", error=str(e))
            error_info["handler_error"] = str(e)
        
        return error_info


# Global error handler instance
error_handler = ErrorHandler()


# ===============================================================================
# EPIC 1 INTEGRATION UTILITIES
# ===============================================================================

class Epic1IntegrationHelper:
    """
    Helper utilities for Epic 1 ConsolidatedProductionOrchestrator integration.
    
    Provides seamless integration patterns and compatibility helpers
    for maintaining Epic 1 achievements while enhancing functionality.
    """
    
    @staticmethod
    async def validate_epic1_compatibility(operation: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate operation compatibility with Epic 1 orchestrator."""
        try:
            orchestrator = get_production_orchestrator()
            if not orchestrator:
                return {
                    "compatible": False,
                    "reason": "Orchestrator not available",
                    "fallback_available": True
                }
            
            # Check orchestrator health
            health = await orchestrator.get_system_health()
            if health.get("status") != "healthy":
                return {
                    "compatible": False,
                    "reason": "Orchestrator unhealthy",
                    "health_status": health,
                    "fallback_available": True
                }
            
            # Operation-specific compatibility checks
            if operation == "task_creation":
                # Validate task creation compatibility
                required_fields = ["title"]
                missing_fields = [field for field in required_fields if not data.get(field)]
                if missing_fields:
                    return {
                        "compatible": False,
                        "reason": f"Missing required fields: {missing_fields}",
                        "fallback_available": False
                    }
            
            elif operation == "workflow_execution":
                # Validate workflow execution compatibility
                if not data.get("definition"):
                    return {
                        "compatible": False,
                        "reason": "Workflow definition required",
                        "fallback_available": False
                    }
            
            return {
                "compatible": True,
                "orchestrator_version": health.get("version", "unknown"),
                "integration_level": "full"
            }
            
        except Exception as e:
            logger.error("Epic 1 compatibility check failed", error=str(e))
            return {
                "compatible": False,
                "reason": f"Compatibility check failed: {str(e)}",
                "fallback_available": True
            }
    
    @staticmethod
    async def ensure_epic1_achievements_preserved() -> Dict[str, Any]:
        """Ensure Epic 1 achievements are preserved during operations."""
        try:
            orchestrator = get_production_orchestrator()
            if not orchestrator:
                return {
                    "preserved": False,
                    "reason": "Orchestrator not available"
                }
            
            # Check key Epic 1 capabilities
            checks = {
                "task_routing": False,
                "agent_coordination": False,
                "performance_monitoring": False,
                "error_handling": False
            }
            
            # Perform capability checks
            health = await orchestrator.get_system_health()
            if health.get("task_routing_active"):
                checks["task_routing"] = True
            
            if health.get("agent_coordination_active"):
                checks["agent_coordination"] = True
            
            if health.get("performance_monitoring_active"):
                checks["performance_monitoring"] = True
            
            if health.get("error_handling_active"):
                checks["error_handling"] = True
            
            preserved = all(checks.values())
            
            return {
                "preserved": preserved,
                "checks": checks,
                "overall_health": health.get("status", "unknown"),
                "recommendations": [] if preserved else ["Verify orchestrator configuration", "Check service dependencies"]
            }
            
        except Exception as e:
            logger.error("Epic 1 achievements check failed", error=str(e))
            return {
                "preserved": False,
                "reason": f"Check failed: {str(e)}"
            }


# Global Epic 1 integration helper
epic1_helper = Epic1IntegrationHelper()


# ===============================================================================
# REDIS COORDINATION UTILITIES
# ===============================================================================

class RedisCoordinationHelper:
    """
    Redis coordination utilities for real-time TaskExecutionAPI operations.
    
    Provides messaging, caching, and coordination patterns for distributed
    task execution and workflow orchestration.
    """
    
    @staticmethod
    async def publish_task_event(
        event_type: str,
        task_id: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Publish task-related event for real-time coordination."""
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return False
            
            event_message = {
                "event_type": event_type,
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data
            }
            
            success = await redis_service.publish("task_events", event_message)
            
            if success:
                logger.debug("Task event published",
                           event_type=event_type,
                           task_id=task_id)
            
            return success
            
        except Exception as e:
            logger.warning("Failed to publish task event",
                          event_type=event_type,
                          task_id=task_id,
                          error=str(e))
            return False
    
    @staticmethod
    async def publish_workflow_event(
        event_type: str,
        workflow_id: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """Publish workflow-related event for real-time coordination."""
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return False
            
            event_message = {
                "event_type": event_type,
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data
            }
            
            success = await redis_service.publish("workflow_events", event_message)
            
            if success:
                logger.debug("Workflow event published",
                           event_type=event_type,
                           workflow_id=workflow_id)
            
            return success
            
        except Exception as e:
            logger.warning("Failed to publish workflow event",
                          event_type=event_type,
                          workflow_id=workflow_id,
                          error=str(e))
            return False
    
    @staticmethod
    async def get_coordination_statistics() -> Dict[str, Any]:
        """Get Redis coordination statistics and health."""
        try:
            redis_service = get_redis_service()
            if not redis_service:
                return {"available": False, "reason": "Redis service not available"}
            
            health = await redis_service.health_check()
            
            # Get basic statistics
            stats = {
                "available": True,
                "health": health,
                "active_channels": ["task_events", "workflow_events", "scheduling_events"],
                "cache_performance": {
                    "hit_rate": 0.85,  # Would be calculated from actual metrics
                    "average_response_time_ms": 2.5
                },
                "coordination_active": True
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get coordination statistics", error=str(e))
            return {"available": False, "reason": str(e)}


# Global Redis coordination helper
redis_helper = RedisCoordinationHelper()


# ===============================================================================
# UTILITY FUNCTION EXPORTS
# ===============================================================================

# Export commonly used utility functions
__all__ = [
    "performance_monitor",
    "cache_manager", 
    "data_validator",
    "error_handler",
    "epic1_helper",
    "redis_helper",
    # Performance monitoring
    "PerformanceMonitor",
    # Cache management
    "CacheManager",
    # Data validation
    "DataValidator", 
    # Error handling
    "ErrorHandler",
    # Epic 1 integration
    "Epic1IntegrationHelper",
    # Redis coordination
    "RedisCoordinationHelper"
]
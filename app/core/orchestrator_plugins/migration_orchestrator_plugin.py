"""
Migration Orchestrator Plugin for LeanVibe Agent Hive 2.0

Epic 1 Phase 2.2: Consolidated plugin architecture
Consolidates orchestrator_migration_adapter.py capabilities into the unified plugin system.

Key Features:
- Backward compatibility layer for legacy orchestrator code
- Seamless migration support from old orchestrator architectures
- API compatibility preservation
- Legacy factory function delegation
- Migration state tracking and validation
- Performance monitoring during migration phases

Epic 1 Performance Targets:
- <10ms legacy API translation overhead
- <5MB memory overhead for compatibility layer
- Zero functionality loss during migration
- Transparent performance preservation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .base_plugin import OrchestratorPlugin, PluginMetadata, PluginError
from ..simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskPriority
from ..logging_service import get_component_logger

logger = get_component_logger("migration_orchestrator_plugin")


class MigrationPhase(Enum):
    """Migration phases for orchestrator consolidation."""
    LEGACY_COMPATIBILITY = "legacy_compatibility"
    GRADUAL_MIGRATION = "gradual_migration"
    FEATURE_PARITY = "feature_parity"
    FULL_CONSOLIDATION = "full_consolidation"
    POST_MIGRATION = "post_migration"


@dataclass
class LegacyAPIMapping:
    """Mapping for legacy API compatibility."""
    legacy_method: str
    new_method: str
    parameter_mapping: Dict[str, str]
    return_value_transform: Optional[str] = None
    deprecation_warning: bool = True


@dataclass
class MigrationStatus:
    """Status of orchestrator migration."""
    phase: MigrationPhase
    legacy_calls_count: int
    migration_errors: List[str]
    compatibility_score: float  # 0.0 to 1.0
    last_updated: datetime


class MigrationOrchestratorPlugin(OrchestratorPlugin):
    """
    Migration orchestrator plugin providing backward compatibility.
    
    Epic 1 Phase 2.2: Consolidation of orchestrator_migration_adapter.py 
    capabilities into the unified plugin architecture for SimpleOrchestrator integration.
    
    Provides:
    - Seamless backward compatibility for legacy orchestrator calls
    - API translation and delegation
    - Migration state tracking and validation
    - Legacy factory function support
    - Performance monitoring with minimal overhead
    """
    
    def __init__(self):
        super().__init__(
            metadata=PluginMetadata(
                name="migration_orchestrator",
                version="2.2.0",
                description="Backward compatibility layer for legacy orchestrator migration",
                author="LeanVibe Agent Hive",
                capabilities=["legacy_compatibility", "api_translation", "migration_tracking"],
                dependencies=["simple_orchestrator"],
                epic_phase="Epic 1 Phase 2.2"
            )
        )
        
        # Migration state
        self.migration_phase = MigrationPhase.LEGACY_COMPATIBILITY
        self.legacy_calls_count = 0
        self.migration_errors: List[str] = []
        self.compatibility_score = 1.0
        
        # Legacy API mappings
        self.api_mappings: Dict[str, LegacyAPIMapping] = {}
        
        # Performance tracking for Epic 1 targets  
        self.operation_times: Dict[str, List[float]] = {}
        self.translation_overhead_ms: List[float] = []
        
        # Legacy state tracking
        self.legacy_agents: Dict[str, Any] = {}
        self.legacy_tasks: Dict[str, Any] = {}
        
        # Compatibility layer cache
        self.translation_cache: Dict[str, Any] = {}
        
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the migration orchestrator plugin."""
        await super().initialize(context)
        
        self.orchestrator = context.get("orchestrator")
        if not isinstance(self.orchestrator, SimpleOrchestrator):
            raise PluginError("MigrationOrchestratorPlugin requires SimpleOrchestrator")
        
        # Initialize legacy API mappings
        await self._initialize_api_mappings()
        
        # Set up compatibility layer
        await self._setup_compatibility_layer()
        
        logger.info("Migration Orchestrator Plugin initialized with backward compatibility")
        
    async def _initialize_api_mappings(self):
        """Initialize legacy API mappings for compatibility."""
        self.api_mappings = {
            # Legacy AgentOrchestrator methods
            "get_agent_orchestrator": LegacyAPIMapping(
                legacy_method="get_agent_orchestrator",
                new_method="get_simple_orchestrator",
                parameter_mapping={},
                return_value_transform="wrap_orchestrator"
            ),
            "initialize_orchestrator": LegacyAPIMapping(
                legacy_method="initialize_orchestrator",
                new_method="initialize",
                parameter_mapping={},
                return_value_transform="none"
            ),
            "shutdown_orchestrator": LegacyAPIMapping(
                legacy_method="shutdown_orchestrator", 
                new_method="shutdown",
                parameter_mapping={},
                return_value_transform="none"
            ),
            
            # Legacy agent management methods
            "spawn_agent": LegacyAPIMapping(
                legacy_method="spawn_agent",
                new_method="spawn_agent",
                parameter_mapping={
                    "role": "role",
                    "agent_id": "agent_id"
                },
                return_value_transform="none"
            ),
            "shutdown_agent": LegacyAPIMapping(
                legacy_method="shutdown_agent",
                new_method="shutdown_agent", 
                parameter_mapping={
                    "agent_id": "agent_id",
                    "graceful": "graceful"
                },
                return_value_transform="none"
            ),
            
            # Legacy task delegation methods
            "delegate_task": LegacyAPIMapping(
                legacy_method="delegate_task",
                new_method="delegate_task",
                parameter_mapping={
                    "task_description": "task_description",
                    "task_type": "task_type",
                    "priority": "priority",
                    "preferred_agent_role": "preferred_agent_role"
                },
                return_value_transform="none"
            ),
            
            # Legacy status methods
            "get_system_status": LegacyAPIMapping(
                legacy_method="get_system_status",
                new_method="get_system_status",
                parameter_mapping={},
                return_value_transform="legacy_status_format"
            )
        }
        
        logger.info(f"Initialized {len(self.api_mappings)} legacy API mappings")
    
    async def _setup_compatibility_layer(self):
        """Set up the compatibility layer for legacy code."""
        # Register compatibility layer with orchestrator
        # This allows the orchestrator to recognize legacy calls
        pass
    
    async def translate_legacy_call(
        self,
        method_name: str,
        args: tuple,
        kwargs: Dict[str, Any]
    ) -> Any:
        """
        Translate legacy orchestrator calls to new API with Epic 1 performance tracking.
        
        Args:
            method_name: Legacy method name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result from new API call
        """
        import time
        start_time_ms = time.time()
        
        try:
            # Check if method has mapping
            if method_name not in self.api_mappings:
                error_msg = f"Unknown legacy method: {method_name}"
                self.migration_errors.append(error_msg)
                raise PluginError(error_msg)
            
            mapping = self.api_mappings[method_name]
            
            # Log deprecation warning if enabled
            if mapping.deprecation_warning:
                logger.warning(f"Legacy method '{method_name}' is deprecated. Use '{mapping.new_method}' instead.")
            
            # Translate parameters
            new_kwargs = await self._translate_parameters(mapping.parameter_mapping, kwargs)
            
            # Check translation cache
            cache_key = f"{mapping.new_method}_{hash(str(new_kwargs))}"
            if cache_key in self.translation_cache:
                cache_entry = self.translation_cache[cache_key]
                if datetime.utcnow() - cache_entry["timestamp"] < timedelta(minutes=1):
                    return cache_entry["result"]
            
            # Call new method
            new_method = getattr(self.orchestrator, mapping.new_method)
            if asyncio.iscoroutinefunction(new_method):
                result = await new_method(*args, **new_kwargs)
            else:
                result = new_method(*args, **new_kwargs)
            
            # Transform return value if needed
            if mapping.return_value_transform:
                result = await self._transform_return_value(result, mapping.return_value_transform)
            
            # Cache result
            self.translation_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.utcnow()
            }
            
            # Track usage
            self.legacy_calls_count += 1
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_translation_time(operation_time_ms)
            
            logger.debug(f"Translated legacy call: {method_name} -> {mapping.new_method} in {operation_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to translate legacy call {method_name}: {e}"
            self.migration_errors.append(error_msg)
            logger.error(error_msg)
            raise PluginError(error_msg) from e
    
    async def _translate_parameters(
        self,
        parameter_mapping: Dict[str, str],
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Translate legacy parameters to new API format."""
        new_kwargs = {}
        
        for legacy_param, new_param in parameter_mapping.items():
            if legacy_param in kwargs:
                new_kwargs[new_param] = kwargs[legacy_param]
        
        # Copy any unmapped parameters
        for key, value in kwargs.items():
            if key not in parameter_mapping and key not in new_kwargs:
                new_kwargs[key] = value
        
        return new_kwargs
    
    async def _transform_return_value(self, result: Any, transform_type: str) -> Any:
        """Transform return values to match legacy API expectations."""
        if transform_type == "wrap_orchestrator":
            # Wrap orchestrator in legacy compatibility wrapper
            return LegacyOrchestratorWrapper(self.orchestrator, self)
        
        elif transform_type == "legacy_status_format":
            # Transform status to legacy format
            if isinstance(result, dict):
                # Add legacy fields for backward compatibility
                legacy_result = result.copy()
                legacy_result["orchestrator_type"] = "simple_orchestrator"
                legacy_result["migration_plugin_active"] = True
                legacy_result["legacy_compatibility"] = {
                    "phase": self.migration_phase.value,
                    "legacy_calls_count": self.legacy_calls_count,
                    "compatibility_score": self.compatibility_score
                }
                return legacy_result
        
        elif transform_type == "none":
            return result
        
        else:
            logger.warning(f"Unknown return value transform: {transform_type}")
            return result
    
    def create_legacy_factory_functions(self) -> Dict[str, Callable]:
        """Create legacy factory functions for backward compatibility."""
        async def get_agent_orchestrator():
            """Legacy factory function - get_agent_orchestrator."""
            return await self.translate_legacy_call("get_agent_orchestrator", (), {})
        
        async def initialize_orchestrator(**kwargs):
            """Legacy factory function - initialize_orchestrator."""
            return await self.translate_legacy_call("initialize_orchestrator", (), kwargs)
        
        async def shutdown_orchestrator(**kwargs):
            """Legacy factory function - shutdown_orchestrator."""
            return await self.translate_legacy_call("shutdown_orchestrator", (), kwargs)
        
        return {
            "get_agent_orchestrator": get_agent_orchestrator,
            "initialize_orchestrator": initialize_orchestrator,
            "shutdown_orchestrator": shutdown_orchestrator
        }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status with Epic 1 performance metrics."""
        import time
        start_time_ms = time.time()
        
        try:
            # Calculate compatibility metrics
            avg_translation_time = sum(self.translation_overhead_ms) / len(self.translation_overhead_ms) if self.translation_overhead_ms else 0.0
            max_translation_time = max(self.translation_overhead_ms) if self.translation_overhead_ms else 0.0
            
            # Calculate memory overhead
            memory_overhead_mb = self._calculate_memory_overhead()
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            
            status = {
                "migration_phase": self.migration_phase.value,
                "legacy_calls_count": self.legacy_calls_count,
                "migration_errors": self.migration_errors[-10:],  # Last 10 errors
                "compatibility_score": self.compatibility_score,
                "performance_metrics": {
                    "avg_translation_time_ms": round(avg_translation_time, 2),
                    "max_translation_time_ms": round(max_translation_time, 2),
                    "translation_count": len(self.translation_overhead_ms),
                    "memory_overhead_mb": round(memory_overhead_mb, 2),
                    "epic1_compliant": {
                        "translation_overhead_under_10ms": avg_translation_time < 10.0,
                        "memory_overhead_under_5mb": memory_overhead_mb < 5.0
                    }
                },
                "api_mappings": {
                    "total_mappings": len(self.api_mappings),
                    "cached_translations": len(self.translation_cache)
                },
                "timestamp": datetime.utcnow().isoformat(),
                "status_operation_time_ms": round(operation_time_ms, 2)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {"error": str(e)}
    
    def _record_translation_time(self, time_ms: float) -> None:
        """Record translation time for Epic 1 performance monitoring."""
        self.translation_overhead_ms.append(time_ms)
        
        # Keep only last 100 measurements for memory efficiency
        if len(self.translation_overhead_ms) > 100:
            self.translation_overhead_ms.pop(0)
        
        # Log performance warnings for Epic 1 targets
        if time_ms > 10.0:
            logger.warning("Translation overhead high",
                         translation_time_ms=time_ms,
                         target_ms=10.0)
    
    def _calculate_memory_overhead(self) -> float:
        """Calculate memory overhead from compatibility layer."""
        try:
            import sys
            
            # Calculate size of compatibility structures
            mappings_size = sys.getsizeof(self.api_mappings)
            cache_size = sys.getsizeof(self.translation_cache)
            errors_size = sys.getsizeof(self.migration_errors)
            times_size = sys.getsizeof(self.translation_overhead_ms)
            
            total_bytes = mappings_size + cache_size + errors_size + times_size
            return total_bytes / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 1.0  # Default estimate
    
    async def validate_migration_compatibility(self) -> Dict[str, Any]:
        """Validate migration compatibility and identify issues."""
        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "compatibility_checks": {},
            "issues_found": [],
            "recommendations": []
        }
        
        try:
            # Check API mapping completeness
            validation_results["compatibility_checks"]["api_mappings"] = {
                "total": len(self.api_mappings),
                "status": "complete" if len(self.api_mappings) >= 5 else "incomplete"
            }
            
            # Check translation performance
            if self.translation_overhead_ms:
                avg_time = sum(self.translation_overhead_ms) / len(self.translation_overhead_ms)
                validation_results["compatibility_checks"]["performance"] = {
                    "avg_translation_ms": round(avg_time, 2),
                    "status": "optimal" if avg_time < 10.0 else "needs_optimization"
                }
                
                if avg_time >= 10.0:
                    validation_results["issues_found"].append("Translation overhead exceeds 10ms target")
                    validation_results["recommendations"].append("Optimize parameter translation and caching")
            
            # Check memory usage
            memory_overhead = self._calculate_memory_overhead()
            validation_results["compatibility_checks"]["memory"] = {
                "overhead_mb": round(memory_overhead, 2),
                "status": "acceptable" if memory_overhead < 5.0 else "high"
            }
            
            if memory_overhead >= 5.0:
                validation_results["issues_found"].append("Memory overhead exceeds 5MB target")
                validation_results["recommendations"].append("Reduce cache sizes and optimize data structures")
            
            # Check error rate
            error_rate = len(self.migration_errors) / max(self.legacy_calls_count, 1)
            validation_results["compatibility_checks"]["error_rate"] = {
                "percentage": round(error_rate * 100, 2),
                "status": "acceptable" if error_rate < 0.01 else "high"
            }
            
            if error_rate >= 0.01:
                validation_results["issues_found"].append("Migration error rate exceeds 1%")
                validation_results["recommendations"].append("Review and fix API mapping issues")
            
            # Overall assessment
            total_checks = len(validation_results["compatibility_checks"])
            passed_checks = len([c for c in validation_results["compatibility_checks"].values() 
                               if c.get("status") in ["complete", "optimal", "acceptable"]])
            
            validation_results["overall"] = {
                "compatibility_score": round(passed_checks / total_checks, 2),
                "status": "ready" if passed_checks == total_checks else "needs_attention",
                "issues_count": len(validation_results["issues_found"])
            }
            
            return validation_results
            
        except Exception as e:
            validation_results["error"] = str(e)
            return validation_results
    
    async def advance_migration_phase(self, target_phase: MigrationPhase) -> Dict[str, Any]:
        """Advance to the next migration phase."""
        current_phase_index = list(MigrationPhase).index(self.migration_phase)
        target_phase_index = list(MigrationPhase).index(target_phase)
        
        if target_phase_index <= current_phase_index:
            return {
                "success": False,
                "message": f"Cannot move backwards from {self.migration_phase.value} to {target_phase.value}"
            }
        
        # Validate readiness for next phase
        validation = await self.validate_migration_compatibility()
        if validation["overall"]["status"] != "ready":
            return {
                "success": False,
                "message": "Migration compatibility issues must be resolved before advancing",
                "validation": validation
            }
        
        old_phase = self.migration_phase
        self.migration_phase = target_phase
        
        logger.info(f"Advanced migration phase: {old_phase.value} -> {target_phase.value}")
        
        return {
            "success": True,
            "old_phase": old_phase.value,
            "new_phase": target_phase.value,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics for Epic 1 monitoring."""
        if not self.translation_overhead_ms:
            return {"no_data": "No translation operations recorded"}
        
        import statistics
        
        return {
            "translation_performance": {
                "avg_ms": round(statistics.mean(self.translation_overhead_ms), 2),
                "max_ms": round(max(self.translation_overhead_ms), 2),
                "min_ms": round(min(self.translation_overhead_ms), 2),
                "count": len(self.translation_overhead_ms),
                "last_ms": round(self.translation_overhead_ms[-1], 2),
                "epic1_compliant": statistics.mean(self.translation_overhead_ms) < 10.0
            },
            "usage_statistics": {
                "legacy_calls_count": self.legacy_calls_count,
                "cached_translations": len(self.translation_cache),
                "migration_errors": len(self.migration_errors)
            },
            "memory_efficiency": {
                "overhead_mb": round(self._calculate_memory_overhead(), 2),
                "epic1_compliant": self._calculate_memory_overhead() < 5.0
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Clear caches and mappings
        self.api_mappings.clear()
        self.translation_cache.clear()
        self.legacy_agents.clear()
        self.legacy_tasks.clear()
        self.operation_times.clear()
        self.translation_overhead_ms.clear()
        self.migration_errors.clear()
        
        await super().cleanup()
        
        logger.info("Migration Orchestrator Plugin cleanup complete")


class LegacyOrchestratorWrapper:
    """Wrapper class to provide legacy orchestrator interface."""
    
    def __init__(self, orchestrator: SimpleOrchestrator, migration_plugin: MigrationOrchestratorPlugin):
        self._orchestrator = orchestrator
        self._migration_plugin = migration_plugin
    
    async def spawn_agent(self, *args, **kwargs):
        """Legacy spawn_agent method."""
        return await self._migration_plugin.translate_legacy_call("spawn_agent", args, kwargs)
    
    async def shutdown_agent(self, *args, **kwargs):
        """Legacy shutdown_agent method."""
        return await self._migration_plugin.translate_legacy_call("shutdown_agent", args, kwargs)
    
    async def delegate_task(self, *args, **kwargs):
        """Legacy delegate_task method."""
        return await self._migration_plugin.translate_legacy_call("delegate_task", args, kwargs)
    
    async def get_system_status(self, *args, **kwargs):
        """Legacy get_system_status method."""
        return await self._migration_plugin.translate_legacy_call("get_system_status", args, kwargs)
    
    # Add other legacy methods as needed
    
    def __getattr__(self, name):
        """Fallback for unmapped methods."""
        if hasattr(self._orchestrator, name):
            attr = getattr(self._orchestrator, name)
            if callable(attr):
                async def wrapper(*args, **kwargs):
                    return await self._migration_plugin.translate_legacy_call(name, args, kwargs)
                return wrapper
            return attr
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def create_migration_orchestrator_plugin() -> MigrationOrchestratorPlugin:
    """Factory function to create migration orchestrator plugin."""
    return MigrationOrchestratorPlugin()


# Export for SimpleOrchestrator integration
__all__ = [
    'MigrationOrchestratorPlugin',
    'MigrationPhase',
    'LegacyAPIMapping',
    'MigrationStatus',
    'LegacyOrchestratorWrapper',
    'create_migration_orchestrator_plugin'
]
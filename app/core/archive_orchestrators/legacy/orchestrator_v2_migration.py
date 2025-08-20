"""
OrchestratorV2 Migration Adapters - Strangler Fig Pattern Implementation
LeanVibe Agent Hive 2.0 - Phase 0 POC Week 2

This module implements migration adapters for gradually transitioning from 35+
legacy orchestrator implementations to OrchestratorV2 using the Strangler Fig pattern.

Migration Strategy (Gemini CLI validated):
1. Create adapters that expose legacy interfaces
2. Internally route to OrchestratorV2 with appropriate plugins
3. Support dark launch / shadow testing
4. Enable gradual cutover with rollback capability
5. Comprehensive validation between old and new systems

Supported Legacy Orchestrators:
- orchestrator.py → Core adapter
- production_orchestrator.py → ProductionPlugin adapter  
- performance_orchestrator.py → PerformancePlugin adapter
- automated_orchestrator.py → AutomationPlugin adapter
- development_orchestrator.py → DevelopmentPlugin adapter
- + 30 more implementations via plugin combinations
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

import structlog

from .orchestrator_v2 import (
    OrchestratorV2, 
    OrchestratorConfig,
    AgentRole, 
    AgentStatus,
    Task,
    TaskExecution,
    AgentInstance
)
from .orchestrator_v2_plugins import (
    ProductionPlugin,
    PerformancePlugin, 
    AutomationPlugin,
    DevelopmentPlugin,
    MonitoringPlugin,
    create_standard_plugin_set,
    create_production_plugin_set,
    create_development_plugin_set
)

logger = structlog.get_logger("orchestrator_migration")

# ================================================================================
# Migration Configuration and Control
# ================================================================================

class MigrationMode(str, Enum):
    """Migration modes for gradual rollout."""
    LEGACY_ONLY = "legacy_only"           # Use only legacy orchestrator
    SHADOW_TESTING = "shadow_testing"     # Run both, compare results
    CANARY_ROLLOUT = "canary_rollout"     # Route percentage to V2
    V2_PRIMARY = "v2_primary"             # V2 primary, legacy fallback
    V2_ONLY = "v2_only"                   # Use only V2

@dataclass
class MigrationConfig:
    """Configuration for migration process."""
    mode: MigrationMode = MigrationMode.SHADOW_TESTING
    v2_traffic_percentage: float = 0.0      # Percentage of traffic to V2
    enable_comparison: bool = True           # Compare V2 vs legacy results
    enable_fallback: bool = True             # Fallback to legacy on V2 failure
    max_response_time_diff_ms: float = 1000  # Max acceptable response time difference
    max_error_rate_diff: float = 0.05        # Max acceptable error rate difference
    comparison_sample_rate: float = 1.0      # Sample rate for comparison (1.0 = all requests)

class MigrationMetrics:
    """Tracks migration performance and comparison metrics."""
    
    def __init__(self):
        self.legacy_metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "avg_response_time_ms": 0.0,
            "response_times": []
        }
        
        self.v2_metrics = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "avg_response_time_ms": 0.0,
            "response_times": []
        }
        
        self.comparison_results = {
            "total_comparisons": 0,
            "identical_results": 0,
            "performance_improvements": 0,
            "performance_regressions": 0,
            "error_differences": 0
        }
    
    def record_legacy_request(self, success: bool, response_time_ms: float):
        """Record legacy orchestrator request metrics."""
        self.legacy_metrics["requests"] += 1
        self.legacy_metrics["response_times"].append(response_time_ms)
        
        if success:
            self.legacy_metrics["successes"] += 1
        else:
            self.legacy_metrics["failures"] += 1
        
        # Update running average
        times = self.legacy_metrics["response_times"]
        self.legacy_metrics["avg_response_time_ms"] = sum(times) / len(times)
        
        # Keep only last 100 times
        if len(times) > 100:
            self.legacy_metrics["response_times"] = times[-100:]
    
    def record_v2_request(self, success: bool, response_time_ms: float):
        """Record V2 orchestrator request metrics."""
        self.v2_metrics["requests"] += 1
        self.v2_metrics["response_times"].append(response_time_ms)
        
        if success:
            self.v2_metrics["successes"] += 1
        else:
            self.v2_metrics["failures"] += 1
        
        # Update running average
        times = self.v2_metrics["response_times"]
        self.v2_metrics["avg_response_time_ms"] = sum(times) / len(times)
        
        # Keep only last 100 times
        if len(times) > 100:
            self.v2_metrics["response_times"] = times[-100:]
    
    def record_comparison(self, results_match: bool, v2_faster: bool, error_diff: bool):
        """Record comparison between legacy and V2."""
        self.comparison_results["total_comparisons"] += 1
        
        if results_match:
            self.comparison_results["identical_results"] += 1
        
        if v2_faster:
            self.comparison_results["performance_improvements"] += 1
        else:
            self.comparison_results["performance_regressions"] += 1
        
        if error_diff:
            self.comparison_results["error_differences"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get migration metrics summary."""
        legacy_error_rate = (
            self.legacy_metrics["failures"] / max(self.legacy_metrics["requests"], 1)
        )
        v2_error_rate = (
            self.v2_metrics["failures"] / max(self.v2_metrics["requests"], 1)
        )
        
        return {
            "legacy_metrics": {
                **self.legacy_metrics,
                "error_rate": legacy_error_rate
            },
            "v2_metrics": {
                **self.v2_metrics,
                "error_rate": v2_error_rate
            },
            "comparison_results": self.comparison_results,
            "migration_health": {
                "v2_performance_improvement": (
                    (self.legacy_metrics["avg_response_time_ms"] - self.v2_metrics["avg_response_time_ms"]) /
                    max(self.legacy_metrics["avg_response_time_ms"], 1) * 100
                ) if self.v2_metrics["avg_response_time_ms"] > 0 else 0,
                "v2_error_rate_improvement": (legacy_error_rate - v2_error_rate) * 100,
                "result_consistency_rate": (
                    self.comparison_results["identical_results"] /
                    max(self.comparison_results["total_comparisons"], 1) * 100
                )
            }
        }

# ================================================================================
# Base Migration Adapter
# ================================================================================

class BaseMigrationAdapter(ABC):
    """
    Base migration adapter implementing Strangler Fig pattern.
    
    Gemini CLI recommendation: Stateless adapters with translation logic.
    """
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.metrics = MigrationMetrics()
        self.orchestrator_v2: Optional[OrchestratorV2] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the adapter and V2 orchestrator."""
        if self._initialized:
            return
        
        # Create V2 orchestrator with appropriate plugins
        plugins = self._get_plugin_configuration()
        orchestrator_config = self._get_orchestrator_config()
        
        self.orchestrator_v2 = OrchestratorV2(orchestrator_config, plugins)
        await self.orchestrator_v2.initialize()
        
        self._initialized = True
        logger.info("Migration adapter initialized", 
                   adapter_type=self.__class__.__name__,
                   migration_mode=self.config.mode.value)
    
    @abstractmethod
    def _get_plugin_configuration(self) -> List[type]:
        """Get plugin configuration for this adapter type."""
        pass
    
    @abstractmethod
    def _get_orchestrator_config(self) -> OrchestratorConfig:
        """Get orchestrator configuration for this adapter type."""
        pass
    
    async def delegate_task_with_migration(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate task through migration layer.
        
        Implements Gemini CLI recommended shadow testing and comparison.
        """
        await self.initialize()
        
        # Convert legacy task format to V2 Task object
        v2_task = self._convert_to_v2_task(task_data)
        
        if self.config.mode == MigrationMode.LEGACY_ONLY:
            return await self._delegate_legacy_only(task_data)
        
        elif self.config.mode == MigrationMode.SHADOW_TESTING:
            return await self._delegate_shadow_testing(task_data, v2_task)
        
        elif self.config.mode == MigrationMode.CANARY_ROLLOUT:
            return await self._delegate_canary_rollout(task_data, v2_task)
        
        elif self.config.mode == MigrationMode.V2_PRIMARY:
            return await self._delegate_v2_primary(task_data, v2_task)
        
        elif self.config.mode == MigrationMode.V2_ONLY:
            return await self._delegate_v2_only(v2_task)
        
        else:
            raise ValueError(f"Unknown migration mode: {self.config.mode}")
    
    async def _delegate_legacy_only(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to legacy orchestrator only."""
        start_time = time.perf_counter()
        
        try:
            result = await self._call_legacy_orchestrator(task_data)
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_legacy_request(True, response_time)
            return result
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_legacy_request(False, response_time)
            raise
    
    async def _delegate_shadow_testing(self, task_data: Dict[str, Any], v2_task: Task) -> Dict[str, Any]:
        """
        Shadow testing: Run both systems, return legacy result, compare outcomes.
        Gemini CLI recommendation: Compare results without impacting production.
        """
        # Always run legacy (primary)
        legacy_start = time.perf_counter()
        legacy_result = None
        legacy_error = None
        
        try:
            legacy_result = await self._call_legacy_orchestrator(task_data)
            legacy_time = (time.perf_counter() - legacy_start) * 1000
            self.metrics.record_legacy_request(True, legacy_time)
        except Exception as e:
            legacy_time = (time.perf_counter() - legacy_start) * 1000
            self.metrics.record_legacy_request(False, legacy_time)
            legacy_error = e
        
        # Run V2 in shadow (async, don't wait for result in production path)
        if self.config.enable_comparison:
            asyncio.create_task(self._run_v2_shadow_test(v2_task, legacy_result, legacy_time))
        
        # Return legacy result (or raise legacy error)
        if legacy_error:
            raise legacy_error
        return legacy_result
    
    async def _run_v2_shadow_test(self, v2_task: Task, legacy_result: Any, legacy_time: float):
        """Run V2 orchestrator as shadow test."""
        v2_start = time.perf_counter()
        v2_result = None
        v2_error = None
        
        try:
            task_id = await self.orchestrator_v2.delegate_task(v2_task)
            # TODO: Wait for task completion and get result
            v2_result = {"task_id": task_id, "status": "completed"}
            v2_time = (time.perf_counter() - v2_start) * 1000
            self.metrics.record_v2_request(True, v2_time)
        except Exception as e:
            v2_time = (time.perf_counter() - v2_start) * 1000
            self.metrics.record_v2_request(False, v2_time)
            v2_error = e
        
        # Compare results
        await self._compare_results(legacy_result, v2_result, legacy_time, v2_time, v2_error)
    
    async def _compare_results(self, legacy_result: Any, v2_result: Any, legacy_time: float, v2_time: float, v2_error: Optional[Exception]):
        """Compare legacy and V2 results for analysis."""
        results_match = self._results_equivalent(legacy_result, v2_result)
        v2_faster = v2_time < legacy_time
        error_diff = (legacy_result is None) != (v2_error is not None)
        
        self.metrics.record_comparison(results_match, v2_faster, error_diff)
        
        # Log significant differences
        time_diff = abs(legacy_time - v2_time)
        if time_diff > self.config.max_response_time_diff_ms:
            logger.warning("Significant response time difference", 
                         legacy_time_ms=legacy_time,
                         v2_time_ms=v2_time,
                         difference_ms=time_diff)
        
        if not results_match:
            logger.warning("Result mismatch between legacy and V2", 
                         legacy_type=type(legacy_result).__name__ if legacy_result else "None",
                         v2_type=type(v2_result).__name__ if v2_result else "None",
                         v2_error=str(v2_error) if v2_error else None)
    
    def _results_equivalent(self, legacy_result: Any, v2_result: Any) -> bool:
        """Check if legacy and V2 results are equivalent."""
        # TODO: Implement sophisticated result comparison
        # For now, just check basic equality
        return legacy_result == v2_result
    
    async def _delegate_canary_rollout(self, task_data: Dict[str, Any], v2_task: Task) -> Dict[str, Any]:
        """Canary rollout: Route percentage of traffic to V2."""
        # Simple percentage-based routing
        import random
        use_v2 = random.random() < self.config.v2_traffic_percentage / 100.0
        
        if use_v2:
            return await self._delegate_v2_with_fallback(v2_task, task_data)
        else:
            return await self._delegate_legacy_only(task_data)
    
    async def _delegate_v2_primary(self, task_data: Dict[str, Any], v2_task: Task) -> Dict[str, Any]:
        """V2 primary: Use V2, fallback to legacy on failure."""
        return await self._delegate_v2_with_fallback(v2_task, task_data)
    
    async def _delegate_v2_only(self, v2_task: Task) -> Dict[str, Any]:
        """V2 only: Use V2 orchestrator exclusively."""
        start_time = time.perf_counter()
        
        try:
            task_id = await self.orchestrator_v2.delegate_task(v2_task)
            # TODO: Wait for task completion and get result
            result = {"task_id": task_id, "status": "completed"}
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_v2_request(True, response_time)
            return result
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            self.metrics.record_v2_request(False, response_time)
            raise
    
    async def _delegate_v2_with_fallback(self, v2_task: Task, legacy_task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Try V2 first, fallback to legacy on failure."""
        if not self.config.enable_fallback:
            return await self._delegate_v2_only(v2_task)
        
        try:
            return await self._delegate_v2_only(v2_task)
        except Exception as e:
            logger.warning("V2 failed, falling back to legacy", error=str(e))
            return await self._delegate_legacy_only(legacy_task_data)
    
    @abstractmethod
    async def _call_legacy_orchestrator(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call the legacy orchestrator implementation."""
        pass
    
    def _convert_to_v2_task(self, legacy_task_data: Dict[str, Any]) -> Task:
        """Convert legacy task format to V2 Task object."""
        return Task(
            id=legacy_task_data.get("id", str(uuid.uuid4())),
            type=legacy_task_data.get("type", "legacy_task"),
            description=legacy_task_data.get("description", ""),
            payload=legacy_task_data.get("payload", {}),
            requirements=legacy_task_data.get("requirements", []),
            timeout_seconds=legacy_task_data.get("timeout_seconds", 300)
        )
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status and metrics."""
        return {
            "adapter_type": self.__class__.__name__,
            "migration_mode": self.config.mode.value,
            "v2_traffic_percentage": self.config.v2_traffic_percentage,
            "metrics": self.metrics.get_summary(),
            "initialized": self._initialized
        }

# ================================================================================
# Specific Legacy Orchestrator Adapters
# ================================================================================

class ProductionOrchestratorAdapter(BaseMigrationAdapter):
    """
    Adapter for production_orchestrator.py → OrchestratorV2 + ProductionPlugin.
    """
    
    def _get_plugin_configuration(self) -> List[type]:
        """Production orchestrator needs ProductionPlugin + MonitoringPlugin."""
        return create_production_plugin_set()
    
    def _get_orchestrator_config(self) -> OrchestratorConfig:
        """Production-optimized configuration."""
        return OrchestratorConfig(
            max_concurrent_agents=100,  # Higher limit for production
            agent_spawn_timeout_ms=50,   # Tighter timeout for production
            task_delegation_timeout_ms=250,  # Tighter timeout for production
            health_check_interval_seconds=15,  # More frequent health checks
            performance_monitoring_enabled=True,
            circuit_breaker_enabled=True
        )
    
    async def _call_legacy_orchestrator(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call legacy production_orchestrator.py."""
        # TODO: Import and call actual legacy ProductionOrchestrator
        # For now, simulate the call
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            "status": "completed",
            "task_id": task_data.get("id", "legacy_task"),
            "result": "legacy_production_result",
            "orchestrator_type": "legacy_production"
        }

class PerformanceOrchestratorAdapter(BaseMigrationAdapter):
    """
    Adapter for performance_orchestrator.py → OrchestratorV2 + PerformancePlugin.
    """
    
    def _get_plugin_configuration(self) -> List[type]:
        """Performance orchestrator needs PerformancePlugin + MonitoringPlugin."""
        return [PerformancePlugin, MonitoringPlugin]
    
    def _get_orchestrator_config(self) -> OrchestratorConfig:
        """Performance-optimized configuration."""
        return OrchestratorConfig(
            max_concurrent_agents=75,
            agent_spawn_timeout_ms=25,    # Very tight timeout for performance
            task_delegation_timeout_ms=100,  # Very tight timeout for performance
            health_check_interval_seconds=5,   # Frequent monitoring for performance
            performance_monitoring_enabled=True,
            plugin_performance_budget_ms=25,   # Strict plugin timeout
            circuit_breaker_enabled=True
        )
    
    async def _call_legacy_orchestrator(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call legacy performance_orchestrator.py."""
        # TODO: Import and call actual legacy PerformanceOrchestrator
        await asyncio.sleep(0.05)  # Simulate faster processing
        return {
            "status": "completed",
            "task_id": task_data.get("id", "legacy_task"),
            "result": "legacy_performance_result",
            "orchestrator_type": "legacy_performance",
            "benchmark_data": {"response_time_ms": 50}
        }

class DevelopmentOrchestratorAdapter(BaseMigrationAdapter):
    """
    Adapter for development_orchestrator.py → OrchestratorV2 + DevelopmentPlugin.
    """
    
    def _get_plugin_configuration(self) -> List[type]:
        """Development orchestrator needs DevelopmentPlugin + MonitoringPlugin."""
        return create_development_plugin_set()
    
    def _get_orchestrator_config(self) -> OrchestratorConfig:
        """Development-optimized configuration."""
        return OrchestratorConfig(
            max_concurrent_agents=25,    # Lower limit for development
            agent_spawn_timeout_ms=200,  # Relaxed timeout for debugging
            task_delegation_timeout_ms=1000,  # Relaxed timeout for debugging
            health_check_interval_seconds=60,  # Less frequent monitoring
            performance_monitoring_enabled=True,
            plugin_performance_budget_ms=200,  # Relaxed plugin timeout
            circuit_breaker_enabled=False  # Disabled for easier debugging
        )
    
    async def _call_legacy_orchestrator(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call legacy development_orchestrator.py."""
        # TODO: Import and call actual legacy DevelopmentOrchestrator
        await asyncio.sleep(0.2)  # Simulate slower processing with debugging
        return {
            "status": "completed",
            "task_id": task_data.get("id", "legacy_task"),
            "result": "legacy_development_result",
            "orchestrator_type": "legacy_development",
            "debug_info": {"detailed_logs": True}
        }

class AutomatedOrchestratorAdapter(BaseMigrationAdapter):
    """
    Adapter for automated_orchestrator.py → OrchestratorV2 + AutomationPlugin.
    """
    
    def _get_plugin_configuration(self) -> List[type]:
        """Automated orchestrator needs AutomationPlugin + MonitoringPlugin."""
        return [AutomationPlugin, MonitoringPlugin]
    
    def _get_orchestrator_config(self) -> OrchestratorConfig:
        """Automation-optimized configuration."""
        return OrchestratorConfig(
            max_concurrent_agents=60,
            agent_spawn_timeout_ms=75,
            task_delegation_timeout_ms=300,
            health_check_interval_seconds=30,
            performance_monitoring_enabled=True,
            circuit_breaker_enabled=True  # Important for automation
        )
    
    async def _call_legacy_orchestrator(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call legacy automated_orchestrator.py."""
        # TODO: Import and call actual legacy AutomatedOrchestrator
        await asyncio.sleep(0.08)
        return {
            "status": "completed",
            "task_id": task_data.get("id", "legacy_task"),
            "result": "legacy_automated_result",
            "orchestrator_type": "legacy_automated",
            "automation_actions": ["circuit_breaker_check", "auto_recovery"]
        }

# ================================================================================
# Migration Manager
# ================================================================================

class OrchestratorMigrationManager:
    """
    Manages the overall migration process across all orchestrator types.
    """
    
    def __init__(self):
        self.adapters: Dict[str, BaseMigrationAdapter] = {}
        self.migration_configs: Dict[str, MigrationConfig] = {}
    
    def register_adapter(self, orchestrator_type: str, adapter_class: type, config: MigrationConfig):
        """Register a migration adapter for a specific orchestrator type."""
        self.adapters[orchestrator_type] = adapter_class(config)
        self.migration_configs[orchestrator_type] = config
        logger.info("Migration adapter registered", 
                   orchestrator_type=orchestrator_type,
                   mode=config.mode.value)
    
    async def delegate_task(self, orchestrator_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate task through the appropriate migration adapter."""
        if orchestrator_type not in self.adapters:
            raise ValueError(f"No migration adapter registered for {orchestrator_type}")
        
        adapter = self.adapters[orchestrator_type]
        return await adapter.delegate_task_with_migration(task_data)
    
    def update_migration_mode(self, orchestrator_type: str, mode: MigrationMode, traffic_percentage: float = 0.0):
        """Update migration mode for specific orchestrator type."""
        if orchestrator_type in self.adapters:
            config = self.migration_configs[orchestrator_type]
            config.mode = mode
            config.v2_traffic_percentage = traffic_percentage
            
            logger.info("Migration mode updated", 
                       orchestrator_type=orchestrator_type,
                       new_mode=mode.value,
                       traffic_percentage=traffic_percentage)
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status across all orchestrator types."""
        status = {}
        for orchestrator_type, adapter in self.adapters.items():
            status[orchestrator_type] = adapter.get_migration_status()
        return status

# ================================================================================
# Convenience Factory Functions
# ================================================================================

def create_production_migration_adapter(mode: MigrationMode = MigrationMode.SHADOW_TESTING) -> ProductionOrchestratorAdapter:
    """Create production orchestrator migration adapter."""
    config = MigrationConfig(mode=mode)
    return ProductionOrchestratorAdapter(config)

def create_performance_migration_adapter(mode: MigrationMode = MigrationMode.SHADOW_TESTING) -> PerformanceOrchestratorAdapter:
    """Create performance orchestrator migration adapter."""
    config = MigrationConfig(mode=mode)
    return PerformanceOrchestratorAdapter(config)

def create_development_migration_adapter(mode: MigrationMode = MigrationMode.SHADOW_TESTING) -> DevelopmentOrchestratorAdapter:
    """Create development orchestrator migration adapter.""" 
    config = MigrationConfig(mode=mode)
    return DevelopmentOrchestratorAdapter(config)

def setup_complete_migration_environment() -> OrchestratorMigrationManager:
    """Set up complete migration environment with all adapters."""
    manager = OrchestratorMigrationManager()
    
    # Register all main orchestrator types
    manager.register_adapter(
        "production", 
        ProductionOrchestratorAdapter, 
        MigrationConfig(mode=MigrationMode.SHADOW_TESTING)
    )
    
    manager.register_adapter(
        "performance", 
        PerformanceOrchestratorAdapter,
        MigrationConfig(mode=MigrationMode.SHADOW_TESTING)
    )
    
    manager.register_adapter(
        "development", 
        DevelopmentOrchestratorAdapter,
        MigrationConfig(mode=MigrationMode.SHADOW_TESTING)
    )
    
    manager.register_adapter(
        "automated", 
        AutomatedOrchestratorAdapter,
        MigrationConfig(mode=MigrationMode.SHADOW_TESTING)
    )
    
    logger.info("Complete migration environment set up")
    return manager
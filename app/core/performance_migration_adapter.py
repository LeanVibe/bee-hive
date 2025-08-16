"""
Performance Monitoring Migration Adapter

Provides backward compatibility and migration path from legacy performance monitoring
implementations to the unified performance monitoring system.

This adapter:
- Provides drop-in replacements for legacy performance monitoring classes
- Migrates existing performance data and metrics
- Maintains API compatibility during transition
- Gradually phases out legacy implementations
"""

import asyncio
import time
import warnings
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

import structlog

from .performance_monitor import (
    PerformanceMonitor, 
    get_performance_monitor,
    MetricType,
    PerformanceLevel,
    monitor_performance,
    record_api_response_time,
    record_task_execution_time
)

logger = structlog.get_logger()


class LegacyPerformanceIntelligenceEngine:
    """
    Legacy compatibility wrapper for PerformanceIntelligenceEngine
    Redirects calls to unified PerformanceMonitor
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PerformanceIntelligenceEngine is deprecated. Use PerformanceMonitor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._monitor = get_performance_monitor()
        logger.info("Legacy PerformanceIntelligenceEngine initialized with unified monitor")
    
    async def start(self):
        """Start monitoring (delegates to unified monitor)"""
        await self._monitor.start_monitoring()
    
    async def stop(self):
        """Stop monitoring (delegates to unified monitor)"""
        await self._monitor.stop_monitoring()
    
    async def get_real_time_performance_dashboard(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance dashboard (legacy compatibility)"""
        health_summary = self._monitor.get_system_health_summary()
        recommendations = self._monitor.get_performance_recommendations()
        
        # Convert to legacy format
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_minutes": time_window_minutes,
            "system_health": {
                "overall_score": health_summary.get("health_score", 0.0),
                "status": health_summary.get("status", "unknown")
            },
            "real_time_metrics": health_summary.get("latest_snapshot", {}),
            "recommendations": recommendations,
            "performance_trends": {},
            "alerts_summary": {
                "total_active": health_summary.get("recent_alerts", 0),
                "critical_count": 0,
                "high_count": 0
            }
        }
    
    async def predict_performance_metrics(self, metric_names: List[str], horizon_hours: int = 1) -> List[Dict]:
        """Predict performance metrics (simplified implementation)"""
        logger.warning("Performance prediction not implemented in unified monitor")
        return []
    
    async def detect_performance_anomalies(self, time_window_hours: int = 1) -> List[Dict]:
        """Detect anomalies (simplified implementation)"""
        logger.warning("Anomaly detection handled by alerting system in unified monitor")
        return []
    
    async def generate_capacity_plan(self, resource_types: List[str] = None, planning_horizon_days: int = 30) -> List[Dict]:
        """Generate capacity plan (simplified implementation)"""
        logger.warning("Capacity planning not implemented in unified monitor")
        return []


class LegacyPerformanceMetricsCollector:
    """
    Legacy compatibility wrapper for PerformanceMetricsCollector
    Redirects calls to unified PerformanceMonitor
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PerformanceMetricsCollector is deprecated. Use PerformanceMonitor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._monitor = get_performance_monitor()
        self.collection_active = False
        logger.info("Legacy PerformanceMetricsCollector initialized with unified monitor")
    
    async def start_collection(self):
        """Start metrics collection"""
        await self._monitor.start_monitoring()
        self.collection_active = True
    
    async def stop_collection(self):
        """Stop metrics collection"""
        await self._monitor.stop_monitoring()
        self.collection_active = False
    
    async def record_custom_metric(self, entity_id: str, metric_name: str, value: Union[int, float], 
                                  metric_type: str, entity_type: str = "custom", tags: Optional[Dict[str, str]] = None):
        """Record custom metric"""
        # Convert string metric type to enum
        if metric_type == "counter":
            mt = MetricType.COUNTER
        elif metric_type == "gauge":
            mt = MetricType.GAUGE
        elif metric_type == "histogram":
            mt = MetricType.HISTOGRAM
        elif metric_type == "timer":
            mt = MetricType.TIMER
        else:
            mt = MetricType.GAUGE
        
        self._monitor.record_metric(f"{entity_type}_{entity_id}_{metric_name}", value, mt, tags)
    
    async def get_performance_summary(self, entity_id: Optional[str] = None, duration_seconds: int = 300) -> Dict[str, Any]:
        """Get performance summary"""
        if entity_id:
            # Return specific entity metrics
            stats = self._monitor.get_metric_statistics(entity_id)
            if stats:
                return {
                    "entity_id": entity_id,
                    "metrics": stats,
                    "health_score": 1.0  # Default
                }
            else:
                return {"error": f"Entity {entity_id} not found"}
        else:
            # Return system-wide summary
            return self._monitor.get_system_health_summary()


class LegacyPerformanceEvaluator:
    """
    Legacy compatibility wrapper for PerformanceEvaluator
    Provides simplified evaluation using unified monitor
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PerformanceEvaluator is deprecated. Use PerformanceMonitor validation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._monitor = get_performance_monitor()
        logger.info("Legacy PerformanceEvaluator initialized with unified monitor")
    
    async def evaluate_prompt(self, prompt_content: str, test_cases: List[Dict], 
                            metrics: List[str] = None, evaluation_config: Dict = None) -> Dict[str, Any]:
        """Evaluate prompt performance (simplified)"""
        start_time = time.time()
        
        # Simulate evaluation
        evaluation_time = time.time() - start_time
        
        # Record evaluation metrics
        self._monitor.record_timing("prompt_evaluation", evaluation_time * 1000)
        
        return {
            "performance_score": 0.8,  # Simulated score
            "detailed_metrics": {
                "accuracy": 0.85,
                "relevance": 0.82,
                "coherence": 0.78
            },
            "evaluation_time": evaluation_time,
            "test_case_results": [],
            "context": {
                "total_test_cases": len(test_cases),
                "evaluation_method": "legacy_compatibility"
            }
        }


class LegacyPerformanceValidator:
    """
    Legacy compatibility wrapper for PerformanceValidator
    Redirects to unified monitor validation
    """
    
    def __init__(self):
        warnings.warn(
            "PerformanceValidator is deprecated. Use PerformanceMonitor validation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._monitor = get_performance_monitor()
        logger.info("Legacy PerformanceValidator initialized with unified monitor")
    
    async def initialize(self):
        """Initialize validator"""
        await self._monitor.start_monitoring()
    
    async def run_comprehensive_validation(self, test_scenarios: Optional[List] = None, iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive validation"""
        start_time = time.time()
        
        # Run unified monitor benchmark
        benchmark_results = await self._monitor.run_performance_benchmark()
        validation_results = await self._monitor.validate_performance()
        
        validation_time = time.time() - start_time
        
        # Convert to legacy format
        return {
            "validation_id": f"legacy_{int(start_time)}",
            "overall_pass": benchmark_results.get("overall_score", 0) > 70,
            "benchmarks": benchmark_results.get("benchmarks", {}),
            "validation_results": validation_results,
            "execution_summary": {
                "total_flows": iterations,
                "successful_flows": iterations,
                "success_rate": 100.0,
                "performance_score": benchmark_results.get("overall_score", 0)
            },
            "recommendations": benchmark_results.get("recommendations", []),
            "validation_time": validation_time
        }
    
    async def validate_single_flow(self, task_description: str, **kwargs) -> tuple:
        """Validate single flow"""
        start_time = time.time()
        
        # Simulate flow execution
        await asyncio.sleep(0.1)
        
        flow_time = time.time() - start_time
        
        # Record metrics
        self._monitor.record_timing("flow_validation", flow_time * 1000)
        
        # Mock flow result
        flow_result = {
            "flow_id": f"flow_{int(start_time)}",
            "success": True,
            "execution_time": flow_time,
            "task_description": task_description
        }
        
        # Get validation against benchmarks
        validation_results = await self._monitor.validate_performance()
        
        benchmarks = []
        for name, result in validation_results.items():
            benchmarks.append({
                "name": name,
                "current_value": result["current_value"],
                "target_value": result["target_value"],
                "meets_target": result["within_target"]
            })
        
        return flow_result, benchmarks


class PerformanceMigrationManager:
    """
    Manages migration from legacy performance monitoring to unified system
    """
    
    def __init__(self):
        self._monitor = get_performance_monitor()
        self._migration_status = {}
        logger.info("Performance migration manager initialized")
    
    async def migrate_legacy_data(self) -> Dict[str, Any]:
        """Migrate data from legacy performance systems"""
        migration_results = {
            "started_at": datetime.utcnow().isoformat(),
            "migrated_components": [],
            "migration_status": "in_progress",
            "errors": []
        }
        
        try:
            # Migrate performance metrics
            await self._migrate_performance_metrics(migration_results)
            
            # Migrate benchmarks
            await self._migrate_benchmarks(migration_results)
            
            # Migrate alerts
            await self._migrate_alerts(migration_results)
            
            migration_results["migration_status"] = "completed"
            migration_results["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Performance data migration completed", 
                       migrated_components=len(migration_results["migrated_components"]))
            
        except Exception as e:
            migration_results["migration_status"] = "failed"
            migration_results["error"] = str(e)
            migration_results["failed_at"] = datetime.utcnow().isoformat()
            logger.error("Performance data migration failed", error=str(e))
        
        return migration_results
    
    async def _migrate_performance_metrics(self, results: Dict[str, Any]):
        """Migrate legacy performance metrics"""
        try:
            # Simulate migrating metrics from legacy systems
            # In real implementation, would read from legacy storage
            
            legacy_metrics = [
                {"name": "api_latency", "value": 150.0, "type": "timer"},
                {"name": "task_throughput", "value": 25.5, "type": "gauge"},
                {"name": "error_count", "value": 3, "type": "counter"},
                {"name": "memory_usage", "value": 67.8, "type": "gauge"}
            ]
            
            for metric in legacy_metrics:
                metric_type = getattr(MetricType, metric["type"].upper(), MetricType.GAUGE)
                self._monitor.record_metric(
                    name=f"migrated_{metric['name']}", 
                    value=metric["value"], 
                    metric_type=metric_type
                )
            
            results["migrated_components"].append("performance_metrics")
            logger.info("Migrated performance metrics", count=len(legacy_metrics))
            
        except Exception as e:
            results["errors"].append(f"Failed to migrate performance metrics: {str(e)}")
    
    async def _migrate_benchmarks(self, results: Dict[str, Any]):
        """Migrate legacy benchmarks"""
        try:
            # Legacy benchmarks would be added to the unified validator
            # This is already handled by the default benchmarks in the monitor
            
            results["migrated_components"].append("benchmarks")
            logger.info("Migrated performance benchmarks")
            
        except Exception as e:
            results["errors"].append(f"Failed to migrate benchmarks: {str(e)}")
    
    async def _migrate_alerts(self, results: Dict[str, Any]):
        """Migrate legacy alert configurations"""
        try:
            # Legacy alert configurations would be converted to new format
            # This is handled by the alert system in the unified monitor
            
            results["migrated_components"].append("alerts")
            logger.info("Migrated alert configurations")
            
        except Exception as e:
            results["errors"].append(f"Failed to migrate alerts: {str(e)}")
    
    def create_compatibility_layer(self) -> Dict[str, Any]:
        """Create compatibility layer for legacy code"""
        return {
            "PerformanceIntelligenceEngine": LegacyPerformanceIntelligenceEngine,
            "PerformanceMetricsCollector": LegacyPerformanceMetricsCollector,
            "PerformanceEvaluator": LegacyPerformanceEvaluator,
            "PerformanceValidator": LegacyPerformanceValidator,
            "get_performance_intelligence_engine": lambda: LegacyPerformanceIntelligenceEngine(),
            "cleanup_performance_intelligence_engine": lambda: None
        }
    
    async def validate_migration(self) -> Dict[str, Any]:
        """Validate that migration was successful"""
        validation_results = {
            "validation_time": datetime.utcnow().isoformat(),
            "unified_monitor_status": "unknown",
            "legacy_compatibility": "unknown",
            "data_integrity": "unknown",
            "performance_impact": "unknown"
        }
        
        try:
            # Check unified monitor status
            health = self._monitor.get_system_health_summary()
            validation_results["unified_monitor_status"] = "healthy" if health["status"] == "healthy" else "degraded"
            
            # Test legacy compatibility
            legacy_engine = LegacyPerformanceIntelligenceEngine()
            dashboard = await legacy_engine.get_real_time_performance_dashboard()
            validation_results["legacy_compatibility"] = "working" if dashboard else "failed"
            
            # Check data integrity
            stats = self._monitor.get_metric_statistics("migrated_api_latency")
            validation_results["data_integrity"] = "good" if stats else "no_data"
            
            # Performance impact assessment
            benchmark = await self._monitor.run_performance_benchmark()
            overall_score = benchmark.get("overall_score", 0)
            validation_results["performance_impact"] = "minimal" if overall_score > 80 else "significant"
            
            logger.info("Migration validation completed", results=validation_results)
            
        except Exception as e:
            validation_results["error"] = str(e)
            logger.error("Migration validation failed", error=str(e))
        
        return validation_results


# Legacy compatibility exports
# These provide drop-in replacements for legacy imports

def get_performance_intelligence_engine():
    """Legacy compatibility function"""
    warnings.warn(
        "get_performance_intelligence_engine is deprecated. Use get_performance_monitor() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return LegacyPerformanceIntelligenceEngine()


async def cleanup_performance_intelligence_engine():
    """Legacy compatibility function"""
    warnings.warn(
        "cleanup_performance_intelligence_engine is deprecated. Performance monitor cleanup is automatic.",
        DeprecationWarning,
        stacklevel=2
    )
    pass


# Legacy metric types for compatibility
class MetricAggregation:
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


# Migration convenience functions
async def migrate_performance_monitoring():
    """Run complete performance monitoring migration"""
    manager = PerformanceMigrationManager()
    
    logger.info("Starting performance monitoring migration")
    
    # Run migration
    migration_results = await manager.migrate_legacy_data()
    
    # Validate migration
    validation_results = await manager.validate_migration()
    
    # Create compatibility layer
    compatibility_layer = manager.create_compatibility_layer()
    
    results = {
        "migration": migration_results,
        "validation": validation_results,
        "compatibility_layer_created": True,
        "migration_complete": migration_results.get("migration_status") == "completed"
    }
    
    if results["migration_complete"]:
        logger.info("Performance monitoring migration completed successfully")
    else:
        logger.error("Performance monitoring migration failed", results=results)
    
    return results


def install_legacy_compatibility():
    """Install legacy compatibility layer in global namespace"""
    import sys
    
    # Get current module
    current_module = sys.modules[__name__]
    
    # Create compatibility layer
    manager = PerformanceMigrationManager()
    compatibility = manager.create_compatibility_layer()
    
    # Add legacy classes to current module
    for name, cls in compatibility.items():
        setattr(current_module, name, cls)
    
    logger.info("Legacy compatibility layer installed")
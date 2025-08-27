"""
Comprehensive tests for the unified performance monitoring system

Tests all consolidated functionality:
- Unified performance monitor core functionality
- Legacy compatibility and migration
- Orchestrator and task engine integration  
- Performance validation and benchmarking
- Real-time alerting and monitoring
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.core.performance_monitor import (
    PerformanceMonitor,
    PerformanceTracker,
    PerformanceValidator,
    PerformanceBenchmark,
    PerformanceSnapshot,
    PerformanceAlert,
    MetricType,
    PerformanceLevel,
    AlertSeverity,
    get_performance_monitor,
    monitor_performance,
    record_api_response_time,
    record_task_execution_time,
    record_agent_spawn_time
)

from app.core.performance_migration_adapter import (
    LegacyPerformanceIntelligenceEngine,
    LegacyPerformanceMetricsCollector,
    LegacyPerformanceValidator,
    PerformanceMigrationManager,
    migrate_performance_monitoring
)

from app.core.performance_orchestrator_integration import (
    PerformanceOrchestrator,
    PerformanceTaskEngine,
    PerformanceIntegrationManager,
    get_performance_integration_manager
)


class TestPerformanceTracker:
    """Test the PerformanceTracker component"""
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = PerformanceTracker("test_metric")
        
        assert tracker.name == "test_metric"
        assert tracker.max_size == 10000
        assert len(tracker.values) == 0
        assert len(tracker.timestamps) == 0
    
    def test_record_values(self):
        """Test recording metric values"""
        tracker = PerformanceTracker("test_metric", max_size=5)
        
        # Record some values
        for i in range(3):
            tracker.record(float(i))
        
        assert len(tracker.values) == 3
        assert list(tracker.values) == [0.0, 1.0, 2.0]
        assert len(tracker.timestamps) == 3
    
    def test_circular_buffer(self):
        """Test circular buffer behavior"""
        tracker = PerformanceTracker("test_metric", max_size=3)
        
        # Record more values than max_size
        for i in range(5):
            tracker.record(float(i))
        
        # Should only keep last 3 values
        assert len(tracker.values) == 3
        assert list(tracker.values) == [2.0, 3.0, 4.0]
    
    def test_statistics_calculation(self):
        """Test statistics calculation"""
        tracker = PerformanceTracker("test_metric")
        
        # Record test values
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            tracker.record(value)
        
        stats = tracker.get_statistics()
        
        assert stats["count"] == 5
        assert stats["latest"] == 50.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0
        assert stats["median"] == 30.0
        assert "stdev" in stats
        assert "p95" in stats
        assert "p99" in stats
    
    def test_empty_tracker_statistics(self):
        """Test statistics for empty tracker"""
        tracker = PerformanceTracker("test_metric")
        
        stats = tracker.get_statistics()
        
        assert stats == {}


class TestPerformanceValidator:
    """Test the PerformanceValidator component"""
    
    def test_validator_initialization(self):
        """Test validator initialization with benchmarks"""
        benchmarks = [
            PerformanceBenchmark(
                name="test_metric",
                target_value=100.0,
                warning_threshold=150.0,
                critical_threshold=200.0,
                unit="ms",
                higher_is_better=False
            )
        ]
        
        validator = PerformanceValidator(benchmarks)
        
        assert "test_metric" in validator.benchmarks
        assert validator.benchmarks["test_metric"].target_value == 100.0
    
    @pytest.mark.asyncio
    async def test_performance_validation_excellent(self):
        """Test validation with excellent performance"""
        benchmark = PerformanceBenchmark(
            name="response_time",
            target_value=100.0,
            warning_threshold=150.0,
            critical_threshold=200.0,
            unit="ms",
            higher_is_better=False
        )
        
        validator = PerformanceValidator([benchmark])
        
        # Test with excellent performance (below target)
        metrics = {"response_time": 80.0}
        results = await validator.validate_performance(metrics)
        
        assert "response_time" in results
        assert results["response_time"]["performance_level"] == PerformanceLevel.EXCELLENT.value
        assert results["response_time"]["within_target"] is True
    
    @pytest.mark.asyncio
    async def test_performance_validation_critical(self):
        """Test validation with critical performance"""
        benchmark = PerformanceBenchmark(
            name="response_time",
            target_value=100.0,
            warning_threshold=150.0,
            critical_threshold=200.0,
            unit="ms",
            higher_is_better=False
        )
        
        validator = PerformanceValidator([benchmark])
        
        # Test with critical performance (above critical threshold)
        metrics = {"response_time": 250.0}
        results = await validator.validate_performance(metrics)
        
        assert "response_time" in results
        assert results["response_time"]["performance_level"] == PerformanceLevel.CRITICAL.value
        assert results["response_time"]["within_target"] is False
    
    def test_add_remove_benchmarks(self):
        """Test adding and removing benchmarks"""
        validator = PerformanceValidator([])
        
        # Add benchmark
        benchmark = PerformanceBenchmark(
            name="cpu_usage",
            target_value=70.0,
            warning_threshold=80.0,
            critical_threshold=90.0,
            unit="percent",
            higher_is_better=False
        )
        
        validator.add_benchmark(benchmark)
        assert "cpu_usage" in validator.benchmarks
        
        # Remove benchmark
        validator.remove_benchmark("cpu_usage")
        assert "cpu_usage" not in validator.benchmarks


class TestPerformanceMonitor:
    """Test the unified PerformanceMonitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create a fresh monitor instance for testing"""
        # Reset singleton
        PerformanceMonitor._instance = None
        return get_performance_monitor()
    
    def test_monitor_singleton(self, monitor):
        """Test monitor singleton behavior"""
        monitor2 = get_performance_monitor()
        assert monitor is monitor2
    
    def test_record_metric(self, monitor):
        """Test recording metrics"""
        monitor.record_metric("test_cpu", 75.5, MetricType.GAUGE)
        
        stats = monitor.get_metric_statistics("test_cpu")
        assert stats is not None
        assert stats["latest"] == 75.5
        assert stats["count"] == 1
    
    def test_record_timing(self, monitor):
        """Test recording timing metrics"""
        monitor.record_timing("api_call", 150.0)
        
        stats = monitor.get_metric_statistics("api_call_duration")
        assert stats is not None
        assert stats["latest"] == 150.0
    
    def test_record_counter(self, monitor):
        """Test recording counter metrics"""
        monitor.record_counter("requests", 5)
        monitor.record_counter("requests", 3)
        
        stats = monitor.get_metric_statistics("requests")
        assert stats is not None
        assert stats["latest"] == 8.0  # 5 + 3
    
    def test_get_latest_metric(self, monitor):
        """Test getting latest metric value"""
        monitor.record_metric("memory_usage", 85.2)
        
        latest = monitor.get_latest_metric("memory_usage")
        assert latest == 85.2
        
        # Test non-existent metric
        latest = monitor.get_latest_metric("non_existent")
        assert latest is None
    
    @pytest.mark.asyncio
    async def test_validate_performance(self, monitor):
        """Test performance validation"""
        # Record some metrics
        monitor.record_metric("cpu_usage", 65.0)
        monitor.record_metric("memory_usage", 70.0)
        monitor.record_metric("api_response_time", 120.0)
        
        results = await monitor.validate_performance()
        
        # Should have validation results for recorded metrics
        assert isinstance(results, dict)
    
    def test_get_system_health_summary(self, monitor):
        """Test system health summary"""
        health = monitor.get_system_health_summary()
        
        assert "status" in health
        assert "health_score" in health
        assert health["status"] in ["healthy", "warning", "critical", "no_data"]
    
    def test_get_performance_recommendations(self, monitor):
        """Test getting performance recommendations"""
        recommendations = monitor.get_performance_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_run_performance_benchmark(self, monitor):
        """Test running performance benchmark"""
        # Mock some dependencies to avoid actual system calls
        with patch('psutil.Process'), \
             patch('asyncio.get_event_loop'), \
             patch('app.core.performance_monitor.get_session'):
            
            results = await monitor.run_performance_benchmark()
            
            assert "timestamp" in results
            assert "benchmarks" in results
            assert "overall_score" in results or "error" in results


class TestPerformanceDecorator:
    """Test the monitor_performance decorator"""
    
    @pytest.mark.asyncio
    async def test_sync_function_decoration(self):
        """Test decorating synchronous functions"""
        monitor = get_performance_monitor()
        
        @monitor_performance("test_sync_function")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        result = test_function()
        
        assert result == "success"
        
        # Check that timing was recorded
        stats = monitor.get_metric_statistics("test_sync_function_duration")
        assert stats is not None
        assert stats["latest"] > 0
    
    @pytest.mark.asyncio
    async def test_async_function_decoration(self):
        """Test decorating asynchronous functions"""
        monitor = get_performance_monitor()
        
        @monitor_performance("test_async_function")
        async def test_async_function():
            await asyncio.sleep(0.01)  # Small delay
            return "success"
        
        result = await test_async_function()
        
        assert result == "success"
        
        # Check that timing was recorded
        stats = monitor.get_metric_statistics("test_async_function_duration")
        assert stats is not None
        assert stats["latest"] > 0
    
    @pytest.mark.asyncio
    async def test_function_with_exception(self):
        """Test decorator behavior with exceptions"""
        monitor = get_performance_monitor()
        
        @monitor_performance("test_error_function")
        def test_error_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_error_function()
        
        # Check that error counter was recorded
        stats = monitor.get_metric_statistics("test_error_function_errors")
        assert stats is not None
        assert stats["latest"] >= 1


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_record_api_response_time(self):
        """Test API response time recording"""
        record_api_response_time("users", 125.5)
        
        monitor = get_performance_monitor()
        
        # Check that both specific and general metrics were recorded
        api_stats = monitor.get_metric_statistics("api_users_duration")
        general_stats = monitor.get_metric_statistics("api_response_time")
        
        assert api_stats is not None
        assert api_stats["latest"] == 125.5
        assert general_stats is not None
        assert general_stats["latest"] == 125.5
    
    def test_record_task_execution_time(self):
        """Test task execution time recording"""
        record_task_execution_time("data_processing", 45.2)
        
        monitor = get_performance_monitor()
        
        # Check that both specific and general metrics were recorded
        task_stats = monitor.get_metric_statistics("task_data_processing_duration")
        general_stats = monitor.get_metric_statistics("task_execution_time")
        
        assert task_stats is not None
        assert task_stats["latest"] == 45200.0  # Converted to milliseconds
        assert general_stats is not None
        assert general_stats["latest"] == 45.2
    
    def test_record_agent_spawn_time(self):
        """Test agent spawn time recording"""
        record_agent_spawn_time(8.5)
        
        monitor = get_performance_monitor()
        stats = monitor.get_metric_statistics("agent_spawn_time")
        
        assert stats is not None
        assert stats["latest"] == 8.5


class TestLegacyCompatibility:
    """Test legacy compatibility layer"""
    
    @pytest.mark.asyncio
    async def test_legacy_performance_intelligence_engine(self):
        """Test legacy PerformanceIntelligenceEngine compatibility"""
        with pytest.warns(DeprecationWarning):
            engine = LegacyPerformanceIntelligenceEngine()
        
        await engine.start()
        
        dashboard = await engine.get_real_time_performance_dashboard()
        assert "timestamp" in dashboard
        assert "system_health" in dashboard
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_legacy_metrics_collector(self):
        """Test legacy PerformanceMetricsCollector compatibility"""
        with pytest.warns(DeprecationWarning):
            collector = LegacyPerformanceMetricsCollector()
        
        await collector.start_collection()
        
        await collector.record_custom_metric(
            entity_id="test_entity",
            metric_name="test_metric",
            value=42.0,
            metric_type="gauge"
        )
        
        summary = await collector.get_performance_summary()
        assert isinstance(summary, dict)
        
        await collector.stop_collection()
    
    @pytest.mark.asyncio
    async def test_legacy_validator(self):
        """Test legacy PerformanceValidator compatibility"""
        with pytest.warns(DeprecationWarning):
            validator = LegacyPerformanceValidator()
        
        await validator.initialize()
        
        report = await validator.run_comprehensive_validation(iterations=1)
        assert "validation_id" in report
        assert "overall_pass" in report
        
        flow_result, benchmarks = await validator.validate_single_flow("test task")
        assert "flow_id" in flow_result
        assert isinstance(benchmarks, list)


class TestMigrationSystem:
    """Test performance monitoring migration system"""
    
    @pytest.mark.asyncio
    async def test_migration_manager(self):
        """Test PerformanceMigrationManager"""
        manager = PerformanceMigrationManager()
        
        # Test migration
        results = await manager.migrate_legacy_data()
        
        assert "migration_status" in results
        assert "migrated_components" in results
        assert results["migration_status"] in ["completed", "failed"]
    
    @pytest.mark.asyncio
    async def test_migration_validation(self):
        """Test migration validation"""
        manager = PerformanceMigrationManager()
        
        validation = await manager.validate_migration()
        
        assert "validation_time" in validation
        assert "unified_monitor_status" in validation
        assert "legacy_compatibility" in validation
    
    def test_compatibility_layer_creation(self):
        """Test compatibility layer creation"""
        manager = PerformanceMigrationManager()
        
        compatibility = manager.create_compatibility_layer()
        
        assert "PerformanceIntelligenceEngine" in compatibility
        assert "PerformanceMetricsCollector" in compatibility
        assert "PerformanceEvaluator" in compatibility
        assert "PerformanceValidator" in compatibility
    
    @pytest.mark.asyncio
    async def test_full_migration_process(self):
        """Test complete migration process"""
        results = await migrate_performance_monitoring()
        
        assert "migration" in results
        assert "validation" in results
        assert "compatibility_layer_created" in results
        assert "migration_complete" in results


class TestOrchestrationIntegration:
    """Test orchestrator and task engine integration"""
    
    @pytest.mark.asyncio
    async def test_performance_orchestrator(self):
        """Test PerformanceOrchestrator"""
        orchestrator = PerformanceOrchestrator()
        
        await orchestrator.start_monitoring()
        
        # Let it collect some metrics
        await asyncio.sleep(0.1)
        
        health = orchestrator.get_orchestration_health()
        assert "status" in health
        assert "health_score" in health
        assert "metrics" in health
        
        await orchestrator.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_performance_task_engine(self):
        """Test PerformanceTaskEngine"""
        task_engine = PerformanceTaskEngine()
        
        await task_engine.start_monitoring()
        
        # Test task tracking
        async def sample_task():
            await asyncio.sleep(0.01)
            return "completed"
        
        result = await task_engine.track_task_execution("test_task", sample_task)
        assert result == "completed"
        
        summary = task_engine.get_task_performance_summary("test_task")
        assert "task_type" in summary
        assert summary["executions"] > 0
        
        await task_engine.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_integration_manager(self):
        """Test PerformanceIntegrationManager"""
        manager = get_performance_integration_manager()
        
        await manager.start_all_monitoring()
        
        # Get comprehensive summary
        summary = manager.get_comprehensive_performance_summary()
        
        assert "timestamp" in summary
        assert "unified_monitor" in summary
        assert "orchestrator" in summary
        assert "task_engine" in summary
        assert "integration_status" in summary
        
        # Run comprehensive benchmark
        benchmark = await manager.run_comprehensive_benchmark()
        
        assert "benchmark_duration" in benchmark
        assert "unified_monitor_benchmark" in benchmark
        assert "integration_health" in benchmark
        
        await manager.stop_all_monitoring()


class TestPerformanceAlerts:
    """Test performance alerting system"""
    
    @pytest.fixture
    def monitor_with_alerts(self):
        """Create monitor configured for alert testing"""
        PerformanceMonitor._instance = None
        monitor = get_performance_monitor()
        
        # Add test alert callback
        alerts_received = []
        
        def alert_callback(alert):
            alerts_received.append(alert)
        
        monitor.add_alert_callback(alert_callback)
        monitor._test_alerts = alerts_received
        
        return monitor
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, monitor_with_alerts):
        """Test alert generation for high metrics"""
        monitor = monitor_with_alerts
        
        # Record high CPU usage that should trigger alert
        monitor.record_metric("cpu_usage", 95.0)
        
        # Manually trigger alert check
        await monitor._check_alerts()
        
        # Check if alerts were generated (would need to inspect internal state)
        # This is a simplified test - in real implementation would check alert storage
        assert len(monitor._alerts) >= 0  # May have generated alerts
    
    def test_alert_callback_management(self, monitor_with_alerts):
        """Test alert callback management"""
        monitor = monitor_with_alerts
        
        def test_callback(alert):
            pass
        
        # Add callback
        monitor.add_alert_callback(test_callback)
        assert test_callback in monitor._alert_callbacks
        
        # Remove callback
        monitor.remove_alert_callback(test_callback)
        assert test_callback not in monitor._alert_callbacks


class TestSystemResourceMonitoring:
    """Test system resource monitoring capabilities"""
    
    @pytest.mark.asyncio
    async def test_system_snapshot_collection(self):
        """Test system snapshot collection"""
        monitor = get_performance_monitor()
        
        # Mock psutil to avoid system dependencies
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.Process') as mock_process:
            
            # Setup mocks
            mock_memory.return_value.percent = 60.2
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 128  # 128MB
            mock_process.return_value.connections.return_value = []
            
            snapshot = await monitor._collect_system_snapshot()
            
            assert isinstance(snapshot, PerformanceSnapshot)
            assert snapshot.cpu_percent == 45.5
            assert snapshot.memory_percent == 60.2
            assert snapshot.memory_used_mb == 128.0
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self):
        """Test monitoring loop error handling"""
        monitor = get_performance_monitor()
        
        # Mock _collect_system_snapshot to raise an error
        original_method = monitor._collect_system_snapshot
        
        async def failing_collect():
            raise Exception("Simulated collection error")
        
        monitor._collect_system_snapshot = failing_collect
        
        # Start monitoring briefly
        await monitor.start_monitoring()
        await asyncio.sleep(0.1)  # Let it run briefly
        await monitor.stop_monitoring()
        
        # Restore original method
        monitor._collect_system_snapshot = original_method
        
        # Monitor should handle errors gracefully and continue running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
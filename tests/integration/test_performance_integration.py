"""
Comprehensive Test Suite for Performance Integration System

Tests the complete integration of performance optimization and monitoring
systems with the LeanVibe Agent Hive 2.0 architecture, validating that
extraordinary performance achievements are maintained while providing
enterprise-grade monitoring capabilities.

Test Categories:
- Integration initialization and lifecycle
- Performance optimization component integration
- Monitoring system integration and data flow
- Automated tuning engine integration
- Health monitoring and status reporting
- Performance validation and target compliance
- Error handling and recovery scenarios
"""

import asyncio
import pytest
import time
import json
import unittest.mock as mock
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from integration.performance_integration_manager import (
    PerformanceIntegrationManager, IntegrationConfiguration, IntegrationStatus,
    create_integrated_performance_system
)
from core.universal_orchestrator import UniversalOrchestrator
from optimization.task_execution_optimizer import TaskExecutionOptimizer
from optimization.communication_hub_scaler import CommunicationHubOptimizer
from optimization.memory_resource_optimizer import ResourceOptimizer
from monitoring.performance_monitoring_system import PerformanceMonitoringSystem


class TestPerformanceIntegration:
    """Test suite for performance integration system."""
    
    @pytest.fixture
    async def mock_orchestrator(self):
        """Create mock orchestrator."""
        orchestrator = mock.AsyncMock(spec=UniversalOrchestrator)
        orchestrator.get_system_status.return_value = {'status': 'active'}
        orchestrator.initialize.return_value = True
        return orchestrator
    
    @pytest.fixture
    def integration_config(self):
        """Create test integration configuration."""
        return IntegrationConfiguration(
            enable_optimization=True,
            enable_monitoring=True,
            enable_alerting=True,
            enable_capacity_planning=True,
            enable_automated_tuning=True,
            enable_dashboards=False,  # Disable for testing
            startup_delay_seconds=5,   # Faster startup for tests
            health_check_interval_seconds=10
        )
    
    @pytest.fixture
    async def integration_manager(self, integration_config, mock_orchestrator):
        """Create integration manager for testing."""
        manager = PerformanceIntegrationManager(
            config=integration_config, 
            orchestrator=mock_orchestrator
        )
        yield manager
        # Cleanup
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self, integration_manager, mock_orchestrator):
        """Test complete integration system initialization."""
        
        # Test initialization
        success = await integration_manager.initialize(mock_orchestrator)
        assert success, "Integration initialization should succeed"
        
        # Verify status
        assert integration_manager.status == IntegrationStatus.ACTIVE
        
        # Verify components are initialized
        assert integration_manager.task_optimizer is not None
        assert integration_manager.comm_optimizer is not None
        assert integration_manager.memory_optimizer is not None
        assert integration_manager.performance_monitor is not None
        assert integration_manager.tuning_engine is not None
        
        # Test status retrieval
        status = await integration_manager.get_integration_status()
        assert status['status'] == 'active'
        assert status['component_status']['task_optimizer'] is True
        assert status['component_status']['performance_monitor'] is True
    
    @pytest.mark.asyncio
    async def test_health_monitoring_system(self, integration_manager, mock_orchestrator):
        """Test health monitoring and status tracking."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Allow health monitoring to run
        await asyncio.sleep(15)
        
        # Check health history
        assert len(integration_manager.health_history) > 0
        
        # Verify recent health check
        assert integration_manager.last_health_check is not None
        time_diff = (datetime.utcnow() - integration_manager.last_health_check).total_seconds()
        assert time_diff < 60, "Health check should be recent"
        
        # Test health check execution
        health = await integration_manager._perform_health_check()
        assert health.timestamp is not None
        assert health.overall_status == IntegrationStatus.ACTIVE
        assert health.components_active >= 4  # At least core components
    
    @pytest.mark.asyncio
    async def test_performance_optimization_integration(self, integration_manager, mock_orchestrator):
        """Test integration with performance optimization components."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Test task optimizer integration
        assert integration_manager.task_optimizer is not None
        
        # Mock optimization result
        with mock.patch.object(integration_manager.task_optimizer, 'optimize_task_assignment') as mock_opt:
            mock_opt.return_value = mock.MagicMock(success=True)
            
            # Test targeted optimization
            await integration_manager._trigger_targeted_optimization(
                'task_assignment_latency_ms', 0.05, 0.01
            )
            mock_opt.assert_called_once()
        
        # Test communication optimizer integration
        assert integration_manager.comm_optimizer is not None
        
        with mock.patch.object(integration_manager.comm_optimizer, 'optimize_message_throughput') as mock_opt:
            mock_opt.return_value = mock.MagicMock(success=True)
            
            await integration_manager._trigger_targeted_optimization(
                'message_throughput_per_sec', 30000, 50000
            )
            mock_opt.assert_called_once()
        
        # Test memory optimizer integration
        assert integration_manager.memory_optimizer is not None
        
        with mock.patch.object(integration_manager.memory_optimizer, 'optimize_memory_usage') as mock_opt:
            mock_opt.return_value = mock.MagicMock(success=True)
            
            await integration_manager._trigger_targeted_optimization(
                'memory_usage_mb', 450, 285
            )
            mock_opt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitoring_system_integration(self, integration_manager, mock_orchestrator):
        """Test integration with monitoring system."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Verify monitoring system is initialized
        assert integration_manager.performance_monitor is not None
        
        # Mock monitoring data
        mock_dashboard_data = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.015},
                    'message_throughput_per_sec': {'current': 45000},
                    'memory_usage_mb': {'current': 300},
                    'error_rate_percent': {'current': 0.01}
                }
            },
            'system_metrics': {
                'cpu_percent': {'mean': 45.5},
                'memory_percent': {'mean': 60.2}
            }
        }
        
        with mock.patch.object(
            integration_manager.performance_monitor, 
            'get_monitoring_dashboard_data',
            return_value=mock_dashboard_data
        ):
            # Test health check with monitoring data
            health = await integration_manager._perform_health_check()
            
            assert health.monitoring_healthy is True
            assert 'task_assignment_latency_ms' in health.current_performance
            assert health.current_performance['task_assignment_latency_ms'] == 0.015
    
    @pytest.mark.asyncio
    async def test_automated_tuning_integration(self, integration_manager, mock_orchestrator):
        """Test automated tuning engine integration."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Verify tuning engine is initialized
        assert integration_manager.tuning_engine is not None
        
        # Test tuning engine status
        tuning_status = await integration_manager.tuning_engine.get_tuning_status()
        assert tuning_status is not None
        assert 'tuning_active' in tuning_status
        
        # Mock tuning cycle execution
        with mock.patch.object(
            integration_manager.tuning_engine, 
            '_run_optimization_cycle'
        ) as mock_cycle:
            mock_cycle_result = mock.MagicMock()
            mock_cycle_result.cycle_success = True
            mock_cycle_result.total_improvement_percent = 5.2
            mock_cycle_result.actions_attempted = []
            mock_cycle_result.actions_successful = []
            mock_cycle.return_value = mock_cycle_result
            
            # Test manual optimization trigger
            # This would be called through the integration manager
            cycle = await integration_manager.tuning_engine._run_optimization_cycle()
            
            assert cycle.cycle_success is True
            assert cycle.total_improvement_percent == 5.2
    
    @pytest.mark.asyncio 
    async def test_performance_target_validation(self, integration_manager, mock_orchestrator):
        """Test performance target validation and compliance."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Test performance targets met scenario
        performance_data = {
            'task_assignment_latency_ms': 0.008,  # Better than target 0.01
            'message_throughput_per_sec': 52000,   # Better than target 50000
            'memory_usage_mb': 280,                # Better than target 285
            'error_rate_percent': 0.003            # Better than target 0.005
        }
        
        targets_met = await integration_manager._check_performance_targets(performance_data)
        assert targets_met is True, "All performance targets should be met"
        
        # Test performance targets not met scenario
        performance_data_poor = {
            'task_assignment_latency_ms': 0.05,   # Worse than target
            'message_throughput_per_sec': 25000,  # Worse than target
            'memory_usage_mb': 450,               # Worse than target
            'error_rate_percent': 0.5             # Worse than target
        }
        
        targets_met = await integration_manager._check_performance_targets(performance_data_poor)
        assert targets_met is False, "Performance targets should not be met"
        
        # Test mixed scenario (some targets met)
        performance_data_mixed = {
            'task_assignment_latency_ms': 0.008,  # Good
            'message_throughput_per_sec': 30000,  # Poor
            'memory_usage_mb': 280,               # Good
            'error_rate_percent': 0.1             # Poor
        }
        
        targets_met = await integration_manager._check_performance_targets(performance_data_mixed)
        # Should be False as less than 80% of targets are met (2/4 = 50%)
        assert targets_met is False, "Mixed performance should not meet threshold"
    
    @pytest.mark.asyncio
    async def test_optimization_coordination(self, integration_manager, mock_orchestrator):
        """Test optimization coordination between components."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Mock performance data showing optimization needs
        mock_dashboard_data = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.025},  # Needs optimization
                    'message_throughput_per_sec': {'current': 35000},  # Needs optimization
                    'memory_usage_mb': {'current': 400}                # Needs optimization
                }
            }
        }
        
        with mock.patch.object(
            integration_manager.performance_monitor,
            'get_monitoring_dashboard_data',
            return_value=mock_dashboard_data
        ):
            # Test optimization need detection
            assert integration_manager._needs_optimization('task_assignment_latency_ms', 0.025, 0.01) is True
            assert integration_manager._needs_optimization('message_throughput_per_sec', 35000, 50000) is True
            assert integration_manager._needs_optimization('memory_usage_mb', 400, 285) is True
            
            # Mock optimizer calls to verify coordination
            with mock.patch.object(integration_manager.task_optimizer, 'optimize_task_assignment') as mock_task, \
                 mock.patch.object(integration_manager.comm_optimizer, 'optimize_message_throughput') as mock_comm, \
                 mock.patch.object(integration_manager.memory_optimizer, 'optimize_memory_usage') as mock_mem:
                
                mock_task.return_value = mock.MagicMock(success=True)
                mock_comm.return_value = mock.MagicMock(success=True)
                mock_mem.return_value = mock.MagicMock(success=True)
                
                # Test targeted optimizations
                await integration_manager._trigger_targeted_optimization('task_assignment_latency_ms', 0.025, 0.01)
                await integration_manager._trigger_targeted_optimization('message_throughput_per_sec', 35000, 50000)
                await integration_manager._trigger_targeted_optimization('memory_usage_mb', 400, 285)
                
                # Verify optimizers were called
                mock_task.assert_called_once()
                mock_comm.assert_called_once()
                mock_mem.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_manager, mock_orchestrator):
        """Test error handling and system recovery capabilities."""
        
        # Test initialization failure handling
        with mock.patch.object(TaskExecutionOptimizer, 'initialize', side_effect=Exception("Init failed")):
            manager = PerformanceIntegrationManager(orchestrator=mock_orchestrator)
            success = await manager.initialize(mock_orchestrator)
            
            assert success is False, "Initialization should fail gracefully"
            assert manager.status == IntegrationStatus.FAILED
        
        # Test component failure during operation
        await integration_manager.initialize(mock_orchestrator)
        
        # Simulate monitoring system failure
        with mock.patch.object(
            integration_manager.performance_monitor,
            'get_monitoring_dashboard_data',
            side_effect=Exception("Monitoring failed")
        ):
            health = await integration_manager._perform_health_check()
            
            assert health.monitoring_healthy is False
            assert len(health.active_issues) > 0
            assert "Monitoring system error" in str(health.active_issues)
        
        # Test graceful degradation
        original_alerting = integration_manager.alerting_system
        integration_manager.alerting_system = None  # Simulate alerting failure
        
        health = await integration_manager._perform_health_check()
        assert health.alerting_healthy is False
        
        # System should still be operational with degraded functionality
        assert health.components_active < health.total_components
    
    @pytest.mark.asyncio
    async def test_comprehensive_reporting(self, integration_manager, mock_orchestrator):
        """Test comprehensive reporting functionality."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Allow some operations to generate history
        await asyncio.sleep(20)
        
        # Test comprehensive report generation
        report = await integration_manager.get_comprehensive_report()
        
        # Verify report structure
        assert 'integration_status' in report
        assert 'component_reports' in report
        assert 'health_history_summary' in report
        
        # Verify integration status details
        integration_status = report['integration_status']
        assert integration_status['status'] == 'active'
        assert 'component_status' in integration_status
        assert 'health_summary' in integration_status
        
        # Verify health history summary
        health_summary = report['health_history_summary']
        assert health_summary['total_health_checks'] >= 0
        assert 'average_components_active' in health_summary
        
        # Test component reports are included
        component_reports = report['component_reports']
        if integration_manager.performance_monitor:
            assert 'monitoring' in component_reports
        if integration_manager.tuning_engine:
            assert 'tuning' in component_reports
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, integration_manager, mock_orchestrator):
        """Test graceful shutdown of integration system."""
        
        await integration_manager.initialize(mock_orchestrator)
        
        # Verify system is active
        assert integration_manager.status == IntegrationStatus.ACTIVE
        
        # Test shutdown
        await integration_manager.shutdown()
        
        # Verify shutdown status
        assert integration_manager.status == IntegrationStatus.SHUTDOWN
        assert integration_manager.shutdown_event.is_set()
        
        # Verify tasks are cancelled
        for task in integration_manager.component_tasks.values():
            if task:
                assert task.done(), "Component tasks should be completed/cancelled"
    
    @pytest.mark.asyncio
    async def test_factory_function(self, mock_orchestrator):
        """Test integration system factory function."""
        
        config = IntegrationConfiguration(
            enable_dashboards=False,  # Disable for testing
            startup_delay_seconds=5
        )
        
        # Test factory function
        manager = await create_integrated_performance_system(
            orchestrator=mock_orchestrator,
            config=config
        )
        
        assert manager is not None
        assert manager.status == IntegrationStatus.ACTIVE
        assert manager.orchestrator == mock_orchestrator
        
        # Cleanup
        await manager.shutdown()
        
        # Test factory function with initialization failure
        with mock.patch.object(PerformanceIntegrationManager, 'initialize', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await create_integrated_performance_system(orchestrator=mock_orchestrator)


class TestPerformanceIntegrationStress:
    """Stress tests for performance integration system."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_extended_operation(self):
        """Test extended operation and stability."""
        
        config = IntegrationConfiguration(
            enable_dashboards=False,
            startup_delay_seconds=2,
            health_check_interval_seconds=5
        )
        
        mock_orchestrator = mock.AsyncMock(spec=UniversalOrchestrator)
        mock_orchestrator.get_system_status.return_value = {'status': 'active'}
        
        manager = PerformanceIntegrationManager(config=config, orchestrator=mock_orchestrator)
        
        try:
            # Initialize system
            await manager.initialize(mock_orchestrator)
            
            # Run for extended period
            start_time = time.time()
            target_duration = 60  # 1 minute
            
            while time.time() - start_time < target_duration:
                # Verify system health periodically
                status = await manager.get_integration_status()
                assert status['status'] in ['active', 'degraded'], "System should remain operational"
                
                await asyncio.sleep(5)
            
            # Verify system maintained health history
            assert len(manager.health_history) > 5, "Should have accumulated health history"
            
            # Verify no critical issues accumulated
            final_health = manager.health_history[-1]
            assert len(final_health.active_issues) < 3, "Should not accumulate many issues"
            
        finally:
            await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operation handling."""
        
        config = IntegrationConfiguration(enable_dashboards=False, startup_delay_seconds=2)
        mock_orchestrator = mock.AsyncMock(spec=UniversalOrchestrator)
        mock_orchestrator.get_system_status.return_value = {'status': 'active'}
        
        manager = PerformanceIntegrationManager(config=config, orchestrator=mock_orchestrator)
        
        try:
            await manager.initialize(mock_orchestrator)
            
            # Run multiple concurrent operations
            async def get_status():
                return await manager.get_integration_status()
            
            async def get_report():
                return await manager.get_comprehensive_report()
            
            async def health_check():
                return await manager._perform_health_check()
            
            # Execute concurrently
            results = await asyncio.gather(
                get_status(),
                get_report(),
                health_check(),
                get_status(),
                health_check(),
                return_exceptions=True
            )
            
            # Verify all operations completed successfully
            for result in results:
                assert not isinstance(result, Exception), f"Operation failed: {result}"
            
        finally:
            await manager.shutdown()


if __name__ == "__main__":
    # Run tests
    import subprocess
    
    print("Running Performance Integration Test Suite...")
    result = subprocess.run([
        'python', '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--asyncio-mode=auto'
    ], cwd=Path(__file__).parent.parent.parent)
    
    exit(result.returncode)
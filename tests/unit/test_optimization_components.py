"""
Unit Tests for Performance Optimization Components

Tests individual optimization components to ensure they maintain
the extraordinary performance achievements of LeanVibe Agent Hive 2.0
while providing reliable optimization capabilities.

Test Categories:
- TaskExecutionOptimizer unit tests
- CommunicationHubOptimizer unit tests  
- ResourceOptimizer unit tests
- AutomatedTuningEngine unit tests
- Performance baseline tracking
- Optimization result validation
"""

import asyncio
import pytest
import time
import unittest.mock as mock
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from optimization.task_execution_optimizer import TaskExecutionOptimizer, OptimizationResult
from optimization.communication_hub_scaler import CommunicationHubOptimizer, ThroughputResult
from optimization.memory_resource_optimizer import ResourceOptimizer, MemoryOptimizationResult
from optimization.automated_tuning_engine import (
    AutomatedTuningEngine, OptimizationConfiguration, OptimizationStrategy, 
    TuningObjective, PerformanceBaseline
)


class TestTaskExecutionOptimizer:
    """Unit tests for TaskExecutionOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create TaskExecutionOptimizer instance."""
        return TaskExecutionOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        await optimizer.initialize()
        
        # Verify optimizer is initialized
        assert optimizer.initialized is True
        assert optimizer.memory_pool is not None
        assert optimizer.optimization_stats is not None
    
    @pytest.mark.asyncio
    async def test_task_assignment_optimization(self, optimizer):
        """Test task assignment optimization."""
        await optimizer.initialize()
        
        # Test optimization execution
        result = await optimizer.optimize_task_assignment()
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.optimization_type == "task_assignment"
        assert result.latency_improvement_ms >= 0
        assert result.memory_efficiency_improvement_percent >= 0
        
        # Verify performance metrics are within expected ranges
        assert result.final_latency_ms <= 0.1  # Should be sub-100 microseconds
        assert result.memory_allocated_mb <= 50  # Should use minimal memory
    
    @pytest.mark.asyncio
    async def test_memory_pool_optimization(self, optimizer):
        """Test memory pool optimization capabilities."""
        await optimizer.initialize()
        
        # Simulate memory pool usage
        with mock.patch.object(optimizer, '_optimize_memory_pools') as mock_optimize:
            mock_optimize.return_value = {
                'pool_efficiency': 95.5,
                'memory_saved_mb': 15.2,
                'allocation_speed_improvement_percent': 25.0
            }
            
            result = await optimizer.optimize_task_assignment()
            
            assert result.success is True
            assert result.memory_efficiency_improvement_percent > 0
    
    @pytest.mark.asyncio
    async def test_cpu_cache_optimization(self, optimizer):
        """Test CPU cache optimization."""
        await optimizer.initialize()
        
        # Test cache optimization
        with mock.patch.object(optimizer, '_optimize_cpu_cache') as mock_optimize:
            mock_optimize.return_value = {
                'cache_hit_rate': 98.7,
                'cache_misses_reduced': 45,
                'cpu_cycles_saved': 12500
            }
            
            result = await optimizer.optimize_task_assignment()
            
            assert result.success is True
            # CPU cache improvements should reduce latency
            assert result.latency_improvement_ms >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_under_load(self, optimizer):
        """Test optimization performance under simulated load."""
        await optimizer.initialize()
        
        # Simulate high load scenario
        tasks = []
        for _ in range(10):  # Concurrent optimizations
            tasks.append(optimizer.optimize_task_assignment())
        
        results = await asyncio.gather(*tasks)
        
        # Verify all optimizations completed successfully
        for result in results:
            assert result.success is True
            assert result.final_latency_ms <= 0.1  # Maintain performance under load
        
        # Verify optimization history is maintained
        assert len(optimizer.optimization_stats.optimization_history) >= 10


class TestCommunicationHubOptimizer:
    """Unit tests for CommunicationHubOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create CommunicationHubOptimizer instance."""
        return CommunicationHubOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test communication optimizer initialization."""
        await optimizer.initialize()
        
        assert optimizer.initialized is True
        assert optimizer.message_queues is not None
        assert optimizer.connection_pools is not None
    
    @pytest.mark.asyncio
    async def test_throughput_optimization(self, optimizer):
        """Test message throughput optimization."""
        await optimizer.initialize()
        
        result = await optimizer.optimize_message_throughput()
        
        # Verify result structure
        assert isinstance(result, ThroughputResult)
        assert result.success is True
        assert result.optimization_type == "message_throughput"
        assert result.throughput_improvement_percent >= 0
        
        # Verify throughput targets
        assert result.final_throughput_msg_per_sec >= 25000  # Minimum target
        assert result.latency_overhead_ms <= 1.0  # Low latency overhead
    
    @pytest.mark.asyncio 
    async def test_message_batching_optimization(self, optimizer):
        """Test message batching optimization."""
        await optimizer.initialize()
        
        with mock.patch.object(optimizer, '_optimize_message_batching') as mock_batch:
            mock_batch.return_value = {
                'batch_size_optimized': 150,
                'batching_efficiency': 92.3,
                'throughput_improvement': 35.0
            }
            
            result = await optimizer.optimize_message_throughput()
            
            assert result.success is True
            assert result.throughput_improvement_percent > 0
            assert result.message_batching_enabled is True
    
    @pytest.mark.asyncio
    async def test_connection_pool_optimization(self, optimizer):
        """Test connection pool optimization."""
        await optimizer.initialize()
        
        with mock.patch.object(optimizer, '_optimize_connection_pools') as mock_pools:
            mock_pools.return_value = {
                'pool_size_optimized': 25,
                'connection_reuse_rate': 96.8,
                'connection_latency_ms': 0.5
            }
            
            result = await optimizer.optimize_message_throughput()
            
            assert result.success is True
            assert result.connection_pool_optimized is True
            assert result.latency_overhead_ms <= 1.0
    
    @pytest.mark.asyncio
    async def test_high_throughput_scenario(self, optimizer):
        """Test optimization under high throughput requirements."""
        await optimizer.initialize()
        
        # Simulate high throughput requirement (50,000+ msg/sec)
        with mock.patch.object(optimizer, '_measure_current_throughput', return_value=48000):
            result = await optimizer.optimize_message_throughput()
            
            assert result.success is True
            # Should achieve target throughput
            assert result.final_throughput_msg_per_sec >= 50000


class TestResourceOptimizer:
    """Unit tests for ResourceOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create ResourceOptimizer instance.""" 
        return ResourceOptimizer()
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimizer):
        """Test resource optimizer initialization."""
        await optimizer.initialize()
        
        assert optimizer.initialized is True
        assert optimizer.gc_monitor is not None
        assert optimizer.memory_tracker is not None
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, optimizer):
        """Test memory usage optimization."""
        await optimizer.initialize()
        
        result = await optimizer.optimize_memory_usage()
        
        # Verify result structure
        assert isinstance(result, MemoryOptimizationResult)
        assert result.success is True
        assert result.optimization_type == "memory_usage"
        assert result.memory_reduction_mb >= 0
        
        # Verify memory targets
        assert result.final_memory_usage_mb <= 500  # Under peak load limit
        assert result.gc_efficiency_improvement_percent >= 0
    
    @pytest.mark.asyncio
    async def test_garbage_collection_tuning(self, optimizer):
        """Test garbage collection tuning."""
        await optimizer.initialize()
        
        with mock.patch.object(optimizer, '_tune_garbage_collection') as mock_gc:
            mock_gc.return_value = {
                'gc_frequency_optimized': True,
                'gc_pause_time_reduced_ms': 5.2,
                'memory_reclaim_efficiency': 94.1
            }
            
            result = await optimizer.optimize_memory_usage()
            
            assert result.success is True
            assert result.gc_tuning_applied is True
            assert result.gc_efficiency_improvement_percent > 0
    
    @pytest.mark.asyncio
    async def test_object_pooling_optimization(self, optimizer):
        """Test object pooling optimization."""
        await optimizer.initialize()
        
        with mock.patch.object(optimizer, '_optimize_object_pools') as mock_pools:
            mock_pools.return_value = {
                'pool_hit_rate': 97.3,
                'allocation_reduction_percent': 42.0,
                'memory_pool_efficiency': 91.5
            }
            
            result = await optimizer.optimize_memory_usage()
            
            assert result.success is True
            assert result.object_pooling_enabled is True
            assert result.allocation_efficiency_improvement_percent > 0
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, optimizer):
        """Test memory leak detection capabilities."""
        await optimizer.initialize()
        
        # Simulate memory leak detection
        with mock.patch.object(optimizer, '_detect_memory_leaks') as mock_detect:
            mock_detect.return_value = {
                'leaks_detected': 2,
                'leaked_memory_mb': 1.5,
                'leaks_resolved': 2
            }
            
            result = await optimizer.optimize_memory_usage()
            
            assert result.success is True
            # Should have resolved detected leaks
            assert result.memory_reduction_mb >= 1.5


class TestAutomatedTuningEngine:
    """Unit tests for AutomatedTuningEngine."""
    
    @pytest.fixture
    def tuning_config(self):
        """Create tuning configuration."""
        return OptimizationConfiguration(
            strategy=OptimizationStrategy.BALANCED,
            primary_objective=TuningObjective.OVERALL_PERFORMANCE,
            tuning_interval_seconds=60,  # Faster for testing
            evaluation_window_seconds=120,
            rollback_threshold_percent=5.0
        )
    
    @pytest.fixture
    def tuning_engine(self, tuning_config):
        """Create AutomatedTuningEngine instance."""
        return AutomatedTuningEngine(tuning_config)
    
    @pytest.mark.asyncio
    async def test_tuning_engine_initialization(self, tuning_engine):
        """Test tuning engine initialization."""
        success = await tuning_engine.initialize()
        
        assert success is True
        assert tuning_engine.task_optimizer is not None
        assert tuning_engine.comm_optimizer is not None
        assert tuning_engine.memory_optimizer is not None
        assert tuning_engine.baseline_manager is not None
    
    @pytest.mark.asyncio
    async def test_baseline_management(self, tuning_engine):
        """Test performance baseline management."""
        await tuning_engine.initialize()
        
        baseline_manager = tuning_engine.baseline_manager
        
        # Test baseline updates
        for i in range(25):  # Add sufficient data
            baseline_manager.update_metric('test_latency_ms', 0.01 + (i * 0.001))
        
        # Verify baseline calculation
        assert baseline_manager.has_sufficient_data('test_latency_ms', 20) is True
        baseline = baseline_manager.get_baseline('test_latency_ms')
        
        assert baseline is not None
        assert 'mean' in baseline
        assert 'p95' in baseline
        assert 'std' in baseline
        assert baseline['mean'] > 0
    
    @pytest.mark.asyncio
    async def test_optimization_opportunity_detection(self, tuning_engine):
        """Test optimization opportunity detection."""
        await tuning_engine.initialize()
        
        # Mock performance monitoring data
        mock_cycle = mock.MagicMock()
        mock_cycle.baseline_metrics = {
            'task_assignment_latency_ms': 0.05,  # Above target of 0.02
            'message_throughput_per_sec': 35000,  # Below target of 50000
            'memory_usage_mb': 450               # Above target of 400
        }
        
        opportunities = await tuning_engine._identify_optimization_opportunities(mock_cycle)
        
        # Should identify opportunities for all poor-performing metrics
        assert len(opportunities) >= 2  # At least latency and throughput
        
        # Verify opportunity structure
        for opportunity in opportunities:
            assert 'metric_name' in opportunity
            assert 'gap_percent' in opportunity
            assert 'priority' in opportunity
            assert opportunity['gap_percent'] > 5.0  # Significant gaps only
    
    @pytest.mark.asyncio
    async def test_optimization_action_generation(self, tuning_engine):
        """Test optimization action generation."""
        await tuning_engine.initialize()
        
        # Test action generation for different components
        opportunity = {
            'metric_name': 'task_assignment_latency_ms',
            'current_value': 0.05,
            'target_value': 0.02,
            'gap_percent': 150.0,
            'component': 'task_execution'
        }
        
        actions = await tuning_engine._generate_optimization_actions(opportunity)
        
        assert len(actions) >= 1
        for action in actions:
            assert action.component == 'task_execution'
            assert action.confidence > 0
            assert action.expected_impact != ""
            assert action.rationale != ""
    
    @pytest.mark.asyncio
    async def test_optimization_cycle_execution(self, tuning_engine):
        """Test complete optimization cycle execution."""
        await tuning_engine.initialize()
        
        # Mock performance monitor data
        mock_monitor = mock.MagicMock()
        mock_monitor.get_monitoring_dashboard_data.return_value = {
            'application_performance': {
                'metrics': {
                    'task_assignment_latency_ms': {'current': 0.03},
                    'message_throughput_per_sec': {'current': 40000},
                    'memory_usage_mb': {'current': 350}
                }
            }
        }
        
        tuning_engine.performance_monitor = mock_monitor
        
        # Mock optimizer success
        with mock.patch.object(tuning_engine.task_optimizer, 'optimize_task_assignment') as mock_opt:
            mock_opt.return_value = mock.MagicMock(success=True)
            
            # Execute optimization cycle
            cycle = await tuning_engine._run_optimization_cycle()
            
            assert cycle.cycle_id is not None
            assert cycle.start_time is not None
            assert cycle.end_time is not None
            assert len(cycle.baseline_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_rollback_mechanism(self, tuning_engine):
        """Test optimization rollback mechanism."""
        await tuning_engine.initialize()
        
        # Create mock cycle with poor performance
        mock_cycle = mock.MagicMock()
        mock_cycle.total_improvement_percent = -10.0  # Regression
        mock_cycle.actions_successful = [mock.MagicMock()]
        mock_cycle.actions_rolled_back = []
        mock_cycle.notes = []
        
        # Test rollback execution
        await tuning_engine._rollback_optimizations(mock_cycle)
        
        # Verify rollback was attempted
        assert len(mock_cycle.actions_rolled_back) >= 0
        assert len(mock_cycle.notes) > 0
        assert "Rolled back" in str(mock_cycle.notes)
    
    @pytest.mark.asyncio
    async def test_continuous_tuning_lifecycle(self, tuning_engine):
        """Test continuous tuning start/stop lifecycle.""" 
        await tuning_engine.initialize()
        
        # Test starting continuous tuning
        success = await tuning_engine.start_continuous_tuning()
        assert success is True
        assert tuning_engine.tuning_active is True
        
        # Allow brief operation
        await asyncio.sleep(1)
        
        # Test stopping continuous tuning
        await tuning_engine.stop_continuous_tuning()
        assert tuning_engine.tuning_active is False


class TestPerformanceBaseline:
    """Unit tests for PerformanceBaseline."""
    
    @pytest.fixture
    def baseline(self):
        """Create PerformanceBaseline instance."""
        return PerformanceBaseline(window_size=50)
    
    def test_baseline_initialization(self, baseline):
        """Test baseline initialization."""
        assert baseline.window_size == 50
        assert len(baseline.baselines) == 0
    
    def test_metric_updates(self, baseline):
        """Test metric updates and baseline calculation."""
        # Add insufficient data
        for i in range(5):
            baseline.update_metric('test_metric', float(i))
        
        assert not baseline.has_sufficient_data('test_metric', 10)
        assert baseline.get_baseline('test_metric') is None
        
        # Add sufficient data
        for i in range(15):  # Total 20 points
            baseline.update_metric('test_metric', float(i) + 10)
        
        assert baseline.has_sufficient_data('test_metric', 10)
        baseline_stats = baseline.get_baseline('test_metric')
        
        assert baseline_stats is not None
        assert 'mean' in baseline_stats
        assert 'median' in baseline_stats
        assert 'p95' in baseline_stats
        assert baseline_stats['mean'] > 0
    
    def test_window_size_enforcement(self, baseline):
        """Test that window size is enforced."""
        # Add more data than window size
        for i in range(60):  # More than window_size of 50
            baseline.update_metric('test_metric', float(i))
        
        baseline_data = baseline.baselines['test_metric']
        assert len(baseline_data['values']) == 50  # Should be limited to window_size
        assert len(baseline_data['timestamps']) == 50
        
        # Should have most recent values
        values_list = list(baseline_data['values'])
        assert values_list[-1] == 59.0  # Most recent value
        assert values_list[0] >= 10.0   # Older values dropped


if __name__ == "__main__":
    # Run tests
    import subprocess
    
    print("Running Performance Optimization Components Unit Tests...")
    result = subprocess.run([
        'python', '-m', 'pytest', 
        __file__, 
        '-v', 
        '--tb=short',
        '--asyncio-mode=auto'
    ], cwd=Path(__file__).parent.parent.parent)
    
    exit(result.returncode)
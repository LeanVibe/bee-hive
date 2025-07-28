"""
Comprehensive Test Suite for Load Balancing and Capacity Management

Tests all components of the advanced load balancing system including:
- AgentLoadBalancer with multiple strategies
- CapacityManager with scaling decisions
- PerformanceMetricsCollector with real-time monitoring
- AdaptiveScaler with pattern recognition
- ResourceOptimizer with system optimization
- HealthMonitor with degradation detection
- Distributed state management
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from app.core.agent_load_balancer import (
    AgentLoadBalancer, AgentLoadState, LoadBalancingStrategy, LoadBalancingDecision
)
from app.core.capacity_manager import (
    CapacityManager, ScalingDecision, ScalingAction, CapacityTier, ResourceAllocation
)
from app.core.performance_metrics_collector import (
    PerformanceMetricsCollector, MetricType, MetricValue, PerformanceProfile
)
from app.core.adaptive_scaler import (
    AdaptiveScaler, ScalingRule, ScalingTrigger, ScalingPattern
)
from app.core.resource_optimizer import (
    ResourceOptimizer, OptimizationType, OptimizationRule, OptimizationResult
)
from app.core.health_monitor import (
    HealthMonitor, HealthStatus, HealthCheck, HealthCheckType, AlertSeverity
)
from app.core.distributed_load_balancing_state import (
    DistributedLoadBalancingState, ClusterNode
)
from app.models.task import Task, TaskPriority
from app.models.agent import Agent, AgentStatus


class TestAgentLoadBalancer:
    """Test suite for AgentLoadBalancer."""
    
    @pytest.fixture
    async def load_balancer(self):
        """Create load balancer instance."""
        return AgentLoadBalancer()
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return Task(
            id=uuid.uuid4(),
            title="Test Task",
            description="Test task for load balancing",
            priority=TaskPriority.MEDIUM,
            complexity_score=1.0
        )
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agent IDs."""
        return ["agent_1", "agent_2", "agent_3"]
    
    async def test_load_balancer_initialization(self, load_balancer):
        """Test load balancer initializes correctly."""
        assert load_balancer.default_strategy == LoadBalancingStrategy.ADAPTIVE_HYBRID
        assert len(load_balancer.agent_loads) == 0
        assert len(load_balancer.decision_times) == 0
    
    async def test_agent_load_state_calculation(self):
        """Test agent load state calculations."""
        load_state = AgentLoadState(
            agent_id="test_agent",
            active_tasks=3,
            pending_tasks=2,
            context_usage_percent=75.0,
            memory_usage_mb=800.0,
            cpu_usage_percent=60.0,
            error_rate_percent=2.0
        )
        
        load_factor = load_state.calculate_load_factor()
        assert 0.0 <= load_factor <= 2.0
        
        assert load_state.can_handle_task(1.0) is True
        assert load_state.is_overloaded(0.85) is False
        assert load_state.is_underloaded(0.3) is False
    
    async def test_round_robin_selection(self, load_balancer, sample_task, sample_agents):
        """Test round-robin agent selection."""
        # Set up load balancer with sample agent data
        for agent_id in sample_agents:
            load_balancer.agent_loads[agent_id] = AgentLoadState(agent_id=agent_id)
        
        decision = await load_balancer._round_robin_selection(sample_agents, sample_task)
        
        assert decision.selected_agent_id in sample_agents
        assert decision.strategy_used == LoadBalancingStrategy.ROUND_ROBIN
        assert decision.decision_confidence == 0.8
    
    async def test_least_connections_selection(self, load_balancer, sample_task, sample_agents):
        """Test least connections selection strategy."""
        # Set up agents with different loads
        load_balancer.agent_loads["agent_1"] = AgentLoadState(agent_id="agent_1", active_tasks=1)
        load_balancer.agent_loads["agent_2"] = AgentLoadState(agent_id="agent_2", active_tasks=3)
        load_balancer.agent_loads["agent_3"] = AgentLoadState(agent_id="agent_3", active_tasks=2)
        
        decision = await load_balancer._least_connections_selection(sample_agents, sample_task)
        
        # Should select agent with least tasks (agent_1)
        assert decision.selected_agent_id == "agent_1"
        assert decision.strategy_used == LoadBalancingStrategy.LEAST_CONNECTIONS
    
    async def test_load_balancing_metrics(self, load_balancer, sample_agents):
        """Test load balancing metrics collection."""
        # Add some decision times
        load_balancer.decision_times.extend([10.0, 15.0, 20.0, 12.0, 18.0])
        
        # Add agent loads
        for agent_id in sample_agents:
            load_balancer.agent_loads[agent_id] = AgentLoadState(
                agent_id=agent_id,
                active_tasks=2,
                health_score=0.9
            )
        
        metrics = await load_balancer.get_load_balancing_metrics()
        
        assert "decision_metrics" in metrics
        assert "agent_load_summary" in metrics
        assert "system_health" in metrics
        assert metrics["decision_metrics"]["total_decisions"] == 5
        assert metrics["system_health"]["total_agents"] == 3


class TestCapacityManager:
    """Test suite for CapacityManager."""
    
    @pytest.fixture
    async def capacity_manager(self):
        """Create capacity manager instance."""
        load_balancer = Mock()
        return CapacityManager(load_balancer=load_balancer)
    
    async def test_capacity_manager_initialization(self, capacity_manager):
        """Test capacity manager initializes correctly."""
        assert len(capacity_manager.resource_allocations) == 0
        assert len(capacity_manager.scaling_history) == 0
        assert capacity_manager.config["min_agents"] == 2
        assert capacity_manager.config["max_agents"] == 20
    
    async def test_resource_allocation_creation(self, capacity_manager):
        """Test resource allocation for different tiers."""
        allocation = capacity_manager._create_allocation_for_tier("test_agent", CapacityTier.STANDARD)
        
        assert allocation.agent_id == "test_agent"
        assert allocation.tier == CapacityTier.STANDARD
        assert allocation.max_concurrent_tasks == 5
        assert allocation.memory_limit_mb == 1000
    
    async def test_capacity_predictions(self, capacity_manager):
        """Test capacity prediction generation."""
        # Mock historical data
        historical_loads = [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4]
        
        prediction = await capacity_manager._predict_capacity_need(
            historical_loads, 60, len(historical_loads)
        )
        
        assert prediction.timeframe_minutes == 60
        assert 0.0 <= prediction.predicted_load_factor <= 2.0
        assert prediction.recommended_agent_count >= capacity_manager.config["min_agents"]
        assert len(prediction.risk_factors) >= 0
    
    async def test_scaling_decision_execution(self, capacity_manager):
        """Test scaling decision execution."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            target_agents=["new_agent_1", "new_agent_2"],
            resource_changes={},
            reasoning="Test scale up",
            confidence=0.8,
            estimated_impact={"agents_added": 2}
        )
        
        result = await capacity_manager.execute_scaling_decision(decision)
        
        assert result["success"] is True
        assert result["agents_added"] == 2
        assert len(capacity_manager.resource_allocations) == 2


class TestPerformanceMetricsCollector:
    """Test suite for PerformanceMetricsCollector."""
    
    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector instance."""
        return PerformanceMetricsCollector(collection_interval=1.0)
    
    async def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initializes correctly."""
        assert metrics_collector.collection_interval == 1.0
        assert len(metrics_collector.profiles) == 0
        assert len(metrics_collector.global_metrics) == 0
        assert metrics_collector.collection_active is False
    
    async def test_custom_metric_recording(self, metrics_collector):
        """Test custom metric recording."""
        await metrics_collector.record_custom_metric(
            entity_id="test_entity",
            metric_name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            entity_type="test"
        )
        
        assert "test_entity" in metrics_collector.profiles
        profile = metrics_collector.profiles["test_entity"]
        assert "test_metric" in profile.metrics
        assert profile.metrics["test_metric"].values[-1][1] == 42.0
    
    async def test_performance_summary(self, metrics_collector):
        """Test performance summary generation."""
        # Add some test metrics
        await metrics_collector.record_custom_metric(
            "agent_1", "response_time", 100.0, MetricType.GAUGE, "agent"
        )
        await metrics_collector.record_custom_metric(
            "agent_1", "error_rate", 2.0, MetricType.GAUGE, "agent"
        )
        
        summary = await metrics_collector.get_performance_summary("agent_1")
        
        assert summary["entity_id"] == "agent_1"
        assert summary["entity_type"] == "agent"
        assert "metrics" in summary
        assert "response_time" in summary["metrics"]
    
    async def test_system_wide_summary(self, metrics_collector):
        """Test system-wide performance summary."""
        # Add global metrics
        metrics_collector._update_global_metric("system.cpu", 75.0, MetricType.GAUGE)
        metrics_collector._update_global_metric("system.memory", 60.0, MetricType.GAUGE)
        
        # Add agent profiles
        await metrics_collector.record_custom_metric(
            "agent_1", "health_score", 0.9, MetricType.GAUGE, "agent"
        )
        await metrics_collector.record_custom_metric(
            "agent_2", "health_score", 0.7, MetricType.GAUGE, "agent"
        )
        
        summary = await metrics_collector.get_performance_summary()
        
        assert "system_metrics" in summary
        assert "agent_summary" in summary
        assert summary["agent_summary"]["total_agents"] == 2


class TestAdaptiveScaler:
    """Test suite for AdaptiveScaler."""
    
    @pytest.fixture
    async def adaptive_scaler(self):
        """Create adaptive scaler instance."""
        load_balancer = Mock()
        capacity_manager = Mock()
        metrics_collector = Mock()
        
        return AdaptiveScaler(
            load_balancer=load_balancer,
            capacity_manager=capacity_manager,
            metrics_collector=metrics_collector
        )
    
    async def test_adaptive_scaler_initialization(self, adaptive_scaler):
        """Test adaptive scaler initializes correctly."""
        assert len(adaptive_scaler.scaling_rules) > 0
        assert adaptive_scaler.auto_scaling_enabled is True
        assert adaptive_scaler.evaluation_interval == 60
    
    async def test_scaling_rule_evaluation(self, adaptive_scaler):
        """Test scaling rule condition evaluation."""
        rule = ScalingRule(
            name="test_rule",
            trigger=ScalingTrigger.WORKLOAD_BASED,
            condition="avg_load > 0.8 and total_agents < 10",
            action=ScalingAction.SCALE_UP
        )
        
        context = {
            "avg_load": 0.9,
            "total_agents": 5,
            "overloaded_agents": 2
        }
        
        assert rule.evaluate_condition(context) is True
        
        context["avg_load"] = 0.5
        assert rule.evaluate_condition(context) is False
    
    async def test_pattern_detection(self, adaptive_scaler):
        """Test workload pattern detection."""
        # Create pattern detection data
        for i in range(30):
            adaptive_scaler.pattern_detection_window.append({
                "timestamp": datetime.utcnow(),
                "load_factor": 0.5 + (i * 0.01),  # Gradual increase
                "agent_count": 5,
                "response_time": 1000
            })
        
        pattern = adaptive_scaler._detect_workload_pattern()
        assert pattern in [ScalingPattern.GRADUAL_INCREASE, ScalingPattern.STEADY_STATE]
    
    async def test_scaling_decision_generation(self, adaptive_scaler):
        """Test scaling decision generation."""
        rule = ScalingRule(
            name="test_scale_up",
            trigger=ScalingTrigger.WORKLOAD_BASED,
            condition="avg_load > 0.8",
            action=ScalingAction.SCALE_UP
        )
        
        context = {
            "avg_load": 0.9,
            "total_agents": 3,
            "overloaded_agents": 2
        }
        
        decision = await adaptive_scaler._generate_scaling_decision(rule, context)
        
        assert decision.action == ScalingAction.SCALE_UP
        assert len(decision.target_agents) >= 1
        assert decision.confidence > 0.0
    
    async def test_scaling_metrics(self, adaptive_scaler):
        """Test scaling metrics collection."""
        # Add some scaling history
        from app.core.adaptive_scaler import ScalingEvent
        
        event = ScalingEvent(
            timestamp=datetime.utcnow(),
            trigger=ScalingTrigger.WORKLOAD_BASED,
            action=ScalingAction.SCALE_UP,
            context={"avg_load": 0.9},
            decision=Mock(),
            success=True
        )
        
        adaptive_scaler.scaling_history.append(event)
        
        metrics = await adaptive_scaler.get_scaling_metrics()
        
        assert "auto_scaling_enabled" in metrics
        assert "scaling_history" in metrics
        assert metrics["scaling_history"]["total_events"] == 1


class TestResourceOptimizer:
    """Test suite for ResourceOptimizer."""
    
    @pytest.fixture
    async def resource_optimizer(self):
        """Create resource optimizer instance."""
        metrics_collector = Mock()
        return ResourceOptimizer(metrics_collector=metrics_collector)
    
    async def test_resource_optimizer_initialization(self, resource_optimizer):
        """Test resource optimizer initializes correctly."""
        assert len(resource_optimizer.optimization_rules) > 0
        assert resource_optimizer.optimization_active is False
        assert resource_optimizer.monitoring_interval == 60
    
    async def test_optimization_rule_evaluation(self, resource_optimizer):
        """Test optimization rule condition evaluation."""
        rule = OptimizationRule(
            name="test_memory_optimization",
            optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
            trigger_condition="memory_percent > 80",
            action="optimize_memory"
        )
        
        context = {"memory_percent": 85.0, "cpu_percent": 50.0}
        
        assert rule.evaluate_condition(context) is True
        
        context["memory_percent"] = 70.0
        assert rule.evaluate_condition(context) is False
    
    async def test_memory_optimization(self, resource_optimizer):
        """Test memory optimization execution."""
        from app.core.resource_optimizer import ResourceUsage
        
        current_usage = ResourceUsage(
            timestamp=datetime.utcnow(),
            memory_mb=1500.0,
            memory_percent=85.0,
            cpu_percent=60.0,
            disk_read_mb_per_sec=10.0,
            disk_write_mb_per_sec=5.0,
            network_bytes_per_sec=1000000,
            active_connections=15,
            context_usage_percent=80.0
        )
        
        result = await resource_optimizer.optimize_memory(current_usage)
        
        assert result.optimization_type == OptimizationType.MEMORY_OPTIMIZATION
        assert result.success is True
        assert "memory_mb" in result.resources_freed
    
    async def test_resource_metrics(self, resource_optimizer):
        """Test resource metrics collection."""
        # Add some resource history
        from app.core.resource_optimizer import ResourceUsage
        
        usage = ResourceUsage(
            timestamp=datetime.utcnow(),
            memory_mb=800.0,
            memory_percent=70.0,
            cpu_percent=55.0,
            disk_read_mb_per_sec=5.0,
            disk_write_mb_per_sec=3.0,
            network_bytes_per_sec=500000,
            active_connections=10,
            context_usage_percent=60.0
        )
        
        resource_optimizer.resource_history.append(usage)
        
        metrics = await resource_optimizer.get_resource_metrics()
        
        assert "current_usage" in metrics
        assert "trends" in metrics
        assert "optimization_stats" in metrics


class TestHealthMonitor:
    """Test suite for HealthMonitor."""
    
    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor instance."""
        metrics_collector = Mock()
        return HealthMonitor(metrics_collector=metrics_collector)
    
    async def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initializes correctly."""
        assert len(health_monitor.agent_profiles) == 0
        assert health_monitor.monitoring_active is False
        assert health_monitor.check_interval == 30
    
    async def test_agent_profile_initialization(self, health_monitor):
        """Test agent health profile initialization."""
        await health_monitor._initialize_agent_profile("test_agent")
        
        assert "test_agent" in health_monitor.agent_profiles
        profile = health_monitor.agent_profiles["test_agent"]
        assert profile.agent_id == "test_agent"
        assert len(profile.checks) > 0
        assert profile.overall_status == HealthStatus.UNKNOWN
    
    async def test_health_check_execution(self, health_monitor):
        """Test health check execution."""
        await health_monitor._initialize_agent_profile("test_agent")
        profile = health_monitor.agent_profiles["test_agent"]
        
        # Get a health check
        heartbeat_check = profile.checks["heartbeat"]
        
        # Mock the check function
        with patch.object(health_monitor, 'check_agent_heartbeat', return_value=True):
            await health_monitor._execute_health_check("test_agent", heartbeat_check)
        
        assert heartbeat_check.last_check_result is True
        assert heartbeat_check.consecutive_failures == 0
        assert heartbeat_check.consecutive_successes == 1
    
    async def test_health_score_calculation(self, health_monitor):
        """Test agent health score calculation."""
        await health_monitor._initialize_agent_profile("test_agent")
        profile = health_monitor.agent_profiles["test_agent"]
        
        # Set some health checks as passed
        profile.checks["heartbeat"].record_success()
        profile.checks["performance"].record_success()
        profile.checks["resource_usage"].record_failure("High memory usage")
        
        profile.update_health_score()
        
        assert 0.0 <= profile.health_score <= 1.0
        assert profile.overall_status != HealthStatus.UNKNOWN
    
    async def test_health_summary(self, health_monitor):
        """Test health summary generation."""
        # Add some agent profiles
        await health_monitor._initialize_agent_profile("agent_1")
        await health_monitor._initialize_agent_profile("agent_2")
        
        # Set different health states
        health_monitor.agent_profiles["agent_1"].overall_status = HealthStatus.HEALTHY
        health_monitor.agent_profiles["agent_1"].health_score = 0.9
        
        health_monitor.agent_profiles["agent_2"].overall_status = HealthStatus.DEGRADED
        health_monitor.agent_profiles["agent_2"].health_score = 0.6
        
        summary = await health_monitor.get_health_summary()
        
        assert "system_health" in summary
        assert "alerts" in summary
        assert "health_checks" in summary
        assert summary["system_health"]["total_agents"] == 2


class TestDistributedLoadBalancingState:
    """Test suite for DistributedLoadBalancingState."""
    
    @pytest.fixture
    async def distributed_state(self):
        """Create distributed state manager instance."""
        # Mock Redis client
        redis_mock = AsyncMock()
        return DistributedLoadBalancingState(
            redis_client=redis_mock,
            node_id="test_node",
            cluster_name="test_cluster"
        )
    
    async def test_distributed_state_initialization(self, distributed_state):
        """Test distributed state initializes correctly."""
        assert distributed_state.node_id == "test_node"
        assert distributed_state.cluster_name == "test_cluster"
        assert distributed_state.is_leader is False
        assert len(distributed_state.cluster_nodes) == 0
    
    async def test_cluster_node_serialization(self):
        """Test cluster node serialization."""
        node = ClusterNode(
            node_id="test_node",
            hostname="localhost",
            process_id=12345,
            started_at=datetime.utcnow(),
            last_heartbeat=datetime.utcnow(),
            is_leader=True
        )
        
        # Test serialization
        node_dict = node.to_dict()
        assert node_dict["node_id"] == "test_node"
        assert node_dict["is_leader"] is True
        
        # Test deserialization
        restored_node = ClusterNode.from_dict(node_dict)
        assert restored_node.node_id == node.node_id
        assert restored_node.is_leader == node.is_leader
    
    async def test_agent_load_state_storage(self, distributed_state):
        """Test agent load state storage and retrieval."""
        load_state = AgentLoadState(
            agent_id="test_agent",
            active_tasks=3,
            context_usage_percent=75.0,
            health_score=0.9
        )
        
        # Mock Redis operations
        distributed_state.redis_client.hset = AsyncMock()
        distributed_state.redis_client.expire = AsyncMock()
        distributed_state.redis_client.hgetall = AsyncMock(return_value={
            "agent_id": "test_agent",
            "active_tasks": "3",
            "pending_tasks": "0",
            "context_usage_percent": "75.0",
            "memory_usage_mb": "0.0",
            "cpu_usage_percent": "0.0",
            "average_response_time_ms": "0.0",
            "error_rate_percent": "0.0",
            "throughput_tasks_per_hour": "0.0",
            "estimated_capacity": "1.0",
            "utilization_ratio": "0.0",
            "health_score": "0.9",
            "last_updated": load_state.last_updated.isoformat()
        })
        
        # Store and retrieve
        await distributed_state.store_agent_load_state("test_agent", load_state)
        retrieved_state = await distributed_state.get_agent_load_state("test_agent")
        
        assert retrieved_state is not None
        assert retrieved_state.agent_id == "test_agent"
        assert retrieved_state.active_tasks == 3
        assert retrieved_state.health_score == 0.9
    
    async def test_distributed_locking(self, distributed_state):
        """Test distributed locking mechanism."""
        resource = "test_resource"
        
        # Mock successful lock acquisition
        distributed_state.redis_client.set = AsyncMock(return_value=True)
        distributed_state.redis_client.get = AsyncMock(return_value="test_node:lock_id")
        distributed_state.redis_client.eval = AsyncMock(return_value=1)
        
        # Test lock acquisition
        lock_id = await distributed_state.acquire_distributed_lock(resource, timeout=30)
        assert lock_id is not None
        assert lock_id.startswith("test_node:")
        
        # Test lock release
        success = await distributed_state.release_distributed_lock(resource, lock_id)
        assert success is True
    
    async def test_cluster_status(self, distributed_state):
        """Test cluster status retrieval."""
        # Add some mock cluster nodes
        distributed_state.cluster_nodes = {
            "node_1": ClusterNode(
                node_id="node_1",
                hostname="host1",
                process_id=123,
                started_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            ),
            "node_2": ClusterNode(
                node_id="node_2",
                hostname="host2",
                process_id=456,
                started_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
        }
        
        status = await distributed_state.get_cluster_status()
        
        assert status["cluster_name"] == "test_cluster"
        assert status["node_id"] == "test_node"
        assert status["cluster_size"] == 2
        assert len(status["nodes"]) == 2


class TestLoadBalancingIntegration:
    """Integration tests for the complete load balancing system."""
    
    async def test_end_to_end_task_assignment(self):
        """Test complete task assignment flow with load balancing."""
        # Create components
        metrics_collector = PerformanceMetricsCollector()
        load_balancer = AgentLoadBalancer()
        capacity_manager = CapacityManager(load_balancer)
        
        # Mock base orchestrator
        base_orchestrator = Mock()
        base_orchestrator.agents = {
            "agent_1": Mock(status=AgentStatus.ACTIVE, context_window_usage=0.5),
            "agent_2": Mock(status=AgentStatus.ACTIVE, context_window_usage=0.3),
            "agent_3": Mock(status=AgentStatus.IDLE, context_window_usage=0.1)
        }
        
        # Create integration
        from app.core.orchestrator_load_balancing_integration import LoadBalancingOrchestrator
        lb_orchestrator = LoadBalancingOrchestrator(base_orchestrator)
        
        # Test task assignment
        task = Task(
            id=uuid.uuid4(),
            title="Integration Test Task",
            description="Test task for integration",
            priority=TaskPriority.HIGH
        )
        
        # Mock the load balancer selection
        with patch.object(load_balancer, 'select_agent_for_task') as mock_select:
            mock_select.return_value = LoadBalancingDecision(
                selected_agent_id="agent_3",
                strategy_used=LoadBalancingStrategy.LEAST_CONNECTIONS,
                decision_time_ms=25.0,
                agent_scores={"agent_3": 0.9},
                load_factors={"agent_3": 0.1},
                decision_confidence=0.95,
                reasoning="Selected least loaded agent"
            )
            
            selected_agent = await lb_orchestrator.assign_task_with_load_balancing(task)
            
            assert selected_agent == "agent_3"
            mock_select.assert_called_once()
    
    async def test_system_status_integration(self):
        """Test system status with all components."""
        # Create mock components
        load_balancer = Mock()
        load_balancer.get_load_balancing_metrics = AsyncMock(return_value={
            "decision_metrics": {"total_decisions": 10, "average_decision_time_ms": 15.0},
            "agent_load_summary": {},
            "system_health": {"total_agents": 3}
        })
        
        capacity_manager = Mock()
        capacity_manager.get_capacity_metrics = AsyncMock(return_value={
            "total_agents": 3,
            "predictions": {}
        })
        
        health_monitor = Mock()
        health_monitor.get_health_summary = AsyncMock(return_value={
            "system_health": {"total_agents": 3, "status_distribution": {"healthy": 3}}
        })
        
        adaptive_scaler = Mock()
        adaptive_scaler.get_scaling_metrics = AsyncMock(return_value={
            "auto_scaling_enabled": True,
            "scaling_history": {"total_events": 5}
        })
        
        resource_optimizer = Mock()
        resource_optimizer.get_resource_metrics = AsyncMock(return_value={
            "system_health": {"memory_pressure": "normal"}
        })
        
        # Create integration with mocked components
        base_orchestrator = Mock()
        base_orchestrator.agents = {"agent_1": Mock(), "agent_2": Mock(), "agent_3": Mock()}
        base_orchestrator.active_sessions = {}
        base_orchestrator.is_running = True
        base_orchestrator.metrics = {"tasks_completed": 100}
        
        from app.core.orchestrator_load_balancing_integration import LoadBalancingOrchestrator
        lb_orchestrator = LoadBalancingOrchestrator(base_orchestrator)
        
        # Replace components with mocks
        lb_orchestrator.load_balancer = load_balancer
        lb_orchestrator.capacity_manager = capacity_manager
        lb_orchestrator.health_monitor = health_monitor
        lb_orchestrator.adaptive_scaler = adaptive_scaler
        lb_orchestrator.resource_optimizer = resource_optimizer
        lb_orchestrator.load_balancing_active = True
        
        status = await lb_orchestrator.get_system_status()
        
        assert "base_orchestrator" in status
        assert "load_balancing" in status
        assert "component_metrics" in status
        assert status["base_orchestrator"]["agents_count"] == 3
        assert status["load_balancing"]["load_balancing_active"] is True


@pytest.mark.asyncio
class TestPerformanceBenchmarks:
    """Performance benchmark tests for load balancing components."""
    
    async def test_load_balancing_decision_speed(self):
        """Test that load balancing decisions are made within 100ms."""
        load_balancer = AgentLoadBalancer()
        
        # Set up test agents
        agents = [f"agent_{i}" for i in range(10)]
        for agent_id in agents:
            load_balancer.agent_loads[agent_id] = AgentLoadState(
                agent_id=agent_id,
                active_tasks=2,
                context_usage_percent=50.0,
                health_score=0.8
            )
        
        # Create test task
        task = Task(
            id=uuid.uuid4(),
            title="Performance Test Task",
            description="Task for performance testing",
            priority=TaskPriority.MEDIUM
        )
        
        # Measure decision time
        start_time = asyncio.get_event_loop().time()
        
        decision = await load_balancer.select_agent_for_task(
            task=task,
            available_agents=agents,
            strategy=LoadBalancingStrategy.LEAST_CONNECTIONS
        )
        
        end_time = asyncio.get_event_loop().time()
        decision_time_ms = (end_time - start_time) * 1000
        
        assert decision_time_ms < 100.0, f"Decision took {decision_time_ms}ms, should be < 100ms"
        assert decision.selected_agent_id in agents
    
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance."""
        metrics_collector = PerformanceMetricsCollector(collection_interval=0.1)
        
        # Record many metrics quickly
        start_time = asyncio.get_event_loop().time()
        
        for i in range(100):
            await metrics_collector.record_custom_metric(
                entity_id=f"entity_{i % 10}",
                metric_name="test_metric",
                value=float(i),
                metric_type=MetricType.COUNTER
            )
        
        end_time = asyncio.get_event_loop().time()
        total_time_ms = (end_time - start_time) * 1000
        
        # Should be able to record 100 metrics in under 100ms
        assert total_time_ms < 100.0, f"Metrics recording took {total_time_ms}ms for 100 metrics"
    
    async def test_concurrent_agent_monitoring(self):
        """Test concurrent monitoring of multiple agents."""
        health_monitor = HealthMonitor(Mock())
        
        # Initialize profiles for many agents
        agent_count = 50
        for i in range(agent_count):
            await health_monitor._initialize_agent_profile(f"agent_{i}")
        
        # Mock all health check functions to return True
        with patch.object(health_monitor, 'check_agent_heartbeat', return_value=True), \
             patch.object(health_monitor, 'check_agent_performance', return_value=True), \
             patch.object(health_monitor, 'check_resource_usage', return_value=True), \
             patch.object(health_monitor, 'check_error_rate', return_value=True), \
             patch.object(health_monitor, 'check_response_time', return_value=True), \
             patch.object(health_monitor, 'check_memory_leak', return_value=True):
            
            start_time = asyncio.get_event_loop().time()
            
            # Run health checks for all agents
            await health_monitor._run_health_checks()
            
            end_time = asyncio.get_event_loop().time()
            total_time_ms = (end_time - start_time) * 1000
            
            # Should monitor 50 agents in under 1 second
            assert total_time_ms < 1000.0, f"Health monitoring took {total_time_ms}ms for {agent_count} agents"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
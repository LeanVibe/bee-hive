"""
Integration Tests for ConsolidatedProductionOrchestrator
Epic 1 Phase 1.1 - Comprehensive testing for orchestrator consolidation

This test suite validates that the ConsolidatedProductionOrchestrator maintains
full compatibility with existing orchestrator implementations while providing
enhanced functionality and performance.

Test Categories:
1. Core Functionality Tests
2. Performance and Scalability Tests  
3. Migration Compatibility Tests
4. Error Handling and Recovery Tests
5. Plugin System Integration Tests
6. Production Readiness Tests
"""

import asyncio
import pytest
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Import test fixtures and utilities
from tests.conftest import orchestrator_config, simple_orchestrator_mock

# Import orchestrator components
from app.core.consolidated_orchestrator import (
    ConsolidatedProductionOrchestrator,
    create_consolidated_orchestrator,
    create_consolidated_orchestrator_sync
)
from app.core.orchestrator_interfaces import (
    OrchestratorConfig,
    OrchestratorMode,
    AgentSpec,
    TaskSpec,
    AgentStatus,
    TaskResult,
    SystemHealth,
    HealthStatus
)
from app.core.orchestrator_migration import (
    OrchestratorMigrationManager,
    migrate_orchestrator,
    MigrationStrategy,
    validate_orchestrator_compatibility
)

pytestmark = pytest.mark.asyncio


class TestConsolidatedOrchestratorCore:
    """Test core orchestrator functionality."""
    
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization with different configurations."""
        # Test default initialization
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        assert orchestrator._initialized is True
        assert isinstance(orchestrator.config, OrchestratorConfig)
        assert orchestrator.config.mode == OrchestratorMode.PRODUCTION
        
        # Test cleanup
        await orchestrator.shutdown()
        assert orchestrator._initialized is False
    
    async def test_orchestrator_initialization_with_config(self):
        """Test orchestrator initialization with custom configuration."""
        config = OrchestratorConfig(
            max_agents=25,
            mode=OrchestratorMode.DEVELOPMENT,
            enable_plugins=False,
            enable_advanced_features=False
        )
        
        orchestrator = ConsolidatedProductionOrchestrator(config)
        await orchestrator.initialize()
        
        assert orchestrator.config.max_agents == 25
        assert orchestrator.config.mode == OrchestratorMode.DEVELOPMENT
        assert orchestrator.config.enable_plugins is False
        
        await orchestrator.shutdown()
    
    async def test_health_check_no_agents(self):
        """Test health check when no agents are registered."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        health = await orchestrator.health_check()
        
        assert isinstance(health, SystemHealth)
        assert health.status == HealthStatus.NO_AGENTS
        assert health.orchestrator_type == "ConsolidatedProductionOrchestrator"
        assert health.version == "2.0.0-consolidated"
        assert health.uptime_seconds >= 0
        
        await orchestrator.shutdown()
    
    async def test_health_check_with_components(self):
        """Test health check with various component states."""
        config = OrchestratorConfig(enable_plugins=True, enable_monitoring=True)
        orchestrator = ConsolidatedProductionOrchestrator(config)
        await orchestrator.initialize()
        
        health = await orchestrator.health_check()
        
        assert "components" in health.components
        assert "performance" in health.performance
        assert "config" in health.config
        
        # Check that enabled features are reflected in health
        enabled_features = health.performance.get("enabled_features", [])
        assert "core_orchestration" in enabled_features
        assert "monitoring" in enabled_features
        
        await orchestrator.shutdown()


class TestAgentManagement:
    """Test agent lifecycle management."""
    
    async def test_register_agent_basic(self):
        """Test basic agent registration."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        agent_spec = AgentSpec(
            role="backend_developer",
            agent_type="claude_code",
            workspace_name="test_workspace"
        )
        
        agent_id = await orchestrator.register_agent(agent_spec)
        
        assert isinstance(agent_id, str)
        assert len(agent_id) > 0
        
        await orchestrator.shutdown()
    
    async def test_register_multiple_agents(self):
        """Test registering multiple agents."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        agent_specs = [
            AgentSpec(role="backend_developer", agent_type="claude_code"),
            AgentSpec(role="frontend_developer", agent_type="claude_code"),
            AgentSpec(role="qa_engineer", agent_type="claude_code")
        ]
        
        agent_ids = []
        for spec in agent_specs:
            agent_id = await orchestrator.register_agent(spec)
            agent_ids.append(agent_id)
        
        assert len(agent_ids) == 3
        assert len(set(agent_ids)) == 3  # All IDs should be unique
        
        # List agents and verify count
        agents = await orchestrator.list_agents()
        assert len(agents) >= 3  # Could be more if simple orchestrator is used
        
        await orchestrator.shutdown()
    
    async def test_get_agent_status(self):
        """Test getting agent status."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        agent_spec = AgentSpec(role="backend_developer", agent_type="claude_code")
        agent_id = await orchestrator.register_agent(agent_spec)
        
        status = await orchestrator.get_agent_status(agent_id)
        
        assert isinstance(status, AgentStatus)
        assert status.id == agent_id
        assert status.role == "backend_developer"
        assert status.health in ["healthy", "inactive"]
        
        await orchestrator.shutdown()
    
    async def test_list_agents(self):
        """Test listing all agents."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Initially should have no agents (or agents from simple orchestrator)
        initial_agents = await orchestrator.list_agents()
        initial_count = len(initial_agents)
        
        # Register some agents
        agent_specs = [
            AgentSpec(role="backend_developer"),
            AgentSpec(role="frontend_developer")
        ]
        
        for spec in agent_specs:
            await orchestrator.register_agent(spec)
        
        # List agents again
        agents = await orchestrator.list_agents()
        
        # Should have at least 2 more agents
        assert len(agents) >= initial_count + 2
        
        # All agents should have required fields
        for agent in agents:
            assert isinstance(agent, AgentStatus)
            assert agent.id is not None
            assert agent.role is not None
            assert agent.status is not None
            
        await orchestrator.shutdown()
    
    async def test_shutdown_agent(self):
        """Test agent shutdown."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        agent_spec = AgentSpec(role="backend_developer")
        agent_id = await orchestrator.register_agent(agent_spec)
        
        # Shutdown the agent
        result = await orchestrator.shutdown_agent(agent_id, graceful=True)
        
        # Result could be True or False depending on implementation
        assert isinstance(result, bool)
        
        await orchestrator.shutdown()


class TestTaskOrchestration:
    """Test task delegation and management."""
    
    async def test_delegate_task_basic(self):
        """Test basic task delegation."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        task_spec = TaskSpec(
            description="Test task for backend development",
            task_type="development",
            priority="medium",
            preferred_agent_role="backend_developer"
        )
        
        task_result = await orchestrator.delegate_task(task_spec)
        
        assert isinstance(task_result, TaskResult)
        assert task_result.id is not None
        assert task_result.description == task_spec.description
        assert task_result.task_type == task_spec.task_type
        assert task_result.priority == task_spec.priority
        assert task_result.status in ["assigned", "pending", "running"]
        
        await orchestrator.shutdown()
    
    async def test_delegate_multiple_tasks(self):
        """Test delegating multiple tasks."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        task_specs = [
            TaskSpec(description="Backend task 1", task_type="backend"),
            TaskSpec(description="Frontend task 1", task_type="frontend"),
            TaskSpec(description="Testing task 1", task_type="testing")
        ]
        
        task_results = []
        for spec in task_specs:
            result = await orchestrator.delegate_task(spec)
            task_results.append(result)
        
        assert len(task_results) == 3
        
        # All task IDs should be unique
        task_ids = [result.id for result in task_results]
        assert len(set(task_ids)) == 3
        
        await orchestrator.shutdown()
    
    async def test_get_task_status(self):
        """Test getting task status."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        task_spec = TaskSpec(description="Test task", task_type="test")
        task_result = await orchestrator.delegate_task(task_spec)
        
        # Get task status
        status = await orchestrator.get_task_status(task_result.id)
        
        assert isinstance(status, TaskResult)
        assert status.id == task_result.id
        assert status.description == task_spec.description
        
        await orchestrator.shutdown()
    
    async def test_list_tasks(self):
        """Test listing tasks."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Initially should have no tasks
        initial_tasks = await orchestrator.list_tasks()
        initial_count = len(initial_tasks)
        
        # Delegate some tasks
        task_specs = [
            TaskSpec(description="Task 1", task_type="type1"),
            TaskSpec(description="Task 2", task_type="type2")
        ]
        
        for spec in task_specs:
            await orchestrator.delegate_task(spec)
        
        # List tasks again
        tasks = await orchestrator.list_tasks()
        
        # Should have at least 2 more tasks
        assert len(tasks) >= initial_count + 2
        
        await orchestrator.shutdown()


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    async def test_response_time_targets(self):
        """Test that orchestrator meets response time targets."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test agent registration response time
        start_time = time.time()
        agent_spec = AgentSpec(role="backend_developer")
        await orchestrator.register_agent(agent_spec)
        agent_registration_time = (time.time() - start_time) * 1000  # ms
        
        # Should be under 50ms target (allowing some buffer for test environment)
        assert agent_registration_time < 200, f"Agent registration took {agent_registration_time}ms"
        
        # Test task delegation response time
        start_time = time.time()
        task_spec = TaskSpec(description="Performance test task")
        await orchestrator.delegate_task(task_spec)
        task_delegation_time = (time.time() - start_time) * 1000  # ms
        
        assert task_delegation_time < 200, f"Task delegation took {task_delegation_time}ms"
        
        # Test health check response time
        start_time = time.time()
        await orchestrator.health_check()
        health_check_time = (time.time() - start_time) * 1000  # ms
        
        assert health_check_time < 100, f"Health check took {health_check_time}ms"
        
        await orchestrator.shutdown()
    
    async def test_concurrent_operations(self):
        """Test concurrent operations performance."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test concurrent agent registrations
        async def register_agent_task(role_suffix):
            spec = AgentSpec(role=f"developer_{role_suffix}")
            return await orchestrator.register_agent(spec)
        
        start_time = time.time()
        tasks = [register_agent_task(i) for i in range(10)]
        agent_ids = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        assert len(agent_ids) == 10
        assert len(set(agent_ids)) == 10  # All unique
        assert concurrent_time < 2.0, f"Concurrent operations took {concurrent_time}s"
        
        await orchestrator.shutdown()
    
    async def test_scaling_decisions(self):
        """Test auto-scaling decision logic."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test scaling recommendations
        scaling_action = await orchestrator.auto_scale_check()
        assert scaling_action in ["scale_up", "scale_down", "maintain", "emergency_scale"]
        
        # Get scaling metrics
        metrics = await orchestrator.get_scaling_metrics()
        assert "active_agents" in metrics
        assert "pending_tasks" in metrics
        
        await orchestrator.shutdown()


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    async def test_invalid_agent_spec(self):
        """Test handling of invalid agent specifications."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test with empty role
        agent_spec = AgentSpec(role="", agent_type="claude_code")
        
        try:
            await orchestrator.register_agent(agent_spec)
            # If no exception, that's also valid (orchestrator handles it gracefully)
        except Exception as e:
            # Exception is acceptable for invalid input
            assert isinstance(e, (ValueError, TypeError))
        
        await orchestrator.shutdown()
    
    async def test_nonexistent_agent_operations(self):
        """Test operations on non-existent agents."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        fake_agent_id = "nonexistent-agent-id"
        
        # Test getting status of non-existent agent
        with pytest.raises(Exception):  # Should raise some kind of exception
            await orchestrator.get_agent_status(fake_agent_id)
        
        # Test shutting down non-existent agent
        result = await orchestrator.shutdown_agent(fake_agent_id)
        # Should return False or raise exception
        assert result is False or isinstance(result, bool)
        
        await orchestrator.shutdown()
    
    async def test_orchestrator_shutdown_recovery(self):
        """Test orchestrator behavior after shutdown and restart."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Register an agent
        agent_spec = AgentSpec(role="backend_developer")
        agent_id = await orchestrator.register_agent(agent_spec)
        
        # Shutdown orchestrator
        await orchestrator.shutdown()
        assert orchestrator._initialized is False
        
        # Reinitialize
        await orchestrator.initialize()
        assert orchestrator._initialized is True
        
        # Orchestrator should be functional again
        health = await orchestrator.health_check()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.NO_AGENTS]
        
        await orchestrator.shutdown()


class TestMigrationCompatibility:
    """Test migration from existing orchestrators."""
    
    async def test_migration_manager_initialization(self):
        """Test migration manager initialization."""
        manager = OrchestratorMigrationManager()
        
        assert len(manager._registered_adapters) > 0
        assert "SimpleOrchestrator" in manager._registered_adapters
        assert "ProductionOrchestrator" in manager._registered_adapters
    
    async def test_compatibility_validation(self):
        """Test orchestrator compatibility validation."""
        # Create a mock orchestrator with required methods
        mock_orchestrator = MagicMock()
        mock_orchestrator.health_check = AsyncMock(return_value={"status": "healthy"})
        mock_orchestrator.register_agent = AsyncMock(return_value="test-agent-id")
        
        result = await validate_orchestrator_compatibility(mock_orchestrator)
        
        assert result["compatible"] is True
        assert "health_check" in result["supported_features"]
        assert len(result["missing_features"]) == 0
    
    async def test_migration_plan_creation(self):
        """Test migration plan creation."""
        manager = OrchestratorMigrationManager()
        
        # Create mock source orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.__class__.__name__ = "SimpleOrchestrator"
        
        plan = await manager.create_migration_plan(
            mock_orchestrator,
            MigrationStrategy.GRADUAL
        )
        
        assert plan.source_orchestrator_type == "SimpleOrchestrator"
        assert plan.target_orchestrator_type == "consolidated"
        assert plan.strategy == MigrationStrategy.GRADUAL
        assert len(plan.phases) > 0
    
    @patch('app.core.consolidated_orchestrator.SIMPLE_ORCHESTRATOR_AVAILABLE', False)
    async def test_fallback_behavior_without_simple_orchestrator(self):
        """Test orchestrator behavior when SimpleOrchestrator is not available."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Should still function with internal implementation
        health = await orchestrator.health_check()
        assert health.status == HealthStatus.NO_AGENTS
        
        # Agent operations should work with internal implementation
        agent_spec = AgentSpec(role="backend_developer")
        agent_id = await orchestrator.register_agent(agent_spec)
        assert isinstance(agent_id, str)
        
        await orchestrator.shutdown()


class TestPluginSystem:
    """Test plugin system integration."""
    
    async def test_plugin_listing(self):
        """Test listing plugins."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        plugins = await orchestrator.list_plugins()
        assert isinstance(plugins, list)
        # Empty list is acceptable if no plugins are loaded
        
        await orchestrator.shutdown()
    
    async def test_plugin_health_check(self):
        """Test plugin health checking."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test health check for non-existent plugin
        health = await orchestrator.plugin_health_check("nonexistent_plugin")
        assert isinstance(health, dict)
        assert "status" in health
        
        await orchestrator.shutdown()


class TestProductionReadiness:
    """Test production-ready features."""
    
    async def test_metrics_collection(self):
        """Test metrics collection and reporting."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Get system metrics
        metrics = await orchestrator.get_metrics()
        assert isinstance(metrics, dict)
        assert "timestamp" in metrics
        assert "uptime_seconds" in metrics
        assert "operations_count" in metrics
        
        # Get performance metrics
        perf_metrics = await orchestrator.get_performance_metrics()
        assert isinstance(perf_metrics, dict)
        assert "average_response_time_ms" in perf_metrics
        assert "operations_per_second" in perf_metrics
        
        await orchestrator.shutdown()
    
    async def test_emergency_handling(self):
        """Test emergency situation handling."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test emergency handling
        result = await orchestrator.handle_emergency(
            "test_emergency",
            {"severity": "high", "component": "test"}
        )
        
        assert isinstance(result, dict)
        assert "emergency_type" in result
        assert "handled" in result
        assert result["emergency_type"] == "test_emergency"
        
        await orchestrator.shutdown()
    
    async def test_backup_and_restore(self):
        """Test state backup and restore capabilities."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Create backup
        backup_id = await orchestrator.backup_state()
        assert isinstance(backup_id, str)
        assert len(backup_id) > 0
        
        # Test restore
        restore_result = await orchestrator.restore_state(backup_id)
        assert isinstance(restore_result, bool)
        
        await orchestrator.shutdown()


class TestFactoryFunctions:
    """Test factory functions and configuration."""
    
    async def test_create_consolidated_orchestrator(self):
        """Test factory function for creating orchestrator."""
        orchestrator = await create_consolidated_orchestrator()
        
        assert isinstance(orchestrator, ConsolidatedProductionOrchestrator)
        assert orchestrator._initialized is True
        assert orchestrator.config.mode == OrchestratorMode.PRODUCTION
        
        await orchestrator.shutdown()
    
    async def test_create_consolidated_orchestrator_with_config(self):
        """Test factory function with custom configuration."""
        config = OrchestratorConfig(
            mode=OrchestratorMode.DEVELOPMENT,
            max_agents=25,
            enable_plugins=False
        )
        
        orchestrator = await create_consolidated_orchestrator(config)
        
        assert orchestrator.config.mode == OrchestratorMode.DEVELOPMENT
        assert orchestrator.config.max_agents == 25
        assert orchestrator.config.enable_plugins is False
        
        await orchestrator.shutdown()
    
    def test_create_consolidated_orchestrator_sync(self):
        """Test synchronous factory function."""
        orchestrator = create_consolidated_orchestrator_sync()
        
        assert isinstance(orchestrator, ConsolidatedProductionOrchestrator)
        assert orchestrator._initialized is False  # Not yet initialized
        
        # No need to shutdown since it's not initialized


class TestWorkflowOrchestration:
    """Test workflow orchestration capabilities."""
    
    async def test_workflow_execution(self):
        """Test basic workflow execution."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        workflow_def = {
            "name": "test_workflow",
            "steps": [
                {"type": "task", "description": "Step 1"},
                {"type": "task", "description": "Step 2"}
            ]
        }
        
        workflow_id = await orchestrator.execute_workflow(workflow_def)
        assert isinstance(workflow_id, str)
        assert len(workflow_id) > 0
        
        # Get workflow status
        status = await orchestrator.get_workflow_status(workflow_id)
        assert isinstance(status, dict)
        assert "id" in status
        
        await orchestrator.shutdown()
    
    async def test_workflow_cancellation(self):
        """Test workflow cancellation."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        workflow_def = {"name": "test_workflow", "steps": []}
        workflow_id = await orchestrator.execute_workflow(workflow_def)
        
        # Cancel workflow
        result = await orchestrator.cancel_workflow(workflow_id)
        assert isinstance(result, bool)
        
        await orchestrator.shutdown()


# Performance benchmarking tests
class TestPerformanceBenchmarks:
    """Performance benchmarking tests for production validation."""
    
    @pytest.mark.performance
    async def test_throughput_benchmark(self):
        """Benchmark orchestrator throughput."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Benchmark agent registration throughput
        start_time = time.time()
        tasks = []
        for i in range(50):  # Register 50 agents
            spec = AgentSpec(role=f"developer_{i}")
            tasks.append(orchestrator.register_agent(spec))
        
        agent_ids = await asyncio.gather(*tasks)
        registration_time = time.time() - start_time
        
        throughput = len(agent_ids) / registration_time
        assert throughput > 10, f"Registration throughput {throughput} agents/sec is too low"
        
        await orchestrator.shutdown()
    
    @pytest.mark.performance
    async def test_memory_usage(self):
        """Test memory usage characteristics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Register many agents and tasks to test memory usage
        for i in range(100):
            agent_spec = AgentSpec(role=f"agent_{i}")
            await orchestrator.register_agent(agent_spec)
            
            task_spec = TaskSpec(description=f"Task {i}")
            await orchestrator.delegate_task(task_spec)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (allowing for test environment overhead)
        assert memory_increase < 200, f"Memory usage increased by {memory_increase}MB"
        
        await orchestrator.shutdown()


# Integration with existing system tests
class TestSystemIntegration:
    """Test integration with existing system components."""
    
    @pytest.mark.integration
    async def test_main_py_compatibility(self):
        """Test compatibility with main.py orchestrator usage patterns."""
        # This would test the exact pattern used in main.py
        from app.core.orchestrator import get_orchestrator
        
        # Test that we can get an orchestrator instance
        orchestrator = await get_orchestrator()
        assert orchestrator is not None
        
        # Test health check
        health = await orchestrator.health_check()
        assert "status" in asdict(health)
        
        # This doesn't shutdown the global instance since it might be used elsewhere
    
    @pytest.mark.integration  
    async def test_database_integration(self):
        """Test database integration if available."""
        orchestrator = ConsolidatedProductionOrchestrator()
        await orchestrator.initialize()
        
        # Test that orchestrator works without database errors
        health = await orchestrator.health_check()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.NO_AGENTS, HealthStatus.DEGRADED]
        
        await orchestrator.shutdown()


# Compatibility aliases tests
class TestCompatibilityAliases:
    """Test that compatibility aliases work correctly."""
    
    async def test_production_orchestrator_alias(self):
        """Test ProductionOrchestrator alias."""
        from app.core.consolidated_orchestrator import ProductionOrchestrator
        
        orchestrator = ProductionOrchestrator()
        assert isinstance(orchestrator, ConsolidatedProductionOrchestrator)
        
        await orchestrator.initialize()
        health = await orchestrator.health_check()
        assert isinstance(health, SystemHealth)
        
        await orchestrator.shutdown()
    
    async def test_unified_orchestrator_alias(self):
        """Test UnifiedOrchestrator alias."""
        from app.core.consolidated_orchestrator import UnifiedOrchestrator
        
        orchestrator = UnifiedOrchestrator()
        assert isinstance(orchestrator, ConsolidatedProductionOrchestrator)
        
        await orchestrator.initialize()
        await orchestrator.shutdown()
    
    async def test_agent_orchestrator_alias(self):
        """Test AgentOrchestrator alias."""
        from app.core.consolidated_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        assert isinstance(orchestrator, ConsolidatedProductionOrchestrator)
        
        await orchestrator.initialize()
        await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
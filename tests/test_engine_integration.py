"""
Comprehensive Integration Tests for Engine Consolidation System

Tests the consolidated engine coordination layer integration with the
production orchestrator and validates Epic 1 Phase 1.5 requirements.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from app.core.engines.consolidated_engine import (
    EngineCoordinationLayer,
    ConsolidatedWorkflowEngine,
    ConsolidatedTaskExecutionEngine,
    ConsolidatedCommunicationEngine,
    WorkflowExecutionContext,
    TaskExecutionContext,
    CommunicationMessage
)
from app.core.production_orchestrator import (
    ProductionOrchestrator,
    create_production_orchestrator
)
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus
from app.models.session import Session, SessionStatus


class TestEngineCoordinationLayerIntegration:
    """Test suite for engine coordination layer integration."""
    
    @pytest.fixture
    async def engine_coordinator(self):
        """Create engine coordinator for testing."""
        config = {
            "max_concurrent_workflows": 10,
            "max_concurrent_tasks": 20,
            "message_queue_size": 100,
            "execution_timeout_seconds": 300
        }
        coordinator = EngineCoordinationLayer(config)
        await coordinator.initialize()
        yield coordinator
        await coordinator.shutdown()
    
    @pytest.fixture
    async def production_orchestrator(self, engine_coordinator):
        """Create production orchestrator with engine coordination."""
        with patch('app.core.production_orchestrator.get_session') as mock_session, \
             patch('app.core.production_orchestrator.get_redis') as mock_redis:
            
            mock_session.return_value = AsyncMock()
            mock_redis.return_value = AsyncMock()
            
            orchestrator = ProductionOrchestrator(
                engine_config={
                    "max_concurrent_workflows": 10,
                    "max_concurrent_tasks": 20
                }
            )
            await orchestrator.start()
            yield orchestrator
            await orchestrator.shutdown()
    
    async def test_engine_coordinator_initialization(self, engine_coordinator):
        """Test engine coordinator initializes all engines correctly."""
        assert engine_coordinator.workflow_engine is not None
        assert engine_coordinator.task_engine is not None
        assert engine_coordinator.communication_engine is not None
        
        # Verify all engines are initialized
        status = await engine_coordinator.get_status()
        assert status["workflow_engine"]["status"] == "running"
        assert status["task_execution_engine"]["status"] == "running"
        assert status["communication_engine"]["status"] == "running"
        
        # Verify metrics collection
        assert status["total_workflows_processed"] >= 0
        assert status["total_tasks_processed"] >= 0
        assert status["total_messages_processed"] >= 0
    
    async def test_workflow_execution_integration(self, engine_coordinator):
        """Test workflow execution through coordination layer."""
        workflow_config = {
            "workflow_id": "test_workflow_001",
            "workflow_type": "agent_coordination",
            "steps": [
                {"step_id": "step1", "type": "task_assignment", "agent_role": "backend_developer"},
                {"step_id": "step2", "type": "validation", "validation_type": "quality_check"}
            ],
            "metadata": {"priority": "high", "timeout": 300}
        }
        
        # Execute workflow
        result = await engine_coordinator.execute_workflow(
            "test_workflow_001",
            workflow_config
        )
        
        assert result.success is True
        assert result.workflow_id == "test_workflow_001"
        assert result.execution_time_ms > 0
        assert result.steps_completed >= 0
        
        # Verify workflow was processed
        status = await engine_coordinator.get_status()
        assert status["total_workflows_processed"] >= 1
    
    async def test_task_execution_integration(self, engine_coordinator):
        """Test task execution through coordination layer."""
        task_config = {
            "task_id": "test_task_001",
            "task_type": "code_generation",
            "priority": "high",
            "requirements": {
                "language": "python",
                "framework": "fastapi",
                "complexity": "medium"
            },
            "constraints": {
                "max_execution_time": 180,
                "memory_limit_mb": 512
            }
        }
        
        # Execute task
        result = await engine_coordinator.execute_task(
            "test_task_001",
            task_config
        )
        
        assert result.success is True
        assert result.task_id == "test_task_001"
        assert result.execution_time_ms > 0
        
        # Verify task was processed
        status = await engine_coordinator.get_status()
        assert status["total_tasks_processed"] >= 1
    
    async def test_communication_integration(self, engine_coordinator):
        """Test communication engine integration."""
        message = CommunicationMessage(
            message_id="test_msg_001",
            source="test_agent",
            destination="orchestrator", 
            message_type="status_update",
            content={"status": "task_completed", "task_id": "test_task_001"},
            priority="normal",
            timestamp=datetime.utcnow()
        )
        
        # Send message
        result = await engine_coordinator.send_message(message)
        
        assert result.success is True
        assert result.message_id == "test_msg_001"
        assert result.delivery_time_ms > 0
        
        # Verify message was processed
        status = await engine_coordinator.get_status()
        assert status["total_messages_processed"] >= 1
    
    async def test_concurrent_processing(self, engine_coordinator):
        """Test concurrent processing across all engines."""
        # Create multiple workflows, tasks, and messages
        workflows = []
        tasks = []
        messages = []
        
        # Create 5 concurrent workflows
        for i in range(5):
            workflow_config = {
                "workflow_id": f"concurrent_workflow_{i}",
                "workflow_type": "parallel_execution",
                "steps": [{"step_id": f"step_{i}", "type": "processing"}]
            }
            workflows.append(engine_coordinator.execute_workflow(
                f"concurrent_workflow_{i}", 
                workflow_config
            ))
        
        # Create 10 concurrent tasks
        for i in range(10):
            task_config = {
                "task_id": f"concurrent_task_{i}",
                "task_type": "processing",
                "priority": "normal"
            }
            tasks.append(engine_coordinator.execute_task(
                f"concurrent_task_{i}", 
                task_config
            ))
        
        # Create 15 concurrent messages
        for i in range(15):
            message = CommunicationMessage(
                message_id=f"concurrent_msg_{i}",
                source=f"agent_{i}",
                destination="orchestrator",
                message_type="update",
                content={"data": f"message_{i}"},
                timestamp=datetime.utcnow()
            )
            messages.append(engine_coordinator.send_message(message))
        
        # Execute all concurrently
        workflow_results = await asyncio.gather(*workflows, return_exceptions=True)
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        message_results = await asyncio.gather(*messages, return_exceptions=True)
        
        # Verify all completed successfully
        successful_workflows = [r for r in workflow_results if not isinstance(r, Exception) and r.success]
        successful_tasks = [r for r in task_results if not isinstance(r, Exception) and r.success]
        successful_messages = [r for r in message_results if not isinstance(r, Exception) and r.success]
        
        assert len(successful_workflows) >= 4  # Allow for some failures due to load
        assert len(successful_tasks) >= 8
        assert len(successful_messages) >= 12
    
    async def test_production_orchestrator_integration(self, production_orchestrator):
        """Test production orchestrator integration with engines."""
        # Get production status including engine status
        status = await production_orchestrator.get_production_status()
        
        assert "engine_status" in status
        engine_status = status["engine_status"]
        
        # Verify all engines are reported as running
        assert engine_status["workflow_engine"]["status"] == "running"
        assert engine_status["task_execution_engine"]["status"] == "running"
        assert engine_status["communication_engine"]["status"] == "running"
        
        # Verify component health includes engine status
        component_health = status["component_health"]
        assert "workflow_engine" in component_health
        assert "task_execution_engine" in component_health
        assert "communication_engine" in component_health
    
    async def test_engine_health_monitoring(self, engine_coordinator):
        """Test engine health check functionality."""
        health_status = await engine_coordinator.health_check()
        
        # Verify health check structure
        assert "workflow_engine" in health_status
        assert "task_execution_engine" in health_status
        assert "communication_engine" in health_status
        assert "overall_health" in health_status
        
        # Verify individual engine health
        for engine_name in ["workflow_engine", "task_execution_engine", "communication_engine"]:
            engine_health = health_status[engine_name]
            assert "status" in engine_health
            assert "uptime_seconds" in engine_health
            assert "processed_count" in engine_health
            assert "error_count" in engine_health
            assert "average_processing_time_ms" in engine_health
        
        # Verify overall health calculation
        assert health_status["overall_health"] in ["healthy", "degraded", "unhealthy"]
    
    async def test_engine_performance_metrics(self, engine_coordinator):
        """Test engine performance metrics collection."""
        # Execute some operations to generate metrics
        await self.test_workflow_execution_integration(engine_coordinator)
        await self.test_task_execution_integration(engine_coordinator)
        await self.test_communication_integration(engine_coordinator)
        
        # Get performance metrics
        metrics = await engine_coordinator.get_performance_metrics()
        
        # Verify workflow metrics
        workflow_metrics = metrics["workflow_engine"]
        assert workflow_metrics["total_processed"] >= 1
        assert workflow_metrics["average_execution_time_ms"] > 0
        assert workflow_metrics["success_rate_percent"] > 0
        
        # Verify task execution metrics
        task_metrics = metrics["task_execution_engine"]
        assert task_metrics["total_processed"] >= 1
        assert task_metrics["average_execution_time_ms"] > 0
        assert task_metrics["success_rate_percent"] > 0
        
        # Verify communication metrics
        comm_metrics = metrics["communication_engine"]
        assert comm_metrics["total_processed"] >= 1
        assert comm_metrics["average_delivery_time_ms"] > 0
        assert comm_metrics["success_rate_percent"] > 0
    
    async def test_engine_error_handling(self, engine_coordinator):
        """Test engine error handling and recovery."""
        # Test workflow error handling
        invalid_workflow_config = {
            "workflow_id": "invalid_workflow",
            # Missing required fields
        }
        
        result = await engine_coordinator.execute_workflow(
            "invalid_workflow", 
            invalid_workflow_config
        )
        assert result.success is False
        assert result.error_message is not None
        
        # Test task error handling
        invalid_task_config = {
            "task_id": "invalid_task",
            # Missing required fields
        }
        
        result = await engine_coordinator.execute_task(
            "invalid_task", 
            invalid_task_config
        )
        assert result.success is False
        assert result.error_message is not None
        
        # Verify engines remain healthy after errors
        health_status = await engine_coordinator.health_check()
        assert health_status["overall_health"] in ["healthy", "degraded"]  # Should not be unhealthy
    
    async def test_engine_resource_management(self, engine_coordinator):
        """Test engine resource management and limits."""
        # Test workflow concurrency limits
        workflow_tasks = []
        for i in range(15):  # Exceed max_concurrent_workflows (10)
            workflow_config = {
                "workflow_id": f"resource_test_workflow_{i}",
                "workflow_type": "resource_test",
                "steps": [{"step_id": "step1", "type": "delay", "delay_ms": 1000}]
            }
            workflow_tasks.append(engine_coordinator.execute_workflow(
                f"resource_test_workflow_{i}", 
                workflow_config
            ))
        
        # Execute concurrently
        results = await asyncio.gather(*workflow_tasks, return_exceptions=True)
        
        # Some should succeed, some may be queued or rate limited
        successful = [r for r in results if not isinstance(r, Exception) and r.success]
        assert len(successful) >= 5  # At least some should succeed
        
        # Verify resource usage is tracked
        status = await engine_coordinator.get_status()
        assert "active_workflows" in status
        assert "active_tasks" in status
    
    async def test_engine_shutdown_graceful(self, engine_coordinator):
        """Test graceful engine shutdown."""
        # Start some long-running operations
        workflow_config = {
            "workflow_id": "shutdown_test_workflow",
            "workflow_type": "long_running",
            "steps": [{"step_id": "step1", "type": "delay", "delay_ms": 2000}]
        }
        
        workflow_task = asyncio.create_task(
            engine_coordinator.execute_workflow("shutdown_test_workflow", workflow_config)
        )
        
        # Wait a moment for workflow to start
        await asyncio.sleep(0.1)
        
        # Initiate shutdown
        shutdown_task = asyncio.create_task(engine_coordinator.shutdown())
        
        # Wait for both to complete
        await asyncio.gather(workflow_task, shutdown_task, return_exceptions=True)
        
        # Verify engines are shut down
        status = await engine_coordinator.get_status()
        assert status["workflow_engine"]["status"] == "stopped"
        assert status["task_execution_engine"]["status"] == "stopped"
        assert status["communication_engine"]["status"] == "stopped"


class TestEngineConsolidationRequirements:
    """Test consolidation requirements for Epic 1 Phase 1.5."""
    
    @pytest.fixture
    async def engine_coordinator(self):
        """Create engine coordinator for requirements testing."""
        coordinator = EngineCoordinationLayer({})
        await coordinator.initialize()
        yield coordinator
        await coordinator.shutdown()
    
    async def test_performance_requirements(self, engine_coordinator):
        """Test performance requirements are met."""
        # Test workflow compilation time <2s requirement
        start_time = datetime.utcnow()
        workflow_config = {
            "workflow_id": "perf_test_workflow",
            "workflow_type": "complex_workflow",
            "steps": [
                {"step_id": f"step_{i}", "type": "processing"} 
                for i in range(20)  # Complex workflow with many steps
            ]
        }
        
        result = await engine_coordinator.execute_workflow(
            "perf_test_workflow", 
            workflow_config
        )
        
        compilation_time = (datetime.utcnow() - start_time).total_seconds()
        assert compilation_time < 2.0, f"Workflow compilation took {compilation_time}s, exceeds 2s requirement"
        
        # Test task assignment time <100ms requirement
        start_time = datetime.utcnow()
        task_config = {
            "task_id": "perf_test_task",
            "task_type": "simple_assignment",
            "priority": "high"
        }
        
        result = await engine_coordinator.execute_task("perf_test_task", task_config)
        
        assignment_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        assert assignment_time < 100.0, f"Task assignment took {assignment_time}ms, exceeds 100ms requirement"
    
    async def test_consolidation_coverage(self, engine_coordinator):
        """Test that consolidation covers all required engine types."""
        # Verify workflow engine types are supported
        workflow_types = [
            "agent_coordination", "task_pipeline", "data_processing",
            "monitoring_workflow", "deployment_workflow", "testing_workflow",
            "integration_workflow", "notification_workflow"
        ]
        
        for workflow_type in workflow_types:
            workflow_config = {
                "workflow_id": f"coverage_test_{workflow_type}",
                "workflow_type": workflow_type,
                "steps": [{"step_id": "step1", "type": "basic_processing"}]
            }
            
            result = await engine_coordinator.execute_workflow(
                f"coverage_test_{workflow_type}",
                workflow_config
            )
            
            assert result.success, f"Workflow type {workflow_type} not supported"
        
        # Verify task engine types are supported
        task_types = [
            "code_generation", "code_review", "testing", "documentation",
            "deployment", "monitoring", "validation", "optimization",
            "security_scan", "performance_analysis", "integration_test",
            "unit_test"
        ]
        
        for task_type in task_types:
            task_config = {
                "task_id": f"coverage_test_{task_type}",
                "task_type": task_type,
                "priority": "normal"
            }
            
            result = await engine_coordinator.execute_task(
                f"coverage_test_{task_type}",
                task_config
            )
            
            assert result.success, f"Task type {task_type} not supported"
    
    async def test_backward_compatibility(self, engine_coordinator):
        """Test backward compatibility with existing engine interfaces."""
        # Test legacy workflow format support
        legacy_workflow = {
            "id": "legacy_workflow",  # Old field name
            "type": "agent_orchestration",  # Old field name
            "definition": {  # Old nested structure
                "steps": [{"name": "step1", "action": "process"}]
            }
        }
        
        # Should handle legacy format through migration utilities
        result = await engine_coordinator.workflow_engine.migrate_and_execute(legacy_workflow)
        assert result.success, "Legacy workflow format not supported"
        
        # Test legacy task format support
        legacy_task = {
            "id": "legacy_task",  # Old field name
            "type": "code_gen",  # Old abbreviated type
            "params": {"lang": "python"}  # Old parameter structure
        }
        
        # Should handle legacy format through migration utilities
        result = await engine_coordinator.task_engine.migrate_and_execute(legacy_task)
        assert result.success, "Legacy task format not supported"
    
    async def test_scalability_requirements(self, engine_coordinator):
        """Test scalability requirements for 50+ concurrent operations."""
        # Create 60 concurrent operations (exceeds 50 requirement)
        operations = []
        
        # 20 workflows
        for i in range(20):
            workflow_config = {
                "workflow_id": f"scale_workflow_{i}",
                "workflow_type": "scalability_test",
                "steps": [{"step_id": "step1", "type": "lightweight_processing"}]
            }
            operations.append(engine_coordinator.execute_workflow(
                f"scale_workflow_{i}", 
                workflow_config
            ))
        
        # 25 tasks
        for i in range(25):
            task_config = {
                "task_id": f"scale_task_{i}",
                "task_type": "scalability_test",
                "priority": "normal"
            }
            operations.append(engine_coordinator.execute_task(
                f"scale_task_{i}", 
                task_config
            ))
        
        # 15 messages
        for i in range(15):
            message = CommunicationMessage(
                message_id=f"scale_msg_{i}",
                source=f"agent_{i}",
                destination="orchestrator",
                message_type="scalability_test",
                content={"test": True},
                timestamp=datetime.utcnow()
            )
            operations.append(engine_coordinator.send_message(message))
        
        # Execute all concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(*operations, return_exceptions=True)
        total_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Verify at least 80% completed successfully
        successful = [r for r in results if not isinstance(r, Exception) and r.success]
        success_rate = len(successful) / len(operations)
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% requirement"
        
        # Verify reasonable total processing time
        assert total_time < 30.0, f"Total processing time {total_time}s exceeds reasonable limit"
    
    async def test_resource_efficiency(self, engine_coordinator):
        """Test resource efficiency requirements."""
        # Get baseline resource usage
        baseline_status = await engine_coordinator.get_status()
        baseline_memory = baseline_status.get("memory_usage_mb", 0)
        
        # Execute several operations
        operations = []
        for i in range(10):
            workflow_config = {
                "workflow_id": f"resource_workflow_{i}",
                "workflow_type": "resource_test",
                "steps": [{"step_id": "step1", "type": "memory_test"}]
            }
            operations.append(engine_coordinator.execute_workflow(
                f"resource_workflow_{i}", 
                workflow_config
            ))
        
        await asyncio.gather(*operations)
        
        # Check resource usage after operations
        final_status = await engine_coordinator.get_status()
        final_memory = final_status.get("memory_usage_mb", 0)
        
        # Memory usage should not increase by more than 50MB for 10 operations
        memory_increase = final_memory - baseline_memory
        assert memory_increase < 50, f"Memory usage increased by {memory_increase}MB, exceeds 50MB limit"


@pytest.mark.asyncio
class TestEngineIntegrationEndToEnd:
    """End-to-end integration tests for the complete engine system."""
    
    async def test_complete_agent_workflow(self):
        """Test complete agent workflow from orchestration to task completion."""
        # This would test the full flow:
        # 1. ProductionOrchestrator receives request
        # 2. Workflow engine compiles execution plan
        # 3. Task engine assigns tasks to agents
        # 4. Communication engine handles messaging
        # 5. Results are collected and reported
        
        config = {
            "max_concurrent_workflows": 5,
            "max_concurrent_tasks": 10,
            "execution_timeout_seconds": 180
        }
        
        async with EngineCoordinationLayer(config) as coordinator:
            await coordinator.initialize()
            
            # Simulate complete workflow
            workflow_config = {
                "workflow_id": "end_to_end_test",
                "workflow_type": "agent_coordination",
                "steps": [
                    {"step_id": "task_assignment", "type": "task_assignment", "agent_role": "backend_developer"},
                    {"step_id": "code_generation", "type": "task_execution", "task_type": "code_generation"},
                    {"step_id": "code_review", "type": "task_execution", "task_type": "code_review"},
                    {"step_id": "validation", "type": "validation", "validation_type": "integration_test"}
                ],
                "metadata": {"priority": "high", "timeout": 180}
            }
            
            result = await coordinator.execute_workflow("end_to_end_test", workflow_config)
            
            assert result.success, f"End-to-end workflow failed: {result.error_message}"
            assert result.steps_completed >= 3, "Not all workflow steps completed"
            assert result.execution_time_ms < 180000, "Workflow exceeded timeout"
            
            # Verify all engines processed the workflow
            status = await coordinator.get_status()
            assert status["total_workflows_processed"] >= 1
            assert status["total_tasks_processed"] >= 2  # At least task assignment and execution
            assert status["total_messages_processed"] >= 1  # Communication between steps
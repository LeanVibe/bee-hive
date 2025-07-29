"""
Integration Tests for Phase 1 Claude Code Features

Tests the integration of hooks system, slash commands, and extended thinking
with the LeanVibe Agent Hive 2.0 orchestration system.
"""

import asyncio
import json
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from app.core.leanvibe_hooks_system import (
    LeanVibeHooksEngine,
    HookEventType,
    HookDefinition,
    ThinkingDepth,
    initialize_leanvibe_hooks_engine
)
from app.core.slash_commands import SlashCommandsEngine, initialize_slash_commands_engine
from app.core.extended_thinking_engine import (
    ExtendedThinkingEngine,
    ThinkingDepth,
    initialize_extended_thinking_engine
)
from app.core.enhanced_orchestrator_integration import (
    EnhancedOrchestratorIntegration,
    initialize_enhanced_orchestrator_integration
)


class TestPhase1Integration:
    """Test suite for Phase 1 Claude Code integration features."""
    
    @pytest.fixture
    async def temp_project_dir(self):
        """Create temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create hooks directory structure
            hooks_dir = project_path / ".leanvibe" / "hooks"
            hooks_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test hook configuration
            hooks_config = {
                "leanvibe_hooks": {
                    "PreAgentTask": [
                        {
                            "name": "test_validation",
                            "command": "echo 'Test validation passed'",
                            "matcher": "*",
                            "description": "Test validation hook",
                            "execution_mode": "blocking",
                            "timeout_seconds": 10,
                            "required": True
                        }
                    ],
                    "PostAgentTask": [
                        {
                            "name": "test_cleanup",
                            "command": "echo 'Test cleanup completed'",
                            "matcher": "*",
                            "description": "Test cleanup hook",
                            "execution_mode": "async",
                            "timeout_seconds": 5
                        }
                    ]
                }
            }
            
            import yaml
            with open(hooks_dir / "test_hooks.yaml", 'w') as f:
                yaml.dump(hooks_config, f)
            
            yield project_path
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock orchestrator for testing."""
        orchestrator = Mock()
        orchestrator.execute_agent_task = AsyncMock(return_value={
            "success": True,
            "output": "Task completed successfully",
            "agent_id": "test_agent"
        })
        orchestrator.get_all_agents = AsyncMock(return_value=[
            {
                "id": "test_agent",
                "name": "Test Agent",
                "role": "backend_specialist",
                "capabilities": ["coding", "testing"],
                "status": "active"
            },
            {
                "id": "security_agent",
                "name": "Security Agent", 
                "role": "security_specialist",
                "capabilities": ["security", "analysis"],
                "status": "active"
            }
        ])
        orchestrator.get_health_status = AsyncMock(return_value={
            "status": "healthy",
            "agents_active": 2
        })
        return orchestrator
    
    @pytest.fixture
    def mock_communication_bus(self):
        """Create mock communication bus for testing."""
        bus = Mock()
        bus.publish_event = AsyncMock()
        return bus
    
    @pytest.fixture
    def mock_hook_processor(self):
        """Create mock hook processor for testing."""
        processor = Mock()
        processor.process_post_tool_use = AsyncMock()
        return processor
    
    @pytest.fixture
    async def hooks_engine(self, temp_project_dir, mock_orchestrator, mock_communication_bus, mock_hook_processor):
        """Create hooks engine for testing."""
        engine = LeanVibeHooksEngine(
            project_root=temp_project_dir,
            orchestrator=mock_orchestrator,
            communication_bus=mock_communication_bus,
            hook_processor=mock_hook_processor
        )
        await engine.load_hooks_configuration()
        return engine
    
    @pytest.fixture
    async def slash_commands_engine(self, temp_project_dir, mock_orchestrator, mock_communication_bus):
        """Create slash commands engine for testing."""
        engine = SlashCommandsEngine(
            project_root=temp_project_dir,
            orchestrator=mock_orchestrator,
            communication_bus=mock_communication_bus
        )
        await engine.load_commands()
        return engine
    
    @pytest.fixture
    async def thinking_engine(self, mock_orchestrator, mock_communication_bus):
        """Create extended thinking engine for testing."""
        engine = ExtendedThinkingEngine(
            orchestrator=mock_orchestrator,
            communication_bus=mock_communication_bus
        )
        return engine
    
    @pytest.fixture
    async def enhanced_integration(self, mock_orchestrator, hooks_engine, slash_commands_engine, thinking_engine):
        """Create enhanced integration for testing."""
        integration = EnhancedOrchestratorIntegration(
            orchestrator=mock_orchestrator,
            hooks_engine=hooks_engine,
            slash_commands=slash_commands_engine,
            thinking_engine=thinking_engine
        )
        return integration

    @pytest.mark.asyncio
    async def test_hooks_engine_initialization(self, temp_project_dir):
        """Test hooks engine initialization and configuration loading."""
        # Test initialization
        engine = LeanVibeHooksEngine(project_root=temp_project_dir)
        assert engine.project_root == temp_project_dir
        assert engine.hooks_config_dir == temp_project_dir / ".leanvibe" / "hooks"
        
        # Test configuration loading
        await engine.load_hooks_configuration()
        
        # Verify hooks were loaded
        assert HookEventType.PRE_AGENT_TASK in engine.hook_definitions
        assert HookEventType.POST_AGENT_TASK in engine.hook_definitions
        
        pre_task_hooks = engine.hook_definitions[HookEventType.PRE_AGENT_TASK]
        assert len(pre_task_hooks) > 0
        
        # Check for test hook
        test_hook = next((h for h in pre_task_hooks if h.name == "test_validation"), None)
        assert test_hook is not None
        assert test_hook.command == "echo 'Test validation passed'"
        assert test_hook.required is True
    
    @pytest.mark.asyncio
    async def test_hooks_execution(self, hooks_engine):
        """Test hook execution for workflow events."""
        workflow_id = str(uuid.uuid4())
        agent_id = "test_agent"
        session_id = str(uuid.uuid4())
        
        # Test PreAgentTask hook execution
        results = await hooks_engine.execute_workflow_hooks(
            event=HookEventType.PRE_AGENT_TASK,
            workflow_id=workflow_id,
            workflow_data={
                "agent_name": "test_agent",
                "task_type": "coding",
                "description": "Test task"
            },
            agent_id=agent_id,
            session_id=session_id
        )
        
        # Verify hooks were executed
        assert len(results) > 0
        
        # Check for successful execution
        successful_results = [r for r in results if r.success]
        assert len(successful_results) > 0
        
        # Verify performance stats updated
        stats = await hooks_engine.get_performance_stats()
        assert stats["execution_stats"]["hooks_executed"] > 0
    
    @pytest.mark.asyncio
    async def test_slash_commands_execution(self, slash_commands_engine):
        """Test slash commands execution."""
        # Test basic command execution
        result = await slash_commands_engine.execute_command(
            command_str="/status",
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert result.success is True
        assert "System Status" in result.output
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio 
    async def test_agents_command(self, slash_commands_engine, mock_orchestrator):
        """Test /agents slash command."""
        result = await slash_commands_engine.execute_command(
            command_str="/agents list",
            agent_id="test_agent"
        )
        
        assert result.success is True
        assert "Available Agents" in result.output
        assert "Test Agent" in result.output
        assert result.metadata.get("agents_count") == 2
    
    @pytest.mark.asyncio
    async def test_workflow_command(self, slash_commands_engine, mock_orchestrator):
        """Test /workflow slash command."""
        # Mock workflow data
        mock_orchestrator.get_active_workflows = AsyncMock(return_value=[
            {
                "name": "Test Workflow",
                "status": "running",
                "progress": 75,
                "agents": ["agent1", "agent2"]
            }
        ])
        
        result = await slash_commands_engine.execute_command(
            command_str="/workflow list",
            agent_id="test_agent"
        )
        
        assert result.success is True
        assert "Active Workflows" in result.output
        assert "Test Workflow" in result.output
    
    @pytest.mark.asyncio
    async def test_extended_thinking_analysis(self, thinking_engine):
        """Test extended thinking needs analysis."""
        # Test architectural decision task
        thinking_config = await thinking_engine.analyze_thinking_needs(
            task_description="Design a scalable microservices architecture for the payment system",
            task_context={
                "complexity": "high",
                "domain": "architecture",
                "requirements": ["scalability", "security", "performance"]
            }
        )
        
        assert thinking_config is not None
        assert thinking_config["trigger"].value == "architectural_decisions"
        assert thinking_config["thinking_depth"] in [ThinkingDepth.DEEP, ThinkingDepth.COLLABORATIVE]
        assert thinking_config["collaboration_recommended"] is True
        
        # Test simple task (should not require thinking)
        simple_config = await thinking_engine.analyze_thinking_needs(
            task_description="Fix a typo in the documentation",
            task_context={"complexity": "low"}
        )
        
        assert simple_config is None
    
    @pytest.mark.asyncio
    async def test_thinking_session_creation(self, thinking_engine):
        """Test creation and management of thinking sessions."""
        session = await thinking_engine.enable_extended_thinking(
            agent_id="test_agent",
            workflow_id="test_workflow",
            problem_description="Optimize database query performance",
            problem_context={"database": "postgresql", "queries": "slow"},
            thinking_depth=ThinkingDepth.DEEP
        )
        
        assert session.session_id is not None
        assert session.agent_id == "test_agent"
        assert session.thinking_depth == ThinkingDepth.DEEP
        assert session.status == "active"
        
        # Check session status
        status = await thinking_engine.get_session_status(session.session_id)
        assert status is not None
        assert status["status"] == "active"
        assert status["thinking_depth"] == "deep"
    
    @pytest.mark.asyncio
    async def test_enhanced_integration_task_execution(self, enhanced_integration, mock_orchestrator):
        """Test enhanced task execution with all features."""
        task_data = {
            "type": "architectural_design",
            "description": "Design microservices architecture for payment processing",
            "complexity": "high",
            "workflow_id": "test_workflow"
        }
        
        result = await enhanced_integration.execute_enhanced_agent_task(
            agent_id="test_agent",
            task_data=task_data,
            session_id="test_session"
        )
        
        # Verify core execution
        assert result["success"] is True
        assert result["agent_id"] == "test_agent"
        
        # Verify orchestrator was called
        mock_orchestrator.execute_agent_task.assert_called_once()
        
        # Check if thinking insights were added
        if "thinking_insights" in result:
            assert "session_id" in result["thinking_insights"]
            assert "collaborative_solution" in result["thinking_insights"]
    
    @pytest.mark.asyncio
    async def test_enhanced_slash_command_execution(self, enhanced_integration):
        """Test enhanced slash command execution."""
        result = await enhanced_integration.execute_enhanced_slash_command(
            command_str="/agents list",
            agent_id="test_agent",
            session_id="test_session"
        )
        
        assert result["success"] is True
        assert "output" in result
        assert result["execution_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_system_status(self, enhanced_integration):
        """Test enhanced system status reporting."""
        status = await enhanced_integration.get_enhanced_system_status()
        
        assert "enhanced_features" in status
        assert status["enhanced_features"]["enabled"] is True
        
        # Check individual feature status
        features = status["enhanced_features"]
        assert features["hooks_engine"]["available"] is True
        assert features["slash_commands"]["available"] is True
        assert features["extended_thinking"]["available"] is True
    
    @pytest.mark.asyncio
    async def test_quality_gate_execution(self, enhanced_integration):
        """Test quality gate execution with hooks."""
        quality_criteria = {
            "code_quality_threshold": 8.5,
            "test_coverage_threshold": 80,
            "security_scan_required": True
        }
        
        result = await enhanced_integration.execute_quality_gate(
            workflow_id="test_workflow",
            quality_criteria=quality_criteria,
            session_id="test_session"
        )
        
        assert "success" in result
        assert "quality_score" in result
        assert "hooks_executed" in result
        assert result["quality_score"] >= 0.0
        assert result["quality_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_feature_enable_disable(self, enhanced_integration):
        """Test enabling and disabling enhanced features."""
        # Test disabling features
        await enhanced_integration.disable_enhanced_features()
        assert enhanced_integration.enhanced_features_enabled is False
        
        # Test enabling features
        await enhanced_integration.enable_enhanced_features()
        assert enhanced_integration.enhanced_features_enabled is True
    
    @pytest.mark.asyncio
    async def test_hooks_performance_stats(self, hooks_engine):
        """Test hooks performance statistics collection."""
        # Execute some hooks to generate stats
        await hooks_engine.execute_workflow_hooks(
            event=HookEventType.PRE_AGENT_TASK,
            workflow_id="test_workflow",
            workflow_data={"test": "data"},
            agent_id="test_agent",
            session_id="test_session"
        )
        
        stats = await hooks_engine.get_performance_stats()
        
        assert "performance_stats" in stats
        assert "execution_stats" in stats
        assert stats["execution_stats"]["hooks_executed"] > 0
        assert "cache_stats" in stats
        assert "hooks_config" in stats
    
    @pytest.mark.asyncio
    async def test_thinking_performance_stats(self, thinking_engine):
        """Test thinking engine performance statistics."""
        stats = await thinking_engine.get_performance_stats()
        
        assert "performance_stats" in stats
        assert "capabilities" in stats
        assert stats["capabilities"]["collaborative_thinking"] is True
        assert stats["capabilities"]["multi_agent_coordination"] is True
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration(self, enhanced_integration, temp_project_dir):
        """Test complete end-to-end integration scenario."""
        # Simulate complex task requiring all features
        task_data = {
            "type": "complex_integration",
            "description": "Implement secure API authentication with performance optimization",
            "context": {
                "security_requirements": True,
                "performance_critical": True,
                "complexity": "high"
            },
            "workflow_id": "e2e_test_workflow"
        }
        
        # Execute enhanced task
        result = await enhanced_integration.execute_enhanced_agent_task(
            agent_id="test_agent",
            task_data=task_data,
            session_id="e2e_test_session"
        )
        
        # Verify successful execution
        assert result["success"] is True
        
        # Execute slash command
        cmd_result = await enhanced_integration.execute_enhanced_slash_command(
            command_str="/status",
            agent_id="test_agent",
            session_id="e2e_test_session"
        )
        
        assert cmd_result["success"] is True
        
        # Check system status
        status = await enhanced_integration.get_enhanced_system_status()
        assert status["enhanced_features"]["enabled"] is True
        
        # Execute quality gate
        quality_result = await enhanced_integration.execute_quality_gate(
            workflow_id="e2e_test_workflow",
            quality_criteria={"comprehensive_validation": True},
            session_id="e2e_test_session"
        )
        
        assert "quality_score" in quality_result


class TestPhase1Performance:
    """Performance tests for Phase 1 features."""
    
    @pytest.mark.asyncio
    async def test_hooks_performance_overhead(self, hooks_engine):
        """Test that hooks add minimal performance overhead."""
        import time
        
        # Measure baseline execution time
        start_time = time.time()
        for _ in range(10):
            await asyncio.sleep(0.001)  # Simulate work
        baseline_time = time.time() - start_time
        
        # Measure with hooks
        start_time = time.time()
        for i in range(10):
            await hooks_engine.execute_workflow_hooks(
                event=HookEventType.POST_AGENT_TASK,
                workflow_id=f"perf_test_{i}",
                workflow_data={"iteration": i},
                agent_id="perf_test_agent",
                session_id="perf_test_session"
            )
        hooks_time = time.time() - start_time
        
        # Verify overhead is reasonable (should be < 50% overhead)
        overhead_ratio = hooks_time / baseline_time
        assert overhead_ratio < 1.5, f"Hooks overhead too high: {overhead_ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_thinking_session_timeout(self, thinking_engine):
        """Test thinking session timeout handling."""
        # Create session with short timeout
        session = await thinking_engine.enable_extended_thinking(
            agent_id="test_agent",
            workflow_id="timeout_test",
            problem_description="Test timeout handling",
            problem_context={},
            thinking_depth=ThinkingDepth.STANDARD
        )
        
        # Override timeout for testing
        session.thinking_time_limit_seconds = 1
        thinking_engine.active_sessions[session.session_id] = session
        
        # Wait for timeout
        await asyncio.sleep(2)
        
        # Check session status
        status = await thinking_engine.get_session_status(session.session_id)
        assert status is not None  # Session should still be trackable


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
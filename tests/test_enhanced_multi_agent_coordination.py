"""
Comprehensive Tests for Enhanced Multi-Agent Coordination System

This test suite validates the sophisticated multi-agent coordination capabilities
including specialized agents, coordination patterns, real-time collaboration,
and cross-agent learning features.

Test Coverage:
- Enhanced Multi-Agent Coordinator initialization and operation
- Specialized agent roles and capabilities
- Coordination pattern execution and validation
- Team formation and optimization
- Real-time collaboration monitoring
- Cross-agent learning and knowledge sharing
- API endpoint functionality
- Command system integration
"""

import asyncio
import json
import pytest
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

import structlog

# Import the enhanced coordination system
from app.core.enhanced_multi_agent_coordination import (
    EnhancedMultiAgentCoordinator,
    SpecializedAgentRole,
    CoordinationPatternType,
    TaskComplexity,
    CollaborationContext,
    CoordinationPattern,
    SpecializedAgent,
    AgentCapability
)
from app.core.enhanced_agent_implementations import (
    create_specialized_agent,
    BaseEnhancedAgent,
    ArchitectAgent,
    DeveloperAgent,
    TesterAgent,
    ReviewerAgent,
    DevOpsAgent,
    ProductAgent,
    TaskExecution
)
from app.core.enhanced_coordination_commands import (
    EnhancedCoordinationCommands,
    get_coordination_commands
)

logger = structlog.get_logger()


class TestEnhancedMultiAgentCoordinator:
    """Test suite for Enhanced Multi-Agent Coordinator."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = EnhancedMultiAgentCoordinator(temp_dir)
            
            # Mock the message broker and communication service to avoid Redis dependency
            coordinator.message_broker = Mock()
            coordinator.communication_service = Mock()
            coordinator.workflow_engine = Mock()
            coordinator.task_router = Mock()
            coordinator.capability_matcher = Mock()
            
            # Initialize with mocked dependencies
            await coordinator._initialize_specialized_agents()
            
            yield coordinator
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes correctly with all components."""
        assert coordinator is not None
        assert len(coordinator.agents) > 0
        assert len(coordinator.coordination_patterns) > 0
        assert len(coordinator.agent_roles) > 0
        
        # Verify all required agent roles are present
        expected_roles = [
            SpecializedAgentRole.ARCHITECT,
            SpecializedAgentRole.DEVELOPER,
            SpecializedAgentRole.TESTER,
            SpecializedAgentRole.REVIEWER,
            SpecializedAgentRole.DEVOPS,
            SpecializedAgentRole.PRODUCT
        ]
        
        for role in expected_roles:
            assert role in coordinator.agent_roles
            assert len(coordinator.agent_roles[role]) > 0
    
    @pytest.mark.asyncio
    async def test_coordination_patterns_initialization(self, coordinator):
        """Test coordination patterns are properly initialized."""
        expected_patterns = [
            CoordinationPatternType.PAIR_PROGRAMMING,
            CoordinationPatternType.CODE_REVIEW_CYCLE,
            CoordinationPatternType.CONTINUOUS_INTEGRATION,
            CoordinationPatternType.DESIGN_REVIEW,
            CoordinationPatternType.KNOWLEDGE_SHARING
        ]
        
        pattern_types = [pattern.pattern_type for pattern in coordinator.coordination_patterns.values()]
        
        for expected_type in expected_patterns:
            assert expected_type in pattern_types
        
        # Verify pattern structure
        for pattern in coordinator.coordination_patterns.values():
            assert pattern.pattern_id is not None
            assert pattern.name is not None
            assert len(pattern.required_roles) > 0
            assert len(pattern.coordination_steps) > 0
            assert pattern.estimated_duration > 0
    
    @pytest.mark.asyncio
    async def test_create_collaboration(self, coordinator):
        """Test collaboration creation with intelligent agent selection."""
        task_description = "Implement high-performance API endpoint"
        requirements = {
            "language": "python",
            "performance_target": "1000_rps",
            "required_capabilities": ["code_implementation", "performance_optimization"]
        }
        
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="pair_programming_01",
            task_description=task_description,
            requirements=requirements
        )
        
        assert collaboration_id is not None
        assert collaboration_id in coordinator.active_collaborations
        
        collaboration = coordinator.active_collaborations[collaboration_id]
        assert len(collaboration.participants) > 0
        assert collaboration.shared_knowledge["task_description"]["value"] == task_description
        assert collaboration.shared_knowledge["requirements"]["value"] == requirements
    
    @pytest.mark.asyncio
    async def test_agent_selection_optimization(self, coordinator):
        """Test intelligent agent selection based on capabilities and workload."""
        # Create collaboration requiring specific capabilities
        requirements = {
            "required_capabilities": ["system_design", "scalability"],
            "complexity": "high"
        }
        
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="design_review_01",
            task_description="Design scalable microservices architecture",
            requirements=requirements
        )
        
        collaboration = coordinator.active_collaborations[collaboration_id]
        participants = collaboration.participants
        
        # Verify architect agents were selected (they have system_design capability)
        architect_agents = [agent_id for agent_id in participants 
                          if coordinator.agents[agent_id].role == SpecializedAgentRole.ARCHITECT]
        assert len(architect_agents) > 0
        
        # Verify selected agents have relevant capabilities
        for agent_id in participants:
            agent = coordinator.agents[agent_id]
            agent_capabilities = [cap.name for cap in agent.capabilities]
            # At least one required capability should match
            has_relevant_capability = any(cap in agent_capabilities for cap in requirements["required_capabilities"])
            assert has_relevant_capability or agent.role in [SpecializedAgentRole.ARCHITECT, SpecializedAgentRole.PRODUCT]
    
    @pytest.mark.asyncio
    async def test_execute_collaboration_pair_programming(self, coordinator):
        """Test pair programming pattern execution."""
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="pair_programming_01",
            task_description="Implement data validation utility",
            requirements={"complexity": "moderate", "language": "python"}
        )
        
        execution_results = await coordinator.execute_collaboration(collaboration_id)
        
        assert execution_results["success"] is True
        assert execution_results["collaboration_id"] == collaboration_id
        assert len(execution_results["execution_steps"]) > 0
        assert execution_results["execution_time"] > 0
        assert "artifacts_created" in execution_results
        
        # Verify pair programming specific steps were executed
        step_names = [step["step"] for step in execution_results["execution_steps"]]
        expected_steps = ["establish_shared_context", "driver_navigator_assignment", "collaborative_coding"]
        
        for expected_step in expected_steps:
            assert expected_step in step_names
    
    @pytest.mark.asyncio
    async def test_execute_collaboration_code_review(self, coordinator):
        """Test code review cycle pattern execution."""
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="code_review_cycle_01",
            task_description="Review authentication system implementation",
            requirements={"security_focus": True, "review_depth": "comprehensive"}
        )
        
        execution_results = await coordinator.execute_collaboration(collaboration_id)
        
        assert execution_results["success"] is True
        
        # Verify code review specific steps
        step_names = [step["step"] for step in execution_results["execution_steps"]]
        expected_steps = ["code_submission", "automated_analysis", "parallel_reviews", "final_approval"]
        
        for expected_step in expected_steps:
            assert expected_step in step_names
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing_pattern(self, coordinator):
        """Test knowledge sharing pattern execution."""
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="knowledge_sharing_01",
            task_description="Share advanced microservices patterns",
            requirements={"knowledge_domain": "microservices", "audience_level": "senior"}
        )
        
        execution_results = await coordinator.execute_collaboration(collaboration_id)
        
        assert execution_results["success"] is True
        assert execution_results["knowledge_shared"] > 0
        
        # Verify knowledge sharing metrics updated
        assert coordinator.coordination_metrics["knowledge_sharing_events"] > 0
    
    @pytest.mark.asyncio
    async def test_collaboration_context_sharing(self, coordinator):
        """Test collaborative context and knowledge sharing between agents."""
        collaboration_id = await coordinator.create_collaboration(
            pattern_id="pair_programming_01",
            task_description="Complex algorithm implementation",
            requirements={"complexity": "high"}
        )
        
        collaboration = coordinator.active_collaborations[collaboration_id]
        
        # Simulate knowledge sharing
        collaboration.add_knowledge("implementation_approach", "recursive_with_memoization", "developer_1")
        collaboration.add_knowledge("performance_target", "O(n_log_n)", "architect_1")
        collaboration.add_communication("developer_1", "architect_1", "What's the optimal approach?", "question")
        collaboration.add_decision("Use recursive approach with memoization", "Balances readability and performance", ["developer_1", "architect_1"])
        
        assert len(collaboration.shared_knowledge) >= 4  # 2 initial + 2 added
        assert len(collaboration.communication_history) >= 1
        assert len(collaboration.decisions_made) >= 1
        
        # Verify knowledge structure
        impl_knowledge = collaboration.shared_knowledge["implementation_approach"]
        assert impl_knowledge["value"] == "recursive_with_memoization"
        assert impl_knowledge["contributor"] == "developer_1"
    
    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, coordinator):
        """Test agent performance tracking and learning."""
        # Get an agent
        agent_id = list(coordinator.agents.keys())[0]
        agent = coordinator.agents[agent_id]
        
        initial_performance_count = len(agent.performance_history)
        initial_capability_proficiency = agent.capabilities[0].proficiency_level
        
        # Simulate successful task execution
        agent.add_performance_record("test_task", True, 60.0, 0.9)
        
        assert len(agent.performance_history) == initial_performance_count + 1
        
        # Verify capability improvement (should be slight increase)
        updated_proficiency = agent.capabilities[0].proficiency_level
        assert updated_proficiency >= initial_capability_proficiency
    
    @pytest.mark.asyncio
    async def test_coordination_status_and_metrics(self, coordinator):
        """Test coordination status reporting and metrics collection."""
        status = coordinator.get_coordination_status()
        
        assert "active_collaborations" in status
        assert "total_agents" in status
        assert "available_agents" in status
        assert "coordination_patterns" in status
        assert "metrics" in status
        assert "agent_workloads" in status
        
        assert status["total_agents"] > 0
        assert status["coordination_patterns"] > 0
        assert isinstance(status["metrics"], dict)
        assert isinstance(status["agent_workloads"], dict)
    
    @pytest.mark.asyncio
    async def test_demonstrate_coordination_patterns(self, coordinator):
        """Test comprehensive coordination patterns demonstration."""
        # This is a comprehensive test that may take longer
        demo_results = await coordinator.demonstrate_coordination_patterns()
        
        assert demo_results["demonstration_id"] is not None
        assert "patterns_demonstrated" in demo_results
        assert len(demo_results["patterns_demonstrated"]) > 0
        assert "success_rate" in demo_results
        assert "total_execution_time" in demo_results
        
        # Verify all patterns were attempted
        demonstrated_patterns = [p["pattern_id"] for p in demo_results["patterns_demonstrated"]]
        for pattern_id in coordinator.coordination_patterns.keys():
            assert pattern_id in demonstrated_patterns


class TestSpecializedAgentImplementations:
    """Test suite for specialized agent implementations."""
    
    @pytest.fixture
    def workspace_dir(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_create_specialized_agents(self, workspace_dir):
        """Test creation of all specialized agent types."""
        for role in SpecializedAgentRole:
            agent = create_specialized_agent(role, f"{role.value}_test", workspace_dir)
            
            assert isinstance(agent, BaseEnhancedAgent)
            assert agent.role == role
            assert len(agent.capabilities) > 0
            assert agent.workspace_dir.exists()
    
    @pytest.mark.asyncio
    async def test_architect_agent_capabilities(self, workspace_dir):
        """Test architect agent specific capabilities."""
        agent = ArchitectAgent("architect_test", workspace_dir)
        await agent.initialize()
        
        capability_names = [cap.name for cap in agent.capabilities]
        expected_capabilities = ["system_design", "architecture_review", "technical_leadership"]
        
        for expected_cap in expected_capabilities:
            assert expected_cap in capability_names
        
        # Test architecture-specific task execution
        task = {
            "id": "arch_test_1",
            "type": "design",
            "description": "Design scalable microservices architecture",
            "requirements": {"scope": "enterprise", "scalability": "high"}
        }
        
        execution_result = await agent.execute_task(task)
        
        assert execution_result.status == "completed"
        assert execution_result.quality_score > 0.8
        assert len(execution_result.artifacts_created) > 0
    
    @pytest.mark.asyncio
    async def test_developer_agent_implementation(self, workspace_dir):
        """Test developer agent code implementation capabilities."""
        agent = DeveloperAgent("developer_test", workspace_dir)
        await agent.initialize()
        
        # Test code implementation task
        task = {
            "id": "dev_test_1",
            "type": "implement",
            "function_name": "data_processor",
            "description": "Implement high-performance data processing function",
            "requirements": {"language": "python", "performance": "optimized"}
        }
        
        execution_result = await agent.execute_task(task)
        
        assert execution_result.status == "completed"
        assert execution_result.quality_score > 0.8
        assert len(execution_result.artifacts_created) > 0
        
        # Verify code files were created
        for artifact_path in execution_result.artifacts_created:
            assert Path(artifact_path).exists()
            if artifact_path.endswith('.py'):
                # Verify it's valid Python code
                with open(artifact_path, 'r') as f:
                    code_content = f.read()
                    assert 'def ' in code_content  # Contains function definition
                    assert 'import' in code_content or 'from' in code_content  # Contains imports
    
    @pytest.mark.asyncio
    async def test_agent_learning_and_improvement(self, workspace_dir):
        """Test agent learning from task execution."""
        agent = DeveloperAgent("learning_test", workspace_dir)
        await agent.initialize()
        
        initial_insights_count = len(agent.learning_insights)
        
        # Execute multiple tasks to trigger learning
        for i in range(3):
            task = {
                "id": f"learning_task_{i}",
                "type": "implementation",
                "description": f"Implementation task {i}",
                "requirements": {"complexity": "moderate"}
            }
            
            await agent.execute_task(task)
        
        # Verify learning insights were updated
        assert len(agent.learning_insights) >= initial_insights_count
        assert len(agent.performance_history) == 3


class TestEnhancedCoordinationAPI:
    """Test suite for Enhanced Coordination API endpoints."""
    
    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator for API testing."""
        coordinator = Mock(spec=EnhancedMultiAgentCoordinator)
        
        # Mock coordination patterns
        mock_pattern = Mock()
        mock_pattern.pattern_id = "test_pattern_01"
        mock_pattern.name = "Test Pattern"
        mock_pattern.pattern_type = CoordinationPatternType.PAIR_PROGRAMMING
        mock_pattern.estimated_duration = 60
        mock_pattern.success_metrics = {"quality": 0.9}
        mock_pattern.required_roles = [SpecializedAgentRole.DEVELOPER]
        
        coordinator.coordination_patterns = {"test_pattern_01": mock_pattern}
        
        # Mock collaboration creation and execution
        coordinator.create_collaboration = AsyncMock(return_value="test_collaboration_123")
        coordinator.execute_collaboration = AsyncMock(return_value={
            "success": True,
            "collaboration_id": "test_collaboration_123",
            "execution_time": 45.0,
            "collaboration_efficiency": 0.92,
            "knowledge_shared": 5,
            "artifacts_created": ["test_artifact.py"],
            "participants": ["developer_1", "reviewer_1"]
        })
        
        # Mock status
        coordinator.get_coordination_status = Mock(return_value={
            "active_collaborations": 2,
            "total_agents": 12,
            "available_agents": 10,
            "coordination_patterns": 5,
            "metrics": {"total_collaborations": 50, "successful_collaborations": 47},
            "agent_workloads": {"developer_1": {"current_workload": 0.3}}
        })
        
        return coordinator
    
    @pytest.mark.asyncio
    async def test_execute_coordination_pattern_endpoint(self, mock_coordinator):
        """Test coordination pattern execution API endpoint."""
        # This would typically be tested with FastAPI TestClient
        # For now, we'll test the core logic
        
        from app.api.v1.enhanced_coordination_api import _execute_pattern_background
        
        # Test background execution
        await _execute_pattern_background(
            mock_coordinator,
            "test_pattern_01",
            "Test task description",
            {"complexity": "moderate"},
            None
        )
        
        # Verify coordinator methods were called
        mock_coordinator.create_collaboration.assert_called_once()
        mock_coordinator.execute_collaboration.assert_called_once()
    
    def test_coordination_status_response_format(self, mock_coordinator):
        """Test coordination status response format."""
        status = mock_coordinator.get_coordination_status()
        
        # Verify expected status fields
        required_fields = [
            "active_collaborations", "total_agents", "available_agents",
            "coordination_patterns", "metrics", "agent_workloads"
        ]
        
        for field in required_fields:
            assert field in status


class TestEnhancedCoordinationCommands:
    """Test suite for Enhanced Coordination Commands."""
    
    @pytest.fixture
    def commands_system(self):
        """Create commands system for testing."""
        return get_coordination_commands()
    
    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator for command testing."""
        coordinator = Mock(spec=EnhancedMultiAgentCoordinator)
        
        # Mock agents
        mock_agent = Mock()
        mock_agent.role = SpecializedAgentRole.DEVELOPER
        mock_agent.specialization_score = 0.85
        mock_agent.capabilities = [Mock(name="code_implementation")]
        mock_agent.current_workload = 0.3
        mock_agent.is_available = True
        
        coordinator.agents = {"developer_1": mock_agent}
        coordinator.agent_roles = {SpecializedAgentRole.DEVELOPER: ["developer_1"]}
        
        # Mock _calculate_agent_suitability method
        coordinator._calculate_agent_suitability = Mock(return_value=0.8)
        
        return coordinator
    
    @pytest.mark.asyncio
    async def test_team_formation_command(self, commands_system):
        """Test team formation command execution."""
        parameters = {
            "project_name": "Test Project",
            "roles": ["developer", "tester", "reviewer"],
            "description": "Test project description",
            "duration": 180
        }
        
        with patch('app.core.enhanced_coordination_commands.get_enhanced_coordinator') as mock_get_coordinator:
            mock_coordinator = Mock()
            mock_coordinator.agent_roles = {
                SpecializedAgentRole.DEVELOPER: ["developer_1"],
                SpecializedAgentRole.TESTER: ["tester_1"], 
                SpecializedAgentRole.REVIEWER: ["reviewer_1"]
            }
            
            # Mock agents
            for agent_id in ["developer_1", "tester_1", "reviewer_1"]:
                mock_agent = Mock()
                mock_agent.specialization_score = 0.85
                mock_agent.capabilities = []
                mock_agent.current_workload = 0.2
                mock_agent.is_available = True
                mock_coordinator.agents = {agent_id: mock_agent}
            
            mock_coordinator._calculate_agent_suitability = Mock(return_value=0.8)
            mock_get_coordinator.return_value = mock_coordinator
            
            result = await commands_system.handle_team_formation(parameters)
            
            assert result["success"] is True
            assert result["command"] == "coord:team-form"
            assert "result" in result
            assert result["result"]["project_name"] == "Test Project"
            assert len(result["result"]["team_members"]) == 3
    
    @pytest.mark.asyncio
    async def test_pattern_execution_command(self, commands_system):
        """Test pattern execution command."""
        parameters = {
            "pattern_id": "pair_programming_01",
            "task_description": "Implement feature X",
            "requirements": {"complexity": "moderate"},
            "async_mode": False
        }
        
        with patch('app.core.enhanced_coordination_commands.get_enhanced_coordinator') as mock_get_coordinator:
            mock_coordinator = Mock()
            
            # Mock pattern
            mock_pattern = Mock()
            mock_pattern.name = "Pair Programming"
            mock_pattern.pattern_type = CoordinationPatternType.PAIR_PROGRAMMING
            mock_pattern.estimated_duration = 60
            
            mock_coordinator.coordination_patterns = {"pair_programming_01": mock_pattern}
            mock_coordinator.create_collaboration = AsyncMock(return_value="collab_123")
            mock_coordinator.execute_collaboration = AsyncMock(return_value={
                "success": True,
                "execution_time": 45.0
            })
            
            mock_get_coordinator.return_value = mock_coordinator
            
            result = await commands_system.handle_pattern_execution(parameters)
            
            assert result["success"] is True
            assert result["command"] == "coord:pattern-exec"
            assert result["result"]["pattern_id"] == "pair_programming_01"
            assert result["result"]["status"] == "completed"
    
    @pytest.mark.asyncio 
    async def test_coordination_status_command(self, commands_system):
        """Test coordination status command."""
        parameters = {
            "detailed": True,
            "format": "json"
        }
        
        with patch('app.core.enhanced_coordination_commands.get_enhanced_coordinator') as mock_get_coordinator:
            mock_coordinator = Mock()
            mock_coordinator.get_coordination_status = Mock(return_value={
                "active_collaborations": 3,
                "total_agents": 12,
                "available_agents": 9,
                "coordination_patterns": 5,
                "metrics": {"total_collaborations": 100, "successful_collaborations": 95},
                "agent_workloads": {"agent_1": {"current_workload": 0.4}}
            })
            
            mock_get_coordinator.return_value = mock_coordinator
            
            result = await commands_system.handle_coordination_status(parameters)
            
            assert result["success"] is True
            assert result["command"] == "coord:status"
            assert "result" in result
            
            # For JSON format, should return complete status
            status = result["result"]
            assert "active_collaborations" in status
            assert "total_agents" in status
    
    def test_commands_registration(self, commands_system):
        """Test that all coordination commands are properly registered."""
        registered_commands = commands_system.get_registered_commands()
        
        expected_commands = [
            "coord:team-form",
            "coord:collaborate", 
            "coord:pattern-exec",
            "coord:demo",
            "coord:status",
            "coord:analytics",
            "coord:agents",
            "coord:patterns"
        ]
        
        registered_command_names = [cmd["name"] for cmd in registered_commands]
        
        for expected_cmd in expected_commands:
            assert expected_cmd in registered_command_names
        
        # Verify all commands have proper structure
        for cmd in registered_commands:
            assert cmd["name"] is not None
            assert cmd["description"] is not None
            assert cmd["category"] == "Enhanced Coordination"
            assert cmd["handler"] is not None
            assert isinstance(cmd["parameters"], dict)


class TestIntegrationScenarios:
    """Integration tests for complete coordination scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_collaboration_workflow(self):
        """Test complete end-to-end collaboration workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize coordinator
            coordinator = EnhancedMultiAgentCoordinator(temp_dir)
            
            # Mock dependencies to avoid Redis/external services
            coordinator.message_broker = Mock()
            coordinator.communication_service = Mock()
            coordinator.workflow_engine = Mock()
            coordinator.task_router = Mock()
            coordinator.capability_matcher = Mock()
            
            await coordinator._initialize_specialized_agents()
            
            # Create collaboration
            collaboration_id = await coordinator.create_collaboration(
                pattern_id="pair_programming_01",
                task_description="Implement comprehensive data validation system",
                requirements={
                    "language": "python",
                    "complexity": "high",
                    "required_capabilities": ["code_implementation", "testing"]
                }
            )
            
            # Execute collaboration
            results = await coordinator.execute_collaboration(collaboration_id)
            
            # Verify end-to-end results
            assert results["success"] is True
            assert results["collaboration_id"] == collaboration_id
            assert len(results["execution_steps"]) > 0
            assert results["execution_time"] > 0
            assert len(results["artifacts_created"]) > 0
            
            # Verify coordination metrics were updated
            status = coordinator.get_coordination_status()
            assert status["metrics"]["total_collaborations"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_collaborations(self):
        """Test handling multiple concurrent collaborations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = EnhancedMultiAgentCoordinator(temp_dir)
            
            # Mock dependencies
            coordinator.message_broker = Mock()
            coordinator.communication_service = Mock()
            coordinator.workflow_engine = Mock()
            coordinator.task_router = Mock()
            coordinator.capability_matcher = Mock()
            
            await coordinator._initialize_specialized_agents()
            
            # Create multiple collaborations
            collaboration_tasks = []
            
            for i in range(3):
                collaboration_id = await coordinator.create_collaboration(
                    pattern_id="pair_programming_01",
                    task_description=f"Task {i+1}: Implement feature module {i+1}",
                    requirements={"complexity": "moderate"}
                )
                
                collaboration_tasks.append(
                    coordinator.execute_collaboration(collaboration_id)
                )
            
            # Execute all collaborations concurrently
            results = await asyncio.gather(*collaboration_tasks, return_exceptions=True)
            
            # Verify all collaborations completed successfully
            successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
            assert len(successful_results) == 3
            
            # Verify system handled concurrent load
            status = coordinator.get_coordination_status()
            assert status["metrics"]["total_collaborations"] >= 3


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for coordination system."""
    
    @pytest.mark.asyncio
    async def test_coordination_performance_benchmarks(self):
        """Test coordination system performance benchmarks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = EnhancedMultiAgentCoordinator(temp_dir)
            
            # Mock dependencies for performance testing
            coordinator.message_broker = Mock()
            coordinator.communication_service = Mock()
            coordinator.workflow_engine = Mock()
            coordinator.task_router = Mock()
            coordinator.capability_matcher = Mock()
            
            await coordinator._initialize_specialized_agents()
            
            # Benchmark collaboration creation
            start_time = time.time()
            
            collaboration_ids = []
            for i in range(10):
                collaboration_id = await coordinator.create_collaboration(
                    pattern_id="pair_programming_01",
                    task_description=f"Performance test task {i}",
                    requirements={"complexity": "simple"}
                )
                collaboration_ids.append(collaboration_id)
            
            creation_time = time.time() - start_time
            
            # Verify performance targets
            assert creation_time < 5.0  # Should create 10 collaborations in under 5 seconds
            assert len(collaboration_ids) == 10
            
            # Benchmark single collaboration execution 
            start_time = time.time()
            result = await coordinator.execute_collaboration(collaboration_ids[0])
            execution_time = time.time() - start_time
            
            assert result["success"] is True
            assert execution_time < 10.0  # Single collaboration should complete in under 10 seconds
    
    @pytest.mark.asyncio
    async def test_agent_scalability(self):
        """Test system scalability with large number of agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            coordinator = EnhancedMultiAgentCoordinator(temp_dir)
            
            # Mock dependencies
            coordinator.message_broker = Mock()
            coordinator.communication_service = Mock()
            coordinator.workflow_engine = Mock()
            coordinator.task_router = Mock()
            coordinator.capability_matcher = Mock()
            
            await coordinator._initialize_specialized_agents()
            
            initial_agent_count = len(coordinator.agents)
            
            # Add more agents to test scalability
            for role in SpecializedAgentRole:
                for i in range(5):  # Add 5 more agents per role
                    agent_id = f"{role.value}_extra_{i}"
                    
                    # Create mock agent
                    agent = SpecializedAgent(
                        agent_id=agent_id,
                        role=role,
                        status=coordinator.agents[list(coordinator.agents.keys())[0]].status,
                        capabilities=coordinator.agents[list(coordinator.agents.keys())[0]].capabilities.copy()
                    )
                    
                    coordinator.agents[agent_id] = agent
                    coordinator.agent_roles[role].append(agent_id)
            
            # Verify scalability
            final_agent_count = len(coordinator.agents)
            assert final_agent_count > initial_agent_count
            
            # Test that system still functions with more agents
            collaboration_id = await coordinator.create_collaboration(
                pattern_id="pair_programming_01",
                task_description="Scalability test task",
                requirements={"complexity": "moderate"}
            )
            
            result = await coordinator.execute_collaboration(collaboration_id)
            assert result["success"] is True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])
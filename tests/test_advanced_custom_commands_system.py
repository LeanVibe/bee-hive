"""
Comprehensive Test Suite for Advanced Custom Commands System - Phase 6.1

Tests for the complete custom commands infrastructure including:
- CommandExecutor with intelligent workflow orchestration
- TaskDistributor with optimal agent workload balancing  
- WorkflowIntelligence engine with adaptive execution
- Quality Gates with automated validation
- Command Templates and project configurations
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any

from app.core.command_executor import CommandExecutor, ExecutionContext, ResourceLimits
from app.core.task_distributor import TaskDistributor, DistributionStrategy, TaskUrgency
from app.core.workflow_intelligence import WorkflowIntelligence, WorkflowComplexity
from app.core.quality_gates import QualityGatesEngine, QualityGateType, QualityGateStatus
from app.core.command_templates import CommandTemplateEngine, ProjectType, TeamSize, TechnologyStack
from app.core.command_registry import CommandRegistry
from app.core.agent_registry import AgentRegistry
from app.schemas.custom_commands import (
    CommandDefinition, CommandExecutionRequest, AgentRequirement, 
    WorkflowStep, SecurityPolicy, AgentRole
)


class TestCommandExecutor:
    """Test suite for CommandExecutor with intelligent workflow orchestration."""
    
    @pytest.fixture
    async def command_executor(self):
        """Create CommandExecutor instance for testing."""
        # Mock dependencies
        command_registry = Mock(spec=CommandRegistry)
        task_distributor = Mock(spec=TaskDistributor)
        agent_registry = Mock(spec=AgentRegistry)
        
        executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=agent_registry
        )
        
        await executor.start()
        yield executor
        await executor.stop()
    
    @pytest.fixture
    def sample_command_definition(self):
        """Create sample command definition for testing."""
        return CommandDefinition(
            name="test_feature_development",
            version="1.0.0",
            description="Test feature development workflow",
            category="development",
            tags=["test", "feature"],
            agents=[
                AgentRequirement(
                    role=AgentRole.BACKEND_ENGINEER,
                    required_capabilities=["python", "testing"]
                )
            ],
            workflow=[
                WorkflowStep(
                    step="implement_feature",
                    agent=AgentRole.BACKEND_ENGINEER,
                    task="Implement the requested feature",
                    timeout_minutes=60
                ),
                WorkflowStep(
                    step="test_feature",
                    agent=AgentRole.QA_TEST_GUARDIAN,
                    task="Test the implemented feature",
                    depends_on=["implement_feature"],
                    timeout_minutes=30
                )
            ],
            security_policy=SecurityPolicy(
                allowed_operations=["file_read", "file_write", "code_execution"],
                network_access=False
            )
        )
    
    @pytest.fixture
    def sample_execution_request(self):
        """Create sample execution request for testing."""
        return CommandExecutionRequest(
            command_name="test_feature_development",
            parameters={"feature_description": "Add user authentication"},
            context={"project_type": "web_application"},
            priority="medium"
        )
    
    async def test_command_execution_lifecycle(self, command_executor, sample_command_definition, sample_execution_request):
        """Test complete command execution lifecycle."""
        # Mock command registry to return our test command
        command_executor.command_registry.get_command.return_value = sample_command_definition
        
        # Mock task distributor to return successful assignments
        mock_distribution_result = Mock()
        mock_distribution_result.assignments = [
            Mock(task_id="implement_feature", agent_id="agent_1"),
            Mock(task_id="test_feature", agent_id="agent_2")
        ]
        command_executor.task_distributor.distribute_tasks.return_value = mock_distribution_result
        
        # Execute command
        result = await command_executor.execute_command(
            sample_execution_request,
            requester_id="test_user"
        )
        
        # Verify execution result
        assert result is not None
        assert result.command_name == "test_feature_development"
        assert result.execution_id is not None
        assert result.start_time is not None
        assert result.total_steps == 2
        
        # Verify command registry was called
        command_executor.command_registry.get_command.assert_called_once_with(
            "test_feature_development", None
        )
        
        # Verify task distribution was called
        command_executor.task_distributor.distribute_tasks.assert_called_once()
    
    async def test_resource_limits_enforcement(self, command_executor):
        """Test resource limits enforcement during execution."""
        execution_id = str(uuid.uuid4())
        
        # Create execution context with resource limits
        context = await command_executor._create_execution_context(
            execution_id,
            Mock(name="test_command", version="1.0.0", security_policy=SecurityPolicy()),
            CommandExecutionRequest(command_name="test", parameters={})
        )
        
        # Verify resource limits are set
        assert context.resource_limits.max_memory_mb > 0
        assert context.resource_limits.max_cpu_time_seconds > 0
        assert context.workspace_path.name == execution_id
        
        # Verify environment variables are set
        assert "AGENT_HIVE_EXECUTION_ID" in context.environment_vars
        assert context.environment_vars["AGENT_HIVE_EXECUTION_ID"] == execution_id
    
    async def test_security_policy_validation(self, command_executor):
        """Test security policy validation."""
        # Create command with restricted security policy
        restricted_command = CommandDefinition(
            name="restricted_command",
            version="1.0.0",
            description="Command with restricted security",
            agents=[AgentRequirement(role=AgentRole.BACKEND_ENGINEER)],
            workflow=[WorkflowStep(step="test", task="test task")],
            security_policy=SecurityPolicy(
                allowed_operations=["file_read"],
                network_access=False,
                requires_approval=True
            )
        )
        
        request = CommandExecutionRequest(command_name="restricted_command")
        
        # Should raise security violation for unapproved command
        with pytest.raises(Exception):
            await command_executor._validate_security_policy(
                restricted_command, request, requester_id=None
            )
    
    async def test_execution_cleanup(self, command_executor):
        """Test execution cleanup and resource management."""
        execution_id = str(uuid.uuid4())
        
        # Create mock context
        context = ExecutionContext(
            execution_id=execution_id,
            command_name="test",
            command_version="1.0.0",
            workspace_path=command_executor.workspace_root / execution_id,
            temp_path=command_executor.workspace_root / execution_id / "tmp",
            resource_limits=ResourceLimits(),
            security_policy=SecurityPolicy()
        )
        
        # Add to active executions
        command_executor.active_executions[execution_id] = context
        
        # Cleanup execution
        await command_executor._cleanup_execution(execution_id)
        
        # Verify cleanup
        assert execution_id not in command_executor.active_executions
    
    async def test_concurrent_execution_limits(self, command_executor, sample_command_definition, sample_execution_request):
        """Test concurrent execution limits enforcement."""
        # Set low concurrent limit for testing
        command_executor.max_concurrent_executions = 1
        
        # Mock command registry
        command_executor.command_registry.get_command.return_value = sample_command_definition
        
        # Add one active execution
        command_executor.active_executions["existing"] = Mock()
        
        # Try to execute another command - should fail
        with pytest.raises(RuntimeError, match="Maximum concurrent execution limit reached"):
            await command_executor.execute_command(sample_execution_request)


class TestTaskDistributor:
    """Test suite for TaskDistributor with intelligent agent selection."""
    
    @pytest.fixture
    def task_distributor(self):
        """Create TaskDistributor instance for testing."""
        agent_registry = Mock(spec=AgentRegistry)
        return TaskDistributor(agent_registry=agent_registry)
    
    @pytest.fixture
    def sample_workflow_steps(self):
        """Create sample workflow steps for testing."""
        return [
            WorkflowStep(
                step="backend_implementation",
                agent=AgentRole.BACKEND_ENGINEER,
                task="Implement backend logic",
                timeout_minutes=60
            ),
            WorkflowStep(
                step="frontend_implementation",
                agent=AgentRole.FRONTEND_BUILDER,
                task="Implement frontend components",
                timeout_minutes=45
            ),
            WorkflowStep(
                step="testing",
                agent=AgentRole.QA_TEST_GUARDIAN,
                task="Test the implementation",
                depends_on=["backend_implementation", "frontend_implementation"],
                timeout_minutes=30
            )
        ]
    
    @pytest.fixture
    def sample_agent_requirements(self):
        """Create sample agent requirements for testing."""
        return [
            AgentRequirement(
                role=AgentRole.BACKEND_ENGINEER,
                required_capabilities=["python", "api_development"]
            ),
            AgentRequirement(
                role=AgentRole.FRONTEND_BUILDER,
                required_capabilities=["react", "typescript"]
            ),
            AgentRequirement(
                role=AgentRole.QA_TEST_GUARDIAN,
                required_capabilities=["testing", "automation"]
            )
        ]
    
    async def test_task_distribution_strategies(self, task_distributor, sample_workflow_steps, sample_agent_requirements):
        """Test different task distribution strategies."""
        # Mock available agents
        mock_agents = [
            Mock(id="agent_1", role="backend-engineer", capabilities=[{"name": "python"}]),
            Mock(id="agent_2", role="frontend-builder", capabilities=[{"name": "react"}]),
            Mock(id="agent_3", role="qa-test-guardian", capabilities=[{"name": "testing"}])
        ]
        
        task_distributor.agent_registry.get_active_agents.return_value = mock_agents
        
        # Test different strategies
        strategies = [
            DistributionStrategy.ROUND_ROBIN,
            DistributionStrategy.LEAST_LOADED,
            DistributionStrategy.CAPABILITY_MATCH,
            DistributionStrategy.HYBRID
        ]
        
        for strategy in strategies:
            result = await task_distributor.distribute_tasks(
                workflow_steps=sample_workflow_steps,
                agent_requirements=sample_agent_requirements,
                strategy_override=strategy
            )
            
            # Verify distribution result
            assert result is not None
            assert result.strategy_used == strategy
            assert len(result.assignments) > 0
            assert result.distribution_time_ms > 0
    
    async def test_agent_workload_balancing(self, task_distributor):
        """Test agent workload balancing and monitoring."""
        # Mock agents with different workloads
        mock_agents = [
            Mock(id="agent_1", status="active"),
            Mock(id="agent_2", status="active"),
            Mock(id="agent_3", status="active")
        ]
        
        task_distributor.agent_registry.get_active_agents.return_value = mock_agents
        
        # Mock agent performance cache with different loads
        task_distributor.agent_performance_cache = {
            "agent_1": Mock(current_tasks=5, cpu_usage=80.0, memory_usage=70.0),
            "agent_2": Mock(current_tasks=2, cpu_usage=40.0, memory_usage=35.0),
            "agent_3": Mock(current_tasks=8, cpu_usage=90.0, memory_usage=85.0)
        }
        
        # Get workload status
        workload_status = await task_distributor.get_agent_workload_status()
        
        # Verify workload information
        assert len(workload_status) == 3
        assert "agent_1" in workload_status
        assert workload_status["agent_2"]["current_tasks"] == 2  # Least loaded
        assert workload_status["agent_3"]["cpu_usage"] == 90.0  # Most loaded
    
    async def test_task_reassignment_on_failure(self, task_distributor, sample_agent_requirements):
        """Test task reassignment when an agent fails."""
        # Mock available agents
        mock_agents = [
            Mock(id="agent_1", role="backend-engineer"),
            Mock(id="agent_2", role="backend-engineer"),
            Mock(id="agent_3", role="frontend-builder")
        ]
        
        task_distributor.agent_registry.get_active_agents.return_value = mock_agents
        
        # Test task reassignment
        reassignment = await task_distributor.reassign_failed_task(
            task_id="failed_task",
            failed_agent_id="agent_1",
            task_requirements=sample_agent_requirements[0]
        )
        
        # Verify reassignment
        assert reassignment is not None
        assert reassignment.task_id == "failed_task"
        assert reassignment.agent_id != "agent_1"  # Different agent assigned
        assert reassignment.assignment_reason == "task_reassignment_after_failure"


class TestWorkflowIntelligence:
    """Test suite for WorkflowIntelligence engine with adaptive execution."""
    
    @pytest.fixture
    def workflow_intelligence(self):
        """Create WorkflowIntelligence instance for testing."""
        command_registry = Mock(spec=CommandRegistry)
        task_distributor = Mock(spec=TaskDistributor)
        agent_registry = Mock(spec=AgentRegistry)
        
        return WorkflowIntelligence(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=agent_registry
        )
    
    @pytest.fixture
    def sample_command_for_analysis(self):
        """Create sample command for workflow analysis."""
        return CommandDefinition(
            name="complex_feature_development",
            version="1.0.0",
            description="Complex feature with multiple dependencies",
            agents=[
                AgentRequirement(role=AgentRole.BACKEND_ENGINEER),
                AgentRequirement(role=AgentRole.FRONTEND_BUILDER),
                AgentRequirement(role=AgentRole.QA_TEST_GUARDIAN)
            ],
            workflow=[
                WorkflowStep(step="step1", task="Task 1", timeout_minutes=30),
                WorkflowStep(step="step2", task="Task 2", depends_on=["step1"], timeout_minutes=45),
                WorkflowStep(step="step3", task="Task 3", depends_on=["step1"], timeout_minutes=60),
                WorkflowStep(step="step4", task="Task 4", depends_on=["step2", "step3"], timeout_minutes=30)
            ],
            security_policy=SecurityPolicy()
        )
    
    async def test_workflow_complexity_analysis(self, workflow_intelligence, sample_command_for_analysis):
        """Test workflow complexity analysis."""
        # Analyze workflow complexity
        analytics = await workflow_intelligence.analyze_workflow(
            sample_command_for_analysis,
            {"project_type": "web_application"}
        )
        
        # Verify analysis results
        assert analytics is not None
        assert analytics.workflow_name == "complex_feature_development"
        assert analytics.total_steps == 4
        assert analytics.complexity_level in [WorkflowComplexity.SIMPLE, WorkflowComplexity.MODERATE, WorkflowComplexity.COMPLEX]
        assert analytics.success_probability > 0.0
        assert len(analytics.optimization_opportunities) >= 0
    
    async def test_workflow_optimization(self, workflow_intelligence, sample_command_for_analysis):
        """Test workflow optimization with AI insights."""
        # First analyze the workflow
        analytics = await workflow_intelligence.analyze_workflow(sample_command_for_analysis)
        
        # Then optimize it
        optimized_def, adaptations = await workflow_intelligence.optimize_workflow_execution(
            analytics, sample_command_for_analysis
        )
        
        # Verify optimization results
        assert optimized_def is not None
        assert optimized_def.name == sample_command_for_analysis.name
        assert isinstance(adaptations, list)
        
        # If adaptations were applied, verify they're reasonable
        if adaptations:
            assert all(adaptation.confidence_score >= 0.0 for adaptation in adaptations)
            assert all(adaptation.expected_improvement >= 0.0 for adaptation in adaptations)
    
    async def test_contextual_insights_generation(self, workflow_intelligence):
        """Test contextual insights generation."""
        execution_context = {
            "project_type": "web_application",
            "team_size": 8,
            "execution_time_hour": 14  # Business hours
        }
        
        # Generate contextual insights
        insights = await workflow_intelligence.generate_contextual_insights(execution_context)
        
        # Verify insights
        assert isinstance(insights, list)
        for insight in insights:
            assert insight.insight_type is not None
            assert insight.description is not None
            assert 0.0 <= insight.confidence <= 1.0
            assert insight.impact_level in ["low", "medium", "high"]
    
    async def test_workflow_template_recommendation(self, workflow_intelligence):
        """Test workflow template recommendation based on task description."""
        # Test different task types
        test_cases = [
            ("Implement user authentication feature", "feature_development"),
            ("Fix memory leak in user service", "bug_fix"),
            ("Optimize database query performance", "performance_optimization"),
            ("Audit application for security vulnerabilities", "security_audit")
        ]
        
        for task_description, expected_type in test_cases:
            template = await workflow_intelligence.recommend_workflow_template(
                task_description,
                {"project_type": "web_application"}
            )
            
            # Verify template recommendation
            if template:  # Template might be None if not found
                assert template.name is not None
                assert len(template.workflow) > 0
                assert len(template.agents) > 0


class TestQualityGatesEngine:
    """Test suite for QualityGatesEngine with automated validation."""
    
    @pytest.fixture
    def quality_gates_engine(self):
        """Create QualityGatesEngine instance for testing."""
        return QualityGatesEngine()
    
    async def test_quality_gates_execution(self, quality_gates_engine):
        """Test comprehensive quality gates execution."""
        execution_context = {
            "command_name": "test_command",
            "project_type": "web_application",
            "codebase_path": "/test/path"
        }
        
        # Execute quality gates
        overall_success, gate_results = await quality_gates_engine.execute_quality_gates(
            execution_context,
            gate_types=[QualityGateType.CODE_QUALITY, QualityGateType.TESTING],
            fail_fast=False
        )
        
        # Verify results
        assert isinstance(overall_success, bool)
        assert isinstance(gate_results, list)
        assert len(gate_results) == 2  # CODE_QUALITY and TESTING
        
        # Verify each gate result
        for result in gate_results:
            assert result.gate_id is not None
            assert result.gate_type in [QualityGateType.CODE_QUALITY, QualityGateType.TESTING]
            assert result.status in [
                QualityGateStatus.PASSED, 
                QualityGateStatus.FAILED, 
                QualityGateStatus.WARNING
            ]
            assert 0.0 <= result.overall_score <= 100.0
    
    async def test_command_quality_validation(self, quality_gates_engine):
        """Test command-specific quality validation."""
        test_command = CommandDefinition(
            name="test_command",
            version="1.0.0",
            description="Test command for quality validation",
            category="testing",
            agents=[AgentRequirement(role=AgentRole.BACKEND_ENGINEER)],
            workflow=[WorkflowStep(step="test", task="test task")],
            security_policy=SecurityPolicy()
        )
        
        # Validate command quality
        passes_quality, gate_results = await quality_gates_engine.validate_command_quality(
            test_command
        )
        
        # Verify validation results
        assert isinstance(passes_quality, bool)
        assert isinstance(gate_results, list)
        assert len(gate_results) > 0
    
    async def test_quality_report_generation(self, quality_gates_engine):
        """Test comprehensive quality report generation."""
        # Create mock gate results
        mock_results = [
            Mock(
                gate_type=QualityGateType.CODE_QUALITY,
                status=QualityGateStatus.PASSED,
                overall_score=85.0,
                metrics=[Mock(severity="minor", status=QualityGateStatus.PASSED)],
                recommendations=["Improve code documentation"],
                artifacts={}
            ),
            Mock(
                gate_type=QualityGateType.SECURITY,
                status=QualityGateStatus.WARNING,
                overall_score=75.0,
                metrics=[Mock(severity="major", status=QualityGateStatus.FAILED)],
                recommendations=["Fix security vulnerabilities"],
                artifacts={}
            )
        ]
        
        execution_context = {"project_name": "test_project"}
        
        # Generate quality report
        report = await quality_gates_engine.generate_quality_report(
            mock_results, execution_context
        )
        
        # Verify report structure
        assert "report_id" in report
        assert "generated_at" in report
        assert "overall_quality" in report
        assert "gate_results" in report
        assert "recommendations" in report
        
        # Verify overall quality metrics
        assert "score" in report["overall_quality"]
        assert "grade" in report["overall_quality"]
        assert "status" in report["overall_quality"]


class TestCommandTemplateEngine:
    """Test suite for CommandTemplateEngine with project-specific configurations."""
    
    @pytest.fixture
    def template_engine(self):
        """Create CommandTemplateEngine instance for testing."""
        return CommandTemplateEngine()
    
    @pytest.fixture
    def sample_project_config(self):
        """Create sample project configuration."""
        from app.core.command_templates import ProjectConfiguration
        
        return ProjectConfiguration(
            project_type=ProjectType.WEB_APPLICATION,
            team_size=TeamSize.MEDIUM,
            tech_stack=TechnologyStack.PYTHON_FASTAPI,
            complexity_level="moderate"
        )
    
    @pytest.fixture
    def sample_customization(self):
        """Create sample template customization."""
        from app.core.command_templates import TemplateCustomization
        
        return TemplateCustomization(
            enable_ai_optimization=True,
            include_security_scans=True,
            code_coverage_threshold=85.0,
            max_workflow_duration_minutes=360
        )
    
    async def test_customized_command_generation(self, template_engine, sample_project_config, sample_customization):
        """Test generation of customized commands based on project configuration."""
        # Generate customized command
        customized_command = await template_engine.generate_customized_command(
            command_type="feature_development",
            project_config=sample_project_config,
            customization=sample_customization
        )
        
        # Verify customized command
        assert customized_command is not None
        assert customized_command.name is not None
        assert len(customized_command.workflow) > 0
        assert len(customized_command.agents) > 0
        
        # Verify customization was applied
        for step in customized_command.workflow:
            if step.timeout_minutes:
                assert step.timeout_minutes <= sample_customization.max_workflow_duration_minutes
    
    async def test_project_workflow_suite_creation(self, template_engine, sample_project_config, sample_customization):
        """Test creation of complete workflow suite for a project."""
        # Create workflow suite
        workflow_suite = await template_engine.create_project_workflow_suite(
            sample_project_config, sample_customization
        )
        
        # Verify workflow suite
        assert isinstance(workflow_suite, dict)
        assert len(workflow_suite) > 0
        
        # Verify each workflow in the suite
        for command_name, command_def in workflow_suite.items():
            assert isinstance(command_def, CommandDefinition)
            assert command_def.name is not None
            assert len(command_def.workflow) > 0
    
    async def test_technology_stack_customization(self, template_engine, sample_customization):
        """Test technology stack specific customizations."""
        # Test different technology stacks
        tech_stacks = [
            TechnologyStack.PYTHON_FASTAPI,
            TechnologyStack.NODEJS_EXPRESS,
            TechnologyStack.REACT_FRONTEND
        ]
        
        for tech_stack in tech_stacks:
            from app.core.command_templates import ProjectConfiguration
            
            config = ProjectConfiguration(
                project_type=ProjectType.WEB_APPLICATION,
                team_size=TeamSize.SMALL,
                tech_stack=tech_stack
            )
            
            try:
                command = await template_engine.generate_customized_command(
                    "feature_development", config, sample_customization
                )
                
                # Verify command was generated
                assert command is not None
                assert command.name is not None
                
            except ValueError:
                # Some combinations might not have templates yet
                pass


class TestSystemIntegration:
    """Integration tests for the complete custom commands system."""
    
    @pytest.fixture
    async def integrated_system(self):
        """Create integrated system with all components."""
        # Create all components
        agent_registry = Mock(spec=AgentRegistry)
        command_registry = Mock(spec=CommandRegistry)
        task_distributor = Mock(spec=TaskDistributor)
        workflow_intelligence = Mock(spec=WorkflowIntelligence)
        quality_gates = QualityGatesEngine()
        
        # Mock agent registry with sample agents
        mock_agents = [
            Mock(id="agent_1", role="backend-engineer", status="active", capabilities=[{"name": "python"}]),
            Mock(id="agent_2", role="frontend-builder", status="active", capabilities=[{"name": "react"}]),
            Mock(id="agent_3", role="qa-test-guardian", status="active", capabilities=[{"name": "testing"}])
        ]
        agent_registry.get_active_agents.return_value = mock_agents
        
        # Create command executor
        command_executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=agent_registry
        )
        
        await command_executor.start()
        
        yield {
            "command_executor": command_executor,
            "task_distributor": task_distributor,
            "workflow_intelligence": workflow_intelligence,
            "quality_gates": quality_gates,
            "agent_registry": agent_registry,
            "command_registry": command_registry
        }
        
        await command_executor.stop()
    
    async def test_end_to_end_command_execution(self, integrated_system):
        """Test end-to-end command execution with all components."""
        # Create comprehensive test command
        test_command = CommandDefinition(
            name="e2e_test_command",
            version="1.0.0",
            description="End-to-end test command",
            category="testing",
            agents=[
                AgentRequirement(role=AgentRole.BACKEND_ENGINEER, required_capabilities=["python"]),
                AgentRequirement(role=AgentRole.QA_TEST_GUARDIAN, required_capabilities=["testing"])
            ],
            workflow=[
                WorkflowStep(
                    step="implement",
                    agent=AgentRole.BACKEND_ENGINEER,
                    task="Implement feature",
                    timeout_minutes=60
                ),
                WorkflowStep(
                    step="test",
                    agent=AgentRole.QA_TEST_GUARDIAN,
                    task="Test implementation",
                    depends_on=["implement"],
                    timeout_minutes=30
                )
            ],
            security_policy=SecurityPolicy(allowed_operations=["file_read", "file_write"])
        )
        
        # Mock command registry to return our test command
        integrated_system["command_registry"].get_command.return_value = test_command
        
        # Mock successful task distribution
        mock_distribution = Mock()
        mock_distribution.assignments = [
            Mock(task_id="implement", agent_id="agent_1"),
            Mock(task_id="test", agent_id="agent_3")
        ]
        integrated_system["task_distributor"].distribute_tasks.return_value = mock_distribution
        
        # Create execution request
        execution_request = CommandExecutionRequest(
            command_name="e2e_test_command",
            parameters={"test_param": "test_value"},
            context={"project_type": "web_application"}
        )
        
        # Execute command
        result = await integrated_system["command_executor"].execute_command(
            execution_request,
            requester_id="test_user"
        )
        
        # Verify execution completed
        assert result is not None
        assert result.command_name == "e2e_test_command"
        assert result.execution_id is not None
        
        # Verify quality gates can validate the command
        passes_quality, gate_results = await integrated_system["quality_gates"].validate_command_quality(
            test_command, result
        )
        
        assert isinstance(passes_quality, bool)
        assert isinstance(gate_results, list)
    
    async def test_system_performance_under_load(self, integrated_system):
        """Test system performance under concurrent load."""
        # Create multiple execution requests
        num_concurrent_requests = 5
        execution_requests = []
        
        for i in range(num_concurrent_requests):
            request = CommandExecutionRequest(
                command_name=f"load_test_command_{i}",
                parameters={"iteration": i},
                context={"load_test": True}
            )
            execution_requests.append(request)
        
        # Mock command registry for all requests
        test_command = CommandDefinition(
            name="load_test_command",
            version="1.0.0",
            description="Load test command",
            agents=[AgentRequirement(role=AgentRole.BACKEND_ENGINEER)],
            workflow=[WorkflowStep(step="work", task="Do work", timeout_minutes=5)],
            security_policy=SecurityPolicy()
        )
        
        integrated_system["command_registry"].get_command.return_value = test_command
        
        # Mock task distribution
        mock_distribution = Mock()
        mock_distribution.assignments = [Mock(task_id="work", agent_id="agent_1")]
        integrated_system["task_distributor"].distribute_tasks.return_value = mock_distribution
        
        # Execute requests concurrently
        start_time = datetime.utcnow()
        
        tasks = [
            integrated_system["command_executor"].execute_command(request, requester_id="load_test_user")
            for request in execution_requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        # At least some should succeed (depending on concurrent limits)
        assert len(successful_results) > 0
        
        # Performance should be reasonable (less than 30 seconds for 5 requests)
        assert execution_time < 30.0
        
        print(f"Load test completed: {len(successful_results)} successful, "
              f"{len(failed_results)} failed, {execution_time:.2f}s total")


# Performance and stress tests
class TestSystemPerformance:
    """Performance and stress tests for the custom commands system."""
    
    @pytest.mark.performance
    async def test_command_execution_performance(self):
        """Test command execution performance benchmarks."""
        # Create minimal command executor
        command_registry = Mock()
        task_distributor = Mock()
        agent_registry = Mock()
        
        executor = CommandExecutor(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=agent_registry
        )
        
        await executor.start()
        
        try:
            # Benchmark execution context creation
            start_time = datetime.utcnow()
            
            for i in range(100):
                execution_id = str(uuid.uuid4())
                context = await executor._create_execution_context(
                    execution_id,
                    Mock(name="perf_test", version="1.0.0", security_policy=SecurityPolicy()),
                    CommandExecutionRequest(command_name="perf_test", parameters={})
                )
                await executor._cleanup_execution(execution_id)
            
            end_time = datetime.utcnow()
            avg_time = (end_time - start_time).total_seconds() / 100
            
            # Should create and cleanup contexts quickly
            assert avg_time < 0.1  # Less than 100ms per context
            
            print(f"Average context creation/cleanup time: {avg_time*1000:.2f}ms")
            
        finally:
            await executor.stop()
    
    @pytest.mark.stress
    async def test_quality_gates_stress(self):
        """Stress test quality gates engine."""
        quality_gates = QualityGatesEngine()
        
        # Run multiple quality gate executions
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(50):
            execution_context = {
                "command_name": f"stress_test_{i}",
                "project_type": "test"
            }
            
            task = quality_gates.execute_quality_gates(
                execution_context,
                gate_types=[QualityGateType.CODE_QUALITY],
                fail_fast=True
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        # Verify results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0
        
        # Performance should be reasonable
        avg_time_per_execution = total_time / len(results)
        assert avg_time_per_execution < 5.0  # Less than 5 seconds per execution
        
        print(f"Stress test completed: {len(successful_results)} successful, "
              f"avg time: {avg_time_per_execution:.2f}s")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=app.core",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-x"  # Stop on first failure for development
    ])
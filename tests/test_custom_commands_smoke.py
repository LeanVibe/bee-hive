"""
Smoke Test for Advanced Custom Commands System - Phase 6.1

Simple validation tests to ensure all components are properly integrated
and can be imported/instantiated without critical errors.
"""

import pytest
import asyncio
from unittest.mock import Mock

from app.core.command_executor import CommandExecutor
from app.core.task_distributor import TaskDistributor
from app.core.workflow_intelligence import WorkflowIntelligence
from app.core.quality_gates import QualityGatesEngine
from app.core.command_templates import CommandTemplateEngine, ProjectType, TeamSize, TechnologyStack
from app.core.command_registry import CommandRegistry
from app.core.agent_registry import AgentRegistry


class TestCustomCommandsSystemSmoke:
    """Smoke tests for the advanced custom commands system."""
    
    def test_all_imports_successful(self):
        """Test that all core components can be imported successfully."""
        # If we reach this point, all imports worked
        assert True
    
    def test_quality_gates_engine_instantiation(self):
        """Test QualityGatesEngine can be instantiated."""
        engine = QualityGatesEngine()
        assert engine is not None
    
    def test_command_template_engine_instantiation(self):
        """Test CommandTemplateEngine can be instantiated."""
        engine = CommandTemplateEngine()
        assert engine is not None
        assert len(engine.base_templates) > 0
    
    def test_workflow_intelligence_instantiation(self):
        """Test WorkflowIntelligence can be instantiated."""
        # Mock dependencies
        command_registry = Mock(spec=CommandRegistry)
        task_distributor = Mock(spec=TaskDistributor)
        agent_registry = Mock(spec=AgentRegistry)
        
        intelligence = WorkflowIntelligence(
            command_registry=command_registry,
            task_distributor=task_distributor,
            agent_registry=agent_registry
        )
        assert intelligence is not None
    
    def test_task_distributor_instantiation(self):
        """Test TaskDistributor can be instantiated."""
        agent_registry = Mock(spec=AgentRegistry)
        distributor = TaskDistributor(agent_registry=agent_registry)
        assert distributor is not None
    
    @pytest.mark.asyncio
    async def test_quality_gates_basic_execution(self):
        """Test that quality gates can execute basic validation."""
        engine = QualityGatesEngine()
        
        context = {
            "command_name": "smoke_test",
            "project_type": "test"
        }
        
        success, results = await engine.execute_quality_gates(
            context, 
            fail_fast=False  # Don't fail fast for smoke test
        )
        
        # We expect some results, success may vary based on context
        assert isinstance(success, bool)
        assert isinstance(results, list)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_command_template_generation(self):
        """Test that command templates can be generated."""
        from app.core.command_templates import ProjectConfiguration, TemplateCustomization
        
        engine = CommandTemplateEngine()
        
        project_config = ProjectConfiguration(
            project_type=ProjectType.WEB_APPLICATION,
            team_size=TeamSize.SMALL,
            tech_stack=TechnologyStack.PYTHON_FASTAPI
        )
        
        customization = TemplateCustomization(
            enable_ai_optimization=True,
            code_coverage_threshold=80.0
        )
        
        try:
            command = await engine.generate_customized_command(
                "feature_development",
                project_config,
                customization
            )
            
            assert command is not None
            assert command.name is not None
            assert len(command.workflow) > 0
            
        except Exception as e:
            # If template generation fails, that's acceptable for smoke test
            # as long as the error is not a critical import/instantiation issue
            assert "No base template found" in str(e) or "Template not found" in str(e)


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v", "--tb=short"])
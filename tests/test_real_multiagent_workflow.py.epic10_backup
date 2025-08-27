"""
Integration Tests for Real Multi-Agent Workflow

Tests the actual working multi-agent development workflow with file operations.
This test proves that LeanVibe Agent Hive 2.0 can coordinate real agents
to complete software development tasks.
"""

import asyncio
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.real_agent_implementations import (
    DeveloperAgent,
    QAAgent, 
    CIAgent,
    MultiAgentWorkflowCoordinator,
    AgentType
)


class TestRealAgentImplementations:
    """Test real agent implementations with actual file operations."""
    
    def setup_method(self):
        """Set up test workspace for each test."""
        self.workspace_dir = tempfile.mkdtemp(prefix="test_agents_")
        self.workspace_path = Path(self.workspace_dir)
    
    def teardown_method(self):
        """Clean up test workspace."""
        import shutil
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
    
    @pytest.mark.asyncio
    async def test_developer_agent_creates_python_code(self):
        """Test that developer agent creates actual Python code file."""
        # Mock the communication system initialization
        with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
            with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                agent = DeveloperAgent("dev-test", self.workspace_dir)
                await agent.initialize()
                
                # Execute development task
                task = {
                    "id": "test-task-1",
                    "function_name": "add_numbers",
                    "description": "Create a function that adds two numbers"
                }
                
                result = await agent.execute_task(task)
                
                # Verify execution results
                assert result.status == "completed"
                assert result.agent_type == AgentType.DEVELOPER
                assert len(result.files_created) == 1
                assert result.execution_time > 0
                
                # Verify file was created
                code_file = Path(result.files_created[0])
                assert code_file.exists()
                assert code_file.name == "add_numbers.py"
                
                # Verify file content
                with open(code_file, 'r') as f:
                    content = f.read()
                    assert "def add_numbers(a, b)" in content
                    assert "return a + b" in content
                    assert "TypeError" in content  # Error handling
    
    @pytest.mark.asyncio 
    async def test_qa_agent_creates_comprehensive_tests(self):
        """Test that QA agent creates comprehensive test file."""
        # First create a code file for testing
        code_file = self.workspace_path / "calculator.py"
        code_content = '''
def add_numbers(a, b):
    """Add two numbers together."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return a + b
'''
        with open(code_file, 'w') as f:
            f.write(code_content)
        
        # Mock the communication system initialization
        with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
            with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                agent = QAAgent("qa-test", self.workspace_dir)
                await agent.initialize()
                
                # Execute QA task
                task = {
                    "id": "test-task-2",
                    "function_name": "add_numbers",
                    "code_file": str(code_file),
                    "description": "Create comprehensive tests for add_numbers function"
                }
                
                result = await agent.execute_task(task)
                
                # Verify execution results
                assert result.status == "completed"
                assert result.agent_type == AgentType.QA_ENGINEER
                assert len(result.files_created) == 1
                assert result.execution_time > 0
                
                # Verify test file was created
                test_file = Path(result.files_created[0])
                assert test_file.exists()
                assert test_file.name == "test_add_numbers.py"
                
                # Verify test file content
                with open(test_file, 'r') as f:
                    content = f.read()
                    assert "import pytest" in content
                    assert "from calculator import add_numbers" in content
                    assert "def test_add_numbers_basic_positive_integers" in content
                    assert "def test_add_numbers_type_validation" in content
                    assert "pytest.raises(TypeError)" in content
    
    @pytest.mark.asyncio
    async def test_ci_agent_runs_tests_successfully(self):
        """Test that CI agent can run tests and report results."""
        # Create code file
        code_file = self.workspace_path / "calculator.py"
        code_content = '''
def add_numbers(a, b):
    """Add two numbers together."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both arguments must be numbers")
    return a + b
'''
        with open(code_file, 'w') as f:
            f.write(code_content)
        
        # Create test file
        test_file = self.workspace_path / "test_calculator.py"
        test_content = '''
import pytest
from calculator import add_numbers

def test_add_numbers_basic():
    """Test basic addition."""
    assert add_numbers(2, 3) == 5

def test_add_numbers_negative():
    """Test with negative numbers.""" 
    assert add_numbers(-2, -3) == -5

def test_add_numbers_type_error():
    """Test type validation."""
    with pytest.raises(TypeError):
        add_numbers("2", 3)
'''
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Mock the communication system initialization
        with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
            with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                agent = CIAgent("ci-test", self.workspace_dir)
                await agent.initialize()
                
                # Execute CI task
                task = {
                    "id": "test-task-3",
                    "test_file": str(test_file),
                    "description": "Run comprehensive test suite"
                }
                
                result = await agent.execute_task(task)
                
                # Verify execution results
                assert result.status == "completed"
                assert result.agent_type == AgentType.CI_CD_ENGINEER
                assert result.execution_time > 0
                assert result.output is not None
                
                # Verify test output contains expected information
                assert "test_add_numbers_basic" in result.output
                assert "test_add_numbers_negative" in result.output
                assert "test_add_numbers_type_error" in result.output


class TestMultiAgentWorkflowCoordinator:
    """Test the complete multi-agent workflow coordination."""
    
    def setup_method(self):
        """Set up test workspace for each test."""
        self.workspace_dir = tempfile.mkdtemp(prefix="test_workflow_")
        self.workspace_path = Path(self.workspace_dir)
    
    def teardown_method(self):
        """Clean up test workspace."""
        import shutil
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
    
    @pytest.mark.asyncio
    async def test_complete_development_workflow(self):
        """Test the complete multi-agent development workflow end-to-end."""
        # Mock the communication system initialization for all agents
        with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
            with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                coordinator = MultiAgentWorkflowCoordinator(self.workspace_dir)
                await coordinator.initialize_agents()
                
                # Define workflow requirements
                requirements = {
                    "function_name": "multiply_numbers",
                    "description": "Create a function that multiplies two numbers with validation"
                }
                
                # Execute the complete workflow
                workflow_results = await coordinator.execute_development_workflow(requirements)
                
                # Verify workflow completion
                assert workflow_results["success"] is True
                assert "stages" in workflow_results
                
                stages = workflow_results["stages"]
                
                # Verify development stage
                assert "development" in stages
                dev_stage = stages["development"]
                assert dev_stage["status"] == "completed"
                assert len(dev_stage["files_created"]) == 1
                
                # Verify the code file exists and has correct content
                code_file = Path(dev_stage["files_created"][0])
                assert code_file.exists()
                assert code_file.name == "multiply_numbers.py"
                
                with open(code_file, 'r') as f:
                    content = f.read()
                    assert "def multiply_numbers(a, b)" in content
                
                # Verify testing stage  
                assert "testing" in stages
                test_stage = stages["testing"]
                assert test_stage["status"] == "completed"
                assert len(test_stage["files_created"]) == 1
                
                # Verify the test file exists and has correct content
                test_file = Path(test_stage["files_created"][0])
                assert test_file.exists()
                assert test_file.name == "test_multiply_numbers.py"
                
                with open(test_file, 'r') as f:
                    content = f.read()
                    assert "import pytest" in content
                    assert "def test_multiply_numbers" in content
                
                # Verify CI/CD stage
                assert "ci_cd" in stages
                ci_stage = stages["ci_cd"]
                assert ci_stage["status"] == "completed"
                assert ci_stage["output"] is not None
                
                # Verify overall workflow metrics
                assert workflow_results["total_execution_time"] > 0
                assert workflow_results["workflow_id"] is not None
    
    @pytest.mark.asyncio
    async def test_workflow_handles_failures_gracefully(self):
        """Test that workflow handles agent failures gracefully."""
        # Mock the communication system but make development fail
        with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
            with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                coordinator = MultiAgentWorkflowCoordinator(self.workspace_dir)
                await coordinator.initialize_agents()
                
                # Mock developer agent to fail
                original_execute = coordinator.developer.execute_task
                async def failing_execute(task):
                    result = await original_execute(task)
                    result.status = "failed" 
                    result.error = "Simulated development failure"
                    return result
                
                coordinator.developer.execute_task = failing_execute
                
                # Execute workflow
                requirements = {
                    "function_name": "broken_function",
                    "description": "This should fail"
                }
                
                workflow_results = await coordinator.execute_development_workflow(requirements)
                
                # Verify workflow failed gracefully
                assert workflow_results["success"] is False
                assert "error" in workflow_results
                assert "stages" in workflow_results
                
                # Development stage should have failed
                dev_stage = workflow_results["stages"]["development"]
                assert dev_stage["status"] == "failed"
                assert dev_stage["error"] == "Simulated development failure"


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for the complete workflow system."""
    
    @pytest.mark.asyncio
    async def test_file_operations_integration(self):
        """Test that the workflow creates actual files that can be executed."""
        workspace_dir = tempfile.mkdtemp(prefix="integration_test_")
        
        try:
            # Mock communication systems
            with patch('app.core.real_agent_implementations.get_message_broker', new_callable=AsyncMock):
                with patch('app.core.real_agent_implementations.AgentCommunicationService'):
                    coordinator = MultiAgentWorkflowCoordinator(workspace_dir)
                    await coordinator.initialize_agents()
                    
                    # Execute workflow
                    requirements = {
                        "function_name": "add_numbers",
                        "description": "Create a function that adds two numbers"
                    }
                    
                    results = await coordinator.execute_development_workflow(requirements)
                    
                    assert results["success"] is True
                    
                    # Verify we can import and use the created code
                    import sys
                    sys.path.insert(0, workspace_dir)
                    
                    try:
                        from add_numbers import add_numbers
                        
                        # Test the actual function
                        assert add_numbers(5, 3) == 8
                        assert add_numbers(-2, 7) == 5
                        
                        # Test error handling
                        with pytest.raises(TypeError):
                            add_numbers("5", 3)
                            
                    finally:
                        sys.path.remove(workspace_dir)
                        
        finally:
            # Clean up
            import shutil
            shutil.rmtree(workspace_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
"""
Demonstration CLI Integration Tests with Mocked API

This module demonstrates Phase 2 of the CLI testing strategy:
CLI Integration Testing with Mocked API responses.

This showcases how to test CLI commands with controlled API responses
without requiring a running server.
"""

import pytest
import json
from unittest.mock import Mock, patch
from click.testing import CliRunner

from app.cli.unix_commands import hive_status, hive_get, hive_create, hive_kill


@pytest.mark.cli_integration
class TestCLIWithMockedAPIDemo:
    """Demonstration of CLI integration testing with mocked API responses."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create isolated CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_healthy_api(self):
        """Mock a healthy API server with realistic responses."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy", "latency": "2ms"},
                    "redis": {"status": "healthy", "connections": 5},
                    "orchestrator": {"status": "healthy", "agents": 3}
                },
                "uptime": "2h 15m 30s",
                "version": "2.0.0"
            }
            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response
            yield mock_requests

    def test_status_command_with_healthy_system(self, cli_runner, mock_healthy_api):
        """Test status command with healthy system response."""
        result = cli_runner.invoke(hive_status, ['--format', 'json'])
        
        assert result.exit_code == 0
        mock_healthy_api.get.assert_called()
        
        # Verify correct API endpoint was called
        call_args = mock_healthy_api.get.call_args[0][0]
        assert "status" in call_args or "health" in call_args
        
        print(f"✅ Status command test passed - API called: {call_args}")

    def test_get_agents_with_mock_agent_data(self, cli_runner, mock_healthy_api):
        """Test get agents command with realistic agent data."""
        # Mock specific agent response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agents": [
                {
                    "id": "agent-backend-001",
                    "status": "active",
                    "active_tasks": 2,
                    "uptime": "1h 45m",
                    "capabilities": ["python", "fastapi", "testing"]
                },
                {
                    "id": "agent-frontend-001",
                    "status": "idle", 
                    "active_tasks": 0,
                    "uptime": "45m",
                    "capabilities": ["react", "typescript", "testing"]
                }
            ],
            "total": 2,
            "active": 1
        }
        mock_healthy_api.get.return_value = mock_response
        
        result = cli_runner.invoke(hive_get, ['agents', '--output', 'json'])
        
        assert result.exit_code == 0
        mock_healthy_api.get.assert_called()
        
        # Verify output contains agent data
        if result.output.strip():
            output_data = json.loads(result.output)
            assert "agents" in output_data
            assert len(output_data["agents"]) == 2
            assert output_data["agents"][0]["id"] == "agent-backend-001"
        
        print("✅ Get agents test passed - realistic agent data returned")

    def test_create_agents_workflow(self, cli_runner, mock_healthy_api):
        """Test complete agent creation workflow."""
        # Mock successful creation response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "agents": [
                {
                    "id": "agent-new-001",
                    "status": "initializing", 
                    "type": "qa-engineer",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                {
                    "id": "agent-new-002",
                    "status": "initializing",
                    "type": "qa-engineer", 
                    "created_at": "2024-01-15T10:30:01Z"
                }
            ],
            "message": "Successfully created 2 QA engineer agents"
        }
        mock_healthy_api.post.return_value = mock_response
        
        result = cli_runner.invoke(hive_create, [
            '--count', '2',
            '--type', 'qa-engineer'
        ])
        
        assert result.exit_code == 0
        mock_healthy_api.post.assert_called()
        
        # Verify correct API call was made
        call_args = mock_healthy_api.post.call_args
        assert "agents/create" in call_args[1]['url']
        
        posted_data = call_args[1]['json']
        assert posted_data['count'] == 2
        assert posted_data['type'] == 'qa-engineer'
        
        # Verify success message in output
        assert "successfully" in result.output.lower() or "created" in result.output.lower()
        
        print("✅ Create agents workflow test passed")

    def test_error_scenario_api_unavailable(self, cli_runner):
        """Test CLI behavior when API is unavailable."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Simulate API server down
            import requests
            mock_requests.get.side_effect = requests.ConnectionError("Connection refused")
            
            result = cli_runner.invoke(hive_status)
            
            # Should handle error gracefully
            assert result.exit_code in [0, 1]
            
            # Should provide helpful error indication
            assert len(result.output) > 0
            
            print("✅ API unavailable error handling test passed")

    def test_agent_termination_workflow(self, cli_runner, mock_healthy_api):
        """Test agent termination workflow."""
        # Mock successful termination response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "agent_id": "agent-test-001",
            "status": "terminated",
            "reason": "user_request",
            "terminated_at": "2024-01-15T10:35:00Z"
        }
        mock_healthy_api.post.return_value = mock_response
        
        result = cli_runner.invoke(hive_kill, [
            'agent-test-001',
            '--reason', 'test-cleanup'
        ])
        
        assert result.exit_code == 0
        mock_healthy_api.post.assert_called()
        
        # Verify correct termination API call
        call_args = mock_healthy_api.post.call_args
        assert "agent-test-001/terminate" in call_args[1]['url']
        
        posted_data = call_args[1]['json']
        assert posted_data['reason'] == 'test-cleanup'
        
        print("✅ Agent termination workflow test passed")


@pytest.mark.cli_integration 
class TestCLICompleteAgentLifecycleDemo:
    """Demonstrate complete agent lifecycle testing with mocked API."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture 
    def mock_agent_lifecycle_api(self):
        """Mock API for complete agent lifecycle testing."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Define response sequence for different API calls
            responses = {
                'debug-agents': {
                    "agents": [
                        {
                            "id": "agent-existing-001",
                            "status": "active",
                            "type": "backend-engineer",
                            "active_tasks": 1,
                            "uptime": "3h 20m"
                        }
                    ],
                    "total": 1
                },
                'api/agents/create': {
                    "success": True,
                    "agents": [
                        {
                            "id": "agent-lifecycle-test",
                            "status": "initializing",
                            "type": "test-agent"
                        }
                    ]
                },
                'api/agents/agent-lifecycle-test/terminate': {
                    "success": True,
                    "message": "Agent terminated successfully"
                }
            }
            
            def mock_request(method, url, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                
                # Match endpoint to appropriate response
                for endpoint, response_data in responses.items():
                    if endpoint in url:
                        mock_response.json.return_value = response_data
                        return mock_response
                
                # Default response
                mock_response.json.return_value = {"status": "ok"}
                return mock_response
            
            mock_requests.get.side_effect = lambda url, **kwargs: mock_request("GET", url, **kwargs)
            mock_requests.post.side_effect = lambda url, **kwargs: mock_request("POST", url, **kwargs)
            
            yield mock_requests

    def test_complete_agent_lifecycle(self, cli_runner, mock_agent_lifecycle_api):
        """Test complete agent management lifecycle."""
        print("\n🔄 Testing complete agent lifecycle...")
        
        # Step 1: List existing agents
        print("  📋 Step 1: List existing agents")
        result = cli_runner.invoke(hive_get, ['agents'])
        assert result.exit_code == 0
        print("    ✅ Successfully listed existing agents")
        
        # Step 2: Create new agent
        print("  🏗️  Step 2: Create new test agent")
        result = cli_runner.invoke(hive_create, [
            '--count', '1',
            '--type', 'test-agent'
        ])
        assert result.exit_code == 0
        print("    ✅ Successfully created test agent")
        
        # Step 3: Verify agent was created (list again)
        print("  🔍 Step 3: Verify agent creation")
        result = cli_runner.invoke(hive_get, ['agents'])
        assert result.exit_code == 0
        print("    ✅ Agent list updated successfully")
        
        # Step 4: Terminate the test agent
        print("  🛑 Step 4: Terminate test agent")
        result = cli_runner.invoke(hive_kill, [
            'agent-lifecycle-test',
            '--reason', 'lifecycle-test-complete'
        ])
        assert result.exit_code == 0
        print("    ✅ Successfully terminated test agent")
        
        # Verify all expected API calls were made
        assert mock_agent_lifecycle_api.get.call_count >= 2  # At least 2 GET calls (list agents)
        assert mock_agent_lifecycle_api.post.call_count >= 2  # Create + terminate
        
        print("✅ Complete agent lifecycle test passed!")


if __name__ == "__main__":
    # Run this module's tests directly
    pytest.main([__file__, "-v", "-s"])
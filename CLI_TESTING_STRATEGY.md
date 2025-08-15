# LeanVibe Agent Hive 2.0 - Comprehensive CLI Testing Strategy

## Executive Summary

Building on our solid foundation of **44 passing unit tests** and **54 passing API tests**, this strategy designs comprehensive CLI testing for LeanVibe Agent Hive 2.0's three CLI interfaces:
- **`agent-hive`** - Professional management CLI
- **`hive`** - Unix-philosophy kubectl-style commands  
- **`lv`** - Enhanced developer experience CLI

## CLI Implementation Analysis

### CLI Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 CLI Testing Pyramid                     │
├─────────────────────────────────────────────────────────┤
│  E2E CLI Workflows (against running API)               │ ◄─── New
├─────────────────────────────────────────────────────────┤
│  CLI Integration Tests (with mocked API)               │ ◄─── New
├─────────────────────────────────────────────────────────┤
│  CLI Unit Tests (command parsing & logic)              │ ◄─── Fix existing
├─────────────────────────────────────────────────────────┤
│  REST API Tests (54 passing) ✅                        │
├─────────────────────────────────────────────────────────┤
│  Unit Tests (44 passing) ✅                            │
└─────────────────────────────────────────────────────────┘
```

### CLI Integration Patterns Identified

**HTTP-Based Integration:**
```python
# CLI Commands → HTTP API Pattern
class HiveContext:
    api_base: str = "http://localhost:8000"
    
    def api_call(self, endpoint: str, method: str = "GET") -> Optional[Dict]:
        url = f"{self.api_base}/{endpoint.lstrip('/')}"
        response = requests.get(url, timeout=5)
        return response.json() if response.status_code == 200 else None

# Example Usage in Commands:
hive_status() → ctx.api_call("status")
hive_get("agents") → ctx.api_call("debug-agents") 
hive_logs() → ctx.api_call("logs")
hive_create() → ctx.api_call("api/agents/create", method="POST")
```

**Entry Points Discovered:**
- **Primary**: `agent-hive` → `app.cli:main` (professional interface)
- **Unix**: `hive` → `app.cli.main:hive_cli` (kubectl-style)
- **DX**: `lv` → `app.dx_cli:main` (developer experience)

## CLI Testing Strategy Design

### Phase 1: CLI Unit Testing Framework ⚡

**Objective**: Test CLI command parsing, validation, and logic without external dependencies

```python
# tests/cli/test_cli_units.py
import pytest
from click.testing import CliRunner
from unittest.mock import Mock, patch

from app.cli.unix_commands import hive_status, hive_get, HiveContext
from app.cli.main import hive_cli

class TestCLIUnitFramework:
    """Unit testing for CLI command parsing and validation."""
    
    @pytest.fixture
    def cli_runner(self):
        """Isolated CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_context(self):
        """Mock HiveContext for isolation."""
        with patch('app.cli.unix_commands.ctx') as mock_ctx:
            mock_ctx.api_call.return_value = {"status": "healthy"}
            yield mock_ctx

    def test_hive_status_command_parsing(self, cli_runner):
        """Test status command argument parsing."""
        # Test valid formats
        result = cli_runner.invoke(hive_status, ['--format', 'json'])
        assert result.exit_code == 0
        
        result = cli_runner.invoke(hive_status, ['--watch'])
        assert result.exit_code == 0
        
        # Test invalid format
        result = cli_runner.invoke(hive_status, ['--format', 'invalid'])
        assert result.exit_code != 0

    def test_hive_get_resource_validation(self, cli_runner):
        """Test resource type validation."""
        valid_resources = ['agents', 'tasks', 'workflows', 'metrics']
        
        for resource in valid_resources:
            result = cli_runner.invoke(hive_get, [resource])
            assert result.exit_code == 0
        
        # Test invalid resource
        result = cli_runner.invoke(hive_get, ['invalid-resource'])
        assert result.exit_code == 0  # Should pass through to API

    def test_hive_create_agent_config_parsing(self, cli_runner):
        """Test agent creation configuration parsing."""
        # Test basic creation
        result = cli_runner.invoke(hive_create, ['--count', '3', '--type', 'worker'])
        assert result.exit_code == 0
        
        # Test with config file
        with cli_runner.isolated_filesystem():
            config_file = "agent-config.json"
            with open(config_file, 'w') as f:
                json.dump({"type": "specialist", "capabilities": ["code", "test"]}, f)
            
            result = cli_runner.invoke(hive_create, ['--config', config_file])
            assert result.exit_code == 0

    @patch('app.cli.unix_commands.ctx')
    def test_context_api_base_configuration(self, mock_ctx, cli_runner):
        """Test API base URL configuration."""
        # Test default configuration
        context = HiveContext()
        assert context.api_base == "http://localhost:8000"
        
        # Test that commands use the context
        mock_ctx.api_call.return_value = {"healthy": True}
        result = cli_runner.invoke(hive_status)
        mock_ctx.api_call.assert_called_with("status")

    def test_error_handling_command_structure(self, cli_runner):
        """Test CLI error handling and help systems."""
        # Test main CLI help
        result = cli_runner.invoke(hive_cli, ['--help'])
        assert result.exit_code == 0
        assert "LeanVibe Agent Hive" in result.output
        
        # Test individual command help
        result = cli_runner.invoke(hive_status, ['--help'])
        assert result.exit_code == 0
        assert "status" in result.output.lower()

    def test_output_format_consistency(self, cli_runner, mock_context):
        """Test output format consistency across commands."""
        # JSON output should be parseable
        result = cli_runner.invoke(hive_status, ['--format', 'json'])
        if result.exit_code == 0 and result.output.strip():
            try:
                json.loads(result.output)
            except json.JSONDecodeError:
                pytest.fail("JSON output is not valid JSON")

class TestAgentHiveCLIUnits:
    """Unit tests for main agent-hive CLI."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @patch('app.cli.AgentHiveCLI')
    def test_start_command_logic(self, mock_cli_class, cli_runner):
        """Test start command logic without external dependencies."""
        mock_cli = Mock()
        mock_cli.check_system_health.return_value = False
        mock_cli.start_services.return_value = True
        mock_cli_class.return_value = mock_cli
        
        from app.cli import start
        result = cli_runner.invoke(start)
        
        mock_cli.check_system_health.assert_called()
        mock_cli.start_services.assert_called_with(quick=False)

    @patch('app.cli.AgentHiveCLI')
    def test_status_command_logic(self, mock_cli_class, cli_runner):
        """Test status command logic."""
        mock_cli = Mock()
        mock_cli.check_system_health.return_value = True
        mock_cli.get_system_status.return_value = {
            "components": {
                "database": {"status": "healthy", "details": "Connected"}
            }
        }
        mock_cli_class.return_value = mock_cli
        
        from app.cli import status
        result = cli_runner.invoke(status)
        
        assert result.exit_code == 0
        mock_cli.check_system_health.assert_called()

class TestDXCLIUnits:
    """Unit tests for lv developer experience CLI."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    def test_intelligent_command_suggester_init(self):
        """Test command suggester initialization."""
        from app.dx_cli import IntelligentCommandSuggester
        
        suggester = IntelligentCommandSuggester()
        assert hasattr(suggester, 'usage_history')
        assert hasattr(suggester, 'context_patterns')
        assert isinstance(suggester.context_patterns, dict)
```

### Phase 2: CLI Integration Testing with Mocked API 🔌

**Objective**: Test CLI commands with controlled API responses

```python
# tests/cli/test_cli_integration_mocked.py
import pytest
import json
from unittest.mock import Mock, patch
from click.testing import CliRunner

class TestCLIWithMockedAPI:
    """Integration tests with mocked API responses."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_healthy_api(self):
        """Mock a healthy API server."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy"},
                    "redis": {"status": "healthy"}
                }
            }
            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response
            yield mock_requests
    
    @pytest.fixture
    def mock_failed_api(self):
        """Mock a failed API server."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection failed")
            mock_requests.post.side_effect = Exception("Connection failed")
            yield mock_requests

    def test_status_with_healthy_api(self, cli_runner, mock_healthy_api):
        """Test status command with healthy API."""
        from app.cli.unix_commands import hive_status
        
        result = cli_runner.invoke(hive_status, ['--format', 'json'])
        
        assert result.exit_code == 0
        mock_healthy_api.get.assert_called()
        
        # Verify API endpoint called correctly
        call_args = mock_healthy_api.get.call_args[0][0]
        assert "status" in call_args or "health" in call_args

    def test_get_agents_with_mock_data(self, cli_runner, mock_healthy_api):
        """Test get agents command with mocked agent data."""
        # Mock agent data response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "agents": [
                {
                    "id": "agent-001",
                    "status": "active",
                    "active_tasks": 3,
                    "uptime": "2h 30m"
                },
                {
                    "id": "agent-002", 
                    "status": "idle",
                    "active_tasks": 0,
                    "uptime": "1h 15m"
                }
            ]
        }
        mock_healthy_api.get.return_value = mock_response
        
        from app.cli.unix_commands import hive_get
        result = cli_runner.invoke(hive_get, ['agents', '--output', 'json'])
        
        assert result.exit_code == 0
        mock_healthy_api.get.assert_called()
        
        # Verify output contains agent data
        if result.output.strip():
            output_data = json.loads(result.output)
            assert "agents" in output_data
            assert len(output_data["agents"]) == 2

    def test_create_agents_with_api_response(self, cli_runner, mock_healthy_api):
        """Test creating agents with API response."""
        # Mock successful creation response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "agents": [
                {"id": "agent-new-001", "status": "initializing"},
                {"id": "agent-new-002", "status": "initializing"}
            ]
        }
        mock_healthy_api.post.return_value = mock_response
        
        from app.cli.unix_commands import hive_create
        result = cli_runner.invoke(hive_create, ['--count', '2', '--type', 'worker'])
        
        assert result.exit_code == 0
        mock_healthy_api.post.assert_called()
        
        # Verify correct API endpoint and data
        call_args = mock_healthy_api.post.call_args
        assert "agents/create" in call_args[1]['url'] or "api/agents/create" in call_args[1]['url']
        assert call_args[1]['json']['count'] == 2
        assert call_args[1]['json']['type'] == 'worker'

    def test_error_handling_with_failed_api(self, cli_runner, mock_failed_api):
        """Test CLI error handling when API is unavailable."""
        from app.cli.unix_commands import hive_status
        
        result = cli_runner.invoke(hive_status)
        
        # Should handle error gracefully
        assert result.exit_code in [0, 1]  # Might succeed with error message or fail cleanly
        
        # Should show helpful error message
        assert any(word in result.output.lower() for word in [
            "error", "failed", "connection", "unavailable", "not responding"
        ])

    def test_logs_command_streaming_simulation(self, cli_runner, mock_healthy_api):
        """Test logs command with streaming simulation."""
        # Mock logs response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entries": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "level": "INFO",
                    "component": "orchestrator", 
                    "message": "Agent deployment successful"
                },
                {
                    "timestamp": "2024-01-15T10:30:05Z",
                    "level": "DEBUG",
                    "component": "agent-001",
                    "message": "Task processing started"
                }
            ]
        }
        mock_healthy_api.get.return_value = mock_response
        
        from app.cli.unix_commands import hive_logs
        result = cli_runner.invoke(hive_logs, ['--lines', '10'])
        
        assert result.exit_code == 0
        mock_healthy_api.get.assert_called()

    def test_config_command_persistence(self, cli_runner):
        """Test configuration command with file operations."""
        from app.cli.unix_commands import hive_config
        
        with cli_runner.isolated_filesystem():
            # Set configuration
            result = cli_runner.invoke(hive_config, ['api.base', 'http://test:9000'])
            assert result.exit_code == 0
            
            # Get configuration
            result = cli_runner.invoke(hive_config, ['api.base'])
            assert result.exit_code == 0
            assert 'http://test:9000' in result.output
            
            # List all configuration
            result = cli_runner.invoke(hive_config, ['--list'])
            assert result.exit_code == 0

class TestCLIAgentLifecycleWorkflow:
    """Test complete agent lifecycle via CLI with mocked API."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_api_sequence(self):
        """Mock API sequence for agent lifecycle."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Define response sequence for different endpoints
            responses = {
                'debug-agents': {
                    "agents": [
                        {"id": "agent-001", "status": "active", "active_tasks": 2}
                    ]
                },
                'api/agents/create': {
                    "success": True,
                    "agents": [{"id": "agent-new", "status": "initializing"}]
                },
                'api/agents/agent-001/terminate': {
                    "success": True,
                    "message": "Agent terminated"
                }
            }
            
            def mock_request(method, url, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                
                # Match endpoint to response
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

    def test_complete_agent_lifecycle(self, cli_runner, mock_api_sequence):
        """Test complete agent management workflow."""
        from app.cli.unix_commands import hive_get, hive_create, hive_kill
        
        # 1. List existing agents
        result = cli_runner.invoke(hive_get, ['agents'])
        assert result.exit_code == 0
        
        # 2. Create new agent
        result = cli_runner.invoke(hive_create, ['--count', '1', '--type', 'test'])
        assert result.exit_code == 0
        
        # 3. Terminate agent (using known agent ID from mock)
        result = cli_runner.invoke(hive_kill, ['agent-001'])
        assert result.exit_code == 0
        
        # Verify all API calls were made
        assert mock_api_sequence.get.call_count >= 1
        assert mock_api_sequence.post.call_count >= 1
```

### Phase 3: CLI End-to-End Testing with Running API 🚀

**Objective**: Test CLI commands against actual running API server

```python
# tests/cli/test_cli_e2e_integration.py
import pytest
import time
import subprocess
import requests
from click.testing import CliRunner

@pytest.mark.integration
class TestCLIEndToEndIntegration:
    """End-to-end CLI testing with running API server."""
    
    @pytest.fixture(scope="class")
    def running_api_server(self):
        """Start test API server for CLI integration testing."""
        # Start server in background
        process = subprocess.Popen([
            "python", "-m", "uvicorn", "app.main:app",
            "--host", "127.0.0.1", "--port", "8002", "--log-level", "error"
        ])
        
        # Wait for server to start
        for _ in range(30):
            try:
                response = requests.get("http://127.0.0.1:8002/health", timeout=1)
                if response.status_code == 200:
                    break
            except:
                time.sleep(1)
        else:
            process.terminate()
            pytest.fail("Test server failed to start")
        
        yield "http://127.0.0.1:8002"
        
        # Cleanup
        process.terminate()
        process.wait()

    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    @pytest.fixture
    def test_config(self, cli_runner, running_api_server):
        """Configure CLI to use test server."""
        from app.cli.unix_commands import hive_config
        
        with cli_runner.isolated_filesystem():
            result = cli_runner.invoke(hive_config, ['api.base', running_api_server])
            assert result.exit_code == 0
            yield running_api_server

    def test_cli_system_health_check(self, cli_runner, test_config):
        """Test CLI system health check against running API."""
        from app.cli.unix_commands import hive_status
        
        result = cli_runner.invoke(hive_status, ['--format', 'json'])
        
        assert result.exit_code == 0
        if result.output.strip():
            import json
            status_data = json.loads(result.output)
            # Should contain health information from real API
            assert isinstance(status_data, dict)

    def test_cli_doctor_diagnostics(self, cli_runner, test_config):
        """Test CLI doctor command diagnostics."""
        from app.cli.unix_commands import hive_doctor
        
        result = cli_runner.invoke(hive_doctor)
        
        assert result.exit_code == 0
        # Should provide system diagnostic information
        assert any(word in result.output.lower() for word in [
            "system", "health", "check", "status"
        ])

    def test_cli_metrics_collection(self, cli_runner, test_config):
        """Test CLI metrics collection from running system."""
        from app.cli.unix_commands import hive_metrics
        
        result = cli_runner.invoke(hive_metrics, ['--format', 'json'])
        
        # Should succeed or fail gracefully
        assert result.exit_code in [0, 1]

    def test_cli_version_information(self, cli_runner, test_config):
        """Test CLI version command against running API."""
        from app.cli.unix_commands import hive_version
        
        result = cli_runner.invoke(hive_version, ['--format', 'json'])
        
        assert result.exit_code == 0
        if result.output.strip():
            import json
            version_data = json.loads(result.output)
            assert 'version' in version_data

    def test_cli_error_scenarios_with_real_api(self, cli_runner, test_config):
        """Test CLI error handling with real API error responses."""
        from app.cli.unix_commands import hive_get
        
        # Test with invalid resource (should handle gracefully)
        result = cli_runner.invoke(hive_get, ['nonexistent-resource'])
        
        # Should either succeed with empty results or fail with helpful error
        assert result.exit_code in [0, 1]

class TestCLIPerformanceWithRunningAPI:
    """Test CLI performance characteristics."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    @pytest.mark.performance
    def test_cli_response_times(self, cli_runner, running_api_server):
        """Test CLI command response times."""
        from app.cli.unix_commands import hive_status, hive_get, hive_version
        
        commands_to_test = [
            (hive_status, []),
            (hive_get, ['agents']),
            (hive_version, [])
        ]
        
        for command, args in commands_to_test:
            start_time = time.time()
            result = cli_runner.invoke(command, args)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # CLI commands should be responsive (< 5 seconds)
            assert response_time < 5.0, f"Command {command.name} took {response_time:.2f}s"
            
            # Log performance for monitoring
            print(f"Command {command.name}: {response_time:.3f}s")

    @pytest.mark.performance
    def test_cli_help_performance(self, cli_runner):
        """Test CLI help command performance (should be instant)."""
        from app.cli.main import hive_cli
        
        start_time = time.time()
        result = cli_runner.invoke(hive_cli, ['--help'])
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Help should be nearly instant
        assert response_time < 1.0, f"Help command took {response_time:.2f}s"
        assert result.exit_code == 0
```

### Phase 4: CLI Workflow and Usability Testing 🎯

**Objective**: Test complete user workflows and usability aspects

```python
# tests/cli/test_cli_workflows.py
import pytest
import tempfile
import json
from pathlib import Path
from click.testing import CliRunner

class TestCLIUsabilityAndWorkflows:
    """Test CLI usability and complete workflows."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_cli_help_quality_and_completeness(self, cli_runner):
        """Test quality and completeness of CLI help."""
        from app.cli.main import hive_cli
        from app.cli.unix_commands import hive_help
        
        # Main CLI help
        result = cli_runner.invoke(hive_cli, ['--help'])
        assert result.exit_code == 0
        
        help_text = result.output
        
        # Should include essential information
        required_sections = ["Usage:", "Commands:", "Options:"]
        for section in required_sections:
            assert section in help_text, f"Missing section: {section}"
        
        # Should mention key commands
        key_commands = ["status", "get", "create", "logs"]
        for command in key_commands:
            assert command in help_text.lower(), f"Missing command in help: {command}"
        
        # Test help command
        result = cli_runner.invoke(hive_help)
        assert result.exit_code == 0
        assert len(result.output) > 100  # Should provide substantial help

    def test_cli_error_message_quality(self, cli_runner):
        """Test quality of CLI error messages."""
        from app.cli.unix_commands import hive_get, hive_config
        
        # Test invalid format option
        result = cli_runner.invoke(hive_get, ['agents', '--output', 'invalid-format'])
        assert result.exit_code != 0
        
        error_text = result.output
        
        # Error should be helpful
        assert len(error_text) > 0
        assert not error_text.isspace()
        
        # Should suggest alternatives or provide guidance
        helpful_words = ["usage", "help", "valid", "available", "try", "options"]
        assert any(word in error_text.lower() for word in helpful_words)

    def test_cli_output_consistency(self, cli_runner):
        """Test consistency of CLI output formats."""
        from app.cli.unix_commands import hive_status, hive_get, hive_version
        
        json_commands = [
            (hive_status, ['--format', 'json']),
            (hive_get, ['agents', '--output', 'json']),
            (hive_version, ['--format', 'json'])
        ]
        
        for command, args in json_commands:
            result = cli_runner.invoke(command, args)
            
            if result.exit_code == 0 and result.output.strip():
                try:
                    json.loads(result.output)
                except json.JSONDecodeError:
                    pytest.fail(f"Command {command.name} produced invalid JSON")

    def test_cli_configuration_workflow(self, cli_runner):
        """Test complete configuration management workflow."""
        from app.cli.unix_commands import hive_config
        
        with cli_runner.isolated_filesystem():
            # Test setting configuration
            config_items = [
                ('api.base', 'http://custom:9000'),
                ('max_agents', '15'),
                ('log_level', 'DEBUG')
            ]
            
            for key, value in config_items:
                result = cli_runner.invoke(hive_config, [key, value])
                assert result.exit_code == 0, f"Failed to set {key}={value}"
            
            # Test getting configuration
            for key, expected_value in config_items:
                result = cli_runner.invoke(hive_config, [key])
                assert result.exit_code == 0
                assert expected_value in result.output
            
            # Test listing all configuration
            result = cli_runner.invoke(hive_config, ['--list'])
            assert result.exit_code == 0
            
            # Should show all our configured items
            for key, value in config_items:
                assert f"{key}={value}" in result.output

    def test_cli_workspace_interaction_workflow(self, cli_runner):
        """Test CLI workspace and project interaction patterns."""
        from app.cli.unix_commands import hive_init, hive_config
        
        with cli_runner.isolated_filesystem():
            # Test initialization
            result = cli_runner.invoke(hive_init, ['--quick'])
            assert result.exit_code == 0
            
            # Should create configuration
            result = cli_runner.invoke(hive_config, ['--list'])
            assert result.exit_code == 0
            assert "api_base" in result.output

class TestCLIAgentManagementWorkflow:
    """Test complete agent management workflows."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()
    
    @pytest.fixture
    def mock_agent_api(self):
        """Mock agent management API responses."""
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Create response sequence for agent management workflow
            responses = {
                'debug-agents': {
                    "agents": [
                        {
                            "id": "agent-001",
                            "status": "active", 
                            "active_tasks": 2,
                            "uptime": "2h 30m",
                            "capabilities": ["code", "test", "debug"]
                        }
                    ],
                    "total": 1
                },
                'api/agents/create': {
                    "success": True,
                    "agents": [
                        {
                            "id": "agent-new-001",
                            "status": "initializing",
                            "type": "worker"
                        }
                    ]
                },
                'api/agents/agent-new-001/scale': {
                    "success": True,
                    "message": "Agent scaled to 3 replicas"
                },
                'api/agents/agent-new-001/terminate': {
                    "success": True,
                    "message": "Agent terminated successfully"
                }
            }
            
            def mock_request(method, url, **kwargs):
                mock_response = Mock()
                mock_response.status_code = 200
                
                for endpoint, response_data in responses.items():
                    if endpoint in url:
                        mock_response.json.return_value = response_data
                        return mock_response
                
                mock_response.json.return_value = {"status": "ok"}
                return mock_response
            
            mock_requests.get.side_effect = lambda url, **kwargs: mock_request("GET", url, **kwargs)
            mock_requests.post.side_effect = lambda url, **kwargs: mock_request("POST", url, **kwargs)
            
            yield mock_requests

    def test_complete_agent_management_workflow(self, cli_runner, mock_agent_api):
        """Test end-to-end agent management workflow."""
        from app.cli.unix_commands import hive_get, hive_create, hive_scale, hive_kill
        
        # Step 1: List current agents
        result = cli_runner.invoke(hive_get, ['agents', '--output', 'table'])
        assert result.exit_code == 0
        
        # Step 2: Create new agent
        result = cli_runner.invoke(hive_create, [
            '--count', '1',
            '--type', 'worker'
        ])
        assert result.exit_code == 0
        assert "successfully" in result.output.lower()
        
        # Step 3: Scale agent (if supported)
        result = cli_runner.invoke(hive_scale, ['agent-new-001', '3'])
        assert result.exit_code == 0
        
        # Step 4: Check status after scaling
        result = cli_runner.invoke(hive_get, ['agents'])
        assert result.exit_code == 0
        
        # Step 5: Terminate agent
        result = cli_runner.invoke(hive_kill, ['agent-new-001', '--reason', 'test-cleanup'])
        assert result.exit_code == 0

class TestCLIMonitoringWorkflow:
    """Test monitoring and debugging workflows."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_monitoring_and_debugging_workflow(self, cli_runner):
        """Test complete monitoring and debugging workflow."""
        from app.cli.unix_commands import hive_status, hive_logs, hive_metrics, hive_debug, hive_doctor
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Mock healthy system responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "components": {
                    "database": {"status": "healthy"},
                    "redis": {"status": "healthy"},
                    "agents": {"count": 3, "active": 2}
                },
                "metrics": {
                    "requests_per_second": 45,
                    "memory_usage": "512MB",
                    "cpu_usage": "25%"
                },
                "logs": {
                    "entries": [
                        {
                            "timestamp": "2024-01-15T10:30:00Z",
                            "level": "INFO",
                            "message": "System healthy"
                        }
                    ]
                }
            }
            mock_requests.get.return_value = mock_response
            
            # Step 1: Check overall system status
            result = cli_runner.invoke(hive_status)
            assert result.exit_code == 0
            
            # Step 2: Get detailed metrics
            result = cli_runner.invoke(hive_metrics, ['--format', 'json'])
            assert result.exit_code == 0
            
            # Step 3: Check recent logs
            result = cli_runner.invoke(hive_logs, ['--lines', '20'])
            assert result.exit_code == 0
            
            # Step 4: Debug specific component
            result = cli_runner.invoke(hive_debug, ['system'])
            assert result.exit_code == 0
            
            # Step 5: Run health diagnostics
            result = cli_runner.invoke(hive_doctor)
            assert result.exit_code == 0
```

### Phase 5: CLI Performance and Error Handling Testing ⚡

**Objective**: Test CLI performance characteristics and error scenarios

```python
# tests/cli/test_cli_performance_errors.py
import pytest
import time
import threading
from click.testing import CliRunner
from unittest.mock import Mock, patch

class TestCLIPerformanceCharacteristics:
    """Test CLI performance and responsiveness."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    @pytest.mark.performance
    def test_cli_startup_time(self, cli_runner):
        """Test CLI startup and initialization time."""
        from app.cli.main import hive_cli
        
        # Test help command (should be instant)
        start_time = time.time()
        result = cli_runner.invoke(hive_cli, ['--help'])
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        assert result.exit_code == 0
        assert startup_time < 1.0, f"CLI startup took {startup_time:.2f}s"

    @pytest.mark.performance
    def test_command_response_times(self, cli_runner):
        """Test individual command response times."""
        from app.cli.unix_commands import hive_version, hive_help
        
        commands_to_test = [
            (hive_version, []),
            (hive_help, []),
        ]
        
        for command, args in commands_to_test:
            start_time = time.time()
            result = cli_runner.invoke(command, args)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Local commands should be very fast
            assert response_time < 2.0, f"{command.name} took {response_time:.2f}s"
            assert result.exit_code == 0

    @pytest.mark.performance
    def test_api_dependent_command_timeouts(self, cli_runner):
        """Test API-dependent commands with timeouts."""
        from app.cli.unix_commands import hive_status
        
        # Mock slow API response
        with patch('app.cli.unix_commands.requests') as mock_requests:
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # Simulate network delay
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "healthy"}
                return mock_response
            
            mock_requests.get.side_effect = slow_response
            
            start_time = time.time()
            result = cli_runner.invoke(hive_status)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should complete in reasonable time
            assert response_time < 10.0, f"Status command took {response_time:.2f}s"

class TestCLIErrorHandlingScenarios:
    """Test CLI error handling and recovery scenarios."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_network_error_handling(self, cli_runner):
        """Test handling of network connectivity errors."""
        from app.cli.unix_commands import hive_status
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Network unreachable")
            
            result = cli_runner.invoke(hive_status)
            
            # Should handle error gracefully, not crash
            assert result.exit_code in [0, 1]
            
            # Should provide helpful error message
            error_words = ["error", "failed", "connection", "network", "unavailable"]
            assert any(word in result.output.lower() for word in error_words)

    def test_api_server_down_handling(self, cli_runner):
        """Test handling when API server is down."""
        from app.cli.unix_commands import hive_get
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Simulate connection refused
            import requests
            mock_requests.get.side_effect = requests.ConnectionError("Connection refused")
            
            result = cli_runner.invoke(hive_get, ['agents'])
            
            # Should fail gracefully
            assert result.exit_code in [0, 1]
            
            # Should suggest solutions
            suggestion_words = ["start", "check", "running", "server", "available"]
            assert any(word in result.output.lower() for word in suggestion_words)

    def test_invalid_api_response_handling(self, cli_runner):
        """Test handling of invalid API responses."""
        from app.cli.unix_commands import hive_status
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            # Mock invalid JSON response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_requests.get.return_value = mock_response
            
            result = cli_runner.invoke(hive_status)
            
            # Should handle invalid response gracefully
            assert result.exit_code in [0, 1]

    def test_permission_denied_handling(self, cli_runner):
        """Test handling of permission denied errors."""
        from app.cli.unix_commands import hive_config
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            
            result = cli_runner.invoke(hive_config, ['api.base', 'http://test:8000'])
            
            assert result.exit_code in [1, 2]  # Should indicate error
            assert "permission" in result.output.lower()

    def test_disk_space_error_handling(self, cli_runner):
        """Test handling of disk space errors during config operations."""
        from app.cli.unix_commands import hive_config
        
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = OSError("No space left on device")
            
            result = cli_runner.invoke(hive_config, ['api.base', 'http://test:8000'])
            
            assert result.exit_code != 0
            assert any(word in result.output.lower() for word in ["space", "disk", "storage"])

    def test_concurrent_command_execution_safety(self, cli_runner):
        """Test safety of concurrent CLI command execution."""
        from app.cli.unix_commands import hive_version
        
        results = []
        errors = []
        
        def run_command():
            try:
                result = cli_runner.invoke(hive_version)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple commands concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=run_command)
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=10)
        
        # All should succeed
        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == 5
        assert all(result.exit_code == 0 for result in results)

class TestCLIEdgeCasesAndCornerCases:
    """Test CLI edge cases and corner case scenarios."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_extremely_long_command_arguments(self, cli_runner):
        """Test handling of extremely long command arguments."""
        from app.cli.unix_commands import hive_create
        
        # Test very long agent type name
        long_type = "a" * 1000
        result = cli_runner.invoke(hive_create, ['--type', long_type])
        
        # Should handle gracefully (either succeed or fail with helpful error)
        assert result.exit_code in [0, 1, 2]

    def test_unicode_and_special_characters(self, cli_runner):
        """Test handling of unicode and special characters."""
        from app.cli.unix_commands import hive_config
        
        with cli_runner.isolated_filesystem():
            # Test unicode in configuration values
            unicode_value = "测试-αβγ-🚀"
            result = cli_runner.invoke(hive_config, ['test_key', unicode_value])
            
            # Should handle unicode gracefully
            assert result.exit_code in [0, 1]

    def test_empty_and_null_input_handling(self, cli_runner):
        """Test handling of empty and null inputs."""
        from app.cli.unix_commands import hive_get
        
        # Test with empty resource name
        result = cli_runner.invoke(hive_get, [''])
        
        # Should handle empty input gracefully
        assert result.exit_code in [0, 1, 2]

    def test_configuration_file_corruption_recovery(self, cli_runner):
        """Test recovery from corrupted configuration files."""
        from app.cli.unix_commands import hive_config
        
        with cli_runner.isolated_filesystem():
            # Create corrupted config file
            config_dir = Path.home() / ".config" / "agent-hive"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_file = config_dir / "config.json"
            
            with open(config_file, 'w') as f:
                f.write("invalid json content {")
            
            # CLI should recover or provide helpful error
            result = cli_runner.invoke(hive_config, ['--list'])
            
            # Should either recover with defaults or provide clear error
            assert result.exit_code in [0, 1]
```

## Implementation Roadmap

### Week 1: Foundation & Unit Testing ✅
- **Fix existing CLI test imports** - Update test_cli_functionality.py 
- **CLI Unit Test Framework** - Command parsing, validation, help systems
- **Error Handling Tests** - Invalid arguments, permission errors, etc.
- **Performance Baseline** - CLI startup time, help command responsiveness

### Week 2: Integration Testing 🔌
- **Mock API Integration** - Test CLI with controlled API responses
- **Agent Lifecycle Workflows** - Create, scale, terminate agent sequences
- **Configuration Management** - End-to-end config workflow testing  
- **Error Scenario Testing** - Network failures, API downtime, invalid responses

### Week 3: End-to-End & Production Testing 🚀
- **Running API Integration** - CLI against real test API server
- **Complete User Workflows** - Full monitoring, debugging, management sequences
- **Performance Testing** - Response times, concurrent execution, stress scenarios
- **Usability Testing** - Help quality, error message clarity, output consistency

## Expected Outcomes & Success Metrics

### Testing Coverage Goals
- **CLI Unit Tests**: 95% coverage on command parsing and logic
- **CLI Integration Tests**: 90% coverage on API interaction patterns  
- **E2E Workflow Tests**: 100% coverage on primary user workflows
- **Error Handling Tests**: 100% coverage on failure scenarios

### Performance Targets
- **CLI Startup Time**: < 1 second for help/version commands
- **API Command Response**: < 5 seconds for network operations
- **Error Recovery**: Graceful handling with helpful messages
- **Concurrent Execution**: Safe multi-user CLI usage

### Quality Assurance Validation
- **Help System Quality**: Comprehensive, accurate, and helpful documentation
- **Error Message Quality**: Clear, actionable error messages with suggestions
- **Output Consistency**: Reliable JSON/table output formats
- **Configuration Persistence**: Robust config file management

## Integration with Existing Test Suite

This CLI testing strategy integrates seamlessly with the existing test foundation:

**Building On Solid Foundation:**
- ✅ **44 Unit Tests Passing** - Provides confidence in core functionality  
- ✅ **54 API Tests Passing** - Ensures CLI has stable API backend to test against
- ✅ **219 API Routes Discovered** - Rich API surface for CLI integration testing

**Test Execution Strategy:**
```bash
# Run complete test suite including new CLI tests
pytest tests/ -v --cov=app --cov-report=html

# Run only CLI tests
pytest tests/cli/ -v -m "not performance"

# Run CLI performance tests separately  
pytest tests/cli/ -v -m performance

# Run CLI integration tests with running API
pytest tests/cli/test_cli_e2e_integration.py -v -m integration
```

**CI/CD Integration:**
- CLI tests run in parallel with existing unit/API tests
- Performance tests run in dedicated CI environment
- E2E tests run against ephemeral test environments
- Results integrate with existing coverage reporting

This comprehensive CLI testing strategy ensures robust, reliable, and user-friendly command-line interfaces that maintain the high quality standards established by the existing test suite.
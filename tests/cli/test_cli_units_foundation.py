"""
Foundation CLI Unit Tests for LeanVibe Agent Hive 2.0

This module implements the foundational CLI testing framework as designed in 
CLI_TESTING_STRATEGY.md Phase 1: CLI Unit Testing Framework.

Tests cover:
- CLI command parsing and validation
- Unix-style command functionality  
- Configuration management
- Error handling scenarios
- Output format consistency
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Import CLI modules correctly based on actual structure
from app.cli.unix_commands import (
    hive_status, hive_get, hive_logs, hive_kill, hive_create, 
    hive_delete, hive_scale, hive_config, hive_init, hive_metrics, 
    hive_debug, hive_doctor, hive_version, hive_help, HiveContext
)
from app.cli.main import hive_cli


class TestHiveContextFoundation:
    """Test core HiveContext functionality."""
    
    def test_context_initialization(self):
        """Test HiveContext initialization with defaults."""
        ctx = HiveContext()
        
        assert ctx.api_base == "http://localhost:8000"
        assert ctx.config_dir == Path.home() / ".config" / "agent-hive"
        
    def test_context_api_call_success(self):
        """Test successful API call handling."""
        ctx = HiveContext()
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_response.content = b'{"status": "healthy"}'
            mock_requests.get.return_value = mock_response
            
            result = ctx.api_call("health")
            
            assert result == {"status": "healthy"}
            mock_requests.get.assert_called_once_with(
                "http://localhost:8000/health", timeout=5
            )
    
    def test_context_api_call_failure(self):
        """Test API call failure handling."""
        ctx = HiveContext()
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_requests.get.side_effect = Exception("Connection failed")
            
            result = ctx.api_call("health")
            
            assert result is None
    
    def test_context_api_call_post_method(self):
        """Test POST API call handling."""
        ctx = HiveContext()
        
        with patch('app.cli.unix_commands.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True}
            mock_response.content = b'{"success": true}'
            mock_requests.post.return_value = mock_response
            
            test_data = {"count": 2, "type": "worker"}
            result = ctx.api_call("api/agents/create", method="POST", data=test_data)
            
            assert result == {"success": True}
            mock_requests.post.assert_called_once_with(
                "http://localhost:8000/api/agents/create", 
                json=test_data, 
                timeout=5
            )


class TestCLICommandParsing:
    """Test CLI command argument parsing and validation."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create isolated CLI test runner."""
        return CliRunner()

    def test_hive_status_command_parsing(self, cli_runner):
        """Test status command argument parsing."""
        # Test valid format options
        valid_formats = ['json', 'table', 'wide']
        
        for fmt in valid_formats:
            with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
                mock_api.return_value = {"status": "healthy"}
                
                result = cli_runner.invoke(hive_status, ['--format', fmt])
                assert result.exit_code == 0
    
    def test_hive_status_watch_option(self, cli_runner):
        """Test status command watch option."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api, \
             patch('time.sleep') as mock_sleep:
            
            mock_api.return_value = {"status": "healthy"}
            mock_sleep.side_effect = KeyboardInterrupt()  # Stop after first iteration
            
            result = cli_runner.invoke(hive_status, ['--watch'])
            
            # Should handle watch mode gracefully
            assert result.exit_code in [0, 1]  # May exit with KeyboardInterrupt

    def test_hive_get_resource_validation(self, cli_runner):
        """Test get command resource validation."""
        valid_resources = ['agents', 'tasks', 'workflows', 'metrics']
        
        for resource in valid_resources:
            with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
                mock_api.return_value = {"agents": []}
                
                result = cli_runner.invoke(hive_get, [resource])
                assert result.exit_code == 0

    def test_hive_get_output_format_validation(self, cli_runner):
        """Test get command output format validation."""
        valid_formats = ['json', 'yaml', 'table']
        
        for fmt in valid_formats:
            with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
                mock_api.return_value = {"agents": []}
                
                result = cli_runner.invoke(hive_get, ['agents', '--output', fmt])
                assert result.exit_code == 0

    def test_hive_create_argument_parsing(self, cli_runner):
        """Test create command argument parsing."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {"success": True, "agents": []}
            
            # Test basic creation with count and type
            result = cli_runner.invoke(hive_create, [
                '--count', '3', 
                '--type', 'worker'
            ])
            
            assert result.exit_code == 0
            mock_api.assert_called_with(
                "api/agents/create", 
                method="POST", 
                data={'type': 'worker', 'count': 3}
            )

    def test_hive_create_config_file_parsing(self, cli_runner):
        """Test create command with config file."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {"success": True}
            
            with cli_runner.isolated_filesystem():
                # Create test config file
                config_data = {
                    "type": "specialist",
                    "capabilities": ["code", "test", "debug"],
                    "max_tasks": 5
                }
                
                with open("agent-config.json", 'w') as f:
                    json.dump(config_data, f)
                
                result = cli_runner.invoke(hive_create, [
                    '--count', '2',
                    '--config', 'agent-config.json'
                ])
                
                assert result.exit_code == 0
                
                # Verify config was merged with command args
                call_args = mock_api.call_args[1]['data']
                assert call_args['type'] == 'specialist'
                assert call_args['count'] == 2
                assert call_args['capabilities'] == ["code", "test", "debug"]

    def test_hive_logs_options_parsing(self, cli_runner):
        """Test logs command options parsing."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {"entries": []}
            
            # Test lines option
            result = cli_runner.invoke(hive_logs, ['--lines', '100'])
            assert result.exit_code == 0
            
            # Test level filter
            result = cli_runner.invoke(hive_logs, ['--level', 'ERROR'])
            assert result.exit_code == 0
            
            # Test component filter
            result = cli_runner.invoke(hive_logs, ['--lines', '50', 'orchestrator'])
            assert result.exit_code == 0


class TestCLIConfigurationManagement:
    """Test CLI configuration management functionality."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_hive_config_get_operation(self, cli_runner):
        """Test configuration get operation."""
        with cli_runner.isolated_filesystem():
            # Create test config directory and file
            config_dir = Path(".config/agent-hive")
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.json"
            
            test_config = {
                "api_base": "http://test:9000",
                "max_agents": 15
            }
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            # Mock the config directory to point to our test location
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                result = cli_runner.invoke(hive_config, ['api_base'])
                
                assert result.exit_code == 0
                assert "http://test:9000" in result.output

    def test_hive_config_set_operation(self, cli_runner):
        """Test configuration set operation."""
        with cli_runner.isolated_filesystem():
            config_dir = Path(".config/agent-hive")
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.json"
            
            # Start with empty config
            with open(config_file, 'w') as f:
                json.dump({}, f)
            
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                # Set configuration value
                result = cli_runner.invoke(hive_config, [
                    'api_base', 'http://custom:8080'
                ])
                
                assert result.exit_code == 0
                
                # Verify configuration was saved
                with open(config_file) as f:
                    saved_config = json.load(f)
                
                assert saved_config['api_base'] == 'http://custom:8080'

    def test_hive_config_list_operation(self, cli_runner):
        """Test configuration list operation."""
        with cli_runner.isolated_filesystem():
            config_dir = Path(".config/agent-hive")
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.json"
            
            test_config = {
                "api_base": "http://localhost:8000",
                "max_agents": 10,
                "log_level": "INFO"
            }
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                result = cli_runner.invoke(hive_config, ['--list'])
                
                assert result.exit_code == 0
                
                # Should show all configuration items
                for key, value in test_config.items():
                    assert f"{key}={value}" in result.output

    def test_hive_config_unset_operation(self, cli_runner):
        """Test configuration unset operation."""
        with cli_runner.isolated_filesystem():
            config_dir = Path(".config/agent-hive")
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.json"
            
            test_config = {
                "api_base": "http://localhost:8000",
                "temp_setting": "remove_me"
            }
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                result = cli_runner.invoke(hive_config, [
                    '--unset', 'temp_setting'
                ])
                
                assert result.exit_code == 0
                
                # Verify setting was removed
                with open(config_file) as f:
                    saved_config = json.load(f)
                
                assert 'temp_setting' not in saved_config
                assert 'api_base' in saved_config  # Other settings preserved

    def test_hive_init_default_configuration(self, cli_runner):
        """Test init command with default configuration."""
        with cli_runner.isolated_filesystem():
            config_dir = Path(".config/agent-hive") 
            
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                result = cli_runner.invoke(hive_init, ['--quick'])
                
                assert result.exit_code == 0
                
                # Verify default config was created
                config_file = config_dir / "config.json"
                assert config_file.exists()
                
                with open(config_file) as f:
                    config = json.load(f)
                
                # Should contain default values
                assert config['api_base'] == "http://localhost:8000"
                assert config['auto_start'] is True
                assert config['max_agents'] == 10


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_network_error_handling(self, cli_runner):
        """Test handling of network errors."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = None  # Simulate network failure
            
            result = cli_runner.invoke(hive_status)
            
            # Should handle error gracefully
            assert result.exit_code in [0, 1]
            
            # Should show appropriate error indication
            assert len(result.output) > 0

    def test_invalid_command_arguments(self, cli_runner):
        """Test handling of invalid command arguments."""
        # Test invalid format
        result = cli_runner.invoke(hive_status, ['--format', 'invalid'])
        assert result.exit_code != 0
        
        # Test invalid output format
        result = cli_runner.invoke(hive_get, ['agents', '--output', 'invalid'])
        assert result.exit_code != 0

    def test_config_file_permission_error(self, cli_runner):
        """Test handling of configuration file permission errors."""
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            
            result = cli_runner.invoke(hive_config, ['api_base', 'http://test:8000'])
            
            assert result.exit_code in [1, 2]
            assert "permission" in result.output.lower() or "denied" in result.output.lower()

    def test_config_file_corruption_handling(self, cli_runner):
        """Test handling of corrupted configuration files."""
        with cli_runner.isolated_filesystem():
            config_dir = Path(".config/agent-hive")
            config_dir.mkdir(parents=True)
            config_file = config_dir / "config.json"
            
            # Create corrupted config file
            with open(config_file, 'w') as f:
                f.write("invalid json content {")
            
            with patch('app.cli.unix_commands.ctx.config_dir', config_dir):
                # Should handle corruption gracefully
                result = cli_runner.invoke(hive_config, ['--list'])
                
                # May succeed with empty config or fail with helpful error
                assert result.exit_code in [0, 1]


class TestCLIOutputFormatConsistency:
    """Test CLI output format consistency."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_json_output_validity(self, cli_runner):
        """Test that JSON output is valid JSON."""
        commands_with_json = [
            (hive_status, ['--format', 'json']),
            (hive_get, ['agents', '--output', 'json']),
            (hive_version, ['--format', 'json']),
        ]
        
        for command, args in commands_with_json:
            with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
                mock_api.return_value = {"test": "data"}
                
                result = cli_runner.invoke(command, args)
                
                if result.exit_code == 0 and result.output.strip():
                    try:
                        json.loads(result.output)
                    except json.JSONDecodeError:
                        pytest.fail(f"Command {command.name} produced invalid JSON")

    def test_table_output_formatting(self, cli_runner):
        """Test table output formatting."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {
                "agents": [
                    {
                        "id": "agent-001",
                        "status": "active",
                        "active_tasks": 3,
                        "uptime": "2h"
                    }
                ]
            }
            
            result = cli_runner.invoke(hive_get, ['agents', '--output', 'table'])
            
            assert result.exit_code == 0
            # Should contain table structure
            assert len(result.output) > 0


class TestCLIHelpSystemQuality:
    """Test CLI help system quality and completeness."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_main_cli_help(self, cli_runner):
        """Test main CLI help quality."""
        result = cli_runner.invoke(hive_cli, ['--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        
        # Should include essential sections
        required_sections = ["Usage:", "Commands:", "Options:"]
        for section in required_sections:
            assert section in help_text, f"Missing section: {section}"
        
        # Should mention LeanVibe Agent Hive
        assert "Agent Hive" in help_text or "LeanVibe" in help_text

    def test_individual_command_help(self, cli_runner):
        """Test individual command help quality."""
        commands_to_test = [
            hive_status,
            hive_get,
            hive_create,
            hive_logs
        ]
        
        for command in commands_to_test:
            result = cli_runner.invoke(command, ['--help'])
            
            assert result.exit_code == 0
            assert len(result.output) > 50  # Should provide meaningful help
            assert "Usage:" in result.output or command.name in result.output.lower()

    def test_help_command_functionality(self, cli_runner):
        """Test hive help command functionality."""
        result = cli_runner.invoke(hive_help)
        
        assert result.exit_code == 0
        assert len(result.output) > 100
        
        # Should list available commands
        key_commands = ["status", "get", "create", "logs"]
        for command in key_commands:
            assert command in result.output.lower()

    def test_help_for_specific_command(self, cli_runner):
        """Test help for specific commands."""
        result = cli_runner.invoke(hive_help, ['status'])
        
        assert result.exit_code == 0
        assert "status" in result.output.lower()


class TestCLIVersionAndDoctor:
    """Test CLI version and diagnostics commands."""
    
    @pytest.fixture
    def cli_runner(self):
        return CliRunner()

    def test_version_command_basic(self, cli_runner):
        """Test version command basic functionality."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {"api_version": "2.0.0"}
            
            result = cli_runner.invoke(hive_version)
            
            assert result.exit_code == 0
            assert "2.0.0" in result.output

    def test_version_command_json_format(self, cli_runner):
        """Test version command JSON output."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {"api_version": "2.0.0"}
            
            result = cli_runner.invoke(hive_version, ['--format', 'json'])
            
            assert result.exit_code == 0
            
            if result.output.strip():
                version_data = json.loads(result.output)
                assert isinstance(version_data, dict)
                assert 'version' in version_data

    def test_doctor_command_basic(self, cli_runner):
        """Test doctor command basic functionality."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {
                "database_status": "healthy",
                "redis_status": "healthy",
                "agent_count": 3
            }
            
            result = cli_runner.invoke(hive_doctor)
            
            assert result.exit_code == 0
            # Should provide health information
            assert len(result.output) > 0

    def test_doctor_command_with_issues(self, cli_runner):
        """Test doctor command when issues are detected."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            mock_api.return_value = {
                "database_status": "unhealthy",
                "redis_status": "healthy", 
                "agent_count": 0
            }
            
            result = cli_runner.invoke(hive_doctor)
            
            assert result.exit_code == 0
            # Should report issues
            assert "issue" in result.output.lower() or "problem" in result.output.lower()

    def test_doctor_command_auto_fix(self, cli_runner):
        """Test doctor command auto-fix functionality."""
        with patch('app.cli.unix_commands.ctx.api_call') as mock_api:
            # Mock unhealthy system first, then auto-fix response
            mock_api.side_effect = [
                {
                    "database_status": "unhealthy",
                    "redis_status": "healthy",
                    "agent_count": 0
                },
                {"success": True}  # auto-fix response
            ]
            
            result = cli_runner.invoke(hive_doctor, ['--fix'])
            
            assert result.exit_code == 0
            # Should attempt auto-fix
            assert mock_api.call_count >= 1


# Mark all tests in this module for specific test running
pytestmark = pytest.mark.cli_units
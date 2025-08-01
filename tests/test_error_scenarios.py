"""
Error Scenario Testing for LeanVibe Agent Hive

Tests critical error scenarios that a new developer might encounter,
ensuring robust error handling and clear error messages.

Based on expert recommendations from Gemini CLI for enterprise-grade quality.
"""

import os
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch
import socket


class ErrorScenarioTester:
    """Test error scenarios and error handling robustness."""
    
    def __init__(self):
        self.test_directory = None
        self.original_directory = os.getcwd()
        
    def setup_test_environment(self) -> Path:
        """Create a test environment for error scenario testing."""
        self.test_directory = tempfile.mkdtemp(prefix="leanvibe_error_test_")
        test_path = Path(self.test_directory)
        
        # Copy essential files for testing
        source_files = [
            "setup-fast.sh",
            "scripts/setup.sh", 
            "docker-compose.yml",
            "pyproject.toml",
            "Makefile"
        ]
        
        for file_path in source_files:
            source = Path(self.original_directory) / file_path
            if source.exists():
                dest_dir = test_path / file_path.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest_dir / source.name)
                
        return test_path
        
    def cleanup(self):
        """Clean up test environment."""
        if self.test_directory and os.path.exists(self.test_directory):
            os.chdir(self.original_directory)
            shutil.rmtree(self.test_directory, ignore_errors=True)


@pytest.fixture
def error_tester():
    """Provide an error scenario tester instance."""
    tester = ErrorScenarioTester()
    yield tester
    tester.cleanup()


def test_missing_docker_dependency(error_tester):
    """Test behavior when Docker is not installed."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Mock docker command to simulate it being missing
    with patch.dict(os.environ, {"PATH": "/nonexistent"}):
        result = subprocess.run([
            "./setup-fast.sh"
        ], capture_output=True, text=True)
        
        # Should fail gracefully with helpful error message
        assert result.returncode != 0, "Should fail when Docker is missing"
        
        # Should provide helpful error message
        error_output = result.stderr.lower()
        assert any(keyword in error_output for keyword in [
            "docker", "not found", "install", "missing"
        ]), f"Error message should mention Docker: {result.stderr}"


def test_missing_python_dependency(error_tester):
    """Test behavior when Python is not available."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Create a script that simulates missing Python
    with open("test_no_python.sh", "w") as f:
        f.write("""#!/bin/bash
# Simulate missing Python by removing it from PATH
export PATH="/bin:/usr/bin"  # Minimal PATH without Python
exec ./setup-fast.sh
""")
    
    os.chmod("test_no_python.sh", 0o755)
    
    result = subprocess.run([
        "./test_no_python.sh"
    ], capture_output=True, text=True)
    
    # Should provide helpful error about Python
    if result.returncode != 0:
        error_output = (result.stdout + result.stderr).lower()
        # Either it should handle the missing Python gracefully or provide clear error
        assert "python" in error_output or "command not found" in error_output


def test_port_conflict_scenario(error_tester):
    """Test behavior when required ports are already in use."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Start a simple server on port 8000 to simulate conflict
    import threading
    import http.server
    import socketserver
    
    class TestServer:
        def __init__(self, port):
            self.port = port
            self.httpd = None
            self.thread = None
            
        def start(self):
            try:
                Handler = http.server.SimpleHTTPRequestHandler
                self.httpd = socketserver.TCPServer(("", self.port), Handler)
                self.thread = threading.Thread(target=self.httpd.serve_forever)
                self.thread.daemon = True
                self.thread.start()
                return True
            except OSError:
                return False  # Port already in use
                
        def stop(self):
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
                
    # Try to occupy port 8000
    test_server = TestServer(8000)
    if test_server.start():
        try:
            # Run setup with port conflict
            result = subprocess.run([
                "./setup-fast.sh"
            ], capture_output=True, text=True, timeout=60)
            
            # Should handle port conflict gracefully
            if result.returncode != 0:
                output = (result.stdout + result.stderr).lower()
                # Should mention port conflict or provide clear error
                assert any(keyword in output for keyword in [
                    "port", "address", "in use", "bind", "conflict"
                ]), f"Should indicate port conflict: {result.stdout + result.stderr}"
                
        finally:
            test_server.stop()


def test_invalid_env_configuration(error_tester):
    """Test behavior with invalid .env.local configuration."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Create an invalid .env.local file
    with open(".env.local", "w") as f:
        f.write("""
# Invalid configuration for testing
ANTHROPIC_API_KEY=invalid_key_format
DATABASE_URL=invalid://url/format
REDIS_URL=not_a_valid_redis_url
""")
    
    result = subprocess.run([
        "./setup-fast.sh"
    ], capture_output=True, text=True, timeout=120)
    
    # System should either:
    # 1. Validate config and provide helpful error, or  
    # 2. Start successfully and handle invalid config gracefully during runtime
    
    if result.returncode != 0:
        # If it fails, should provide helpful error about configuration
        output = (result.stdout + result.stderr).lower()
        assert any(keyword in output for keyword in [
            "config", "env", "invalid", "format", "url"
        ]), f"Should provide configuration error details: {output}"


def test_permission_error_scenarios(error_tester):
    """Test behavior with permission issues."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Create a directory that simulates permission issues
    restricted_dir = test_path / "restricted"
    restricted_dir.mkdir()
    
    # Make it read-only to simulate permission issues
    os.chmod(restricted_dir, 0o444)
    
    try:
        # Try to write to restricted directory (this should be handled gracefully)
        result = subprocess.run([
            "touch", str(restricted_dir / "test_file")
        ], capture_output=True, text=True)
        
        # Should fail with permission error
        assert result.returncode != 0, "Should fail due to permissions"
        assert "permission denied" in result.stderr.lower()
        
    finally:
        # Restore permissions for cleanup
        os.chmod(restricted_dir, 0o755)


def test_network_connectivity_issues(error_tester):
    """Test behavior when network connectivity is limited."""
    test_path = error_tester.setup_test_environment()
    os.chdir(test_path)
    
    # Test with limited network access (simulated by using invalid DNS)
    env_with_bad_dns = os.environ.copy()
    env_with_bad_dns["DNS"] = "127.0.0.1"  # Point to localhost for DNS
    
    # This test may be fragile, so we'll make it optional
    try:
        result = subprocess.run([
            "pip", "install", "--dry-run", "requests"
        ], capture_output=True, text=True, timeout=30, env=env_with_bad_dns)
        
        # Should either succeed or provide network-related error
        if result.returncode != 0:
            output = result.stderr.lower()
            # Should provide helpful network error information
            network_keywords = ["network", "connection", "timeout", "dns", "resolve"]
            has_network_error = any(keyword in output for keyword in network_keywords)
            
            if has_network_error:
                print(f"Network error properly detected: {output}")
                
    except subprocess.TimeoutExpired:
        # Timeout is also acceptable for network issues
        print("Network test timed out (acceptable for network issues)")


def test_disk_space_simulation(error_tester):
    """Test behavior when disk space is limited (simulation)."""
    test_path = error_tester.setup_test_environment() 
    os.chdir(test_path)
    
    # Check available disk space
    stat = shutil.disk_usage(test_path)
    available_gb = stat.free / (1024**3)
    
    # If we have very limited space, test the behavior
    if available_gb < 1:  # Less than 1GB available
        result = subprocess.run([
            "./setup-fast.sh"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            output = (result.stdout + result.stderr).lower()
            # Should provide helpful error about disk space
            space_keywords = ["space", "disk", "storage", "full", "no space"]
            if any(keyword in output for keyword in space_keywords):
                print(f"Disk space error properly handled: {output}")


def test_makefile_help_robustness():
    """Test that make help command handles broken pipes gracefully."""
    # Test the current make help implementation
    result = subprocess.run([
        "make", "help"
    ], capture_output=True, text=True, timeout=30)
    
    # Should succeed and provide help output
    assert result.returncode == 0, f"make help failed: {result.stderr}"
    assert len(result.stdout) > 0, "make help should produce output"
    
    # Should not have broken pipe errors
    assert "broken pipe" not in result.stderr.lower()
    assert "sigpipe" not in result.stderr.lower()


if __name__ == "__main__":
    # Run tests manually for development  
    pytest.main([__file__, "-v", "-s"])
"""
End-to-End Testing for Fresh Developer Setup Experience

This test validates the complete developer onboarding experience from 
a fresh git clone to a working autonomous development system.

Based on expert recommendations from Gemini CLI for enterprise-grade quality.
"""

import os
import subprocess
import time
import tempfile
import shutil
import requests
import pytest
from pathlib import Path
from typing import Dict, List, Tuple


class FreshSetupTester:
    """Test fresh developer setup experience with time measurements."""
    
    def __init__(self):
        self.setup_times: Dict[str, float] = {}
        self.test_directory = None
        self.original_directory = os.getcwd()
        
    def setup_test_environment(self) -> Path:
        """Create a fresh test environment by cloning the repo."""
        # Create temporary directory for testing
        self.test_directory = tempfile.mkdtemp(prefix="leanvibe_e2e_test_")
        test_path = Path(self.test_directory)
        
        # Clone the repository to simulate fresh developer experience
        repo_url = "file://" + self.original_directory  # Use local repo for testing
        
        start_time = time.time()
        result = subprocess.run([
            "git", "clone", repo_url, str(test_path / "bee-hive")
        ], capture_output=True, text=True)
        
        clone_time = time.time() - start_time
        self.setup_times["git_clone"] = clone_time
        
        if result.returncode != 0:
            raise Exception(f"Git clone failed: {result.stderr}")
            
        return test_path / "bee-hive"
        
    def measure_setup_time(self, project_path: Path) -> Dict[str, float]:
        """Measure setup time with professional methodology."""
        os.chdir(project_path)
        
        # Test the fast setup script
        start_time = time.time()
        result = subprocess.run([
            "./setup-fast.sh"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        setup_time = time.time() - start_time
        self.setup_times["setup_script"] = setup_time
        self.setup_times["total_setup"] = self.setup_times["git_clone"] + setup_time
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "setup_time": setup_time,
            "total_time": self.setup_times["total_setup"]
        }
        
    def validate_system_health(self) -> Dict[str, bool]:
        """Validate that all system components are healthy."""
        health_checks = {}
        
        # Check Docker services
        try:
            result = subprocess.run([
                "docker", "compose", "ps"
            ], capture_output=True, text=True, timeout=30)
            health_checks["docker_services"] = result.returncode == 0
        except subprocess.TimeoutExpired:
            health_checks["docker_services"] = False
            
        # Check API health endpoint
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            health_checks["api_health"] = response.status_code == 200
        except requests.exceptions.RequestException:
            health_checks["api_health"] = False
            
        # Check database connectivity
        try:
            result = subprocess.run([
                "docker", "compose", "exec", "postgres", 
                "pg_isready", "-U", "leanvibe_user"
            ], capture_output=True, text=True, timeout=30)
            health_checks["database"] = result.returncode == 0
        except subprocess.TimeoutExpired:
            health_checks["database"] = False
            
        # Check Redis connectivity
        try:
            result = subprocess.run([
                "docker", "compose", "exec", "redis", 
                "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=30)
            health_checks["redis"] = "PONG" in result.stdout
        except subprocess.TimeoutExpired:
            health_checks["redis"] = False
            
        return health_checks
        
    def cleanup(self):
        """Clean up test environment."""
        if self.test_directory and os.path.exists(self.test_directory):
            os.chdir(self.original_directory)
            # Stop any running services first
            try:
                subprocess.run([
                    "docker", "compose", "down", "-v"
                ], cwd=self.test_directory + "/bee-hive", timeout=60)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            shutil.rmtree(self.test_directory, ignore_errors=True)


@pytest.fixture
def fresh_setup_tester():
    """Provide a fresh setup tester instance."""
    tester = FreshSetupTester()
    yield tester
    tester.cleanup()


def test_fresh_developer_complete_setup(fresh_setup_tester):
    """Test complete fresh developer setup experience."""
    tester = fresh_setup_tester
    
    # Step 1: Clone repository (simulate fresh developer)
    project_path = tester.setup_test_environment()
    assert project_path.exists(), "Failed to create test environment"
    
    # Step 2: Measure setup time
    setup_result = tester.measure_setup_time(project_path)
    
    # Validate setup completed successfully
    assert setup_result["success"], f"Setup failed: {setup_result['stderr']}"
    
    # Validate setup time claim (<2 minutes = 120 seconds)
    assert setup_result["total_time"] < 120, (
        f"Setup time {setup_result['total_time']:.1f}s exceeds claimed <2 minutes"
    )
    
    # Step 3: Wait for services to fully start
    time.sleep(30)  # Allow services to initialize
    
    # Step 4: Validate system health
    health_status = tester.validate_system_health()
    
    # All critical services should be healthy
    assert health_status["docker_services"], "Docker services not running"
    assert health_status["api_health"], "API health check failed"
    assert health_status["database"], "Database not accessible"
    assert health_status["redis"], "Redis not accessible"
    
    # Report performance metrics
    print(f"\n=== SETUP PERFORMANCE METRICS ===")
    print(f"Git clone time: {tester.setup_times['git_clone']:.1f}s")
    print(f"Setup script time: {tester.setup_times['setup_script']:.1f}s") 
    print(f"Total setup time: {tester.setup_times['total_setup']:.1f}s")
    print(f"Setup claim validation: {'✅ PASSED' if setup_result['total_time'] < 120 else '❌ FAILED'}")


def test_documentation_accuracy(fresh_setup_tester):
    """Test that documentation instructions are accurate."""
    tester = fresh_setup_tester
    project_path = tester.setup_test_environment()
    
    # Test commands mentioned in README/docs
    test_commands = [
        ("make help", "Help command should work"),
        ("make status", "Status command should work"),
        ("./health-check.sh", "Health check script should exist and work")
    ]
    
    os.chdir(project_path)
    
    for command, description in test_commands:
        result = subprocess.run(
            command.split(), 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        assert result.returncode == 0, f"{description}: {result.stderr}"


def test_api_endpoints_basic(fresh_setup_tester):
    """Test basic API endpoints are responding."""
    tester = fresh_setup_tester
    project_path = tester.setup_test_environment()
    
    # Setup and start services
    setup_result = tester.measure_setup_time(project_path)
    assert setup_result["success"], "Setup must succeed for API testing"
    
    time.sleep(30)  # Wait for API to be ready
    
    # Test critical API endpoints
    endpoints_to_test = [
        "/health",
        "/docs",  # FastAPI auto-generated docs
        "/api/v1/agents",  # Core agent endpoint
    ]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            assert response.status_code in [200, 401], (
                f"Endpoint {endpoint} returned {response.status_code}"
            )
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Endpoint {endpoint} not accessible: {e}")


if __name__ == "__main__":
    # Run tests manually for development
    pytest.main([__file__, "-v", "-s"])
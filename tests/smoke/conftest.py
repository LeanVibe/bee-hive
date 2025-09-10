"""
Smoke Tests Configuration

Specialized configuration for smoke tests with optimizations for:
- Fast execution (<30 seconds total)
- Reliable test isolation
- Minimal resource usage
- Clear failure reporting
"""

import pytest
import asyncio
import os
from typing import AsyncGenerator

# Base fixtures are automatically available from parent conftest.py

# Database fixtures for smoke tests
@pytest.fixture
async def test_engine():
    """Create test database engine with proper configuration for smoke tests."""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool
    
    # Use SQLite for fast smoke tests - no complex schema needed for connection pool tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # No need to create complex tables for connection pool testing
    # The test just needs to execute "SELECT 1" queries
    
    yield engine
    
    # Cleanup
    await engine.dispose()

# Override environment for smoke tests
@pytest.fixture(autouse=True, scope="session")
def smoke_test_environment():
    """Configure environment specifically for smoke tests."""
    os.environ.update({
        "TESTING": "true",
        "LOG_LEVEL": "ERROR",  # Reduce noise
        "SKIP_STARTUP_INIT": "true",
        "DISABLE_METRICS": "true",  # Speed up tests
        "CACHE_DISABLED": "true",  # Avoid cache interactions
        "WEBHOOKS_DISABLED": "true",  # Avoid external calls
        "SMTP_DISABLED": "true",  # Avoid email sending
        "PERFORMANCE_MONITORING_DISABLED": "true",
    })


# Smoke test specific fixtures
@pytest.fixture(scope="function")
def performance_timer():
    """Utility fixture for timing test operations."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
            return self
        
        def stop(self):
            self.end_time = time.time()
            return self
        
        @property
        def elapsed_ms(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return (self.end_time - self.start_time) * 1000
    
    return Timer()


@pytest.fixture(scope="function")
async def isolated_orchestrator():
    """Create isolated orchestrator instance for testing."""
    from app.core.simple_orchestrator import create_simple_orchestrator
    
    orchestrator = create_simple_orchestrator()
    yield orchestrator
    
    # Clean up any agents
    try:
        status = await orchestrator.get_system_status()
        agent_details = status.get("agents", {}).get("details", {})
        for agent_id in agent_details.keys():
            try:
                await orchestrator.shutdown_agent(agent_id, graceful=True)
            except Exception:
                pass  # Best effort cleanup
    except Exception:
        pass  # Orchestrator might be in error state


# Performance assertion helpers
def assert_performance(operation_time_ms: float, target_ms: float, operation_name: str = "Operation"):
    """Helper to assert performance targets with clear messages."""
    assert operation_time_ms < target_ms, (
        f"{operation_name} took {operation_time_ms:.2f}ms, "
        f"exceeds {target_ms}ms performance target"
    )


def assert_response_valid(response, expected_status_codes=None):
    """Helper to validate API responses."""
    if expected_status_codes is None:
        expected_status_codes = [200]
    
    assert response.status_code in expected_status_codes, (
        f"Response status {response.status_code} not in expected {expected_status_codes}"
    )
    
    # Basic response validation
    if response.status_code == 200:
        try:
            data = response.json()
            assert data is not None
        except Exception as e:
            # Some endpoints might return non-JSON
            if response.headers.get("content-type", "").startswith("application/json"):
                raise AssertionError(f"Expected JSON response but got parse error: {e}")


# Test collection hooks for better organization
def pytest_collection_modifyitems(config, items):
    """Modify test collection for smoke tests."""
    # Add markers based on test names/paths
    for item in items:
        # Add performance marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add fast marker to tests that should be very quick
        if any(keyword in item.name.lower() for keyword in ["health", "status", "ping"]):
            item.add_marker(pytest.mark.fast)
        
        # Add slow marker to integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add API marker to API tests
        if "api" in item.name.lower() or "endpoint" in item.name.lower():
            item.add_marker(pytest.mark.api)
        
        # Add orchestrator marker
        if "orchestrator" in item.name.lower():
            item.add_marker(pytest.mark.orchestrator)
        
        # Add database marker
        if "database" in item.name.lower() or "db" in item.name.lower():
            item.add_marker(pytest.mark.database)
        
        # Add redis marker
        if "redis" in item.name.lower():
            item.add_marker(pytest.mark.redis)


# Reporting hooks for better test output
def pytest_report_header(config):
    """Add custom header to test report."""
    return [
        "LeanVibe Agent Hive 2.0 - Smoke Test Suite",
        "=========================================",
        "Target: <30 seconds total execution time",
        "Focus: Critical functionality validation",
        ""
    ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary to test report."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        errors = len(terminalreporter.stats.get('error', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        terminalreporter.write_sep("=", "SMOKE TEST SUMMARY")
        terminalreporter.write_line(f"✅ Passed: {passed}")
        terminalreporter.write_line(f"❌ Failed: {failed}")
        terminalreporter.write_line(f"🚫 Errors: {errors}")
        terminalreporter.write_line(f"⏭️  Skipped: {skipped}")
        
        if exitstatus == 0:
            terminalreporter.write_line("\n🎉 All smoke tests passed! System is ready for use.")
        else:
            terminalreporter.write_line("\n⚠️  Some smoke tests failed. Check system health before proceeding.")

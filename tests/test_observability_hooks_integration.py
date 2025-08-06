"""
Comprehensive test suite for LeanVibe Agent Hive 2.0 Observability Hooks Integration

Tests the complete hook system including script execution, event processing,
database integration, and performance optimization.
"""

import asyncio
import json
import os
import pytest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import test components
from app.observability.hooks.hooks_config import HookConfig, get_hook_config, reload_hook_config
from app.observability.hooks.hooks_integration import (
    HookEventProcessor,
    HookScriptExecutor, 
    HookIntegrationManager,
    get_hook_integration_manager
)
from app.models.observability import AgentEvent, EventType


class TestHookConfig:
    """Test hook configuration system."""
    
    def test_hook_config_initialization(self):
        """Test basic hook configuration initialization."""
        config = HookConfig()
        
        assert config.environment in ("development", "production", "testing")
        assert config.performance is not None
        assert config.security is not None
        assert config.integration is not None
        assert config.session is not None
    
    def test_environment_specific_config(self):
        """Test environment-specific configuration loading."""
        # Test production config
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            prod_config = HookConfig()
            assert prod_config.environment == "production"
            assert prod_config.performance.slow_tool_threshold_ms == 3000
            assert prod_config.performance.error_rate_threshold_percent == 5.0
        
        # Test development config
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            dev_config = HookConfig()
            assert dev_config.environment == "development"
            assert dev_config.performance.slow_tool_threshold_ms == 10000
            assert dev_config.performance.error_rate_threshold_percent == 20.0
    
    def test_session_id_generation(self):
        """Test session ID generation methods."""
        config = HookConfig()
        
        # Test environment-based ID
        with patch.dict(os.environ, {"CLAUDE_SESSION_ID": "test-session-123"}):
            session_id = config.get_session_id()
            assert session_id == "test-session-123"
        
        # Test fallback ID generation
        with patch.dict(os.environ, {}, clear=True):
            session_id = config.get_session_id()
            assert session_id is not None
            assert len(session_id) > 0
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = HookConfig()
        validation_results = config.validate_configuration()
        
        assert isinstance(validation_results, dict)
        assert "performance_thresholds_valid" in validation_results
        assert "batch_sizes_valid" in validation_results
        assert "security_config_valid" in validation_results
    
    def test_event_capture_filtering(self):
        """Test event capture filtering logic."""
        config = HookConfig()
        
        # Test enabled events
        assert config.should_capture_event("PreToolUse") == config.enable_pre_tool_use
        assert config.should_capture_event("PostToolUse") == config.enable_post_tool_use
        assert config.should_capture_event("SessionStart") == config.enable_session_lifecycle


class TestHookEventProcessor:
    """Test hook event processor."""
    
    @pytest.fixture
    def event_processor(self):
        """Create event processor for testing."""
        return HookEventProcessor()
    
    @pytest.mark.asyncio
    async def test_process_event_success(self, event_processor):
        """Test successful event processing."""
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        payload = {"tool_name": "test_tool", "success": True}
        
        with patch.object(event_processor, '_store_event_in_database', return_value=12345):
            with patch.object(event_processor, '_publish_to_redis_stream'):
                with patch.object(event_processor, '_update_prometheus_metrics'):
                    with patch.object(event_processor, '_integrate_with_observability_middleware'):
                        
                        event_id = await event_processor.process_event(
                            session_id=session_id,
                            agent_id=agent_id,
                            event_type=EventType.POST_TOOL_USE,
                            payload=payload,
                            latency_ms=100
                        )
                        
                        assert event_id == "12345"
    
    @pytest.mark.asyncio
    async def test_process_event_database_failure(self, event_processor):
        """Test event processing with database failure."""
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        payload = {"tool_name": "test_tool", "success": False}
        
        with patch.object(event_processor, '_store_event_in_database', side_effect=Exception("DB Error")):
            with patch.object(event_processor, '_publish_to_redis_stream'):
                with patch.object(event_processor, '_update_prometheus_metrics'):
                    with patch.object(event_processor, '_integrate_with_observability_middleware'):
                        
                        with pytest.raises(Exception):
                            await event_processor.process_event(
                                session_id=session_id,
                                agent_id=agent_id,
                                event_type=EventType.POST_TOOL_USE,
                                payload=payload
                            )


class TestHookScriptExecutor:
    """Test hook script executor."""
    
    @pytest.fixture
    def script_executor(self):
        """Create script executor for testing."""
        return HookScriptExecutor()
    
    @pytest.fixture
    def mock_script_file(self):
        """Create a mock executable script file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""#!/usr/bin/env python3
import sys
import json

# Read input data
if len(sys.argv) > 1:
    tool_name = sys.argv[1]
    print(f"Hook executed for {tool_name}: event-123")
else:
    data = json.loads(sys.stdin.read())
    tool_name = data.get('tool_name', 'unknown')
    print(f"Hook executed for {tool_name}: event-456")

sys.exit(0)
""")
            script_path = Path(f.name)
        
        script_path.chmod(0o755)  # Make executable
        yield script_path
        
        # Cleanup
        script_path.unlink()
    
    @pytest.mark.asyncio
    async def test_execute_pre_tool_use_hook(self, script_executor, mock_script_file):
        """Test pre-tool-use hook execution."""
        with patch.object(script_executor.config, 'pre_tool_use_script', mock_script_file):
            with patch.object(script_executor.config, 'enable_pre_tool_use', True):
                
                result = await script_executor.execute_pre_tool_use_hook(
                    tool_name="test_tool",
                    parameters={"param1": "value1"},
                    session_id="session-123",
                    agent_id="agent-456"
                )
                
                assert result is not None
                assert "Hook executed for test_tool" in result
    
    @pytest.mark.asyncio
    async def test_execute_post_tool_use_hook(self, script_executor, mock_script_file):
        """Test post-tool-use hook execution."""
        with patch.object(script_executor.config, 'post_tool_use_script', mock_script_file):
            with patch.object(script_executor.config, 'enable_post_tool_use', True):
                
                result = await script_executor.execute_post_tool_use_hook(
                    tool_name="test_tool",
                    success=True,
                    execution_time_ms=150,
                    result="tool result data",
                    session_id="session-123",
                    agent_id="agent-456"
                )
                
                assert result is not None
                assert "Hook executed for test_tool" in result
    
    @pytest.mark.asyncio
    async def test_execute_session_lifecycle_hook(self, script_executor, mock_script_file):
        """Test session lifecycle hook execution."""
        with patch.object(script_executor.config, 'session_lifecycle_script', mock_script_file):
            with patch.object(script_executor.config, 'enable_session_lifecycle', True):
                
                result = await script_executor.execute_session_lifecycle_hook(
                    event_type="session_start",
                    session_id="session-123",
                    agent_id="agent-456"
                )
                
                assert result is not None
                assert "Hook executed for session_start" in result
    
    @pytest.mark.asyncio
    async def test_script_timeout(self, script_executor):
        """Test script execution timeout handling."""
        # Create a script that hangs
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""#!/usr/bin/env python3
import time
time.sleep(60)  # Sleep longer than timeout
""")
            script_path = Path(f.name)
        
        script_path.chmod(0o755)
        
        try:
            with patch.object(script_executor.config, 'pre_tool_use_script', script_path):
                with patch.object(script_executor.config, 'enable_pre_tool_use', True):
                    
                    result = await script_executor.execute_pre_tool_use_hook(
                        tool_name="test_tool",
                        parameters={}
                    )
                    
                    assert result is None  # Should timeout and return None
        finally:
            script_path.unlink()
    
    @pytest.mark.asyncio
    async def test_script_error_handling(self, script_executor):
        """Test script error handling."""
        # Create a script that fails
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""#!/usr/bin/env python3
import sys
print("Error message", file=sys.stderr)
sys.exit(1)
""")
            script_path = Path(f.name)
        
        script_path.chmod(0o755)
        
        try:
            with patch.object(script_executor.config, 'pre_tool_use_script', script_path):
                with patch.object(script_executor.config, 'enable_pre_tool_use', True):
                    
                    result = await script_executor.execute_pre_tool_use_hook(
                        tool_name="test_tool",
                        parameters={}
                    )
                    
                    assert result is None  # Should handle error and return None
        finally:
            script_path.unlink()


class TestHookIntegrationManager:
    """Test hook integration manager."""
    
    @pytest.fixture
    def integration_manager(self):
        """Create integration manager for testing."""
        return HookIntegrationManager()
    
    @pytest.mark.asyncio
    async def test_capture_tool_execution_complete(self, integration_manager):
        """Test complete tool execution capture."""
        with patch.object(integration_manager.script_executor, 'execute_pre_tool_use_hook', return_value="pre-123"):
            with patch.object(integration_manager.script_executor, 'execute_post_tool_use_hook', return_value="post-456"):
                
                results = await integration_manager.capture_tool_execution(
                    tool_name="test_tool",
                    parameters={"param1": "value1"},
                    execution_result={
                        "success": True,
                        "execution_time_ms": 200,
                        "result": "tool output"
                    },
                    session_id="session-123",
                    agent_id="agent-456"
                )
                
                assert results["pre_event_id"] == "pre-123"
                assert results["post_event_id"] == "post-456"
    
    @pytest.mark.asyncio
    async def test_capture_session_lifecycle_event(self, integration_manager):
        """Test session lifecycle event capture."""
        with patch.object(integration_manager.script_executor, 'execute_session_lifecycle_hook', return_value="session-789"):
            
            result = await integration_manager.capture_session_lifecycle_event(
                event_type="session_start",
                session_id="session-123",
                agent_id="agent-456",
                context_data={"key": "value"}
            )
            
            assert result == "session-789"
    
    @pytest.mark.asyncio
    async def test_get_integration_status(self, integration_manager):
        """Test integration status reporting."""
        with patch('app.observability.hooks.hooks_integration.get_db_session'):
            with patch('app.core.redis.get_redis_client'):
                
                status = await integration_manager.get_integration_status()
                
                assert "timestamp" in status
                assert "environment" in status
                assert "hooks_enabled" in status
                assert "script_validation" in status
                assert "integration_health" in status
                assert "configuration" in status
                
                # Check hooks status
                hooks_enabled = status["hooks_enabled"]
                assert "pre_tool_use" in hooks_enabled
                assert "post_tool_use" in hooks_enabled
                assert "session_lifecycle" in hooks_enabled
                assert "error_hooks" in hooks_enabled
    
    def test_enable_disable_hooks(self, integration_manager):
        """Test enabling and disabling hooks."""
        # Test enabling specific hooks
        integration_manager.enable_hooks(["pre_tool_use", "post_tool_use"])
        assert integration_manager.config.enable_pre_tool_use is True
        assert integration_manager.config.enable_post_tool_use is True
        
        # Test disabling specific hooks
        integration_manager.disable_hooks(["pre_tool_use"])
        assert integration_manager.config.enable_pre_tool_use is False
        assert integration_manager.config.enable_post_tool_use is True  # Should remain enabled
        
        # Test disabling all hooks
        integration_manager.disable_hooks()
        assert integration_manager.config.enable_pre_tool_use is False
        assert integration_manager.config.enable_post_tool_use is False
        assert integration_manager.config.enable_session_lifecycle is False
        assert integration_manager.config.enable_error_hooks is False


class TestHookIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_tool_execution_flow(self):
        """Test complete tool execution flow with real hook scripts."""
        integration_manager = get_hook_integration_manager()
        
        # Test pre-tool-use capture
        results = await integration_manager.capture_tool_execution(
            tool_name="test_integration_tool",
            parameters={
                "action": "test",
                "data": {"key": "value"}
            },
            execution_result={
                "success": True,
                "execution_time_ms": 250,
                "result": "Integration test successful"
            },
            session_id="integration-test-session",
            agent_id="integration-test-agent"
        )
        
        # Verify both events were captured
        assert results["pre_event_id"] is not None or not integration_manager.config.enable_pre_tool_use
        assert results["post_event_id"] is not None or not integration_manager.config.enable_post_tool_use
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_session_lifecycle_flow(self):
        """Test complete session lifecycle flow."""
        integration_manager = get_hook_integration_manager()
        
        session_id = f"test-session-{uuid.uuid4()}"
        agent_id = f"test-agent-{uuid.uuid4()}"
        
        # Test session start
        start_result = await integration_manager.capture_session_lifecycle_event(
            event_type="session_start",
            session_id=session_id,
            agent_id=agent_id,
            context_data={"test": "integration"}
        )
        
        # Test session end
        end_result = await integration_manager.capture_session_lifecycle_event(
            event_type="session_end",
            session_id=session_id,
            agent_id=agent_id,
            reason="integration_test_complete"
        )
        
        # Verify events were captured if lifecycle hooks are enabled
        if integration_manager.config.enable_session_lifecycle:
            assert start_result is not None
            assert end_result is not None
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integration_health_check(self):
        """Test integration health and status reporting."""
        integration_manager = get_hook_integration_manager()
        
        status = await integration_manager.get_integration_status()
        
        # Verify status structure
        assert "timestamp" in status
        assert "environment" in status
        assert "hooks_enabled" in status
        assert "script_validation" in status
        assert "integration_health" in status
        assert "configuration" in status
        
        # Check that at least some components are healthy or have expected errors
        health = status["integration_health"]
        assert isinstance(health, dict)
        
        # Check script validation
        validation = status["script_validation"]
        for script_name in ["pre_tool_use", "post_tool_use", "session_lifecycle"]:
            assert script_name in validation
            script_info = validation[script_name]
            assert "exists" in script_info
            assert "executable" in script_info
            assert "path" in script_info


class TestHookPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_batch_event_processing(self):
        """Test batch event processing performance."""
        event_processor = HookEventProcessor()
        
        # Create multiple test events
        events = []
        for i in range(10):
            events.append({
                "session_id": uuid.uuid4(),
                "agent_id": uuid.uuid4(),
                "event_type": EventType.POST_TOOL_USE,
                "payload": {"tool_name": f"test_tool_{i}", "success": True},
                "latency_ms": 100 + i
            })
        
        # Mock database and Redis operations
        with patch.object(event_processor, '_store_event_in_database', return_value=12345):
            with patch.object(event_processor, '_publish_to_redis_stream'):
                with patch.object(event_processor, '_update_prometheus_metrics'):
                    with patch.object(event_processor, '_integrate_with_observability_middleware'):
                        
                        # Process events and measure time
                        import time
                        start_time = time.time()
                        
                        tasks = []
                        for event in events:
                            tasks.append(event_processor.process_event(**event))
                        
                        results = await asyncio.gather(*tasks)
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        # Verify all events processed successfully
                        assert len(results) == 10
                        assert all(result is not None for result in results)
                        
                        # Performance should be reasonable (less than 1 second for 10 mock events)
                        assert total_time < 1.0
    
    @pytest.mark.asyncio
    async def test_concurrent_hook_execution(self):
        """Test concurrent hook script execution."""
        script_executor = HookScriptExecutor()
        
        # Create multiple concurrent hook executions
        tasks = []
        for i in range(5):
            tasks.append(
                script_executor.execute_pre_tool_use_hook(
                    tool_name=f"concurrent_tool_{i}",
                    parameters={"index": i},
                    session_id=f"session-{i}",
                    agent_id=f"agent-{i}"
                )
            )
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that no exceptions occurred (or expected ones for missing scripts)
        for result in results:
            assert not isinstance(result, Exception) or "not found" in str(result)


# Performance benchmarks
@pytest.mark.benchmark
class TestHookPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_event_processing_throughput(self):
        """Benchmark event processing throughput."""
        event_processor = HookEventProcessor()
        
        # Mock all external dependencies
        with patch.object(event_processor, '_store_event_in_database', return_value=12345):
            with patch.object(event_processor, '_publish_to_redis_stream'):
                with patch.object(event_processor, '_update_prometheus_metrics'):
                    with patch.object(event_processor, '_integrate_with_observability_middleware'):
                        
                        # Benchmark processing 100 events
                        event_count = 100
                        
                        import time
                        start_time = time.time()
                        
                        for i in range(event_count):
                            await event_processor.process_event(
                                session_id=uuid.uuid4(),
                                agent_id=uuid.uuid4(),
                                event_type=EventType.POST_TOOL_USE,
                                payload={"tool_name": f"benchmark_tool_{i}", "success": True},
                                latency_ms=50
                            )
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        # Calculate throughput
                        throughput = event_count / total_time
                        
                        print(f"Event processing throughput: {throughput:.2f} events/second")
                        
                        # Should be able to process at least 100 events/second
                        assert throughput > 100


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
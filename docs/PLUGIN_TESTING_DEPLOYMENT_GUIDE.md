# LeanVibe Plugin Testing and Deployment Guide

**Version:** 2.3.0  
**Epic 2 Phase 2.3: Developer SDK & Documentation**

Comprehensive guide for testing, validating, and deploying plugins developed with the LeanVibe Plugin SDK.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Unit Testing](#unit-testing)
3. [Integration Testing](#integration-testing)
4. [Performance Testing](#performance-testing)
5. [Security Testing](#security-testing)
6. [End-to-End Testing](#end-to-end-testing)
7. [Test Automation](#test-automation)
8. [Deployment Strategies](#deployment-strategies)
9. [Production Deployment](#production-deployment)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Rollback and Recovery](#rollback-and-recovery)
12. [Best Practices](#best-practices)

## Testing Overview

### Testing Philosophy

The LeanVibe Plugin SDK emphasizes comprehensive testing to ensure:

- **Epic 1 Compliance**: Response times <50ms, memory usage <80MB
- **Reliability**: Plugins handle errors gracefully and recover automatically
- **Security**: No vulnerabilities or data leaks
- **Compatibility**: Works across different environments and configurations
- **Performance**: Maintains consistent performance under load

### Testing Pyramid

```
     /\
    /E2E\       - End-to-End Tests (few, slow, comprehensive)
   /____\
  /      \
 /Integration\ - Integration Tests (moderate, medium speed)
/____________\
/            \
/  Unit Tests  \ - Unit Tests (many, fast, focused)
/______________\
```

### Test Types and Coverage Goals

| Test Type | Coverage Goal | Speed | Scope |
|-----------|---------------|-------|-------|
| Unit Tests | 90%+ | Fast (<1s) | Individual functions/methods |
| Integration Tests | 80%+ | Medium (1-10s) | Plugin components together |
| Performance Tests | 100% critical paths | Medium (5-30s) | Epic 1 compliance |
| Security Tests | 100% attack vectors | Slow (30s-5m) | Vulnerability scanning |
| E2E Tests | Key workflows | Slow (1-30m) | Full system integration |

## Unit Testing

### Setting Up Unit Tests

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from leanvibe.plugin_sdk.testing import PluginTestFramework
from leanvibe.plugin_sdk import PluginConfig, TaskInterface, TaskResult
from your_plugin import YourPlugin

class TestYourPlugin:
    """Comprehensive unit tests for YourPlugin."""
    
    @pytest.fixture
    async def plugin_config(self):
        """Create test plugin configuration."""
        return PluginConfig(
            name="TestPlugin",
            version="1.0.0",
            description="Test plugin configuration",
            parameters={
                "batch_size": 100,
                "timeout_seconds": 30,
                "enable_retries": True,
                "test_mode": True
            }
        )
    
    @pytest.fixture
    async def plugin(self, plugin_config):
        """Create and initialize plugin for testing."""
        plugin = YourPlugin(plugin_config)
        await plugin.initialize()
        return plugin
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        return TaskInterface(
            task_id="test_task_001",
            task_type="process_data",
            parameters={
                "input_data": [
                    {"id": 1, "name": "Test Item 1", "value": 100},
                    {"id": 2, "name": "Test Item 2", "value": 200}
                ]
            }
        )
    
    async def test_plugin_initialization(self, plugin_config):
        """Test plugin initialization."""
        plugin = YourPlugin(plugin_config)
        
        # Test initial state
        assert plugin.config == plugin_config
        assert not plugin.is_initialized
        
        # Test initialization
        await plugin.initialize()
        assert plugin.is_initialized
        
        # Test cleanup
        await plugin.cleanup()
        assert not plugin.is_initialized
    
    async def test_successful_task_execution(self, plugin, sample_task):
        """Test successful task execution."""
        result = await plugin.handle_task(sample_task)
        
        assert result.success is True
        assert result.plugin_id == plugin.plugin_id
        assert result.task_id == sample_task.task_id
        assert result.data is not None
        assert result.error is None
        
        # Verify Epic 1 compliance
        assert result.execution_time_ms < 50  # <50ms requirement
    
    async def test_invalid_task_type(self, plugin):
        """Test handling of invalid task types."""
        invalid_task = TaskInterface(
            task_id="invalid_test",
            task_type="invalid_operation",
            parameters={}
        )
        
        result = await plugin.handle_task(invalid_task)
        
        assert result.success is False
        assert "invalid" in result.error.lower()
        assert result.error_code == "INVALID_TASK_TYPE"
    
    async def test_missing_parameters(self, plugin):
        """Test handling of missing required parameters."""
        task_missing_data = TaskInterface(
            task_id="missing_params_test",
            task_type="process_data",
            parameters={}  # Missing required input_data
        )
        
        result = await plugin.handle_task(task_missing_data)
        
        assert result.success is False
        assert "missing" in result.error.lower() or "required" in result.error.lower()
    
    async def test_invalid_parameters(self, plugin):
        """Test handling of invalid parameter types."""
        task_invalid_data = TaskInterface(
            task_id="invalid_params_test",
            task_type="process_data",
            parameters={
                "input_data": "not_a_list"  # Should be a list
            }
        )
        
        result = await plugin.handle_task(task_invalid_data)
        
        assert result.success is False
        assert "invalid" in result.error.lower() or "type" in result.error.lower()
    
    @patch('your_plugin.external_service_call')
    async def test_external_service_failure(self, mock_service, plugin, sample_task):
        """Test handling of external service failures."""
        # Mock external service to raise exception
        mock_service.side_effect = Exception("Service unavailable")
        
        result = await plugin.handle_task(sample_task)
        
        # Plugin should handle the error gracefully
        assert result.success is False
        assert "service" in result.error.lower() or "unavailable" in result.error.lower()
        
        # Verify service was called
        mock_service.assert_called_once()
    
    async def test_task_timeout(self, plugin_config):
        """Test task timeout handling."""
        # Create plugin with very short timeout
        plugin_config.parameters["timeout_seconds"] = 1
        plugin = YourPlugin(plugin_config)
        await plugin.initialize()
        
        # Create task that takes longer than timeout
        long_running_task = TaskInterface(
            task_id="timeout_test",
            task_type="slow_operation",
            parameters={"delay_seconds": 5}
        )
        
        result = await plugin.handle_task(long_running_task)
        
        assert result.success is False
        assert "timeout" in result.error.lower()
    
    async def test_memory_usage(self, plugin, sample_task):
        """Test memory usage compliance."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execute task
        result = await plugin.handle_task(sample_task)
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify Epic 1 memory compliance
        assert memory_increase < 80  # <80MB increase
        assert result.success is True
    
    async def test_concurrent_tasks(self, plugin):
        """Test concurrent task execution."""
        tasks = [
            TaskInterface(
                task_id=f"concurrent_test_{i}",
                task_type="process_data",
                parameters={"input_data": [{"id": i, "value": i * 10}]}
            )
            for i in range(10)
        ]
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            *[plugin.handle_task(task) for task in tasks],
            return_exceptions=True
        )
        
        # Verify all tasks completed successfully
        successful_results = [r for r in results if isinstance(r, TaskResult) and r.success]
        assert len(successful_results) == 10
        
        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0
    
    async def test_plugin_state_isolation(self, plugin_config):
        """Test that plugin instances don't share state."""
        plugin1 = YourPlugin(plugin_config)
        plugin2 = YourPlugin(plugin_config)
        
        await plugin1.initialize()
        await plugin2.initialize()
        
        # Modify state in plugin1
        plugin1._test_state = "modified"
        
        # Verify plugin2 is unaffected
        assert not hasattr(plugin2, '_test_state')
        
        await plugin1.cleanup()
        await plugin2.cleanup()
```

### Running Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run unit tests with coverage
pytest tests/unit/ -v --cov=your_plugin --cov-report=html --cov-report=term

# Run specific test class
pytest tests/unit/test_your_plugin.py::TestYourPlugin -v

# Run tests with performance profiling
pytest tests/unit/ -v --durations=10
```

## Integration Testing

### Plugin Integration Tests

```python
import pytest
import asyncio
from leanvibe.plugin_sdk.testing import PluginTestFramework
from leanvibe.plugin_sdk import PluginConfig
from your_plugin import YourPlugin

class TestPluginIntegration:
    """Integration tests for plugin components."""
    
    @pytest.fixture
    async def test_framework(self):
        """Create test framework."""
        framework = PluginTestFramework()
        yield framework
        await framework.cleanup()
    
    @pytest.fixture
    async def plugin_with_dependencies(self, test_framework):
        """Create plugin with mock dependencies."""
        config = PluginConfig(
            name="IntegrationTestPlugin",
            version="1.0.0",
            parameters={
                "database_url": "sqlite:///:memory:",
                "cache_url": "redis://localhost:6379/15",  # Test DB
                "external_api_url": "http://localhost:8080/api/test"
            }
        )
        
        plugin = YourPlugin(config)
        await test_framework.setup_plugin_environment(plugin)
        
        return plugin
    
    async def test_database_integration(self, plugin_with_dependencies):
        """Test database integration."""
        plugin = plugin_with_dependencies
        
        # Test database operations
        test_data = {"id": 1, "data": "test_value"}
        
        # Test save operation
        save_task = TaskInterface(
            task_id="db_save_test",
            task_type="save_data",
            parameters={"data": test_data}
        )
        
        save_result = await plugin.handle_task(save_task)
        assert save_result.success is True
        
        # Test retrieve operation
        retrieve_task = TaskInterface(
            task_id="db_retrieve_test",
            task_type="get_data",
            parameters={"id": 1}
        )
        
        retrieve_result = await plugin.handle_task(retrieve_task)
        assert retrieve_result.success is True
        assert retrieve_result.data["data"] == "test_value"
    
    async def test_cache_integration(self, plugin_with_dependencies):
        """Test cache integration."""
        plugin = plugin_with_dependencies
        
        # Test cache operations
        cache_task = TaskInterface(
            task_id="cache_test",
            task_type="cache_data",
            parameters={
                "key": "test_key",
                "value": "test_value",
                "ttl": 300
            }
        )
        
        cache_result = await plugin.handle_task(cache_task)
        assert cache_result.success is True
        
        # Test cache retrieval
        get_task = TaskInterface(
            task_id="cache_get_test",
            task_type="get_cached_data",
            parameters={"key": "test_key"}
        )
        
        get_result = await plugin.handle_task(get_task)
        assert get_result.success is True
        assert get_result.data["value"] == "test_value"
    
    async def test_external_api_integration(self, plugin_with_dependencies):
        """Test external API integration."""
        plugin = plugin_with_dependencies
        
        # Mock external API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"result": "success"}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            api_task = TaskInterface(
                task_id="api_test",
                task_type="call_external_api",
                parameters={
                    "endpoint": "/test",
                    "payload": {"test": "data"}
                }
            )
            
            api_result = await plugin.handle_task(api_task)
            assert api_result.success is True
            assert api_result.data["result"] == "success"
    
    async def test_plugin_lifecycle_integration(self, test_framework):
        """Test complete plugin lifecycle."""
        config = PluginConfig(
            name="LifecycleTestPlugin",
            version="1.0.0",
            parameters={"test_param": "test_value"}
        )
        
        plugin = YourPlugin(config)
        
        # Test initialization
        initialization_result = await test_framework.test_plugin_initialization(plugin)
        assert initialization_result.success is True
        
        # Test task execution
        task_result = await test_framework.test_plugin_task(
            plugin,
            task_type="process_data",
            parameters={"input_data": [{"test": "data"}]},
            expected_success=True
        )
        assert task_result.passed is True
        
        # Test cleanup
        cleanup_result = await test_framework.test_plugin_cleanup(plugin)
        assert cleanup_result.success is True
    
    async def test_error_propagation(self, plugin_with_dependencies):
        """Test error propagation through plugin stack."""
        plugin = plugin_with_dependencies
        
        # Test that database errors are properly handled
        with patch('your_plugin.database.execute') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            db_task = TaskInterface(
                task_id="db_error_test",
                task_type="save_data",
                parameters={"data": {"test": "data"}}
            )
            
            result = await plugin.handle_task(db_task)
            assert result.success is False
            assert "database" in result.error.lower()
            assert result.error_code == "DATABASE_ERROR"
```

## Performance Testing

### Epic 1 Compliance Testing

```python
import pytest
import time
import psutil
import asyncio
from statistics import mean, median
from leanvibe.plugin_sdk.testing import PluginTestFramework

class TestPerformanceCompliance:
    """Test Epic 1 performance compliance."""
    
    @pytest.fixture
    async def performance_framework(self):
        """Create performance testing framework."""
        framework = PluginTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_response_time_compliance(self, performance_framework):
        """Test <50ms response time requirement."""
        plugin = YourPlugin(PluginConfig(
            name="PerformanceTest",
            version="1.0.0",
            parameters={"batch_size": 100}
        ))
        
        await performance_framework.setup_plugin_environment(plugin)
        
        # Test multiple iterations for statistical validity
        response_times = []
        iterations = 100
        
        for i in range(iterations):
            task = TaskInterface(
                task_id=f"perf_test_{i}",
                task_type="process_data",
                parameters={"input_data": [{"id": j, "value": j} for j in range(10)]}
            )
            
            start_time = time.perf_counter()
            result = await plugin.handle_task(task)
            end_time = time.perf_counter()
            
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
            
            assert result.success is True
            assert response_time_ms < 50  # Epic 1 requirement
        
        # Statistical analysis
        avg_response_time = mean(response_times)
        median_response_time = median(response_times)
        max_response_time = max(response_times)
        
        print(f"Response Time Statistics:")
        print(f"  Average: {avg_response_time:.2f}ms")
        print(f"  Median: {median_response_time:.2f}ms") 
        print(f"  Maximum: {max_response_time:.2f}ms")
        print(f"  All iterations < 50ms: {max_response_time < 50}")
        
        # Verify performance characteristics
        assert avg_response_time < 25  # Target average well below limit
        assert max_response_time < 50  # Hard limit
        assert len([t for t in response_times if t > 40]) < iterations * 0.05  # <5% over 40ms
    
    async def test_memory_usage_compliance(self, performance_framework):
        """Test <80MB memory usage requirement."""
        plugin = YourPlugin(PluginConfig(
            name="MemoryTest",
            version="1.0.0",
            parameters={"batch_size": 1000}
        ))
        
        await performance_framework.setup_plugin_environment(plugin)
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execute memory-intensive task
        large_dataset = [{"id": i, "data": f"data_{i}" * 100} for i in range(10000)]
        
        memory_task = TaskInterface(
            task_id="memory_test",
            task_type="process_data",
            parameters={"input_data": large_dataset}
        )
        
        result = await plugin.handle_task(memory_task)
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Increase: {memory_increase:.2f}MB")
        
        assert result.success is True
        assert memory_increase < 80  # Epic 1 requirement
        assert memory_increase < 50  # Target well below limit
    
    async def test_throughput_performance(self, performance_framework):
        """Test plugin throughput under load."""
        plugin = YourPlugin(PluginConfig(
            name="ThroughputTest",
            version="1.0.0",
            parameters={"batch_size": 500}
        ))
        
        await performance_framework.setup_plugin_environment(plugin)
        
        # Create multiple concurrent tasks
        tasks = []
        task_count = 50
        start_time = time.perf_counter()
        
        for i in range(task_count):
            task = TaskInterface(
                task_id=f"throughput_test_{i}",
                task_type="process_data",
                parameters={"input_data": [{"id": j, "value": j} for j in range(100)]}
            )
            tasks.append(plugin.handle_task(task))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = task_count / total_time
        
        print(f"Throughput Performance:")
        print(f"  Total tasks: {task_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} tasks/second")
        
        # Verify all tasks succeeded
        successful_tasks = sum(1 for r in results if r.success)
        assert successful_tasks == task_count
        
        # Verify throughput meets requirements
        assert throughput > 10  # Minimum 10 tasks/second
    
    async def test_stress_testing(self, performance_framework):
        """Test plugin under stress conditions."""
        plugin = YourPlugin(PluginConfig(
            name="StressTest",
            version="1.0.0",
            parameters={"batch_size": 1000}
        ))
        
        await performance_framework.setup_plugin_environment(plugin)
        
        # Gradually increase load
        load_levels = [10, 25, 50, 100, 200]
        results = {}
        
        for load_level in load_levels:
            print(f"Testing load level: {load_level} concurrent tasks")
            
            tasks = []
            start_time = time.perf_counter()
            
            for i in range(load_level):
                task = TaskInterface(
                    task_id=f"stress_test_{load_level}_{i}",
                    task_type="process_data",
                    parameters={"input_data": [{"id": j} for j in range(50)]}
                )
                tasks.append(plugin.handle_task(task))
            
            # Execute tasks
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Analyze results
            successful = sum(1 for r in task_results if isinstance(r, TaskResult) and r.success)
            failed = sum(1 for r in task_results if isinstance(r, TaskResult) and not r.success)
            exceptions = sum(1 for r in task_results if isinstance(r, Exception))
            
            total_time = end_time - start_time
            success_rate = successful / load_level
            
            results[load_level] = {
                "successful": successful,
                "failed": failed,
                "exceptions": exceptions,
                "total_time": total_time,
                "success_rate": success_rate,
                "throughput": successful / total_time
            }
            
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Throughput: {successful / total_time:.2f} tasks/second")
            
            # Verify acceptable performance under stress
            assert success_rate > 0.95  # >95% success rate
            assert exceptions == 0  # No unhandled exceptions
```

### Load Testing

```python
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor

async def load_test_runner():
    """Run comprehensive load tests."""
    
    # Initialize multiple plugin instances
    plugin_configs = [
        PluginConfig(
            name=f"LoadTestPlugin_{i}",
            version="1.0.0",
            parameters={"instance_id": i}
        )
        for i in range(5)
    ]
    
    plugins = [YourPlugin(config) for config in plugin_configs]
    
    # Initialize all plugins
    for plugin in plugins:
        await plugin.initialize()
    
    try:
        # Run sustained load test
        duration_seconds = 300  # 5 minutes
        target_rps = 100  # 100 requests per second
        
        print(f"Starting load test: {target_rps} RPS for {duration_seconds} seconds")
        
        start_time = time.time()
        tasks_completed = 0
        tasks_failed = 0
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Create batch of tasks
            batch_tasks = []
            for _ in range(target_rps):
                plugin = random.choice(plugins)
                task = TaskInterface(
                    task_id=f"load_test_{tasks_completed}",
                    task_type="process_data",
                    parameters={
                        "input_data": [
                            {"id": i, "value": random.randint(1, 1000)}
                            for i in range(random.randint(10, 100))
                        ]
                    }
                )
                batch_tasks.append(plugin.handle_task(task))
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Count results
            for result in batch_results:
                if isinstance(result, TaskResult) and result.success:
                    tasks_completed += 1
                else:
                    tasks_failed += 1
            
            # Wait for next second
            batch_duration = time.time() - batch_start
            if batch_duration < 1.0:
                await asyncio.sleep(1.0 - batch_duration)
        
        total_duration = time.time() - start_time
        actual_rps = tasks_completed / total_duration
        failure_rate = tasks_failed / (tasks_completed + tasks_failed)
        
        print(f"Load test completed:")
        print(f"  Duration: {total_duration:.2f}s")
        print(f"  Tasks completed: {tasks_completed}")
        print(f"  Tasks failed: {tasks_failed}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  Failure rate: {failure_rate:.2%}")
        
        # Verify load test results
        assert actual_rps > target_rps * 0.9  # Within 10% of target
        assert failure_rate < 0.01  # <1% failure rate
        
    finally:
        # Cleanup plugins
        for plugin in plugins:
            await plugin.cleanup()

# Run load test
asyncio.run(load_test_runner())
```

## Security Testing

### Security Validation Tests

```python
import pytest
from leanvibe.plugin_sdk.testing import SecurityTestFramework

class TestPluginSecurity:
    """Security tests for plugin vulnerabilities."""
    
    @pytest.fixture
    async def security_framework(self):
        """Create security testing framework."""
        framework = SecurityTestFramework()
        yield framework
        await framework.cleanup()
    
    async def test_input_validation(self, security_framework):
        """Test input validation and sanitization."""
        plugin = YourPlugin(PluginConfig(name="SecurityTest", version="1.0.0"))
        await plugin.initialize()
        
        # Test SQL injection attempts
        sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; SELECT * FROM sensitive_data; --"
        ]
        
        for payload in sql_injection_payloads:
            task = TaskInterface(
                task_id="sql_injection_test",
                task_type="process_data",
                parameters={"input_data": [{"name": payload}]}
            )
            
            result = await plugin.handle_task(task)
            
            # Plugin should either reject or sanitize malicious input
            if result.success:
                # Verify output is sanitized
                assert payload not in str(result.data)
            else:
                # Verify appropriate error handling
                assert "invalid" in result.error.lower() or "forbidden" in result.error.lower()
    
    async def test_xss_prevention(self, security_framework):
        """Test XSS prevention."""
        plugin = YourPlugin(PluginConfig(name="XSSTest", version="1.0.0"))
        await plugin.initialize()
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            task = TaskInterface(
                task_id="xss_test",
                task_type="process_data",
                parameters={"input_data": [{"content": payload}]}
            )
            
            result = await plugin.handle_task(task)
            
            if result.success:
                # Verify dangerous content is escaped or removed
                output_str = str(result.data).lower()
                assert "<script" not in output_str
                assert "javascript:" not in output_str
                assert "alert(" not in output_str
    
    async def test_path_traversal_prevention(self, security_framework):
        """Test path traversal prevention."""
        plugin = YourPlugin(PluginConfig(name="PathTraversalTest", version="1.0.0"))
        await plugin.initialize()
        
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for payload in path_traversal_payloads:
            task = TaskInterface(
                task_id="path_traversal_test",
                task_type="read_file",
                parameters={"file_path": payload}
            )
            
            result = await plugin.handle_task(task)
            
            # Should reject path traversal attempts
            assert result.success is False
            assert "invalid" in result.error.lower() or "forbidden" in result.error.lower()
    
    async def test_sensitive_data_handling(self, security_framework):
        """Test sensitive data handling."""
        plugin = YourPlugin(PluginConfig(name="DataSecurityTest", version="1.0.0"))
        await plugin.initialize()
        
        sensitive_data = {
            "password": "secret123",
            "api_key": "sk_live_abcd1234567890",
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111"
        }
        
        task = TaskInterface(
            task_id="sensitive_data_test",
            task_type="process_data",
            parameters={"input_data": [sensitive_data]}
        )
        
        result = await plugin.handle_task(task)
        
        if result.success:
            output_str = str(result.data)
            
            # Verify sensitive data is not exposed in output
            assert "secret123" not in output_str
            assert "sk_live_" not in output_str
            assert "123-45-6789" not in output_str
            assert "4111-1111-1111-1111" not in output_str
    
    async def test_resource_limits(self, security_framework):
        """Test resource consumption limits."""
        plugin = YourPlugin(PluginConfig(name="ResourceLimitTest", version="1.0.0"))
        await plugin.initialize()
        
        # Test extremely large input
        large_input = [{"data": "x" * 1000000} for _ in range(1000)]  # ~1GB of data
        
        task = TaskInterface(
            task_id="resource_limit_test",
            task_type="process_data",
            parameters={"input_data": large_input}
        )
        
        start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        result = await plugin.handle_task(task)
        
        end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_increase = end_memory - start_memory
        
        # Plugin should either handle large input efficiently or reject it
        if result.success:
            assert memory_increase < 80  # Epic 1 memory limit
        else:
            assert "too large" in result.error.lower() or "limit" in result.error.lower()
    
    async def test_authentication_bypass(self, security_framework):
        """Test authentication bypass attempts."""
        plugin = YourPlugin(PluginConfig(
            name="AuthTest", 
            version="1.0.0",
            parameters={"require_auth": True}
        ))
        await plugin.initialize()
        
        # Test without authentication
        task = TaskInterface(
            task_id="auth_bypass_test",
            task_type="protected_operation",
            parameters={"data": "sensitive_operation"}
        )
        
        result = await plugin.handle_task(task)
        
        # Should reject unauthenticated requests
        assert result.success is False
        assert "auth" in result.error.lower() or "unauthorized" in result.error.lower()
```

### Automated Security Scanning

```bash
#!/bin/bash

# Security scanning script for plugin deployment

echo "🔒 Starting security scans for plugin deployment..."

# 1. Static code analysis
echo "📝 Running static code analysis..."
bandit -r your_plugin/ -f json -o security_report.json

# 2. Dependency vulnerability scan
echo "📦 Scanning dependencies for vulnerabilities..."
safety check --json --output dependency_vulnerabilities.json

# 3. Secrets detection
echo "🔐 Scanning for hardcoded secrets..."
truffleHog --regex --entropy=False your_plugin/ > secrets_report.txt

# 4. License compliance
echo "📄 Checking license compliance..."
pip-licenses --format=json --output-file=license_report.json

# 5. Code quality analysis
echo "📊 Running code quality analysis..."
pylint your_plugin/ --output-format=json > code_quality_report.json

# 6. Security linting
echo "🛡️ Running security-focused linting..."
semgrep --config=auto your_plugin/ --json --output=semgrep_report.json

echo "✅ Security scans completed. Review reports before deployment."
```

## Test Automation

### Continuous Integration Pipeline

```yaml
# .github/workflows/plugin-testing.yml
name: Plugin Testing Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=your_plugin --cov-report=xml --cov-fail-under=90
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
      
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    
    - name: Run integration tests
      env:
        TEST_DATABASE_URL: postgresql://postgres:testpass@localhost:5432/testdb
        TEST_REDIS_URL: redis://localhost:6379/15
      run: |
        pytest tests/integration/ -v --durations=10

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --durations=0
    
    - name: Epic 1 compliance check
      run: |
        python scripts/check_epic1_compliance.py

  security-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety semgrep
    
    - name: Run security scans
      run: |
        # Static analysis
        bandit -r your_plugin/ -f json -o bandit_report.json
        
        # Dependency scan
        safety check --json --output safety_report.json
        
        # Semgrep analysis
        semgrep --config=auto your_plugin/ --json --output=semgrep_report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit_report.json
          safety_report.json
          semgrep_report.json

  deployment-test:
    runs-on: ubuntu-latest
    needs: [integration-tests, performance-tests, security-tests]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Test plugin packaging
      run: |
        python setup.py sdist bdist_wheel
        pip install dist/*.whl
    
    - name: Test deployment readiness
      run: |
        python scripts/deployment_readiness_check.py
```

### Test Configuration Management

```python
# conftest.py - Global test configuration
import pytest
import asyncio
from typing import AsyncGenerator
from leanvibe.plugin_sdk.testing import PluginTestFramework

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_database() -> AsyncGenerator[str, None]:
    """Set up test database."""
    # Create test database
    database_url = "sqlite:///test.db"
    
    # Initialize database schema
    await initialize_test_database(database_url)
    
    yield database_url
    
    # Cleanup
    await cleanup_test_database(database_url)

@pytest.fixture(scope="function")
async def clean_database(test_database):
    """Provide clean database for each test."""
    await clear_test_data(test_database)
    yield test_database
    await clear_test_data(test_database)

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "test_mode": True,
        "log_level": "DEBUG",
        "timeout_seconds": 10,
        "retry_attempts": 1
    }

@pytest.fixture
async def mock_external_services():
    """Mock external service dependencies."""
    mocks = {}
    
    # Mock HTTP services
    with patch('aiohttp.ClientSession') as mock_session:
        mocks['http'] = mock_session
        
        # Mock Redis
        with patch('redis.Redis') as mock_redis:
            mocks['redis'] = mock_redis
            
            yield mocks

async def initialize_test_database(database_url: str):
    """Initialize test database schema."""
    # Implementation depends on your database setup
    pass

async def cleanup_test_database(database_url: str):
    """Clean up test database."""
    # Implementation depends on your database setup
    pass

async def clear_test_data(database_url: str):
    """Clear test data from database."""
    # Implementation depends on your database setup
    pass
```

## Deployment Strategies

### Blue-Green Deployment

```python
#!/usr/bin/env python3
"""
Blue-Green deployment script for LeanVibe plugins.
"""

import asyncio
import logging
from typing import Dict, List
from dataclasses import dataclass
from leanvibe.plugin_sdk import PluginConfig
from leanvibe.plugin_sdk.deployment import DeploymentManager

@dataclass
class DeploymentConfig:
    plugin_name: str
    version: str
    blue_environment: str
    green_environment: str
    health_check_url: str
    rollback_threshold: float = 0.05  # 5% error rate triggers rollback

class BlueGreenDeployment:
    """Blue-Green deployment manager for plugins."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_manager = DeploymentManager()
        self.logger = logging.getLogger(__name__)
    
    async def deploy(self, plugin_package_path: str) -> bool:
        """Execute blue-green deployment."""
        
        self.logger.info(f"Starting blue-green deployment of {self.config.plugin_name}")
        
        try:
            # Step 1: Determine current active environment
            active_env = await self._get_active_environment()
            target_env = self._get_target_environment(active_env)
            
            self.logger.info(f"Active environment: {active_env}")
            self.logger.info(f"Target environment: {target_env}")
            
            # Step 2: Deploy to target environment
            deployment_success = await self._deploy_to_environment(
                plugin_package_path, target_env
            )
            
            if not deployment_success:
                self.logger.error("Deployment to target environment failed")
                return False
            
            # Step 3: Run health checks on target environment
            health_check_passed = await self._run_health_checks(target_env)
            
            if not health_check_passed:
                self.logger.error("Health checks failed on target environment")
                await self._cleanup_environment(target_env)
                return False
            
            # Step 4: Run canary tests
            canary_success = await self._run_canary_tests(target_env)
            
            if not canary_success:
                self.logger.error("Canary tests failed")
                await self._cleanup_environment(target_env)
                return False
            
            # Step 5: Switch traffic to target environment
            switch_success = await self._switch_traffic(active_env, target_env)
            
            if not switch_success:
                self.logger.error("Traffic switch failed")
                await self._cleanup_environment(target_env)
                return False
            
            # Step 6: Monitor for issues and potentially rollback
            monitoring_success = await self._monitor_deployment(target_env, active_env)
            
            if not monitoring_success:
                self.logger.warning("Monitoring detected issues, rolling back")
                await self._rollback(active_env, target_env)
                return False
            
            # Step 7: Cleanup old environment
            await self._cleanup_environment(active_env)
            
            self.logger.info("Blue-green deployment completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed with error: {e}")
            return False
    
    async def _get_active_environment(self) -> str:
        """Determine currently active environment."""
        # Check which environment is receiving traffic
        blue_active = await self.deployment_manager.is_environment_active(
            self.config.blue_environment
        )
        
        return self.config.blue_environment if blue_active else self.config.green_environment
    
    def _get_target_environment(self, active_env: str) -> str:
        """Get target environment for deployment."""
        if active_env == self.config.blue_environment:
            return self.config.green_environment
        else:
            return self.config.blue_environment
    
    async def _deploy_to_environment(self, package_path: str, environment: str) -> bool:
        """Deploy plugin to specified environment."""
        try:
            self.logger.info(f"Deploying plugin to {environment}")
            
            # Install plugin package
            install_success = await self.deployment_manager.install_plugin(
                package_path, environment
            )
            
            if not install_success:
                return False
            
            # Configure plugin
            config_success = await self.deployment_manager.configure_plugin(
                self.config.plugin_name, environment
            )
            
            if not config_success:
                return False
            
            # Start plugin
            start_success = await self.deployment_manager.start_plugin(
                self.config.plugin_name, environment
            )
            
            return start_success
            
        except Exception as e:
            self.logger.error(f"Deployment to {environment} failed: {e}")
            return False
    
    async def _run_health_checks(self, environment: str) -> bool:
        """Run health checks on target environment."""
        self.logger.info(f"Running health checks on {environment}")
        
        try:
            # Basic connectivity check
            connectivity_ok = await self.deployment_manager.check_connectivity(environment)
            if not connectivity_ok:
                return False
            
            # Plugin-specific health check
            health_ok = await self.deployment_manager.check_plugin_health(
                self.config.plugin_name, environment
            )
            if not health_ok:
                return False
            
            # Performance baseline check
            performance_ok = await self.deployment_manager.check_performance_baseline(
                self.config.plugin_name, environment
            )
            
            return performance_ok
            
        except Exception as e:
            self.logger.error(f"Health checks failed: {e}")
            return False
    
    async def _run_canary_tests(self, environment: str) -> bool:
        """Run canary tests with real traffic."""
        self.logger.info(f"Running canary tests on {environment}")
        
        try:
            # Divert 5% of traffic to target environment
            canary_success = await self.deployment_manager.setup_canary_traffic(
                environment, traffic_percentage=5
            )
            
            if not canary_success:
                return False
            
            # Monitor canary for 5 minutes
            monitoring_duration = 300  # 5 minutes
            monitoring_success = await self.deployment_manager.monitor_canary(
                environment, 
                duration_seconds=monitoring_duration,
                error_threshold=self.config.rollback_threshold
            )
            
            # Remove canary traffic
            await self.deployment_manager.remove_canary_traffic(environment)
            
            return monitoring_success
            
        except Exception as e:
            self.logger.error(f"Canary tests failed: {e}")
            return False
    
    async def _switch_traffic(self, from_env: str, to_env: str) -> bool:
        """Switch traffic from one environment to another."""
        self.logger.info(f"Switching traffic from {from_env} to {to_env}")
        
        try:
            # Gradual traffic switch: 0% -> 25% -> 50% -> 75% -> 100%
            traffic_percentages = [25, 50, 75, 100]
            
            for percentage in traffic_percentages:
                self.logger.info(f"Switching {percentage}% traffic to {to_env}")
                
                switch_success = await self.deployment_manager.switch_traffic(
                    to_env, traffic_percentage=percentage
                )
                
                if not switch_success:
                    return False
                
                # Monitor for 30 seconds at each level
                await asyncio.sleep(30)
                
                # Check for issues
                metrics = await self.deployment_manager.get_environment_metrics(to_env)
                error_rate = metrics.get('error_rate', 0)
                
                if error_rate > self.config.rollback_threshold:
                    self.logger.error(f"Error rate {error_rate} exceeds threshold")
                    return False
            
            self.logger.info("Traffic switch completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Traffic switch failed: {e}")
            return False
    
    async def _monitor_deployment(self, new_env: str, old_env: str) -> bool:
        """Monitor deployment for issues."""
        self.logger.info(f"Monitoring deployment on {new_env}")
        
        monitoring_duration = 600  # 10 minutes
        check_interval = 30  # 30 seconds
        
        for i in range(0, monitoring_duration, check_interval):
            try:
                # Get metrics from new environment
                metrics = await self.deployment_manager.get_environment_metrics(new_env)
                
                # Check error rate
                error_rate = metrics.get('error_rate', 0)
                if error_rate > self.config.rollback_threshold:
                    self.logger.error(f"Error rate {error_rate} exceeds threshold")
                    return False
                
                # Check response times
                avg_response_time = metrics.get('avg_response_time_ms', 0)
                if avg_response_time > 50:  # Epic 1 requirement
                    self.logger.error(f"Response time {avg_response_time}ms exceeds Epic 1 limit")
                    return False
                
                # Check memory usage
                memory_usage = metrics.get('memory_usage_mb', 0)
                if memory_usage > 80:  # Epic 1 requirement
                    self.logger.error(f"Memory usage {memory_usage}MB exceeds Epic 1 limit")
                    return False
                
                self.logger.info(f"Monitoring check {i//check_interval + 1}: OK")
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring check failed: {e}")
                return False
        
        self.logger.info("Monitoring completed successfully")
        return True
    
    async def _rollback(self, old_env: str, new_env: str) -> bool:
        """Rollback to previous environment."""
        self.logger.info(f"Rolling back from {new_env} to {old_env}")
        
        try:
            # Switch traffic back to old environment
            rollback_success = await self.deployment_manager.switch_traffic(
                old_env, traffic_percentage=100
            )
            
            if rollback_success:
                self.logger.info("Rollback completed successfully")
            else:
                self.logger.error("Rollback failed")
            
            return rollback_success
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    async def _cleanup_environment(self, environment: str) -> bool:
        """Clean up specified environment."""
        self.logger.info(f"Cleaning up environment {environment}")
        
        try:
            cleanup_success = await self.deployment_manager.cleanup_environment(
                environment, self.config.plugin_name
            )
            
            return cleanup_success
            
        except Exception as e:
            self.logger.error(f"Environment cleanup failed: {e}")
            return False

# Usage example
async def main():
    config = DeploymentConfig(
        plugin_name="YourPlugin",
        version="1.2.0",
        blue_environment="production-blue",
        green_environment="production-green",
        health_check_url="https://api.example.com/health",
        rollback_threshold=0.02  # 2% error rate
    )
    
    deployment = BlueGreenDeployment(config)
    success = await deployment.deploy("path/to/plugin-package.tar.gz")
    
    if success:
        print("✅ Deployment completed successfully")
    else:
        print("❌ Deployment failed")

if __name__ == "__main__":
    asyncio.run(main())
```

### Rolling Deployment

```python
class RollingDeployment:
    """Rolling deployment strategy for plugins."""
    
    def __init__(self, instances: List[str], config: DeploymentConfig):
        self.instances = instances
        self.config = config
        self.deployment_manager = DeploymentManager()
        self.logger = logging.getLogger(__name__)
    
    async def deploy(self, plugin_package_path: str) -> bool:
        """Execute rolling deployment across instances."""
        
        self.logger.info(f"Starting rolling deployment across {len(self.instances)} instances")
        
        successful_instances = []
        failed_instances = []
        
        for i, instance in enumerate(self.instances):
            self.logger.info(f"Deploying to instance {i+1}/{len(self.instances)}: {instance}")
            
            try:
                # Deploy to single instance
                instance_success = await self._deploy_to_instance(
                    plugin_package_path, instance
                )
                
                if instance_success:
                    successful_instances.append(instance)
                    self.logger.info(f"✅ Instance {instance} deployed successfully")
                    
                    # Brief pause between deployments
                    await asyncio.sleep(30)
                else:
                    failed_instances.append(instance)
                    self.logger.error(f"❌ Instance {instance} deployment failed")
                    
                    # Decide whether to continue or abort
                    if len(failed_instances) > len(self.instances) * 0.2:  # >20% failure rate
                        self.logger.error("Too many failures, aborting rolling deployment")
                        await self._rollback_instances(successful_instances)
                        return False
                
            except Exception as e:
                self.logger.error(f"Instance {instance} deployment error: {e}")
                failed_instances.append(instance)
        
        # Final validation
        if len(successful_instances) >= len(self.instances) * 0.8:  # 80% success rate
            self.logger.info("Rolling deployment completed successfully")
            return True
        else:
            self.logger.error("Rolling deployment failed, rolling back")
            await self._rollback_instances(successful_instances)
            return False
    
    async def _deploy_to_instance(self, package_path: str, instance: str) -> bool:
        """Deploy to a single instance."""
        try:
            # Remove instance from load balancer
            await self.deployment_manager.remove_from_load_balancer(instance)
            
            # Deploy plugin
            deploy_success = await self.deployment_manager.deploy_plugin(
                package_path, instance
            )
            
            if not deploy_success:
                await self.deployment_manager.add_to_load_balancer(instance)
                return False
            
            # Health check
            health_ok = await self.deployment_manager.check_instance_health(instance)
            
            if not health_ok:
                await self.deployment_manager.rollback_instance(instance)
                await self.deployment_manager.add_to_load_balancer(instance)
                return False
            
            # Add back to load balancer
            await self.deployment_manager.add_to_load_balancer(instance)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Instance deployment failed: {e}")
            await self.deployment_manager.add_to_load_balancer(instance)
            return False
    
    async def _rollback_instances(self, instances: List[str]) -> bool:
        """Rollback specified instances."""
        self.logger.info(f"Rolling back {len(instances)} instances")
        
        rollback_tasks = [
            self.deployment_manager.rollback_instance(instance)
            for instance in instances
        ]
        
        results = await asyncio.gather(*rollback_tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        
        self.logger.info(f"Rollback completed: {success_count}/{len(instances)} successful")
        
        return success_count == len(instances)
```

## Best Practices

### Testing Best Practices

1. **Test Pyramid Structure**
   - Many fast unit tests (90%+)
   - Moderate integration tests (80%+)
   - Few comprehensive E2E tests (key flows)

2. **Epic 1 Compliance**
   - Every test verifies <50ms response time
   - Memory usage tests for <80MB limit
   - Performance regression detection

3. **Test Isolation**
   - No shared state between tests
   - Clean database/cache for each test
   - Mock external dependencies

4. **Error Testing**
   - Test all error conditions
   - Verify error messages and codes
   - Test timeout and resource exhaustion

5. **Security Testing**
   - Input validation for all parameters
   - Authentication and authorization
   - Sensitive data handling

### Deployment Best Practices

1. **Gradual Rollouts**
   - Use blue-green or rolling deployments
   - Monitor metrics during deployment
   - Automated rollback on issues

2. **Health Checks**
   - Comprehensive health endpoints
   - Deep health checks (dependencies)
   - Performance baseline validation

3. **Monitoring**
   - Real-time metrics collection
   - Alerting on threshold violations
   - Distributed tracing for debugging

4. **Rollback Strategy**
   - Automated rollback triggers
   - Quick rollback procedures
   - Data migration considerations

5. **Documentation**
   - Deployment runbooks
   - Troubleshooting guides
   - Performance baselines

This comprehensive guide provides everything needed to test, validate, and deploy plugins developed with the LeanVibe SDK while maintaining Epic 1 performance standards and ensuring production reliability.
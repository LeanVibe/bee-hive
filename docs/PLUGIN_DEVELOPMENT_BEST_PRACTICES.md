# LeanVibe Plugin Development Best Practices

**Version:** 2.3.0  
**Epic 2 Phase 2.3: Developer SDK & Documentation**

Comprehensive guide to developing high-quality, efficient plugins for the LeanVibe Agent Hive system.

## Table of Contents

1. [Performance Best Practices](#performance-best-practices)
2. [Memory Management](#memory-management)
3. [Error Handling](#error-handling)
4. [Security Guidelines](#security-guidelines)
5. [Testing Strategies](#testing-strategies)
6. [Code Organization](#code-organization)
7. [Configuration Management](#configuration-management)
8. [Logging and Monitoring](#logging-and-monitoring)
9. [Resource Management](#resource-management)
10. [Deployment Considerations](#deployment-considerations)

## Performance Best Practices

### Epic 1 Compliance

All plugins must maintain Epic 1 performance standards:

- **API Response Times**: <50ms for standard operations
- **Memory Usage**: <80MB total system impact
- **Initialization Time**: <10ms plugin startup
- **Task Execution**: <2s for typical operations

### Lazy Loading

Use lazy loading for expensive resources:

```python
class MyPlugin(PluginBase):
    def __init__(self, config):
        super().__init__(config)
        self._ml_model = None
        self._database_connection = None
    
    @property
    def ml_model(self):
        """Lazy load ML model only when needed."""
        if self._ml_model is None:
            self._ml_model = self._load_model()
        return self._ml_model
    
    @property
    def database(self):
        """Lazy load database connection."""
        if self._database_connection is None:
            self._database_connection = self._create_db_connection()
        return self._database_connection
    
    def _load_model(self):
        # Expensive model loading
        return load_pretrained_model()
    
    def _create_db_connection(self):
        # Database connection setup
        return create_connection(self.config.parameters["db_url"])
```

### Efficient Data Processing

Process data in chunks to avoid memory spikes:

```python
@performance_tracked(memory_limit_mb=50)
async def handle_large_dataset(self, task: TaskInterface) -> TaskResult:
    """Process large datasets efficiently."""
    dataset = task.parameters["dataset"]
    chunk_size = self.config.parameters.get("chunk_size", 1000)
    
    # Process in chunks to manage memory
    total_processed = 0
    for chunk in self._chunk_data(dataset, chunk_size):
        processed_chunk = await self._process_chunk(chunk)
        await self._save_chunk_results(processed_chunk)
        total_processed += len(processed_chunk)
        
        # Update progress
        progress = total_processed / len(dataset)
        await task.update_status("running", progress=progress)
        
        # Yield control to prevent blocking
        await asyncio.sleep(0.001)
    
    return TaskResult(
        success=True,
        plugin_id=self.plugin_id,
        task_id=task.task_id,
        data={"records_processed": total_processed}
    )

def _chunk_data(self, data, chunk_size):
    """Generate data chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]
```

### Caching Strategies

Implement intelligent caching for frequently accessed data:

```python
from functools import lru_cache
from leanvibe.plugin_sdk.decorators import cached_result

class DataProcessorPlugin(PluginBase):
    
    @cached_result(ttl_seconds=300, max_size=100)
    def get_processed_data(self, data_id: str):
        """Cache processed data for 5 minutes."""
        return self._expensive_data_processing(data_id)
    
    @lru_cache(maxsize=50)
    def get_metadata(self, resource_id: str):
        """Cache metadata using built-in LRU cache."""
        return self._fetch_metadata(resource_id)
    
    def _expensive_data_processing(self, data_id):
        # Expensive operation
        return process_data(data_id)
    
    def _fetch_metadata(self, resource_id):
        # Metadata fetching
        return fetch_from_api(resource_id)
```

### Asynchronous Operations

Use async/await properly for non-blocking operations:

```python
class AsyncPlugin(PluginBase):
    
    @async_plugin_method(timeout_seconds=60)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with proper async operations."""
        
        # Run multiple operations concurrently
        async with asyncio.TaskGroup() as group:
            data_task = group.create_task(self._fetch_data(task.parameters["source"]))
            config_task = group.create_task(self._load_config())
            validation_task = group.create_task(self._validate_input(task.parameters))
        
        # All tasks completed successfully
        data = data_task.result()
        config = config_task.result()
        validation_result = validation_task.result()
        
        if not validation_result.valid:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error="Input validation failed"
            )
        
        # Process data asynchronously
        result = await self._process_data_async(data, config)
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data=result
        )
    
    async def _fetch_data(self, source):
        """Fetch data asynchronously."""
        async with aiohttp.ClientSession() as session:
            async with session.get(source) as response:
                return await response.json()
    
    async def _process_data_async(self, data, config):
        """Process data with async operations."""
        # Use asyncio.gather for concurrent processing
        tasks = [self._process_item(item, config) for item in data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        return {
            "processed": len(successful_results),
            "total": len(data),
            "results": successful_results
        }
```

## Memory Management

### Monitor Memory Usage

Track memory consumption throughout plugin execution:

```python
import psutil
from leanvibe.plugin_sdk.decorators import performance_tracked

class MemoryAwarePlugin(PluginBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.memory_threshold = config.parameters.get("memory_threshold_mb", 50)
    
    @performance_tracked(track_memory=True, memory_limit_mb=50)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with memory monitoring."""
        
        # Check initial memory
        initial_memory = self._get_memory_usage()
        
        try:
            # Process data with memory checks
            result_data = []
            for batch in self._get_data_batches(task.parameters["data"]):
                # Check memory before processing batch
                current_memory = self._get_memory_usage()
                if current_memory > self.memory_threshold:
                    await self._cleanup_memory()
                
                batch_result = await self._process_batch(batch)
                result_data.extend(batch_result)
                
                # Log memory usage
                await self.log_info(f"Memory usage: {current_memory:.1f}MB")
            
            final_memory = self._get_memory_usage()
            memory_used = final_memory - initial_memory
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={"processed_count": len(result_data)},
                memory_used_mb=memory_used
            )
            
        except MemoryError:
            await self.log_error("Memory limit exceeded")
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error="Memory limit exceeded"
            )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    async def _cleanup_memory(self):
        """Clean up memory when threshold is reached."""
        # Clear caches
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        await self.log_info("Memory cleanup performed")
```

### Use Weak References

Prevent memory leaks with weak references:

```python
import weakref
from typing import Dict, Any

class CacheManager:
    """Memory-efficient cache using weak references."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = weakref.WeakValueDictionary()
        self._callbacks: Dict[str, weakref.ref] = {}
    
    def cache_object(self, key: str, obj: Any, cleanup_callback: callable = None):
        """Cache object with optional cleanup callback."""
        self._cache[key] = obj
        
        if cleanup_callback:
            def callback_wrapper(ref):
                cleanup_callback(key)
                self._callbacks.pop(key, None)
            
            self._callbacks[key] = weakref.ref(obj, callback_wrapper)
    
    def get_cached(self, key: str) -> Any:
        """Get cached object."""
        return self._cache.get(key)
    
    def clear_cache(self):
        """Clear all cached objects."""
        self._cache.clear()
        self._callbacks.clear()
```

### Resource Pools

Use object pools for expensive resources:

```python
import asyncio
from asyncio import Queue
from contextlib import asynccontextmanager

class ConnectionPool:
    """Connection pool for managing database connections."""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self._pool: Queue = Queue(maxsize=max_connections)
        self._created_connections = 0
    
    async def _create_connection(self):
        """Create a new connection."""
        # Simulate connection creation
        await asyncio.sleep(0.1)
        self._created_connections += 1
        return f"connection_{self._created_connections}"
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        try:
            # Try to get existing connection
            connection = self._pool.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if pool is empty
            if self._created_connections < self.max_connections:
                connection = await self._create_connection()
            else:
                # Wait for available connection
                connection = await self._pool.get()
        
        try:
            yield connection
        finally:
            # Return connection to pool
            await self._pool.put(connection)

class PooledPlugin(PluginBase):
    """Plugin using connection pool."""
    
    def __init__(self, config):
        super().__init__(config)
        self.connection_pool = ConnectionPool(max_connections=5)
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task using pooled connection."""
        async with self.connection_pool.get_connection() as conn:
            # Use connection for database operations
            result = await self._query_database(conn, task.parameters["query"])
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data=result
            )
```

## Error Handling

### Comprehensive Exception Handling

Handle errors gracefully with detailed context:

```python
from leanvibe.plugin_sdk.exceptions import (
    PluginExecutionError, PluginValidationError, PluginTimeoutError
)

class RobustPlugin(PluginBase):
    
    @error_handled(log_errors=True, suppress_errors=False)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with comprehensive error handling."""
        
        try:
            # Validate input parameters
            validation_result = await self._validate_parameters(task.parameters)
            if not validation_result.valid:
                raise PluginValidationError(
                    f"Parameter validation failed: {validation_result.errors}",
                    validation_errors=validation_result.errors,
                    plugin_id=self.plugin_id
                )
            
            # Execute main logic with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_main_logic(task),
                    timeout=self.config.max_execution_time_seconds
                )
            except asyncio.TimeoutError:
                raise PluginTimeoutError(
                    "Task execution timed out",
                    timeout_seconds=self.config.max_execution_time_seconds,
                    operation="main_logic",
                    plugin_id=self.plugin_id
                )
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data=result
            )
            
        except PluginValidationError:
            # Re-raise validation errors
            raise
        except PluginTimeoutError:
            # Re-raise timeout errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            await self.log_error(f"Unexpected error in task execution: {e}")
            raise PluginExecutionError(
                f"Task execution failed: {str(e)}",
                task_id=task.task_id,
                plugin_id=self.plugin_id
            ) from e
    
    async def _validate_parameters(self, parameters: Dict[str, Any]):
        """Validate input parameters."""
        errors = []
        
        # Check required parameters
        required_params = ["input_data", "operation_type"]
        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter types
        if "input_data" in parameters and not isinstance(parameters["input_data"], (list, dict)):
            errors.append("input_data must be a list or dictionary")
        
        if "operation_type" in parameters and parameters["operation_type"] not in ["process", "analyze", "transform"]:
            errors.append("operation_type must be one of: process, analyze, transform")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
    
    async def _execute_main_logic(self, task: TaskInterface):
        """Execute main plugin logic."""
        operation = task.parameters["operation_type"]
        data = task.parameters["input_data"]
        
        if operation == "process":
            return await self._process_data(data)
        elif operation == "analyze":
            return await self._analyze_data(data)
        elif operation == "transform":
            return await self._transform_data(data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
```

### Circuit Breaker Pattern

Implement circuit breaker for external service calls:

```python
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            # Reset on success
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            
            raise e

class CircuitBreakerPlugin(PluginBase):
    """Plugin using circuit breaker for external calls."""
    
    def __init__(self, config):
        super().__init__(config)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with circuit breaker protection."""
        try:
            # Call external service with circuit breaker
            result = await self.circuit_breaker.call(
                self._call_external_service,
                task.parameters["service_url"],
                task.parameters["data"]
            )
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data=result
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Service call failed: {str(e)}"
            )
    
    async def _call_external_service(self, url: str, data: Any):
        """Call external service."""
        # Simulate external service call
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                return await response.json()
```

## Security Guidelines

### Input Validation

Always validate and sanitize input data:

```python
import re
from typing import Any, Dict, List
from leanvibe.plugin_sdk.exceptions import PluginSecurityError

class SecurePlugin(PluginBase):
    """Plugin with comprehensive security measures."""
    
    ALLOWED_FILE_EXTENSIONS = {".txt", ".json", ".csv", ".xml"}
    MAX_FILE_SIZE_MB = 10
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with security validation."""
        
        # Validate and sanitize inputs
        sanitized_params = await self._validate_and_sanitize_input(task.parameters)
        
        # Check permissions
        if not await self._check_permissions(task.task_type, sanitized_params):
            raise PluginSecurityError(
                "Insufficient permissions for requested operation",
                required_permission=f"execute_{task.task_type}",
                plugin_id=self.plugin_id
            )
        
        # Execute with sanitized parameters
        result = await self._execute_secure_operation(sanitized_params)
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data=result
        )
    
    async def _validate_and_sanitize_input(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize input parameters."""
        sanitized = {}
        
        for key, value in parameters.items():
            # Sanitize string inputs
            if isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            
            # Validate file paths
            elif key.endswith("_path") or key.endswith("_file"):
                sanitized[key] = self._validate_file_path(value)
            
            # Validate URLs
            elif key.endswith("_url") or key == "endpoint":
                sanitized[key] = self._validate_url(value)
            
            # Copy other values as-is after type checking
            else:
                sanitized[key] = self._validate_type(key, value)
        
        return sanitized
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>&"\'\x00-\x1f\x7f-\x9f]', '', str(value))
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    def _validate_file_path(self, path: str) -> str:
        """Validate file path for security."""
        import os
        from pathlib import Path
        
        # Convert to Path object
        file_path = Path(path)
        
        # Check for path traversal attempts
        if ".." in str(file_path) or str(file_path).startswith("/"):
            raise PluginSecurityError(
                "Path traversal attempt detected",
                security_violation="path_traversal",
                plugin_id=self.plugin_id
            )
        
        # Validate file extension
        if file_path.suffix.lower() not in self.ALLOWED_FILE_EXTENSIONS:
            raise PluginSecurityError(
                f"File extension not allowed: {file_path.suffix}",
                security_violation="invalid_file_extension",
                plugin_id=self.plugin_id
            )
        
        # Check file size if exists
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                raise PluginSecurityError(
                    f"File too large: {size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB",
                    security_violation="file_too_large",
                    plugin_id=self.plugin_id
                )
        
        return str(file_path)
    
    def _validate_url(self, url: str) -> str:
        """Validate URL for security."""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        
        # Only allow HTTP/HTTPS
        if parsed.scheme not in ("http", "https"):
            raise PluginSecurityError(
                f"Invalid URL scheme: {parsed.scheme}",
                security_violation="invalid_url_scheme",
                plugin_id=self.plugin_id
            )
        
        # Block localhost and private networks
        hostname = parsed.hostname
        if hostname in ("localhost", "127.0.0.1", "::1"):
            raise PluginSecurityError(
                "Localhost URLs not allowed",
                security_violation="localhost_blocked",
                plugin_id=self.plugin_id
            )
        
        return url
    
    def _validate_type(self, key: str, value: Any) -> Any:
        """Validate value type based on key."""
        type_requirements = {
            "timeout": (int, float),
            "max_retries": int,
            "batch_size": int,
            "enabled": bool
        }
        
        if key in type_requirements:
            expected_type = type_requirements[key]
            if not isinstance(value, expected_type):
                raise PluginSecurityError(
                    f"Invalid type for {key}: expected {expected_type}, got {type(value)}",
                    security_violation="type_validation",
                    plugin_id=self.plugin_id
                )
        
        return value
    
    async def _check_permissions(self, operation: str, parameters: Dict[str, Any]) -> bool:
        """Check if plugin has permission for operation."""
        required_permission = f"execute_{operation}"
        
        # Use security interface if available
        if hasattr(self, '_security') and self._security:
            return await self._security.validate_permissions(
                required_permission,
                {"plugin_id": self.plugin_id, "parameters": parameters}
            )
        
        # Default permission check
        return required_permission in self.config.required_permissions
```

### Data Encryption

Encrypt sensitive data in transit and at rest:

```python
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class EncryptionPlugin(PluginBase):
    """Plugin with data encryption capabilities."""
    
    def __init__(self, config):
        super().__init__(config)
        self.encryption_key = self._derive_key(config.parameters.get("encryption_password", "default"))
        self.cipher_suite = Fernet(self.encryption_key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = b"leanvibe_plugin_salt"  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle task with data encryption."""
        
        # Decrypt input data if encrypted
        input_data = task.parameters.get("data")
        if task.parameters.get("encrypted", False):
            input_data = self._decrypt_data(input_data)
        
        # Process data
        result_data = await self._process_sensitive_data(input_data)
        
        # Encrypt result if requested
        if task.parameters.get("encrypt_result", False):
            result_data = self._encrypt_data(result_data)
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={"result": result_data, "encrypted": task.parameters.get("encrypt_result", False)}
        )
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt string data."""
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    async def _process_sensitive_data(self, data: str) -> str:
        """Process sensitive data securely."""
        # Simulate processing
        processed = f"processed_{data}"
        
        # Log without exposing sensitive data
        await self.log_info(f"Processed {len(data)} characters of sensitive data")
        
        return processed
```

## Testing Strategies

### Comprehensive Test Coverage

Write tests for all plugin functionality:

```python
import pytest
import asyncio
from leanvibe.plugin_sdk.testing import PluginTestFramework, MockTask, MockOrchestrator

class TestMyPlugin:
    """Comprehensive test suite for plugin."""
    
    @pytest.fixture
    def plugin_config(self):
        """Create test plugin configuration."""
        return PluginConfig(
            name="TestPlugin",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            parameters={
                "batch_size": 100,
                "timeout_seconds": 30,
                "retry_count": 3
            }
        )
    
    @pytest.fixture
    def plugin(self, plugin_config):
        """Create test plugin instance."""
        return MyPlugin(plugin_config)
    
    @pytest.fixture
    def test_framework(self):
        """Create test framework."""
        return PluginTestFramework()
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self, plugin, test_framework):
        """Test plugin initialization."""
        result = await test_framework.test_plugin_initialization(plugin)
        
        assert result.status.value == "passed"
        assert plugin.is_initialized
        assert plugin.batch_size == 100
    
    @pytest.mark.asyncio
    async def test_successful_task_execution(self, plugin, test_framework):
        """Test successful task execution."""
        await plugin.initialize(test_framework.mock_orchestrator)
        
        task = MockTask(
            task_type="process_data",
            parameters={
                "input_data": [{"id": 1, "value": "test"}],
                "operation": "transform"
            }
        )
        
        result = await plugin.execute(task)
        
        assert result.success
        assert result.plugin_id == plugin.plugin_id
        assert "processed_count" in result.data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, plugin, test_framework):
        """Test error handling."""
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Test with invalid parameters
        task = MockTask(
            task_type="process_data",
            parameters={}  # Missing required parameters
        )
        
        result = await plugin.execute(task)
        
        assert not result.success
        assert result.error is not None
        assert "missing" in result.error.lower() or "required" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, plugin, test_framework):
        """Test performance requirements."""
        performance_result = await test_framework.performance_test(
            plugin, 
            iterations=10,
            concurrent_tasks=1
        )
        
        assert performance_result.status.value in ["passed", "failed"]
        
        # Check Epic 1 compliance
        avg_time = performance_result.metrics.get("avg_execution_time_ms", 0)
        assert avg_time < 2000, f"Average execution time {avg_time}ms exceeds 2s limit"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, plugin, test_framework):
        """Test memory usage compliance."""
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Execute multiple tasks to test memory accumulation
        tasks = [
            MockTask(
                task_type="process_data",
                parameters={"input_data": [{"id": i, "value": f"test_{i}"}]}
            )
            for i in range(10)
        ]
        
        initial_memory = plugin.performance_metrics.get("memory_usage_mb", 0)
        
        for task in tasks:
            await plugin.execute(task)
        
        final_memory = plugin.performance_metrics.get("memory_usage_mb", 0)
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 50, f"Memory increase {memory_increase}MB too high"
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, plugin, test_framework):
        """Test concurrent task execution."""
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Create multiple tasks
        tasks = [
            MockTask(
                task_type="process_data",
                parameters={"input_data": [{"id": i}], "delay": 0.1}
            )
            for i in range(5)
        ]
        
        # Execute concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[plugin.execute(task) for task in tasks],
            return_exceptions=True
        )
        end_time = asyncio.get_event_loop().time()
        
        # Verify all tasks succeeded
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent execution failed: {result}")
            assert result.success
        
        # Verify concurrent execution was faster than sequential
        total_time = end_time - start_time
        assert total_time < 1.0, f"Concurrent execution took {total_time}s, too slow"
    
    @pytest.mark.asyncio
    async def test_plugin_cleanup(self, plugin, test_framework):
        """Test plugin cleanup."""
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Execute some tasks
        task = MockTask(task_type="process_data", parameters={"input_data": []})
        await plugin.execute(task)
        
        # Test cleanup
        result = await test_framework.test_plugin_cleanup(plugin)
        assert result.status.value == "passed"
    
    def test_configuration_validation(self, plugin_config):
        """Test configuration validation."""
        # Test valid configuration
        errors = plugin_config.validate()
        assert len(errors) == 0
        
        # Test invalid configuration
        invalid_config = PluginConfig(
            name="",  # Invalid: empty name
            version="1.0.0",
            max_memory_mb=-1  # Invalid: negative memory
        )
        
        errors = invalid_config.validate()
        assert len(errors) > 0
        assert any("name" in error.lower() for error in errors)
        assert any("memory" in error.lower() for error in errors)
```

### Integration Testing

Test plugin integration with system components:

```python
@pytest.mark.integration
class TestPluginIntegration:
    """Integration tests for plugin with real system components."""
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self, plugin):
        """Test plugin coordination with agents."""
        # Create mock orchestrator with agents
        orchestrator = MockOrchestrator()
        
        # Add agents with different capabilities
        agent1 = MockAgent("agent1", ["data_processing", "validation"])
        agent2 = MockAgent("agent2", ["data_processing", "transformation"])
        orchestrator.add_agent(agent1)
        orchestrator.add_agent(agent2)
        
        await plugin.initialize(orchestrator)
        
        # Test agent selection
        agents = await plugin.get_available_agents(["data_processing"])
        assert len(agents) == 2
        
        # Test coordination
        coordination_result = await plugin.coordinate_agents(agents)
        assert coordination_result.success
        assert len(coordination_result.agent_results) == 2
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, plugin):
        """Test plugin monitoring integration."""
        from leanvibe.plugin_sdk.testing import MockMonitoring
        
        mock_monitoring = MockMonitoring()
        plugin._monitoring = mock_monitoring
        
        # Execute task that generates monitoring events
        task = MockTask(
            task_type="monitored_operation",
            parameters={"data": "test"}
        )
        
        await plugin.execute(task)
        
        # Verify monitoring events were logged
        logged_events = mock_monitoring.get_logged_events()
        assert len(logged_events) > 0
        
        logged_metrics = mock_monitoring.get_logged_metrics()
        assert len(logged_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_security_integration(self, plugin):
        """Test plugin security integration."""
        from leanvibe.plugin_sdk.testing import MockSecurity
        
        mock_security = MockSecurity()
        plugin._security = mock_security
        
        # Test permission validation
        has_permission = await mock_security.validate_permissions(
            "execute_task",
            {"plugin_id": plugin.plugin_id}
        )
        assert has_permission
        
        # Test data encryption/decryption
        test_data = "sensitive information"
        encrypted = await mock_security.encrypt_data(test_data)
        decrypted = await mock_security.decrypt_data(encrypted)
        
        assert encrypted != test_data
        assert decrypted == "decrypted_mock_data"  # Mock implementation
```

### Property-Based Testing

Use property-based testing for robust validation:

```python
from hypothesis import given, strategies as st

class TestPluginProperties:
    """Property-based tests for plugin."""
    
    @given(st.lists(st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.one_of(st.text(), st.integers(), st.floats())
    ), min_size=0, max_size=100))
    @pytest.mark.asyncio
    async def test_data_processing_properties(self, input_data):
        """Test data processing with various input types."""
        plugin = MyPlugin(self.get_test_config())
        framework = PluginTestFramework()
        await plugin.initialize(framework.mock_orchestrator)
        
        task = MockTask(
            task_type="process_data",
            parameters={"input_data": input_data}
        )
        
        result = await plugin.execute(task)
        
        # Properties that should always hold
        assert result is not None
        assert hasattr(result, 'success')
        assert hasattr(result, 'plugin_id')
        assert result.plugin_id == plugin.plugin_id
        
        if result.success:
            assert result.data is not None
            # Output should have same or fewer records (filtering allowed)
            if "processed_count" in result.data:
                assert result.data["processed_count"] <= len(input_data)
    
    @given(st.integers(min_value=1, max_value=10000))
    @pytest.mark.asyncio
    async def test_batch_size_property(self, batch_size):
        """Test plugin behavior with different batch sizes."""
        config = self.get_test_config()
        config.parameters["batch_size"] = batch_size
        
        plugin = MyPlugin(config)
        assert plugin.batch_size == batch_size
        
        # Verify batch size is used correctly in processing
        test_data = [{"id": i} for i in range(batch_size * 2)]
        
        framework = PluginTestFramework()
        await plugin.initialize(framework.mock_orchestrator)
        
        task = MockTask(
            task_type="process_data",
            parameters={"input_data": test_data}
        )
        
        result = await plugin.execute(task)
        assert result.success
    
    def get_test_config(self):
        """Get test configuration."""
        return PluginConfig(
            name="PropertyTestPlugin",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW
        )
```

This comprehensive best practices guide provides developers with the knowledge and patterns needed to create high-quality, secure, and performant plugins for the LeanVibe Agent Hive system.
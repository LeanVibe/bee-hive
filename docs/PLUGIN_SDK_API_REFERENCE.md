# LeanVibe Plugin SDK API Reference

**Version:** 2.3.0  
**Epic 2 Phase 2.3: Developer SDK & Documentation**

Complete API reference for the LeanVibe Plugin Development Kit, enabling third-party developers to create high-quality plugins effortlessly.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Interfaces](#core-interfaces)
3. [Data Models](#data-models)
4. [Base Plugin Classes](#base-plugin-classes)
5. [Decorators](#decorators)
6. [Testing Framework](#testing-framework)
7. [Development Tools](#development-tools)
8. [Exceptions](#exceptions)
9. [Performance Considerations](#performance-considerations)
10. [Examples](#examples)

## Quick Start

### Installation

```python
# Import the SDK
from leanvibe.plugin_sdk import (
    PluginBase, PluginConfig, TaskInterface, TaskResult,
    plugin_method, performance_tracked
)
```

### Creating Your First Plugin

```python
from leanvibe.plugin_sdk import WorkflowPlugin, PluginConfig, TaskInterface, TaskResult

class MyAwesomePlugin(WorkflowPlugin):
    """A simple example plugin."""
    
    @plugin_method(timeout_seconds=30)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle plugin task execution."""
        # Your plugin logic here
        result_data = {"message": "Task completed successfully"}
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data=result_data
        )

# Create and use the plugin
config = PluginConfig(name="MyAwesomePlugin", version="1.0.0")
plugin = MyAwesomePlugin(config)
```

### Template Generation

```python
from leanvibe.plugin_sdk.tools import PluginGenerator

generator = PluginGenerator()
project_path = generator.create_plugin_project(
    plugin_name="MyWorkflowPlugin",
    plugin_type="workflow",
    output_dir="./my_plugins",
    author_name="Your Name",
    author_email="your.email@example.com"
)
```

## Core Interfaces

### AgentInterface

Protocol for interacting with agents in the system.

```python
from typing import Protocol, List, Dict, Any, Optional

class AgentInterface(Protocol):
    @property
    def agent_id(self) -> str: ...
    
    @property 
    def capabilities(self) -> List[str]: ...
    
    @property
    def status(self) -> str: ...
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult: ...
    
    async def get_context(self) -> Dict[str, Any]: ...
    
    async def send_message(self, message: str, target: Optional[str] = None) -> bool: ...
```

**Example Usage:**

```python
# Get available agents
agents = await plugin.get_available_agents(["data_processing"])

# Execute task on agent
if agents:
    agent = agents[0]
    task = {"task_id": "123", "type": "process", "data": {...}}
    result = await agent.execute_task(task)
    
    # Send message to agent
    await agent.send_message("Task completed", target="coordinator")
```

### TaskInterface

Protocol for task operations within plugins.

```python
class TaskInterface(Protocol):
    @property
    def task_id(self) -> str: ...
    
    @property
    def task_type(self) -> str: ...
    
    @property
    def parameters(self) -> Dict[str, Any]: ...
    
    @property
    def priority(self) -> int: ...
    
    async def update_status(self, status: str, progress: float = 0.0) -> None: ...
    
    async def add_result(self, key: str, value: Any) -> None: ...
    
    async def get_dependency_results(self) -> Dict[str, Any]: ...
```

**Example Usage:**

```python
async def handle_task(self, task: TaskInterface) -> TaskResult:
    # Update task status
    await task.update_status("running", progress=0.1)
    
    # Get task parameters
    input_data = task.parameters.get("input_data", {})
    
    # Process data...
    processed_data = process(input_data)
    
    # Add intermediate results
    await task.add_result("processed", processed_data)
    
    # Update progress
    await task.update_status("running", progress=0.8)
    
    # Get dependency results if needed
    dependencies = await task.get_dependency_results()
    
    # Complete task
    await task.update_status("completed", progress=1.0)
    
    return TaskResult(success=True, plugin_id=self.plugin_id, task_id=task.task_id)
```

### OrchestratorInterface

Protocol for interacting with the orchestrator.

```python
class OrchestratorInterface(Protocol):
    async def get_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[AgentInterface]: ...
    
    async def create_task(self, task_type: str, parameters: Dict[str, Any]) -> TaskInterface: ...
    
    async def schedule_task(self, task: TaskInterface, delay_seconds: int = 0) -> str: ...
    
    async def get_system_metrics(self) -> Dict[str, Any]: ...
    
    async def broadcast_event(self, event: PluginEvent) -> None: ...
```

**Example Usage:**

```python
# Get agents with specific capabilities
data_agents = await orchestrator.get_agents({"capabilities": ["data_processing"]})

# Create subtask
subtask = await orchestrator.create_task(
    task_type="data_analysis",
    parameters={"dataset": "user_interactions"}
)

# Schedule task for later execution
schedule_id = await orchestrator.schedule_task(subtask, delay_seconds=300)

# Get system metrics
metrics = await orchestrator.get_system_metrics()
cpu_usage = metrics.get("cpu_usage", 0)

# Broadcast event
event = PluginEvent(
    event_type="data_processed",
    plugin_id=self.plugin_id,
    data={"records": 1000}
)
await orchestrator.broadcast_event(event)
```

### MonitoringInterface

Protocol for monitoring and observability.

```python
class MonitoringInterface(Protocol):
    async def log_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None: ...
    
    async def log_event(self, event: PluginEvent) -> None: ...
    
    async def create_alert(self, message: str, severity: str = "info") -> None: ...
    
    async def get_performance_data(self, plugin_id: str) -> Dict[str, Any]: ...
```

**Example Usage:**

```python
# Log custom metrics
await monitoring.log_metric("processing_time", 245.7, {"unit": "ms"})
await monitoring.log_metric("records_processed", 1500, {"batch": "daily"})

# Log events
event = PluginEvent(
    event_type="error_rate_spike",
    plugin_id=self.plugin_id,
    data={"error_rate": 0.15, "threshold": 0.10},
    severity=EventSeverity.WARNING
)
await monitoring.log_event(event)

# Create alerts
await monitoring.create_alert(
    "High memory usage detected", 
    severity="warning"
)

# Get performance data
perf_data = await monitoring.get_performance_data(self.plugin_id)
```

## Data Models

### PluginConfig

Configuration and metadata for plugins.

```python
@dataclass
class PluginConfig:
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    plugin_id: Optional[str] = None
    plugin_type: Optional[PluginType] = None
    
    # Configuration parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    # Resource limits  
    max_memory_mb: int = 100
    max_execution_time_seconds: int = 300
    max_concurrent_tasks: int = 10
    
    # Security and permissions
    required_permissions: List[str] = field(default_factory=list)
    security_level: str = "standard"
    sandbox_enabled: bool = True
```

**Example Usage:**

```python
config = PluginConfig(
    name="DataProcessorPlugin",
    version="2.1.0",
    description="Advanced data processing capabilities",
    author="Data Team",
    plugin_type=PluginType.WORKFLOW,
    parameters={
        "batch_size": 1000,
        "timeout_minutes": 30,
        "output_format": "json"
    },
    capabilities=["data_processing", "batch_operations"],
    max_memory_mb=512,
    max_execution_time_seconds=1800,
    required_permissions=["data_read", "data_write"],
    security_level="elevated"
)

# Validate configuration
errors = config.validate()
if errors:
    print(f"Configuration errors: {errors}")
```

### TaskResult

Result of task execution.

```python
@dataclass
class TaskResult:
    success: bool
    plugin_id: str
    task_id: str
    execution_time_ms: float = 0.0
    
    # Result data
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Error information
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    memory_used_mb: float = 0.0
    cpu_time_ms: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
```

**Example Usage:**

```python
# Create successful result
result = TaskResult(
    success=True,
    plugin_id=self.plugin_id,
    task_id=task.task_id,
    execution_time_ms=150.5
)

# Add result data
result.set_result("processed_records", 1500)
result.set_result("output_file", "/tmp/processed_data.json")

# Add artifacts
result.add_artifact("/tmp/processing_log.txt")
result.add_artifact("/tmp/error_report.json")

# Set performance metrics
result.memory_used_mb = 45.2
result.cpu_time_ms = 1250.0

# Mark as completed
result.complete()

# Create error result
error_result = TaskResult(
    success=False,
    plugin_id=self.plugin_id,
    task_id=task.task_id
)
error_result.set_error(
    "Data validation failed", 
    error_code="VALIDATION_ERROR",
    details={"invalid_records": 23, "total_records": 1000}
)
```

### PluginEvent

Plugin event for monitoring and coordination.

```python
@dataclass
class PluginEvent:
    event_type: str
    plugin_id: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    severity: EventSeverity = EventSeverity.INFO
    
    # Context information
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    execution_id: Optional[str] = None
    
    # Tags for filtering and search
    tags: Dict[str, str] = field(default_factory=dict)
```

**Example Usage:**

```python
# Create informational event
info_event = PluginEvent(
    event_type="data_processing_started",
    plugin_id=self.plugin_id,
    data={"batch_size": 1000, "estimated_time": "5m"},
    task_id=task.task_id
)
info_event.add_tag("category", "processing")
info_event.add_tag("priority", "normal")

# Create error event  
error_event = PluginEvent(
    event_type="processing_error",
    plugin_id=self.plugin_id,
    data={"error_message": "Invalid data format", "record_count": 50},
    severity=EventSeverity.ERROR,
    task_id=task.task_id
)

# Create performance event
perf_event = PluginEvent(
    event_type="performance_alert",
    plugin_id=self.plugin_id,
    data={"memory_usage_mb": 150.5, "threshold_mb": 100.0},
    severity=EventSeverity.WARNING
)

# Emit events
await plugin.emit_event(info_event)
await plugin.emit_event(error_event)
await plugin.emit_event(perf_event)
```

## Base Plugin Classes

### PluginBase

Base class for all LeanVibe plugins.

```python
class PluginBase(ABC):
    def __init__(self, config: PluginConfig): ...
    
    @property
    def name(self) -> str: ...
    
    @property  
    def version(self) -> str: ...
    
    @property
    def plugin_type(self) -> PluginType: ...
    
    @property
    def is_running(self) -> bool: ...
    
    @property
    def performance_metrics(self) -> Dict[str, Any]: ...
    
    async def initialize(self, orchestrator: OrchestratorInterface) -> bool: ...
    
    async def execute(self, task: TaskInterface) -> TaskResult: ...
    
    async def coordinate_agents(self, agents: List[AgentInterface]) -> CoordinationResult: ...
    
    async def cleanup(self) -> bool: ...
    
    @abstractmethod
    async def handle_task(self, task: TaskInterface) -> TaskResult: ...
    
    # Lifecycle hooks (optional overrides)
    async def _on_initialize(self) -> None: ...
    async def _before_execute(self, task: TaskInterface) -> None: ...
    async def _after_execute(self, task: TaskInterface, result: TaskResult) -> None: ...
    async def _on_error(self, task: TaskInterface, error: Exception) -> None: ...
    async def _on_cleanup(self) -> None: ...
    
    # Utility methods
    async def log_info(self, message: str, **kwargs) -> None: ...
    async def log_error(self, message: str, **kwargs) -> None: ...
    async def create_alert(self, message: str, severity: str = "info") -> None: ...
    async def get_available_agents(self, capabilities: Optional[List[str]] = None) -> List[AgentInterface]: ...
    async def create_subtask(self, task_type: str, parameters: Dict[str, Any]) -> TaskInterface: ...
```

**Example Implementation:**

```python
class DataProcessorPlugin(PluginBase):
    """Plugin for processing large datasets."""
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.batch_size = config.parameters.get("batch_size", 1000)
        self.processing_stats = {"total_processed": 0, "errors": 0}
    
    async def _on_initialize(self) -> None:
        """Initialize data processor."""
        await self.log_info("Initializing DataProcessorPlugin")
        
        # Validate configuration
        if self.batch_size <= 0:
            raise PluginConfigurationError("batch_size must be positive")
        
        # Initialize processing environment
        self.processing_stats = {"total_processed": 0, "errors": 0}
        
        await self.log_info(f"Initialized with batch_size={self.batch_size}")
    
    @performance_tracked(alert_threshold_ms=5000, memory_limit_mb=200)
    @plugin_method(timeout_seconds=300, max_retries=2)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Process data according to task parameters."""
        dataset = task.parameters.get("dataset")
        if not dataset:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error="Missing required parameter: dataset"
            )
        
        try:
            # Update task status
            await task.update_status("running", progress=0.1)
            
            # Load dataset
            data = await self._load_dataset(dataset)
            await task.update_status("running", progress=0.3)
            
            # Process in batches
            total_records = len(data)
            processed_data = []
            
            for i in range(0, total_records, self.batch_size):
                batch = data[i:i + self.batch_size]
                processed_batch = await self._process_batch(batch)
                processed_data.extend(processed_batch)
                
                # Update progress
                progress = 0.3 + (0.6 * (i + len(batch)) / total_records)
                await task.update_status("running", progress=progress)
            
            # Save results
            output_path = await self._save_results(processed_data)
            await task.update_status("running", progress=0.95)
            
            # Update statistics
            self.processing_stats["total_processed"] += len(processed_data)
            
            # Complete task
            await task.update_status("completed", progress=1.0)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "records_processed": len(processed_data),
                    "output_path": output_path,
                    "processing_stats": self.processing_stats
                }
            )
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            await self.log_error(f"Data processing failed: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                data={"processing_stats": self.processing_stats}
            )
    
    async def _load_dataset(self, dataset: str) -> List[Dict[str, Any]]:
        """Load dataset from source."""
        # Implementation depends on data source
        await self.log_info(f"Loading dataset: {dataset}")
        # Simulate loading...
        return [{"id": i, "value": f"data_{i}"} for i in range(10000)]
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of records."""
        # Apply processing logic
        processed = []
        for record in batch:
            processed_record = {
                **record,
                "processed": True,
                "processed_at": datetime.utcnow().isoformat()
            }
            processed.append(processed_record)
        return processed
    
    async def _save_results(self, data: List[Dict[str, Any]]) -> str:
        """Save processed results."""
        output_path = f"/tmp/processed_data_{uuid.uuid4().hex[:8]}.json"
        # Save data to file...
        await self.log_info(f"Saved {len(data)} records to {output_path}")
        return output_path
    
    async def _on_cleanup(self) -> None:
        """Cleanup resources."""
        await self.log_info("Cleaning up DataProcessorPlugin")
        # Cleanup temporary files, connections, etc.
```

### WorkflowPlugin

Specialized base class for workflow plugins.

```python
class WorkflowPlugin(PluginBase):
    """Base class for workflow plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.WORKFLOW
        super().__init__(config)
```

### MonitoringPlugin

Specialized base class for monitoring plugins.

```python
class MonitoringPlugin(PluginBase):
    """Base class for monitoring plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.MONITORING
        super().__init__(config)
```

### SecurityPlugin

Specialized base class for security plugins.

```python
class SecurityPlugin(PluginBase):
    """Base class for security plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.SECURITY
        super().__init__(config)
```

## Decorators

The SDK provides powerful decorators for enhancing plugin methods.

### @plugin_method

Decorator for plugin methods with error handling and logging.

```python
@plugin_method(
    timeout_seconds=30,
    max_retries=3,
    retry_delay=1.0,
    log_execution=True
)
def process_data(self, data):
    # Method implementation
    pass
```

### @async_plugin_method

Decorator for async plugin methods.

```python
@async_plugin_method(
    timeout_seconds=60,
    max_retries=2,
    retry_delay=2.0
)
async def process_data_async(self, data):
    # Async method implementation
    pass
```

### @performance_tracked

Decorator for tracking method performance.

```python
@performance_tracked(
    track_memory=True,
    track_execution_time=True,
    alert_threshold_ms=1000,
    memory_limit_mb=100
)
def expensive_operation(self, data):
    # Implementation
    pass
```

### @error_handled

Decorator for comprehensive error handling.

```python
@error_handled(
    default_return=[],
    suppress_errors=True,
    log_errors=True
)
def get_data_list(self):
    # Implementation that might fail
    pass
```

### @cached_result

Decorator for caching method results.

```python
@cached_result(
    ttl_seconds=300,
    max_size=100,
    ignore_args=["debug"]
)
def expensive_computation(self, data, debug=False):
    # Expensive computation
    return result
```

### @validate_params

Decorator for parameter validation.

```python
@validate_params(
    data=lambda x: isinstance(x, dict),
    count=lambda x: isinstance(x, int) and x > 0
)
def process_data(self, data, count):
    # Implementation
    pass
```

## Testing Framework

Comprehensive testing utilities for plugin development.

### PluginTestFramework

Main testing framework class.

```python
from leanvibe.plugin_sdk.testing import PluginTestFramework

framework = PluginTestFramework()

# Test plugin initialization
result = await framework.test_plugin_initialization(plugin)

# Test task execution
result = await framework.test_plugin_task_execution(plugin)

# Run performance tests
result = await framework.performance_test(plugin, iterations=10)

# Run integration tests
result = await framework.integration_test(plugin)

# Run full test suite
suite = await framework.run_full_test_suite(plugin)
```

### Mock Implementations

Use mock objects for testing.

```python
from leanvibe.plugin_sdk.testing import MockOrchestrator, MockAgent, MockTask

# Create mock orchestrator
mock_orchestrator = MockOrchestrator()

# Add mock agents
agent1 = MockAgent("agent1", ["capability1", "capability2"])
agent2 = MockAgent("agent2", ["capability2", "capability3"])
mock_orchestrator.add_agent(agent1)
mock_orchestrator.add_agent(agent2)

# Create mock task
mock_task = MockTask(
    task_type="test_task",
    parameters={"input": "test_data"}
)

# Initialize plugin with mocks
await plugin.initialize(mock_orchestrator)

# Execute task
result = await plugin.execute(mock_task)
```

### Test Result Analysis

```python
# Analyze test results
test_suite = await framework.run_full_test_suite(plugin)

print(f"Total tests: {test_suite.total_tests}")
print(f"Passed: {test_suite.passed_tests}")
print(f"Failed: {test_suite.failed_tests}")
print(f"Success rate: {test_suite.get_success_rate():.2%}")

# Get detailed report
report = framework.get_test_report(test_suite)

# Generate HTML report
from leanvibe.plugin_sdk.testing import TestRunner
runner = TestRunner()
html_report = runner.generate_html_report([test_suite])
```

## Development Tools

### PluginGenerator

Generate plugin templates and projects.

```python
from leanvibe.plugin_sdk.tools import PluginGenerator

generator = PluginGenerator()

# Create full plugin project
project_path = generator.create_plugin_project(
    plugin_name="MyAwesomePlugin",
    plugin_type="workflow",
    output_dir="./plugins",
    author_name="Your Name",
    author_email="your.email@example.com"
)

# Quick template generation
template_code = generator.create_plugin_template(
    plugin_name="QuickPlugin",
    plugin_type="monitoring"
)
```

### PluginPackager

Package plugins for distribution.

```python
from leanvibe.plugin_sdk.tools import PluginPackager

packager = PluginPackager()

# Package plugin
package_info = packager.package_plugin(
    plugin_dir="./my_plugin",
    output_dir="./packages",
    include_dependencies=True
)

print(f"Package created: {package_info.package_path}")
print(f"Size: {package_info.size_bytes} bytes")
```

### PerformanceProfiler

Profile plugin performance.

```python
from leanvibe.plugin_sdk.tools import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile plugin execution
profile = await profiler.profile_plugin(
    plugin=my_plugin,
    task=test_task,
    duration_seconds=60.0
)

# Analyze results
print(f"Total execution time: {profile.duration_seconds}s")
print(f"Peak memory: {profile.peak_memory_mb}MB")
print(f"Hotspots: {len(profile.hotspots)}")

# Export profile
profiler.export_profile(profile.profile_id, "profile_report.json")
```

### DebugConsole

Interactive debugging console.

```python
from leanvibe.plugin_sdk.tools import DebugConsole

console = DebugConsole()

# Register plugin for debugging
session_id = console.register_plugin(my_plugin)

# Execute debug commands
result = console.execute_command(session_id, "status")
result = console.execute_command(session_id, "inspect config")
result = console.execute_command(session_id, "watch execution_count")
result = console.execute_command(session_id, "metrics")
```

## Exceptions

Comprehensive error handling with detailed context.

### Base Exception

```python
from leanvibe.plugin_sdk.exceptions import PluginSDKError

try:
    # Plugin operation
    pass
except PluginSDKError as e:
    print(f"Plugin error: {e.message}")
    print(f"Error code: {e.error_code}")
    print(f"Recovery suggestions: {e.recovery_suggestions}")
    print(f"Error details: {e.details}")
```

### Specific Exceptions

```python
from leanvibe.plugin_sdk.exceptions import (
    PluginInitializationError,
    PluginExecutionError,
    PluginValidationError,
    PluginTimeoutError,
    PluginConfigurationError,
    PluginDependencyError,
    PluginSecurityError,
    PluginResourceError
)

# Handle specific error types
try:
    await plugin.initialize(orchestrator)
except PluginInitializationError as e:
    # Handle initialization failure
    pass
except PluginConfigurationError as e:
    # Handle configuration issues
    pass
```

### Error Utilities

```python
from leanvibe.plugin_sdk.exceptions import (
    handle_plugin_error,
    create_validation_error,
    create_timeout_error,
    create_resource_error
)

# Convert generic exceptions
try:
    # Some operation
    pass
except Exception as e:
    plugin_error = handle_plugin_error(e, plugin_id="my_plugin")
    raise plugin_error

# Create specific errors
validation_error = create_validation_error(
    plugin_id="my_plugin",
    validation_errors=["Missing required field", "Invalid value"]
)

timeout_error = create_timeout_error(
    plugin_id="my_plugin",
    operation="data_processing",
    timeout_seconds=30.0
)
```

## Performance Considerations

### Epic 1 Compliance

The SDK is designed to preserve Epic 1 performance achievements:

- **<50ms API response times**: All SDK operations are optimized for speed
- **<80MB memory usage**: Lazy loading and efficient data structures
- **Minimal overhead**: Plugin framework adds <5% overhead

### Best Practices

```python
# Use lazy loading for heavy resources
class MyPlugin(PluginBase):
    def __init__(self, config):
        super().__init__(config)
        self._heavy_resource = None
    
    @property
    def heavy_resource(self):
        if self._heavy_resource is None:
            self._heavy_resource = load_heavy_resource()
        return self._heavy_resource

# Use performance tracking
@performance_tracked(alert_threshold_ms=100)
async def handle_task(self, task):
    # Implementation
    pass

# Cache expensive computations
@cached_result(ttl_seconds=300)
def expensive_computation(self, data):
    # Expensive operation
    return result

# Use context managers for resource management
async def handle_task(self, task):
    async with self.resource_manager() as resource:
        # Use resource
        pass
```

### Memory Management

```python
# Monitor memory usage
@performance_tracked(track_memory=True, memory_limit_mb=50)
def process_large_dataset(self, data):
    # Process data in chunks
    for chunk in self.chunk_data(data, chunk_size=1000):
        self.process_chunk(chunk)
        # Memory is automatically tracked

# Use weak references for cache
import weakref
class MyPlugin(PluginBase):
    def __init__(self, config):
        super().__init__(config)
        self._cache = weakref.WeakValueDictionary()
```

## Examples

### Complete Workflow Plugin

```python
from datetime import datetime
from typing import Dict, List, Any
from leanvibe.plugin_sdk import (
    WorkflowPlugin, PluginConfig, TaskInterface, TaskResult,
    plugin_method, performance_tracked, PluginType
)

class DataPipelinePlugin(WorkflowPlugin):
    """Complete data processing pipeline plugin."""
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.pipeline_steps = config.parameters.get("pipeline_steps", [])
        self.output_format = config.parameters.get("output_format", "json")
    
    async def _on_initialize(self) -> None:
        """Initialize the pipeline."""
        await self.log_info("Initializing DataPipelinePlugin")
        
        if not self.pipeline_steps:
            raise PluginConfigurationError("Pipeline steps not configured")
        
        await self.log_info(f"Configured {len(self.pipeline_steps)} pipeline steps")
    
    @performance_tracked(alert_threshold_ms=30000, memory_limit_mb=200)
    @plugin_method(timeout_seconds=1800, max_retries=2)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Execute data pipeline."""
        input_data = task.parameters.get("input_data")
        if not input_data:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error="No input data provided"
            )
        
        try:
            await task.update_status("running", progress=0.0)
            
            # Execute pipeline steps
            current_data = input_data
            step_results = []
            
            for i, step in enumerate(self.pipeline_steps):
                await self.log_info(f"Executing step {i+1}: {step['name']}")
                
                step_result = await self._execute_pipeline_step(step, current_data, task)
                step_results.append(step_result)
                
                if not step_result["success"]:
                    return TaskResult(
                        success=False,
                        plugin_id=self.plugin_id,
                        task_id=task.task_id,
                        error=f"Pipeline step {i+1} failed: {step_result['error']}",
                        data={"completed_steps": i, "step_results": step_results}
                    )
                
                current_data = step_result["output"]
                progress = (i + 1) / len(self.pipeline_steps)
                await task.update_status("running", progress=progress)
            
            # Save final results
            output_path = await self._save_pipeline_output(current_data)
            
            await task.update_status("completed", progress=1.0)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "output_path": output_path,
                    "records_processed": len(current_data) if isinstance(current_data, list) else 1,
                    "pipeline_steps": len(self.pipeline_steps),
                    "step_results": step_results
                }
            )
            
        except Exception as e:
            await self.log_error(f"Pipeline execution failed: {e}")
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _execute_pipeline_step(self, step: Dict[str, Any], data: Any, task: TaskInterface) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step_type = step.get("type")
        step_params = step.get("parameters", {})
        
        if step_type == "filter":
            return await self._filter_step(data, step_params)
        elif step_type == "transform":
            return await self._transform_step(data, step_params)
        elif step_type == "aggregate":
            return await self._aggregate_step(data, step_params)
        elif step_type == "validate":
            return await self._validate_step(data, step_params)
        else:
            return {
                "success": False,
                "error": f"Unknown step type: {step_type}",
                "output": data
            }
    
    async def _filter_step(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Filter data based on criteria."""
        try:
            filter_field = params.get("field")
            filter_value = params.get("value")
            filter_op = params.get("operation", "equals")
            
            if not filter_field:
                return {"success": False, "error": "Filter field not specified", "output": data}
            
            filtered_data = []
            for record in data:
                if filter_op == "equals" and record.get(filter_field) == filter_value:
                    filtered_data.append(record)
                elif filter_op == "contains" and filter_value in str(record.get(filter_field, "")):
                    filtered_data.append(record)
                elif filter_op == "greater_than" and record.get(filter_field, 0) > filter_value:
                    filtered_data.append(record)
            
            return {
                "success": True,
                "error": None,
                "output": filtered_data,
                "metrics": {
                    "input_count": len(data),
                    "output_count": len(filtered_data),
                    "filtered_count": len(data) - len(filtered_data)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e), "output": data}
    
    async def _transform_step(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Transform data records."""
        try:
            transformations = params.get("transformations", [])
            
            transformed_data = []
            for record in data:
                transformed_record = record.copy()
                
                for transform in transformations:
                    field = transform.get("field")
                    operation = transform.get("operation")
                    value = transform.get("value")
                    
                    if operation == "set":
                        transformed_record[field] = value
                    elif operation == "uppercase" and field in transformed_record:
                        transformed_record[field] = str(transformed_record[field]).upper()
                    elif operation == "lowercase" and field in transformed_record:
                        transformed_record[field] = str(transformed_record[field]).lower()
                    elif operation == "multiply" and field in transformed_record:
                        transformed_record[field] = float(transformed_record[field]) * float(value)
                
                transformed_data.append(transformed_record)
            
            return {
                "success": True,
                "error": None,
                "output": transformed_data,
                "metrics": {
                    "transformations_applied": len(transformations),
                    "records_transformed": len(transformed_data)
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e), "output": data}
    
    async def _aggregate_step(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Aggregate data."""
        try:
            group_by = params.get("group_by")
            aggregations = params.get("aggregations", [])
            
            if not group_by:
                # Global aggregation
                result = {}
                for agg in aggregations:
                    field = agg.get("field")
                    operation = agg.get("operation")
                    
                    values = [record.get(field, 0) for record in data if field in record]
                    
                    if operation == "sum":
                        result[f"{field}_sum"] = sum(values)
                    elif operation == "avg":
                        result[f"{field}_avg"] = sum(values) / len(values) if values else 0
                    elif operation == "count":
                        result[f"{field}_count"] = len(values)
                    elif operation == "max":
                        result[f"{field}_max"] = max(values) if values else 0
                    elif operation == "min":
                        result[f"{field}_min"] = min(values) if values else 0
                
                return {
                    "success": True,
                    "error": None,
                    "output": [result],
                    "metrics": {"input_records": len(data), "output_records": 1}
                }
            else:
                # Group by aggregation
                groups = {}
                for record in data:
                    key = record.get(group_by)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(record)
                
                aggregated_data = []
                for group_key, group_records in groups.items():
                    result = {group_by: group_key}
                    
                    for agg in aggregations:
                        field = agg.get("field")
                        operation = agg.get("operation")
                        
                        values = [record.get(field, 0) for record in group_records if field in record]
                        
                        if operation == "sum":
                            result[f"{field}_sum"] = sum(values)
                        elif operation == "avg":
                            result[f"{field}_avg"] = sum(values) / len(values) if values else 0
                        elif operation == "count":
                            result[f"{field}_count"] = len(values)
                    
                    aggregated_data.append(result)
                
                return {
                    "success": True,
                    "error": None,
                    "output": aggregated_data,
                    "metrics": {
                        "input_records": len(data),
                        "output_groups": len(groups),
                        "output_records": len(aggregated_data)
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e), "output": data}
    
    async def _validate_step(self, data: List[Dict], params: Dict) -> Dict[str, Any]:
        """Validate data quality."""
        try:
            validations = params.get("validations", [])
            validation_errors = []
            
            for i, record in enumerate(data):
                for validation in validations:
                    field = validation.get("field")
                    rule = validation.get("rule")
                    
                    if rule == "required" and (field not in record or record[field] is None):
                        validation_errors.append(f"Record {i}: Missing required field '{field}'")
                    elif rule == "numeric" and field in record:
                        try:
                            float(record[field])
                        except (ValueError, TypeError):
                            validation_errors.append(f"Record {i}: Field '{field}' is not numeric")
                    elif rule == "email" and field in record:
                        import re
                        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                        if not re.match(email_pattern, str(record[field])):
                            validation_errors.append(f"Record {i}: Field '{field}' is not a valid email")
            
            if validation_errors:
                return {
                    "success": False,
                    "error": f"Validation failed with {len(validation_errors)} errors",
                    "output": data,
                    "validation_errors": validation_errors
                }
            else:
                return {
                    "success": True,
                    "error": None,
                    "output": data,
                    "metrics": {"records_validated": len(data), "validation_errors": 0}
                }
        except Exception as e:
            return {"success": False, "error": str(e), "output": data}
    
    async def _save_pipeline_output(self, data: Any) -> str:
        """Save pipeline output to file."""
        import json
        import uuid
        
        output_file = f"/tmp/pipeline_output_{uuid.uuid4().hex[:8]}.{self.output_format}"
        
        if self.output_format == "json":
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif self.output_format == "csv" and isinstance(data, list) and data:
            import csv
            with open(output_file, 'w', newline='') as f:
                if isinstance(data[0], dict):
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        
        await self.log_info(f"Saved pipeline output to {output_file}")
        return output_file

# Usage example
config = PluginConfig(
    name="DataPipelinePlugin",
    version="1.0.0",
    plugin_type=PluginType.WORKFLOW,
    parameters={
        "output_format": "json",
        "pipeline_steps": [
            {
                "name": "Filter Active Users",
                "type": "filter",
                "parameters": {
                    "field": "status",
                    "value": "active",
                    "operation": "equals"
                }
            },
            {
                "name": "Transform Names",
                "type": "transform",
                "parameters": {
                    "transformations": [
                        {"field": "name", "operation": "uppercase"},
                        {"field": "processed_at", "operation": "set", "value": datetime.utcnow().isoformat()}
                    ]
                }
            },
            {
                "name": "Validate Data",
                "type": "validate",
                "parameters": {
                    "validations": [
                        {"field": "email", "rule": "required"},
                        {"field": "email", "rule": "email"},
                        {"field": "age", "rule": "numeric"}
                    ]
                }
            }
        ]
    }
)

plugin = DataPipelinePlugin(config)
```

### Testing Example

```python
import pytest
from leanvibe.plugin_sdk.testing import PluginTestFramework, MockTask

@pytest.mark.asyncio
async def test_data_pipeline_plugin():
    """Test the complete data pipeline plugin."""
    
    # Create test configuration
    config = PluginConfig(
        name="TestDataPipeline",
        version="1.0.0",
        plugin_type=PluginType.WORKFLOW,
        parameters={
            "output_format": "json",
            "pipeline_steps": [
                {
                    "name": "Filter Test",
                    "type": "filter",
                    "parameters": {
                        "field": "active",
                        "value": True,
                        "operation": "equals"
                    }
                }
            ]
        }
    )
    
    # Create plugin
    plugin = DataPipelinePlugin(config)
    
    # Create test framework
    framework = PluginTestFramework()
    
    # Initialize plugin
    await plugin.initialize(framework.mock_orchestrator)
    
    # Create test task
    test_data = [
        {"id": 1, "name": "John", "active": True, "email": "john@example.com"},
        {"id": 2, "name": "Jane", "active": False, "email": "jane@example.com"},
        {"id": 3, "name": "Bob", "active": True, "email": "bob@example.com"}
    ]
    
    task = MockTask(
        task_type="pipeline",
        parameters={"input_data": test_data}
    )
    
    # Execute pipeline
    result = await plugin.execute(task)
    
    # Verify results
    assert result.success
    assert "output_path" in result.data
    assert result.data["records_processed"] == 2  # Only active users
    
    # Run full test suite
    test_suite = await framework.run_full_test_suite(plugin)
    assert test_suite.get_success_rate() > 0.8  # At least 80% tests pass
```

This comprehensive API reference provides developers with everything they need to create powerful, efficient plugins for the LeanVibe Agent Hive system while maintaining Epic 1 performance standards.
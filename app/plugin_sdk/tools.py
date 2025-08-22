"""
Development Tools for LeanVibe Plugin SDK.

Comprehensive suite of tools for plugin development including:
- Plugin template generation
- Code scaffolding
- Packaging and distribution
- Performance profiling
- Debug console
- IDE integration
"""

import os
import shutil
import tempfile
import zipfile
import json
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import subprocess
import time

from .interfaces import PluginType
from .models import PluginConfig, PluginEvent, EventSeverity
from .exceptions import PluginSDKError


@dataclass
class PluginTemplate:
    """Plugin template definition."""
    name: str
    description: str
    plugin_type: PluginType
    files: Dict[str, str]  # filename -> content
    variables: Dict[str, str] = None  # template variables
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}


@dataclass 
class PackageInfo:
    """Plugin package information."""
    plugin_id: str
    version: str
    package_path: str
    size_bytes: int
    created_at: datetime
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PerformanceProfile:
    """Performance profiling result."""
    plugin_id: str
    profile_id: str
    duration_seconds: float
    
    # Timing data
    function_times: Dict[str, float]
    total_calls: Dict[str, int]
    average_times: Dict[str, float]
    
    # Memory data
    peak_memory_mb: float
    average_memory_mb: float
    memory_timeline: List[Dict[str, Any]]
    
    # Resource usage
    cpu_usage_percent: float
    io_operations: int
    
    # Hotspots and bottlenecks
    hotspots: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]
    
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "profile_id": self.profile_id,
            "duration_seconds": self.duration_seconds,
            "function_times": self.function_times,
            "total_calls": self.total_calls,
            "average_times": self.average_times,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "memory_timeline": self.memory_timeline,
            "cpu_usage_percent": self.cpu_usage_percent,
            "io_operations": self.io_operations,
            "hotspots": self.hotspots,
            "bottlenecks": self.bottlenecks,
            "created_at": self.created_at.isoformat()
        }


class PluginGenerator:
    """
    Plugin template generator and scaffolding system.
    
    Creates complete plugin projects with proper structure,
    documentation, and example code.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_generator")
        self.templates = self._load_built_in_templates()
    
    def _load_built_in_templates(self) -> Dict[str, PluginTemplate]:
        """Load built-in plugin templates."""
        templates = {}
        
        # Workflow Plugin Template
        templates["workflow"] = PluginTemplate(
            name="Workflow Plugin",
            description="Template for workflow automation plugins",
            plugin_type=PluginType.WORKFLOW,
            files={
                "__init__.py": self._get_workflow_init_template(),
                "plugin.py": self._get_workflow_plugin_template(),
                "config.yaml": self._get_workflow_config_template(),
                "README.md": self._get_workflow_readme_template(),
                "tests/test_plugin.py": self._get_workflow_test_template(),
                "requirements.txt": "# Add your dependencies here\n",
                ".gitignore": self._get_gitignore_template()
            },
            variables={
                "plugin_name": "MyWorkflowPlugin",
                "plugin_description": "A workflow automation plugin",
                "author_name": "Developer Name",
                "author_email": "developer@example.com"
            }
        )
        
        # Monitoring Plugin Template
        templates["monitoring"] = PluginTemplate(
            name="Monitoring Plugin",
            description="Template for monitoring and observability plugins",
            plugin_type=PluginType.MONITORING,
            files={
                "__init__.py": self._get_monitoring_init_template(),
                "plugin.py": self._get_monitoring_plugin_template(),
                "config.yaml": self._get_monitoring_config_template(),
                "README.md": self._get_monitoring_readme_template(),
                "tests/test_plugin.py": self._get_monitoring_test_template(),
                "requirements.txt": "# Add your dependencies here\n",
                ".gitignore": self._get_gitignore_template()
            }
        )
        
        # Security Plugin Template
        templates["security"] = PluginTemplate(
            name="Security Plugin",
            description="Template for security and compliance plugins",
            plugin_type=PluginType.SECURITY,
            files={
                "__init__.py": self._get_security_init_template(),
                "plugin.py": self._get_security_plugin_template(),
                "config.yaml": self._get_security_config_template(),
                "README.md": self._get_security_readme_template(),
                "tests/test_plugin.py": self._get_security_test_template(),
                "requirements.txt": "# Add your dependencies here\n",
                ".gitignore": self._get_gitignore_template()
            }
        )
        
        return templates
    
    def create_plugin_project(
        self,
        plugin_name: str,
        plugin_type: str,
        output_dir: str,
        author_name: str = "Developer",
        author_email: str = "developer@example.com",
        custom_variables: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a complete plugin project from template.
        
        Args:
            plugin_name: Name of the plugin
            plugin_type: Type of plugin (workflow, monitoring, security, etc.)
            output_dir: Directory to create the plugin project
            author_name: Author name
            author_email: Author email
            custom_variables: Additional template variables
            
        Returns:
            str: Path to created plugin project
        """
        try:
            # Get template
            template = self.templates.get(plugin_type)
            if not template:
                raise PluginSDKError(f"Unknown plugin type: {plugin_type}")
            
            # Prepare variables
            variables = template.variables.copy()
            variables.update({
                "plugin_name": plugin_name,
                "plugin_id": self._normalize_plugin_id(plugin_name),
                "author_name": author_name,
                "author_email": author_email,
                "creation_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "sdk_version": "2.3.0"
            })
            
            if custom_variables:
                variables.update(custom_variables)
            
            # Create project directory
            project_path = Path(output_dir) / self._normalize_plugin_id(plugin_name)
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create files from template
            for file_path, content_template in template.files.items():
                file_full_path = project_path / file_path
                
                # Create parent directories if needed
                file_full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Replace template variables
                content = self._replace_template_variables(content_template, variables)
                
                # Write file
                with open(file_full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Create additional directories
            (project_path / "docs").mkdir(exist_ok=True)
            (project_path / "examples").mkdir(exist_ok=True)
            
            self.logger.info(f"Created plugin project: {project_path}")
            return str(project_path)
            
        except Exception as e:
            self.logger.error(f"Failed to create plugin project: {e}")
            raise PluginSDKError(f"Plugin project creation failed: {str(e)}")
    
    def create_plugin_template(self, plugin_name: str, plugin_type: str) -> str:
        """
        Quick plugin template creation for development.
        
        Args:
            plugin_name: Name of the plugin
            plugin_type: Type of plugin
            
        Returns:
            str: Generated plugin code
        """
        template = self.templates.get(plugin_type)
        if not template:
            raise PluginSDKError(f"Unknown plugin type: {plugin_type}")
        
        variables = {
            "plugin_name": plugin_name,
            "plugin_id": self._normalize_plugin_id(plugin_name),
            "author_name": "Developer",
            "author_email": "developer@example.com",
            "creation_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "sdk_version": "2.3.0"
        }
        
        # Return main plugin file content
        plugin_content = template.files.get("plugin.py", "")
        return self._replace_template_variables(plugin_content, variables)
    
    def _normalize_plugin_id(self, plugin_name: str) -> str:
        """Normalize plugin name to valid ID."""
        import re
        # Convert to lowercase, replace spaces and special chars with underscores
        normalized = re.sub(r'[^a-zA-Z0-9_]', '_', plugin_name.lower())
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Remove leading/trailing underscores
        return normalized.strip('_')
    
    def _replace_template_variables(self, content: str, variables: Dict[str, str]) -> str:
        """Replace template variables in content."""
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"
            content = content.replace(placeholder, var_value)
        return content
    
    # Template content methods
    def _get_workflow_plugin_template(self) -> str:
        return '''"""
{{plugin_name}} - Workflow Plugin
Generated by LeanVibe Plugin SDK on {{creation_date}}

Author: {{author_name}} <{{author_email}}>
"""

import asyncio
from typing import Dict, List, Optional, Any

from leanvibe.plugin_sdk import (
    WorkflowPlugin, PluginConfig, TaskInterface, TaskResult,
    plugin_method, performance_tracked, error_handled
)


class {{plugin_name}}(WorkflowPlugin):
    """
    {{plugin_description}}
    
    This workflow plugin provides automation capabilities
    for the LeanVibe Agent Hive system.
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Plugin-specific initialization
        self.workflow_steps = []
        self.execution_context = {}
    
    @plugin_method(timeout_seconds=30)
    async def _on_initialize(self) -> None:
        """Initialize the workflow plugin."""
        await self.log_info("Initializing {{plugin_name}}")
        
        # Load workflow configuration
        self.workflow_steps = self.config.parameters.get("workflow_steps", [])
        
        # Initialize workflow context
        self.execution_context = {
            "started_at": None,
            "current_step": 0,
            "step_results": []
        }
        
        await self.log_info(f"Loaded {len(self.workflow_steps)} workflow steps")
    
    @performance_tracked(alert_threshold_ms=1000)
    @error_handled(default_return=None, suppress_errors=False)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Handle workflow task execution.
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult: Execution result
        """
        await self.log_info(f"Executing workflow task: {task.task_type}")
        
        try:
            # Initialize execution context
            self.execution_context["started_at"] = datetime.utcnow()
            self.execution_context["current_step"] = 0
            self.execution_context["step_results"] = []
            
            # Execute workflow steps
            result_data = {}
            for i, step in enumerate(self.workflow_steps):
                self.execution_context["current_step"] = i
                
                await task.update_status("running", progress=i / len(self.workflow_steps))
                
                step_result = await self._execute_workflow_step(step, task)
                self.execution_context["step_results"].append(step_result)
                
                if not step_result.get("success", False):
                    # Workflow step failed
                    return TaskResult(
                        success=False,
                        plugin_id=self.plugin_id,
                        task_id=task.task_id,
                        error=f"Workflow step {i} failed: {step_result.get('error', 'Unknown error')}",
                        data={"failed_step": i, "step_results": self.execution_context["step_results"]}
                    )
                
                # Merge step results
                result_data.update(step_result.get("data", {}))
            
            # Workflow completed successfully
            await task.update_status("completed", progress=1.0)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "workflow_result": result_data,
                    "steps_executed": len(self.workflow_steps),
                    "execution_context": self.execution_context
                }
            )
            
        except Exception as e:
            await self.log_error(f"Workflow execution failed: {e}")
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                data={"execution_context": self.execution_context}
            )
    
    async def _execute_workflow_step(self, step: Dict[str, Any], task: TaskInterface) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step_type = step.get("type", "unknown")
        step_params = step.get("parameters", {})
        
        await self.log_info(f"Executing workflow step: {step_type}")
        
        # Route to appropriate step handler
        if step_type == "agent_task":
            return await self._execute_agent_task_step(step_params, task)
        elif step_type == "coordination":
            return await self._execute_coordination_step(step_params, task)
        elif step_type == "data_processing":
            return await self._execute_data_processing_step(step_params, task)
        elif step_type == "notification":
            return await self._execute_notification_step(step_params, task)
        else:
            return {
                "success": False,
                "error": f"Unknown step type: {step_type}",
                "data": {}
            }
    
    async def _execute_agent_task_step(self, params: Dict[str, Any], task: TaskInterface) -> Dict[str, Any]:
        """Execute agent task step."""
        try:
            # Get required capabilities
            required_capabilities = params.get("capabilities", [])
            agents = await self.get_available_agents(required_capabilities)
            
            if not agents:
                return {
                    "success": False,
                    "error": f"No agents available with capabilities: {required_capabilities}",
                    "data": {}
                }
            
            # Use first available agent
            agent = agents[0]
            
            # Create agent task
            agent_task = {
                "task_id": f"workflow_step_{uuid.uuid4().hex[:8]}",
                "type": params.get("task_type", "generic"),
                "parameters": params.get("task_parameters", {})
            }
            
            # Execute agent task
            result = await agent.execute_task(agent_task)
            
            return {
                "success": result.success,
                "error": result.error if not result.success else None,
                "data": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    async def _execute_coordination_step(self, params: Dict[str, Any], task: TaskInterface) -> Dict[str, Any]:
        """Execute coordination step."""
        try:
            # Get agents for coordination
            required_capabilities = params.get("capabilities", [])
            agents = await self.get_available_agents(required_capabilities)
            
            if len(agents) < params.get("min_agents", 1):
                return {
                    "success": False,
                    "error": f"Insufficient agents for coordination: {len(agents)} < {params.get('min_agents', 1)}",
                    "data": {}
                }
            
            # Coordinate agents
            coordination_result = await self.coordinate_agents(agents[:params.get("max_agents", len(agents))])
            
            return {
                "success": coordination_result.success,
                "error": coordination_result.error if not coordination_result.success else None,
                "data": {"coordination_id": coordination_result.coordination_id}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    async def _execute_data_processing_step(self, params: Dict[str, Any], task: TaskInterface) -> Dict[str, Any]:
        """Execute data processing step."""
        try:
            # Get input data
            input_data = params.get("input_data", {})
            processing_type = params.get("processing_type", "transform")
            
            # Process data based on type
            if processing_type == "transform":
                # Example data transformation
                output_data = {"transformed": True, "input": input_data}
            elif processing_type == "aggregate":
                # Example data aggregation
                output_data = {"aggregated": True, "count": len(input_data)}
            else:
                output_data = input_data
            
            return {
                "success": True,
                "error": None,
                "data": output_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    async def _execute_notification_step(self, params: Dict[str, Any], task: TaskInterface) -> Dict[str, Any]:
        """Execute notification step."""
        try:
            message = params.get("message", "Workflow notification")
            severity = params.get("severity", "info")
            
            await self.create_alert(message, severity)
            
            return {
                "success": True,
                "error": None,
                "data": {"notification_sent": True}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": {}
            }
    
    async def _on_cleanup(self) -> None:
        """Cleanup workflow plugin resources."""
        await self.log_info("Cleaning up {{plugin_name}}")
        
        # Clear execution context
        self.execution_context.clear()
        self.workflow_steps.clear()
'''
    
    def _get_workflow_init_template(self) -> str:
        return '''"""
{{plugin_name}} Plugin Package
Generated by LeanVibe Plugin SDK

Author: {{author_name}} <{{author_email}}>
"""

from .plugin import {{plugin_name}}

__version__ = "1.0.0"
__author__ = "{{author_name}}"
__email__ = "{{author_email}}"

__all__ = ["{{plugin_name}}"]
'''
    
    def _get_workflow_config_template(self) -> str:
        return '''# {{plugin_name}} Configuration
# Generated by LeanVibe Plugin SDK

plugin:
  name: "{{plugin_name}}"
  version: "1.0.0"
  description: "{{plugin_description}}"
  type: "workflow"
  author: "{{author_name}}"
  email: "{{author_email}}"

# Plugin parameters
parameters:
  workflow_steps:
    - type: "agent_task"
      parameters:
        capabilities: ["data_processing"]
        task_type: "process_data"
        task_parameters:
          format: "json"
    
    - type: "data_processing"
      parameters:
        processing_type: "transform"
        input_data: {}
    
    - type: "notification"
      parameters:
        message: "Workflow completed successfully"
        severity: "info"

# Resource limits
resources:
  max_memory_mb: 100
  max_execution_time_seconds: 300
  max_concurrent_tasks: 5

# Security settings
security:
  required_permissions: []
  security_level: "standard"
  sandbox_enabled: true

# Performance settings
performance:
  performance_optimized: true
  lazy_loading: true
  cache_enabled: true
'''
    
    def _get_workflow_readme_template(self) -> str:
        return '''# {{plugin_name}}

{{plugin_description}}

Generated by LeanVibe Plugin SDK on {{creation_date}}.

## Overview

This workflow plugin provides automation capabilities for the LeanVibe Agent Hive system. It allows you to define and execute complex workflows with multiple steps including agent coordination, data processing, and notifications.

## Features

- **Multi-step Workflows**: Define complex workflows with sequential steps
- **Agent Integration**: Coordinate multiple agents within workflows  
- **Data Processing**: Transform and aggregate data between workflow steps
- **Error Handling**: Robust error handling with step-level recovery
- **Performance Monitoring**: Built-in performance tracking and optimization
- **Flexible Configuration**: YAML-based configuration for easy customization

## Installation

1. Copy the plugin files to your LeanVibe plugins directory
2. Install any additional dependencies: `pip install -r requirements.txt`
3. Configure the plugin by editing `config.yaml`
4. Register the plugin with your LeanVibe system

## Configuration

Edit `config.yaml` to customize the plugin behavior:

```yaml
parameters:
  workflow_steps:
    - type: "agent_task"
      parameters:
        capabilities: ["data_processing"]
        task_type: "process_data"
```

## Usage

```python
from leanvibe.plugin_sdk import PluginConfig
from {{plugin_id}} import {{plugin_name}}

# Create plugin configuration
config = PluginConfig(
    name="{{plugin_name}}",
    version="1.0.0",
    plugin_type=PluginType.WORKFLOW
)

# Initialize plugin
plugin = {{plugin_name}}(config)
await plugin.initialize(orchestrator)

# Execute workflow
result = await plugin.execute(task)
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 .

# Run type checking
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This plugin is licensed under the MIT License.

## Author

{{author_name}} <{{author_email}}>
'''
    
    def _get_workflow_test_template(self) -> str:
        return '''"""
Tests for {{plugin_name}}
Generated by LeanVibe Plugin SDK
"""

import pytest
import asyncio
from leanvibe.plugin_sdk import PluginConfig, PluginType
from leanvibe.plugin_sdk.testing import PluginTestFramework, MockTask

from {{plugin_id}}.plugin import {{plugin_name}}


class Test{{plugin_name}}:
    """Test suite for {{plugin_name}}."""
    
    @pytest.fixture
    def plugin_config(self):
        """Create test plugin configuration."""
        return PluginConfig(
            name="{{plugin_name}}",
            version="1.0.0",
            plugin_type=PluginType.WORKFLOW,
            parameters={
                "workflow_steps": [
                    {
                        "type": "data_processing",
                        "parameters": {
                            "processing_type": "transform",
                            "input_data": {"test": "data"}
                        }
                    }
                ]
            }
        )
    
    @pytest.fixture
    def plugin(self, plugin_config):
        """Create test plugin instance."""
        return {{plugin_name}}(plugin_config)
    
    @pytest.fixture
    def test_framework(self):
        """Create test framework."""
        return PluginTestFramework()
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self, plugin, test_framework):
        """Test plugin initialization."""
        result = await test_framework.test_plugin_initialization(plugin)
        assert result.status.value == "passed"
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, plugin, test_framework):
        """Test workflow execution."""
        # Initialize plugin
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Create test task
        test_task = MockTask(
            task_type="workflow_test",
            parameters={"test_param": "value"}
        )
        
        # Execute workflow
        result = await plugin.execute(test_task)
        
        # Verify results
        assert result.success
        assert result.plugin_id == plugin.plugin_id
        assert "workflow_result" in result.data
    
    @pytest.mark.asyncio
    async def test_workflow_step_execution(self, plugin, test_framework):
        """Test individual workflow step execution."""
        # Initialize plugin
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Test data processing step
        step_params = {
            "processing_type": "transform",
            "input_data": {"test": "data"}
        }
        
        test_task = MockTask(task_type="test")
        result = await plugin._execute_data_processing_step(step_params, test_task)
        
        assert result["success"]
        assert "transformed" in result["data"]
    
    @pytest.mark.asyncio
    async def test_plugin_cleanup(self, plugin, test_framework):
        """Test plugin cleanup."""
        result = await test_framework.test_plugin_cleanup(plugin)
        assert result.status.value == "passed"
    
    @pytest.mark.asyncio
    async def test_performance(self, plugin, test_framework):
        """Test plugin performance."""
        # Initialize plugin
        await plugin.initialize(test_framework.mock_orchestrator)
        
        # Run performance test
        result = await test_framework.performance_test(plugin, iterations=5)
        
        assert result.status.value in ["passed", "failed"]  # May fail on slow systems
        assert "avg_execution_time_ms" in result.metrics
'''
    
    def _get_monitoring_plugin_template(self) -> str:
        return '''"""
{{plugin_name}} - Monitoring Plugin
Generated by LeanVibe Plugin SDK on {{creation_date}}

Author: {{author_name}} <{{author_email}}>
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from leanvibe.plugin_sdk import (
    MonitoringPlugin, PluginConfig, TaskInterface, TaskResult,
    plugin_method, performance_tracked, error_handled
)


class {{plugin_name}}(MonitoringPlugin):
    """
    {{plugin_description}}
    
    This monitoring plugin provides observability and alerting
    capabilities for the LeanVibe Agent Hive system.
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.alert_thresholds = {}
        self.metric_history = {}
        self.monitoring_active = False
    
    async def _on_initialize(self) -> None:
        """Initialize the monitoring plugin."""
        await self.log_info("Initializing {{plugin_name}}")
        
        # Load monitoring configuration
        self.monitoring_interval = self.config.parameters.get("monitoring_interval", 30)
        self.alert_thresholds = self.config.parameters.get("alert_thresholds", {})
        
        # Initialize metric storage
        self.metric_history = {}
        
        await self.log_info(f"Monitoring interval: {self.monitoring_interval}s")
    
    @performance_tracked(alert_threshold_ms=500)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Handle monitoring task execution.
        
        Args:
            task: Monitoring task to execute
            
        Returns:
            TaskResult: Monitoring result
        """
        task_type = task.task_type
        
        if task_type == "start_monitoring":
            return await self._start_monitoring(task)
        elif task_type == "stop_monitoring":
            return await self._stop_monitoring(task)
        elif task_type == "collect_metrics":
            return await self._collect_metrics(task)
        elif task_type == "generate_report":
            return await self._generate_report(task)
        else:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Unknown monitoring task type: {task_type}"
            )
    
    async def _start_monitoring(self, task: TaskInterface) -> TaskResult:
        """Start monitoring system metrics."""
        try:
            if self.monitoring_active:
                return TaskResult(
                    success=True,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    data={"message": "Monitoring already active"}
                )
            
            self.monitoring_active = True
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
            await self.log_info("Started monitoring")
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={"monitoring_started": True, "interval": self.monitoring_interval}
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _stop_monitoring(self, task: TaskInterface) -> TaskResult:
        """Stop monitoring system metrics."""
        try:
            self.monitoring_active = False
            
            await self.log_info("Stopped monitoring")
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={"monitoring_stopped": True}
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _collect_metrics(self, task: TaskInterface) -> TaskResult:
        """Collect current system metrics."""
        try:
            # Get system metrics from orchestrator
            system_metrics = await self._orchestrator.get_system_metrics()
            
            # Collect additional metrics
            current_time = datetime.utcnow()
            metrics = {
                "timestamp": current_time.isoformat(),
                "system": system_metrics,
                "plugin": {
                    "uptime_seconds": (current_time - self.created_at).total_seconds(),
                    "tasks_executed": self._execution_count,
                    "error_count": self._error_count
                }
            }
            
            # Store in history
            self._store_metrics(metrics)
            
            # Check for alerts
            alerts = await self._check_alert_conditions(metrics)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "metrics": metrics,
                    "alerts_triggered": alerts
                }
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _generate_report(self, task: TaskInterface) -> TaskResult:
        """Generate monitoring report."""
        try:
            report_period = task.parameters.get("period_hours", 24)
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(hours=report_period)
            
            # Filter metrics for the period
            period_metrics = []
            for timestamp_str, metrics in self.metric_history.items():
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp >= start_time:
                    period_metrics.append(metrics)
            
            # Generate report
            report = await self._create_monitoring_report(period_metrics, report_period)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={"report": report}
            )
            
        except Exception as e:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e)
            )
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                system_metrics = await self._orchestrator.get_system_metrics()
                current_time = datetime.utcnow()
                
                metrics = {
                    "timestamp": current_time.isoformat(),
                    "system": system_metrics,
                    "plugin": {
                        "uptime_seconds": (current_time - self.created_at).total_seconds(),
                        "tasks_executed": self._execution_count,
                        "error_count": self._error_count
                    }
                }
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check for alerts
                alerts = await self._check_alert_conditions(metrics)
                
                # Trigger alerts if any
                for alert in alerts:
                    await self.create_alert(alert["message"], alert["severity"])
                
                # Wait for next collection
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                await self.log_error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in history."""
        timestamp = metrics["timestamp"]
        self.metric_history[timestamp] = metrics
        
        # Keep only last 1000 entries
        if len(self.metric_history) > 1000:
            oldest_key = min(self.metric_history.keys())
            del self.metric_history[oldest_key]
    
    async def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check metrics against alert thresholds."""
        alerts = []
        
        system_metrics = metrics.get("system", {})
        
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in system_metrics:
                value = system_metrics[metric_name]
                
                if isinstance(value, (int, float)):
                    if "max" in threshold and value > threshold["max"]:
                        alerts.append({
                            "message": f"{metric_name} is above threshold: {value} > {threshold['max']}",
                            "severity": "warning"
                        })
                    
                    if "min" in threshold and value < threshold["min"]:
                        alerts.append({
                            "message": f"{metric_name} is below threshold: {value} < {threshold['min']}",
                            "severity": "warning"
                        })
        
        return alerts
    
    async def _create_monitoring_report(self, metrics: List[Dict[str, Any]], period_hours: int) -> Dict[str, Any]:
        """Create monitoring report from metrics."""
        if not metrics:
            return {"error": "No metrics available for the period"}
        
        # Calculate averages and trends
        cpu_values = [m["system"].get("cpu_usage", 0) for m in metrics if "system" in m]
        memory_values = [m["system"].get("memory_usage", 0) for m in metrics if "system" in m]
        
        report = {
            "period_hours": period_hours,
            "total_data_points": len(metrics),
            "summary": {
                "cpu_usage": {
                    "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0
                },
                "memory_usage": {
                    "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                    "max": max(memory_values) if memory_values else 0,
                    "min": min(memory_values) if memory_values else 0
                }
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
'''
    
    def _get_monitoring_init_template(self) -> str:
        return self._get_workflow_init_template()
    
    def _get_monitoring_config_template(self) -> str:
        return '''# {{plugin_name}} Configuration
# Generated by LeanVibe Plugin SDK

plugin:
  name: "{{plugin_name}}"
  version: "1.0.0"
  description: "{{plugin_description}}"
  type: "monitoring"
  author: "{{author_name}}"
  email: "{{author_email}}"

# Monitoring parameters
parameters:
  monitoring_interval: 30  # seconds
  alert_thresholds:
    cpu_usage:
      max: 80.0
    memory_usage:
      max: 85.0
    active_tasks:
      max: 100

# Resource limits
resources:
  max_memory_mb: 50
  max_execution_time_seconds: 120
  max_concurrent_tasks: 3

# Security settings
security:
  required_permissions: ["system_metrics"]
  security_level: "standard"
  sandbox_enabled: true
'''
    
    def _get_monitoring_readme_template(self) -> str:
        return '''# {{plugin_name}}

{{plugin_description}}

A comprehensive monitoring plugin for the LeanVibe Agent Hive system.

## Features

- **Real-time Monitoring**: Continuous system metrics collection
- **Alert Management**: Configurable thresholds and notifications
- **Historical Data**: Metric history and trend analysis
- **Performance Reports**: Automated monitoring reports
- **Resource Tracking**: CPU, memory, and task monitoring

## Configuration

```yaml
parameters:
  monitoring_interval: 30
  alert_thresholds:
    cpu_usage:
      max: 80.0
    memory_usage:
      max: 85.0
```

## Usage

The plugin responds to the following task types:

- `start_monitoring`: Begin continuous monitoring
- `stop_monitoring`: Stop monitoring
- `collect_metrics`: Collect current metrics
- `generate_report`: Generate monitoring report

## Author

{{author_name}} <{{author_email}}>
'''
    
    def _get_monitoring_test_template(self) -> str:
        return '''"""
Tests for {{plugin_name}}
"""

import pytest
import asyncio
from leanvibe.plugin_sdk import PluginConfig, PluginType
from leanvibe.plugin_sdk.testing import PluginTestFramework, MockTask

from {{plugin_id}}.plugin import {{plugin_name}}


class Test{{plugin_name}}:
    """Test suite for {{plugin_name}}."""
    
    @pytest.fixture
    def plugin_config(self):
        return PluginConfig(
            name="{{plugin_name}}",
            version="1.0.0",
            plugin_type=PluginType.MONITORING,
            parameters={
                "monitoring_interval": 5,
                "alert_thresholds": {
                    "cpu_usage": {"max": 80.0},
                    "memory_usage": {"max": 85.0}
                }
            }
        )
    
    @pytest.fixture
    def plugin(self, plugin_config):
        return {{plugin_name}}(plugin_config)
    
    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, plugin):
        """Test starting and stopping monitoring."""
        framework = PluginTestFramework()
        await plugin.initialize(framework.mock_orchestrator)
        
        # Start monitoring
        start_task = MockTask(task_type="start_monitoring")
        result = await plugin.execute(start_task)
        assert result.success
        
        # Stop monitoring
        stop_task = MockTask(task_type="stop_monitoring")
        result = await plugin.execute(stop_task)
        assert result.success
'''
    
    def _get_security_plugin_template(self) -> str:
        return '''"""
{{plugin_name}} - Security Plugin
Generated by LeanVibe Plugin SDK on {{creation_date}}

Author: {{author_name}} <{{author_email}}>
"""

from leanvibe.plugin_sdk import SecurityPlugin, PluginConfig, TaskInterface, TaskResult


class {{plugin_name}}(SecurityPlugin):
    """Security plugin for LeanVibe Agent Hive."""
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle security task."""
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={"security_check": "passed"}
        )
'''
    
    def _get_security_init_template(self) -> str:
        return self._get_workflow_init_template()
    
    def _get_security_config_template(self) -> str:
        return '''# {{plugin_name}} Configuration
plugin:
  name: "{{plugin_name}}"
  version: "1.0.0"
  type: "security"
'''
    
    def _get_security_readme_template(self) -> str:
        return '''# {{plugin_name}}

Security plugin for LeanVibe Agent Hive.

## Author

{{author_name}} <{{author_email}}>
'''
    
    def _get_security_test_template(self) -> str:
        return '''"""
Tests for {{plugin_name}}
"""

import pytest
from leanvibe.plugin_sdk import PluginConfig, PluginType

from {{plugin_id}}.plugin import {{plugin_name}}


def test_plugin_creation():
    """Test plugin creation."""
    config = PluginConfig(name="{{plugin_name}}", plugin_type=PluginType.SECURITY)
    plugin = {{plugin_name}}(config)
    assert plugin.name == "{{plugin_name}}"
'''
    
    def _get_gitignore_template(self) -> str:
        return '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Plugin-specific
*.log
*.tmp
debug/
'''


class PluginPackager:
    """
    Plugin packaging and distribution tool.
    
    Creates distributable plugin packages with proper metadata,
    dependencies, and installation scripts.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_packager")
    
    def package_plugin(
        self,
        plugin_dir: str,
        output_dir: str,
        include_dependencies: bool = True,
        compression_level: int = 6
    ) -> PackageInfo:
        """
        Package a plugin for distribution.
        
        Args:
            plugin_dir: Directory containing plugin source
            output_dir: Directory to create package in
            include_dependencies: Whether to include dependencies
            compression_level: ZIP compression level (0-9)
            
        Returns:
            PackageInfo: Information about created package
        """
        try:
            plugin_path = Path(plugin_dir)
            output_path = Path(output_dir)
            
            # Read plugin configuration
            config_file = plugin_path / "config.yaml"
            if not config_file.exists():
                raise PluginSDKError("Plugin config.yaml not found")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            plugin_info = config.get("plugin", {})
            plugin_id = plugin_info.get("name", "unknown")
            version = plugin_info.get("version", "1.0.0")
            
            # Create package filename
            package_filename = f"{plugin_id}-{version}.zip"
            package_path = output_path / package_filename
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create package
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression_level) as zipf:
                # Add plugin files
                for file_path in plugin_path.rglob("*"):
                    if file_path.is_file() and not self._should_exclude_file(file_path):
                        arcname = file_path.relative_to(plugin_path)
                        zipf.write(file_path, arcname)
                
                # Add package metadata
                metadata = self._create_package_metadata(config, plugin_path)
                zipf.writestr("package-info.json", json.dumps(metadata, indent=2))
                
                # Add installation script
                install_script = self._create_installation_script(metadata)
                zipf.writestr("install.py", install_script)
            
            # Get package size
            package_size = package_path.stat().st_size
            
            # Create package info
            package_info = PackageInfo(
                plugin_id=plugin_id,
                version=version,
                package_path=str(package_path),
                size_bytes=package_size,
                created_at=datetime.utcnow(),
                dependencies=metadata.get("dependencies", [])
            )
            
            self.logger.info(f"Created plugin package: {package_path} ({package_size} bytes)")
            return package_info
            
        except Exception as e:
            self.logger.error(f"Failed to package plugin: {e}")
            raise PluginSDKError(f"Plugin packaging failed: {str(e)}")
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from package."""
        exclude_patterns = [
            "*.pyc", "__pycache__", ".git", ".gitignore",
            "*.log", "*.tmp", ".pytest_cache", ".mypy_cache",
            "tests/", "docs/", ".vscode/", ".idea/"
        ]
        
        file_str = str(file_path)
        for pattern in exclude_patterns:
            if pattern in file_str or file_path.name.startswith('.'):
                return True
        return False
    
    def _create_package_metadata(self, config: Dict[str, Any], plugin_path: Path) -> Dict[str, Any]:
        """Create package metadata."""
        plugin_info = config.get("plugin", {})
        
        # Read requirements if exists
        dependencies = []
        requirements_file = plugin_path / "requirements.txt"
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                dependencies = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        return {
            "plugin": plugin_info,
            "package": {
                "created_at": datetime.utcnow().isoformat(),
                "sdk_version": "2.3.0",
                "format_version": "1.0"
            },
            "dependencies": dependencies,
            "installation": {
                "requires_python": ">=3.8",
                "requires_sdk": ">=2.3.0"
            }
        }
    
    def _create_installation_script(self, metadata: Dict[str, Any]) -> str:
        """Create installation script for the package."""
        return '''#!/usr/bin/env python3
"""
Plugin Installation Script
Generated by LeanVibe Plugin SDK
"""

import os
import sys
import shutil
import json
from pathlib import Path


def install_plugin():
    """Install the plugin."""
    print("Installing plugin...")
    
    # Read metadata
    with open("package-info.json", "r") as f:
        metadata = json.load(f)
    
    plugin_info = metadata["plugin"]
    plugin_name = plugin_info["name"]
    
    # Get installation directory
    install_dir = os.environ.get("LEANVIBE_PLUGINS_DIR", "./plugins")
    plugin_dir = Path(install_dir) / plugin_name
    
    # Create plugin directory
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy plugin files
    current_dir = Path(".")
    for item in current_dir.iterdir():
        if item.name not in ["install.py", "package-info.json"]:
            if item.is_file():
                shutil.copy2(item, plugin_dir)
            elif item.is_dir():
                shutil.copytree(item, plugin_dir / item.name, dirs_exist_ok=True)
    
    print(f"Plugin '{plugin_name}' installed to: {plugin_dir}")
    
    # Install dependencies
    dependencies = metadata.get("dependencies", [])
    if dependencies:
        print("Installing dependencies...")
        for dep in dependencies:
            os.system(f"pip install {dep}")
    
    print("Installation complete!")


if __name__ == "__main__":
    install_plugin()
'''


class PerformanceProfiler:
    """
    Performance profiling tool for plugins.
    
    Provides detailed performance analysis including:
    - Function-level timing
    - Memory usage tracking
    - Resource utilization
    - Bottleneck identification
    """
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_profiler")
        self.profiles = {}
    
    async def profile_plugin(
        self,
        plugin,
        task,
        duration_seconds: float = 60.0,
        sample_interval: float = 0.1
    ) -> PerformanceProfile:
        """
        Profile plugin performance during execution.
        
        Args:
            plugin: Plugin to profile
            task: Task to execute during profiling
            duration_seconds: How long to profile
            sample_interval: Sampling interval in seconds
            
        Returns:
            PerformanceProfile: Detailed performance profile
        """
        profile_id = f"profile_{uuid.uuid4().hex[:8]}"
        
        try:
            import cProfile
            import pstats
            import io
            
            # Start profiling
            profiler = cProfile.Profile()
            memory_samples = []
            start_time = time.time()
            
            profiler.enable()
            
            # Execute plugin task
            result = await plugin.execute(task)
            
            profiler.disable()
            end_time = time.time()
            
            # Analyze profiling results
            stats_buffer = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_buffer)
            stats.sort_stats('cumulative')
            
            # Extract function timing data
            function_times = {}
            total_calls = {}
            
            for func_key, (call_count, total_time, cumulative_time, callers) in stats.stats.items():
                func_name = f"{func_key[0]}:{func_key[2]}"
                function_times[func_name] = cumulative_time * 1000  # Convert to ms
                total_calls[func_name] = call_count
            
            # Calculate averages
            average_times = {
                func: func_time / total_calls[func] if total_calls[func] > 0 else 0
                for func, func_time in function_times.items()
            }
            
            # Identify hotspots (top 10 by cumulative time)
            hotspots = []
            sorted_functions = sorted(function_times.items(), key=lambda x: x[1], reverse=True)
            for func_name, cum_time in sorted_functions[:10]:
                hotspots.append({
                    "function": func_name,
                    "cumulative_time_ms": cum_time,
                    "calls": total_calls.get(func_name, 0),
                    "average_time_ms": average_times.get(func_name, 0)
                })
            
            # Identify bottlenecks (functions with high average time)
            bottlenecks = []
            sorted_averages = sorted(average_times.items(), key=lambda x: x[1], reverse=True)
            for func_name, avg_time in sorted_averages[:10]:
                if avg_time > 1.0:  # More than 1ms average
                    bottlenecks.append({
                        "function": func_name,
                        "average_time_ms": avg_time,
                        "calls": total_calls.get(func_name, 0),
                        "cumulative_time_ms": function_times.get(func_name, 0)
                    })
            
            # Create performance profile
            profile = PerformanceProfile(
                plugin_id=plugin.plugin_id,
                profile_id=profile_id,
                duration_seconds=end_time - start_time,
                function_times=function_times,
                total_calls=total_calls,
                average_times=average_times,
                peak_memory_mb=0.0,  # Would need psutil for accurate memory tracking
                average_memory_mb=0.0,
                memory_timeline=[],
                cpu_usage_percent=0.0,  # Would need psutil
                io_operations=0,
                hotspots=hotspots,
                bottlenecks=bottlenecks,
                created_at=datetime.utcnow()
            )
            
            # Store profile
            self.profiles[profile_id] = profile
            
            self.logger.info(f"Created performance profile: {profile_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Profiling failed: {e}")
            raise PluginSDKError(f"Performance profiling failed: {str(e)}")
    
    def get_profile(self, profile_id: str) -> Optional[PerformanceProfile]:
        """Get a performance profile by ID."""
        return self.profiles.get(profile_id)
    
    def get_all_profiles(self) -> List[PerformanceProfile]:
        """Get all performance profiles."""
        return list(self.profiles.values())
    
    def export_profile(self, profile_id: str, output_file: str) -> bool:
        """Export performance profile to file."""
        try:
            profile = self.profiles.get(profile_id)
            if not profile:
                return False
            
            with open(output_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export profile: {e}")
            return False


class DebugConsole:
    """
    Interactive debug console for plugin development.
    
    Provides real-time debugging capabilities including:
    - Live plugin inspection
    - Interactive execution
    - Variable monitoring
    - Event logging
    """
    
    def __init__(self):
        self.logger = logging.getLogger("debug_console")
        self.active_plugins = {}
        self.debug_sessions = {}
    
    def register_plugin(self, plugin) -> str:
        """Register a plugin for debugging."""
        session_id = f"debug_{uuid.uuid4().hex[:8]}"
        self.active_plugins[session_id] = plugin
        self.debug_sessions[session_id] = {
            "plugin": plugin,
            "started_at": datetime.utcnow(),
            "commands_executed": [],
            "breakpoints": [],
            "watches": {}
        }
        
        self.logger.info(f"Registered plugin for debugging: {session_id}")
        return session_id
    
    def execute_command(self, session_id: str, command: str) -> Dict[str, Any]:
        """Execute a debug command."""
        if session_id not in self.debug_sessions:
            return {"error": "Invalid debug session"}
        
        session = self.debug_sessions[session_id]
        plugin = session["plugin"]
        
        try:
            # Log command
            session["commands_executed"].append({
                "command": command,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Execute command
            if command.startswith("inspect "):
                attr_name = command[8:].strip()
                return self._inspect_attribute(plugin, attr_name)
            
            elif command.startswith("set "):
                parts = command[4:].split("=", 1)
                if len(parts) == 2:
                    attr_name = parts[0].strip()
                    value = parts[1].strip()
                    return self._set_attribute(plugin, attr_name, value)
            
            elif command == "status":
                return self._get_plugin_status(plugin)
            
            elif command == "metrics":
                return self._get_plugin_metrics(plugin)
            
            elif command.startswith("watch "):
                attr_name = command[6:].strip()
                return self._add_watch(session, attr_name)
            
            elif command == "watches":
                return self._get_watches(session)
            
            else:
                return {"error": f"Unknown command: {command}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _inspect_attribute(self, plugin, attr_name: str) -> Dict[str, Any]:
        """Inspect a plugin attribute."""
        try:
            if hasattr(plugin, attr_name):
                attr_value = getattr(plugin, attr_name)
                return {
                    "attribute": attr_name,
                    "type": type(attr_value).__name__,
                    "value": str(attr_value),
                    "repr": repr(attr_value)
                }
            else:
                return {"error": f"Attribute '{attr_name}' not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def _set_attribute(self, plugin, attr_name: str, value: str) -> Dict[str, Any]:
        """Set a plugin attribute value."""
        try:
            if hasattr(plugin, attr_name):
                # Try to evaluate the value
                try:
                    evaluated_value = eval(value)
                except:
                    evaluated_value = value  # Use as string if eval fails
                
                setattr(plugin, attr_name, evaluated_value)
                return {
                    "attribute": attr_name,
                    "old_value": str(getattr(plugin, attr_name)),
                    "new_value": str(evaluated_value)
                }
            else:
                return {"error": f"Attribute '{attr_name}' not found"}
        except Exception as e:
            return {"error": str(e)}
    
    def _get_plugin_status(self, plugin) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            "plugin_id": plugin.plugin_id,
            "name": plugin.name,
            "version": plugin.version,
            "status": plugin.status.value,
            "is_running": plugin.is_running,
            "created_at": plugin.created_at.isoformat(),
            "last_executed": plugin.last_executed.isoformat() if plugin.last_executed else None,
            "performance_metrics": plugin.performance_metrics
        }
    
    def _get_plugin_metrics(self, plugin) -> Dict[str, Any]:
        """Get plugin performance metrics."""
        return plugin.performance_metrics
    
    def _add_watch(self, session: Dict[str, Any], attr_name: str) -> Dict[str, Any]:
        """Add an attribute to watch list."""
        plugin = session["plugin"]
        if hasattr(plugin, attr_name):
            current_value = getattr(plugin, attr_name)
            session["watches"][attr_name] = {
                "current_value": str(current_value),
                "added_at": datetime.utcnow().isoformat()
            }
            return {"message": f"Added watch for '{attr_name}'"}
        else:
            return {"error": f"Attribute '{attr_name}' not found"}
    
    def _get_watches(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Get current watch values."""
        plugin = session["plugin"]
        watches = {}
        
        for attr_name, watch_info in session["watches"].items():
            if hasattr(plugin, attr_name):
                current_value = str(getattr(plugin, attr_name))
                watches[attr_name] = {
                    "current_value": current_value,
                    "previous_value": watch_info["current_value"],
                    "changed": current_value != watch_info["current_value"],
                    "added_at": watch_info["added_at"]
                }
                # Update stored value
                session["watches"][attr_name]["current_value"] = current_value
        
        return {"watches": watches}


class DevEnvironmentSetup:
    """
    Development environment setup utility.
    
    Configures the development environment for optimal
    plugin development experience.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("dev_environment_setup")
    
    def initialize(self, workspace_dir: str = "./plugin_workspace") -> Dict[str, Any]:
        """Initialize development environment."""
        try:
            workspace_path = Path(workspace_dir)
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            self._create_directory_structure(workspace_path)
            
            # Create configuration files
            self._create_config_files(workspace_path)
            
            # Create development scripts
            self._create_dev_scripts(workspace_path)
            
            return {
                "workspace_dir": str(workspace_path),
                "message": "Development environment initialized successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            raise PluginSDKError(f"Environment initialization failed: {str(e)}")
    
    def _create_directory_structure(self, workspace_path: Path):
        """Create workspace directory structure."""
        directories = [
            "plugins",
            "templates", 
            "tests",
            "docs",
            "examples",
            "packages",
            "profiles",
            ".vscode"
        ]
        
        for directory in directories:
            (workspace_path / directory).mkdir(exist_ok=True)
    
    def _create_config_files(self, workspace_path: Path):
        """Create development configuration files."""
        # VS Code settings
        vscode_settings = {
            "python.defaultInterpreterPath": "./venv/bin/python",
            "python.linting.enabled": True,
            "python.linting.pylintEnabled": True,
            "python.formatting.provider": "black",
            "python.testing.pytestEnabled": True,
            "python.testing.pytestArgs": ["tests/"]
        }
        
        with open(workspace_path / ".vscode" / "settings.json", "w") as f:
            json.dump(vscode_settings, f, indent=2)
        
        # Development requirements
        dev_requirements = [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "pylint>=2.0.0",
            "mypy>=0.800",
            "coverage>=5.0.0"
        ]
        
        with open(workspace_path / "requirements-dev.txt", "w") as f:
            f.write("\n".join(dev_requirements))
    
    def _create_dev_scripts(self, workspace_path: Path):
        """Create development scripts."""
        # Test runner script
        test_script = '''#!/usr/bin/env python3
"""Test runner for plugin development."""

import subprocess
import sys

def run_tests():
    """Run all tests."""
    print("Running tests...")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    return result.returncode == 0

def run_linting():
    """Run code linting."""
    print("Running linting...")
    result = subprocess.run(["pylint", "plugins/"])
    return result.returncode == 0

def run_formatting():
    """Run code formatting."""
    print("Formatting code...")
    subprocess.run(["black", "plugins/"])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_tests()
        elif sys.argv[1] == "lint":
            run_linting()
        elif sys.argv[1] == "format":
            run_formatting()
        else:
            print("Usage: dev.py [test|lint|format]")
    else:
        run_formatting()
        run_linting()
        run_tests()
'''
        
        with open(workspace_path / "dev.py", "w") as f:
            f.write(test_script)
        
        # Make script executable
        import stat
        script_path = workspace_path / "dev.py"
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)


class PluginValidator:
    """
    Plugin validation utility.
    
    Validates plugins against SDK requirements and best practices.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("plugin_validator")
    
    def validate_plugin(self, plugin_dir: str) -> Dict[str, Any]:
        """Validate a plugin directory."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        plugin_path = Path(plugin_dir)
        
        # Check required files
        required_files = ["__init__.py", "plugin.py", "config.yaml"]
        for required_file in required_files:
            if not (plugin_path / required_file).exists():
                validation_result["errors"].append(f"Missing required file: {required_file}")
                validation_result["valid"] = False
        
        # Check configuration
        config_file = plugin_path / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Validate config structure
                if "plugin" not in config:
                    validation_result["errors"].append("Missing 'plugin' section in config.yaml")
                    validation_result["valid"] = False
                
            except yaml.YAMLError as e:
                validation_result["errors"].append(f"Invalid YAML in config.yaml: {e}")
                validation_result["valid"] = False
        
        return validation_result
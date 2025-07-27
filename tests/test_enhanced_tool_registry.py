"""
Comprehensive tests for Enhanced Tool Registry & Discovery System.

Tests include:
- Tool registration and discovery functionality
- Dynamic plugin architecture
- Input validation with Pydantic schemas
- Security integration and access control
- Usage analytics and health monitoring
- Rate limiting and error handling
- Integration with existing external tools
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.enhanced_tool_registry import (
    EnhancedToolRegistry,
    ToolDefinition,
    ToolCategory,
    ToolAccessLevel,
    ToolInputSchema,
    ToolExecutionResult,
    ToolUsageMetrics,
    ToolHealthStatus,
    get_enhanced_tool_registry,
    discover_available_tools,
    execute_tool_by_id,
    get_agent_tool_recommendations
)
from app.core.enhanced_security_safeguards import ControlDecision


class TestToolInputSchema:
    """Test tool input schema validation."""
    
    def test_basic_tool_schema_creation(self):
        """Test basic tool schema creation and validation."""
        
        class TestSchema(ToolInputSchema):
            name: str
            count: int = 1
            optional_param: str = None
        
        # Valid input
        valid_data = {"name": "test", "count": 5}
        schema_instance = TestSchema(**valid_data)
        assert schema_instance.name == "test"
        assert schema_instance.count == 5
        assert schema_instance.optional_param is None
        
        # Test validation
        with pytest.raises(Exception):  # Should raise validation error
            TestSchema(count="invalid")  # Wrong type
    
    def test_schema_forbids_extra_fields(self):
        """Test that schema forbids extra fields."""
        
        class StrictSchema(ToolInputSchema):
            required_field: str
        
        # Should raise validation error for extra field
        with pytest.raises(Exception):
            StrictSchema(required_field="test", extra_field="not_allowed")


class TestToolDefinition:
    """Test tool definition validation and creation."""
    
    @pytest.fixture
    def sample_tool_schema(self):
        """Sample tool schema for testing."""
        class SampleSchema(ToolInputSchema):
            input_text: str
            count: int = 1
        return SampleSchema
    
    def test_tool_definition_creation(self, sample_tool_schema):
        """Test tool definition creation with all fields."""
        tool_def = ToolDefinition(
            id="test_tool",
            name="Test Tool",
            description="A tool for testing purposes",
            category=ToolCategory.UTILITY,
            function="test_function",
            module_path="test.module",
            input_schema=sample_tool_schema,
            usage_examples=["Example usage"],
            when_to_use="When testing",
            access_level=ToolAccessLevel.PUBLIC,
            timeout_seconds=30
        )
        
        assert tool_def.id == "test_tool"
        assert tool_def.name == "Test Tool"
        assert tool_def.category == ToolCategory.UTILITY
        assert tool_def.access_level == ToolAccessLevel.PUBLIC
        assert tool_def.timeout_seconds == 30
        assert isinstance(tool_def.created_at, datetime)
    
    def test_tool_definition_defaults(self, sample_tool_schema):
        """Test tool definition with default values."""
        tool_def = ToolDefinition(
            id="minimal_tool",
            name="Minimal Tool",
            description="Minimal tool definition",
            category=ToolCategory.UTILITY,
            function="minimal_function",
            module_path="minimal.module",
            input_schema=sample_tool_schema
        )
        
        assert tool_def.access_level == ToolAccessLevel.PUBLIC
        assert tool_def.timeout_seconds == 30
        assert tool_def.version == "1.0.0"
        assert tool_def.usage_examples == []
        assert tool_def.required_permissions == []


class TestEnhancedToolRegistry:
    """Test enhanced tool registry functionality."""
    
    @pytest.fixture
    def registry(self):
        """Tool registry instance for testing."""
        # Create a fresh registry for each test
        from app.core.enhanced_tool_registry import EnhancedToolRegistry
        return EnhancedToolRegistry()
    
    @pytest.fixture
    def sample_tool_schema(self):
        """Sample tool schema for testing."""
        class TestToolSchema(ToolInputSchema):
            message: str
            repeat: int = 1
        return TestToolSchema
    
    @pytest.fixture
    def sample_tool_definition(self, sample_tool_schema):
        """Sample tool definition for testing."""
        return ToolDefinition(
            id="test_echo_tool",
            name="Echo Tool",
            description="Echoes back the input message",
            category=ToolCategory.UTILITY,
            function="echo_function",
            module_path="test.echo",
            input_schema=sample_tool_schema,
            usage_examples=["echo_tool(message='hello', repeat=2)"],
            when_to_use="When you need to echo messages",
            timeout_seconds=10
        )
    
    @pytest.fixture
    def sample_tool_function(self):
        """Sample tool function for testing."""
        async def echo_function(message: str, repeat: int = 1) -> Dict[str, Any]:
            return {
                "result": message * repeat,
                "original_message": message,
                "repeat_count": repeat
            }
        return echo_function
    
    def test_registry_initialization(self, registry):
        """Test registry initialization with core tools."""
        # Registry should initialize with some core tools
        assert len(registry.tools) > 0
        assert len(registry.tool_functions) > 0
        assert len(registry.usage_metrics) > 0
        assert len(registry.health_status) > 0
        
        # Check that core Git tools are registered
        tool_ids = list(registry.tools.keys())
        assert "git_clone" in tool_ids
        assert "git_commit" in tool_ids
    
    def test_register_tool_success(self, registry, sample_tool_definition, sample_tool_function):
        """Test successful tool registration."""
        initial_count = len(registry.tools)
        
        success = registry.register_tool(sample_tool_definition, sample_tool_function)
        
        assert success is True
        assert len(registry.tools) == initial_count + 1
        assert sample_tool_definition.id in registry.tools
        assert sample_tool_definition.id in registry.tool_functions
        assert sample_tool_definition.id in registry.usage_metrics
        assert sample_tool_definition.id in registry.health_status
    
    def test_register_tool_duplicate_id(self, registry, sample_tool_definition, sample_tool_function):
        """Test registration of tool with duplicate ID."""
        # Register tool first time
        success1 = registry.register_tool(sample_tool_definition, sample_tool_function)
        assert success1 is True
        
        # Try to register again without override
        success2 = registry.register_tool(sample_tool_definition, sample_tool_function, override=False)
        assert success2 is False
        
        # Register again with override
        success3 = registry.register_tool(sample_tool_definition, sample_tool_function, override=True)
        assert success3 is True
    
    def test_unregister_tool(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool unregistration."""
        # Register and then unregister
        registry.register_tool(sample_tool_definition, sample_tool_function)
        initial_count = len(registry.tools)
        
        success = registry.unregister_tool(sample_tool_definition.id)
        
        assert success is True
        assert len(registry.tools) == initial_count - 1
        assert sample_tool_definition.id not in registry.tools
        assert sample_tool_definition.id not in registry.tool_functions
        assert sample_tool_definition.id not in registry.usage_metrics
        assert sample_tool_definition.id not in registry.health_status
    
    def test_unregister_nonexistent_tool(self, registry):
        """Test unregistration of non-existent tool."""
        success = registry.unregister_tool("nonexistent_tool")
        assert success is False
    
    def test_discover_tools_no_filters(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool discovery without filters."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        
        discovered_tools = registry.discover_tools()
        
        assert len(discovered_tools) > 0
        tool_ids = [tool.id for tool in discovered_tools]
        assert sample_tool_definition.id in tool_ids
    
    def test_discover_tools_by_category(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool discovery filtered by category."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        
        # Discover utility tools
        utility_tools = registry.discover_tools(category=ToolCategory.UTILITY)
        utility_ids = [tool.id for tool in utility_tools]
        assert sample_tool_definition.id in utility_ids
        
        # Discover version control tools (should not include our utility tool)
        vc_tools = registry.discover_tools(category=ToolCategory.VERSION_CONTROL)
        vc_ids = [tool.id for tool in vc_tools]
        assert sample_tool_definition.id not in vc_ids
    
    def test_discover_tools_by_search_query(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool discovery with search query."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        
        # Search by name
        name_results = registry.discover_tools(search_query="Echo")
        assert len(name_results) > 0
        assert sample_tool_definition.id in [tool.id for tool in name_results]
        
        # Search by description
        desc_results = registry.discover_tools(search_query="echoes")
        assert len(desc_results) > 0
        assert sample_tool_definition.id in [tool.id for tool in desc_results]
        
        # Search with no matches
        no_results = registry.discover_tools(search_query="nonexistent")
        assert sample_tool_definition.id not in [tool.id for tool in no_results]
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, registry, sample_tool_definition, sample_tool_function):
        """Test successful tool execution."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        # Mock security validation
        with patch('app.core.enhanced_tool_registry.validate_agent_action', new_callable=AsyncMock) as mock_security:
            mock_security.return_value = (ControlDecision.ALLOW, "Allowed", {})
            
            result = await registry.execute_tool(
                tool_id=sample_tool_definition.id,
                agent_id=agent_id,
                input_data={"message": "hello", "repeat": 2}
            )
            
            assert result.success is True
            assert result.output["result"] == "hellohello"
            assert result.output["original_message"] == "hello"
            assert result.output["repeat_count"] == 2
            assert result.execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_execute_tool_validation_error(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool execution with input validation error."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        with patch('app.core.enhanced_tool_registry.validate_agent_action', new_callable=AsyncMock) as mock_security:
            mock_security.return_value = (ControlDecision.ALLOW, "Allowed", {})
            
            result = await registry.execute_tool(
                tool_id=sample_tool_definition.id,
                agent_id=agent_id,
                input_data={"message": "hello", "repeat": "invalid"}  # Invalid type
            )
            
            assert result.success is False
            assert "Input validation failed" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_security_denied(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool execution with security denial."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        with patch('app.core.enhanced_tool_registry.validate_agent_action', new_callable=AsyncMock) as mock_security:
            mock_security.return_value = (ControlDecision.DENY, "Access denied", {})
            
            result = await registry.execute_tool(
                tool_id=sample_tool_definition.id,
                agent_id=agent_id,
                input_data={"message": "hello", "repeat": 1}
            )
            
            assert result.success is False
            assert "Security check failed" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, registry, sample_tool_schema):
        """Test tool execution timeout."""
        # Create a tool with very short timeout
        timeout_tool_def = ToolDefinition(
            id="timeout_tool",
            name="Timeout Tool",
            description="Tool that times out",
            category=ToolCategory.UTILITY,
            function="timeout_function",
            module_path="test.timeout",
            input_schema=sample_tool_schema,
            timeout_seconds=1  # Very short timeout
        )
        
        async def slow_function(message: str, repeat: int = 1):
            await asyncio.sleep(2)  # Longer than timeout
            return {"result": "too slow"}
        
        registry.register_tool(timeout_tool_def, slow_function)
        agent_id = uuid.uuid4()
        
        with patch('app.core.enhanced_tool_registry.validate_agent_action', new_callable=AsyncMock) as mock_security:
            mock_security.return_value = (ControlDecision.ALLOW, "Allowed", {})
            
            result = await registry.execute_tool(
                tool_id=timeout_tool_def.id,
                agent_id=agent_id,
                input_data={"message": "hello", "repeat": 1}
            )
            
            assert result.success is False
            assert "timed out" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        """Test execution of non-existent tool."""
        agent_id = uuid.uuid4()
        
        result = await registry.execute_tool(
            tool_id="nonexistent_tool",
            agent_id=agent_id,
            input_data={}
        )
        
        assert result.success is False
        assert "not found" in result.error
    
    def test_rate_limiting(self, registry):
        """Test tool rate limiting functionality."""
        tool_id = "test_tool"
        agent_id = uuid.uuid4()
        
        # Mock tool with rate limit of 2 per minute
        registry.tools[tool_id] = Mock()
        registry.tools[tool_id].rate_limit_per_minute = 2
        
        # First two requests should pass
        assert registry._check_rate_limit(tool_id, agent_id) is True
        assert registry._check_rate_limit(tool_id, agent_id) is True
        
        # Third request should be rate limited
        assert registry._check_rate_limit(tool_id, agent_id) is False
    
    @pytest.mark.asyncio
    async def test_health_check_tool(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool health checking."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        
        health_status = await registry.health_check_tool(sample_tool_definition.id)
        
        assert isinstance(health_status, ToolHealthStatus)
        assert health_status.tool_id == sample_tool_definition.id
        assert health_status.is_healthy is True
        assert health_status.health_score > 0.8
        assert len(health_status.issues) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_nonexistent_tool(self, registry):
        """Test health check of non-existent tool."""
        health_status = await registry.health_check_tool("nonexistent_tool")
        
        assert health_status.is_healthy is False
        assert health_status.health_score == 0.0
        assert "Tool not found" in health_status.issues
    
    @pytest.mark.asyncio
    async def test_health_check_all_tools(self, registry, sample_tool_definition, sample_tool_function):
        """Test health check of all tools."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        
        health_results = await registry.health_check_all_tools()
        
        assert isinstance(health_results, dict)
        assert len(health_results) > 0
        assert sample_tool_definition.id in health_results
        assert all(isinstance(status, ToolHealthStatus) for status in health_results.values())
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, registry, sample_tool_definition, sample_tool_function):
        """Test tool usage metrics tracking."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        # Update metrics manually (would normally happen during execution)
        await registry._update_tool_metrics(
            tool_id=sample_tool_definition.id,
            agent_id=agent_id,
            success=True,
            execution_time_ms=150
        )
        
        metrics = registry.get_tool_metrics(sample_tool_definition.id)
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 1
        assert metrics.failed_executions == 0
        assert metrics.average_execution_time_ms == 150.0
        assert str(agent_id) in metrics.agents_using_tool
        assert metrics.last_used_at is not None
    
    @pytest.mark.asyncio
    async def test_metrics_error_tracking(self, registry, sample_tool_definition, sample_tool_function):
        """Test error tracking in metrics."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        # Update metrics with error
        await registry._update_tool_metrics(
            tool_id=sample_tool_definition.id,
            agent_id=agent_id,
            success=False,
            execution_time_ms=50,
            error_message="Test error occurred"
        )
        
        metrics = registry.get_tool_metrics(sample_tool_definition.id)
        
        assert metrics.total_executions == 1
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 1
        assert len(metrics.error_patterns) == 1
        assert "Test error occurred" in metrics.error_patterns
    
    def test_get_agent_tool_usage(self, registry, sample_tool_definition, sample_tool_function):
        """Test getting tool usage for specific agent."""
        registry.register_tool(sample_tool_definition, sample_tool_function)
        agent_id = uuid.uuid4()
        
        # Add agent to metrics
        metrics = registry.usage_metrics[sample_tool_definition.id]
        metrics.agents_using_tool.add(str(agent_id))
        metrics.successful_executions = 5
        metrics.total_executions = 6
        metrics.last_used_at = datetime.utcnow()
        
        agent_usage = registry.get_agent_tool_usage(agent_id)
        
        assert sample_tool_definition.id in agent_usage
        tool_data = agent_usage[sample_tool_definition.id]
        assert tool_data["tool_name"] == sample_tool_definition.name
        assert tool_data["category"] == sample_tool_definition.category.value
        assert tool_data["success_rate"] == 5/6


class TestConvenienceFunctions:
    """Test convenience functions for tool operations."""
    
    @pytest.mark.asyncio
    async def test_discover_available_tools(self):
        """Test convenience function for tool discovery."""
        with patch('app.core.enhanced_tool_registry.get_enhanced_tool_registry') as mock_get_registry:
            mock_registry = Mock()
            mock_registry.discover_tools.return_value = [Mock(id="test_tool")]
            mock_get_registry.return_value = mock_registry
            
            agent_id = uuid.uuid4()
            tools = await discover_available_tools(agent_id, category=ToolCategory.UTILITY)
            
            assert len(tools) > 0
            mock_registry.discover_tools.assert_called_once_with(
                agent_id=agent_id, 
                category=ToolCategory.UTILITY, 
                search_query=None
            )
    
    @pytest.mark.asyncio
    async def test_execute_tool_by_id(self):
        """Test convenience function for tool execution."""
        with patch('app.core.enhanced_tool_registry.get_enhanced_tool_registry') as mock_get_registry:
            mock_registry = AsyncMock()
            mock_result = ToolExecutionResult(success=True, output="test_result")
            mock_registry.execute_tool.return_value = mock_result
            mock_get_registry.return_value = mock_registry
            
            agent_id = uuid.uuid4()
            result = await execute_tool_by_id("test_tool", agent_id, {"input": "test"})
            
            assert result.success is True
            assert result.output == "test_result"
            mock_registry.execute_tool.assert_called_once_with(
                "test_tool", agent_id, {"input": "test"}
            )
    
    @pytest.mark.asyncio
    async def test_get_agent_tool_recommendations(self):
        """Test tool recommendations for agent."""
        with patch('app.core.enhanced_tool_registry.get_enhanced_tool_registry') as mock_get_registry:
            mock_registry = Mock()
            
            # Mock version control tools for git-related task
            vc_tools = [Mock(id="git_clone", category=ToolCategory.VERSION_CONTROL)]
            mock_registry.discover_tools.return_value = vc_tools
            mock_get_registry.return_value = mock_registry
            
            agent_id = uuid.uuid4()
            recommendations = await get_agent_tool_recommendations(
                agent_id, 
                current_task="Clone the repository"
            )
            
            assert len(recommendations) > 0
            mock_registry.discover_tools.assert_called_with(
                agent_id=agent_id, 
                category=ToolCategory.VERSION_CONTROL
            )


class TestPluginArchitecture:
    """Test plugin architecture functionality."""
    
    @pytest.fixture
    def registry(self):
        """Fresh registry for plugin tests."""
        from app.core.enhanced_tool_registry import EnhancedToolRegistry
        return EnhancedToolRegistry()
    
    def test_register_plugin_manager(self, registry):
        """Test plugin manager registration."""
        async def mock_plugin_manager():
            return []
        
        initial_count = len(registry.plugin_managers)
        registry.register_plugin_manager(mock_plugin_manager)
        
        assert len(registry.plugin_managers) == initial_count + 1
        assert mock_plugin_manager in registry.plugin_managers
    
    @pytest.mark.asyncio
    async def test_discover_plugins(self, registry):
        """Test plugin discovery."""
        # Mock plugin manager that returns tools
        async def mock_plugin_manager():
            class MockSchema(ToolInputSchema):
                test_field: str
            
            return [
                ToolDefinition(
                    id="plugin_tool_1",
                    name="Plugin Tool 1",
                    description="Tool from plugin",
                    category=ToolCategory.UTILITY,
                    function="plugin_function",
                    module_path="plugin.module",
                    input_schema=MockSchema
                )
            ]
        
        registry.register_plugin_manager(mock_plugin_manager)
        discovered_tools = await registry.discover_plugins()
        
        assert len(discovered_tools) > 0
        assert discovered_tools[0].id == "plugin_tool_1"
        assert discovered_tools[0].name == "Plugin Tool 1"


class TestIntegrationAndPerformance:
    """Test integration scenarios and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_executions(self):
        """Test concurrent tool executions."""
        registry = EnhancedToolRegistry()
        
        class ConcurrentSchema(ToolInputSchema):
            value: int
        
        async def concurrent_function(value: int):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": value * 2}
        
        tool_def = ToolDefinition(
            id="concurrent_tool",
            name="Concurrent Tool",
            description="Tool for concurrent testing",
            category=ToolCategory.UTILITY,
            function="concurrent_function",
            module_path="test.concurrent",
            input_schema=ConcurrentSchema
        )
        
        registry.register_tool(tool_def, concurrent_function)
        
        # Execute multiple tools concurrently
        agent_ids = [uuid.uuid4() for _ in range(5)]
        
        with patch('app.core.enhanced_tool_registry.validate_agent_action', new_callable=AsyncMock) as mock_security:
            mock_security.return_value = (ControlDecision.ALLOW, "Allowed", {})
            
            tasks = [
                registry.execute_tool(
                    tool_id="concurrent_tool",
                    agent_id=agent_id,
                    input_data={"value": i}
                )
                for i, agent_id in enumerate(agent_ids)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert all(result.success for result in results)
            assert [result.output["result"] for result in results] == [0, 2, 4, 6, 8]
    
    @pytest.mark.asyncio
    async def test_large_scale_tool_registry(self):
        """Test registry with large number of tools."""
        registry = EnhancedToolRegistry()
        
        class TestSchema(ToolInputSchema):
            input_value: str
        
        async def test_function(input_value: str):
            return {"output": input_value}
        
        # Register many tools
        num_tools = 100
        for i in range(num_tools):
            tool_def = ToolDefinition(
                id=f"tool_{i}",
                name=f"Tool {i}",
                description=f"Test tool number {i}",
                category=ToolCategory.UTILITY,
                function="test_function",
                module_path="test.module",
                input_schema=TestSchema
            )
            
            success = registry.register_tool(tool_def, test_function)
            assert success is True
        
        # Test discovery performance
        all_tools = registry.discover_tools()
        assert len(all_tools) >= num_tools  # Including core tools
        
        # Test category filtering
        utility_tools = registry.discover_tools(category=ToolCategory.UTILITY)
        assert len(utility_tools) == num_tools
        
        # Test search functionality
        search_results = registry.discover_tools(search_query="Tool 5")
        tool_names = [tool.name for tool in search_results]
        # Should find "Tool 5", "Tool 50", "Tool 51", etc.
        assert any("Tool 5" in name for name in tool_names)
    
    def test_memory_efficiency_with_metrics(self):
        """Test memory efficiency of metrics tracking."""
        registry = EnhancedToolRegistry()
        
        class TestSchema(ToolInputSchema):
            data: str
        
        tool_def = ToolDefinition(
            id="memory_test_tool",
            name="Memory Test Tool",
            description="Tool for memory testing",
            category=ToolCategory.UTILITY,
            function="memory_function",
            module_path="test.memory",
            input_schema=TestSchema
        )
        
        def memory_function(data: str):
            return {"result": data}
        
        registry.register_tool(tool_def, memory_function)
        
        # Simulate many metric updates
        agent_id = uuid.uuid4()
        for i in range(1000):
            asyncio.run(registry._update_tool_metrics(
                tool_id="memory_test_tool",
                agent_id=agent_id,
                success=i % 2 == 0,  # Alternate success/failure
                execution_time_ms=100 + i,
                error_message=f"Error {i}" if i % 2 == 1 else None
            ))
        
        metrics = registry.get_tool_metrics("memory_test_tool")
        
        # Check that performance trends are limited to 100 entries
        assert len(metrics.performance_trends) == 100
        
        # Check metrics accuracy
        assert metrics.total_executions == 1000
        assert metrics.successful_executions == 500
        assert metrics.failed_executions == 500
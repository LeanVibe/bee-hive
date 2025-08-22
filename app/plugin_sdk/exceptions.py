"""
Exception classes for LeanVibe Plugin SDK.

Provides comprehensive error handling with detailed context
and recovery suggestions for plugin developers.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime


class PluginSDKError(Exception):
    """
    Base exception for all Plugin SDK errors.
    
    Provides structured error information with context
    and recovery suggestions for developers.
    """
    
    def __init__(
        self, 
        message: str,
        error_code: Optional[str] = None,
        plugin_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SDK_ERROR"
        self.plugin_id = plugin_id
        self.details = details or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.utcnow()
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add additional error detail."""
        self.details[key] = value
    
    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add recovery suggestion."""
        self.recovery_suggestions.append(suggestion)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "plugin_id": self.plugin_id,
            "details": self.details,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp.isoformat()
        }


class PluginInitializationError(PluginSDKError):
    """
    Raised when plugin initialization fails.
    
    Common causes:
    - Missing required configuration
    - Invalid dependencies
    - Insufficient permissions
    - Resource allocation failure
    """
    
    def __init__(self, message: str, plugin_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="PLUGIN_INIT_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Check plugin configuration for required parameters",
            "Verify all dependencies are available",
            "Ensure plugin has necessary permissions",
            "Check system resource availability"
        ])


class PluginExecutionError(PluginSDKError):
    """
    Raised when plugin execution fails.
    
    Common causes:
    - Runtime errors in plugin code
    - Task parameter validation failure
    - Resource exhaustion
    - External service unavailable
    """
    
    def __init__(
        self, 
        message: str, 
        task_id: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_EXEC_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if task_id:
            self.add_detail("task_id", task_id)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Review plugin logs for detailed error information",
            "Validate task parameters and input data",
            "Check system resources and dependencies",
            "Retry with different parameters if appropriate"
        ])


class PluginValidationError(PluginSDKError):
    """
    Raised when plugin validation fails.
    
    Common causes:
    - Invalid plugin configuration
    - Missing required methods
    - Security policy violations
    - Incompatible dependencies
    """
    
    def __init__(
        self, 
        message: str, 
        validation_errors: Optional[List[str]] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_VALIDATION_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if validation_errors:
            self.add_detail("validation_errors", validation_errors)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Review plugin configuration against SDK requirements",
            "Implement all required abstract methods",
            "Check plugin security permissions",
            "Verify dependency versions are compatible"
        ])


class PluginTimeoutError(PluginSDKError):
    """
    Raised when plugin operation times out.
    
    Common causes:
    - Long-running operations without timeout handling
    - Blocking operations in async code
    - External service delays
    - Resource contention
    """
    
    def __init__(
        self, 
        message: str, 
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_TIMEOUT_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if timeout_seconds:
            self.add_detail("timeout_seconds", timeout_seconds)
        if operation:
            self.add_detail("operation", operation)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Increase timeout for long-running operations",
            "Break down large operations into smaller chunks",
            "Use async/await properly for non-blocking operations",
            "Check external service response times",
            "Implement proper timeout handling in plugin code"
        ])


class PluginConfigurationError(PluginSDKError):
    """
    Raised when plugin configuration is invalid.
    
    Common causes:
    - Missing required configuration parameters
    - Invalid parameter values
    - Conflicting configuration options
    - Malformed configuration file
    """
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Optional[Any] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_CONFIG_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if config_key:
            self.add_detail("config_key", config_key)
        if expected_type:
            self.add_detail("expected_type", expected_type)
        if actual_value is not None:
            self.add_detail("actual_value", actual_value)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Check plugin configuration file syntax",
            "Verify all required parameters are provided",
            "Validate parameter types match expected values",
            "Review plugin documentation for configuration requirements"
        ])


class PluginDependencyError(PluginSDKError):
    """
    Raised when plugin dependencies are missing or incompatible.
    
    Common causes:
    - Missing required dependencies
    - Incompatible dependency versions
    - Circular dependencies
    - Failed dependency initialization
    """
    
    def __init__(
        self, 
        message: str, 
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        available_version: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_DEPENDENCY_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if dependency_name:
            self.add_detail("dependency_name", dependency_name)
        if required_version:
            self.add_detail("required_version", required_version)
        if available_version:
            self.add_detail("available_version", available_version)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Install missing dependencies",
            "Update dependencies to compatible versions",
            "Check for circular dependency conflicts",
            "Verify dependency installation and initialization"
        ])


class PluginSecurityError(PluginSDKError):
    """
    Raised when plugin security validation fails.
    
    Common causes:
    - Insufficient permissions
    - Security policy violations
    - Sandbox escape attempts
    - Unauthorized resource access
    """
    
    def __init__(
        self, 
        message: str, 
        security_violation: Optional[str] = None,
        required_permission: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_SECURITY_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if security_violation:
            self.add_detail("security_violation", security_violation)
        if required_permission:
            self.add_detail("required_permission", required_permission)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Request necessary permissions for plugin operation",
            "Review plugin security policy compliance",
            "Use provided SDK interfaces instead of direct system access",
            "Contact system administrator for permission changes"
        ])


class PluginResourceError(PluginSDKError):
    """
    Raised when plugin resource limits are exceeded.
    
    Common causes:
    - Memory limit exceeded
    - Too many concurrent operations
    - Disk space exhaustion
    - Network resource limits
    """
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        limit_value: Optional[Any] = None,
        current_usage: Optional[Any] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_RESOURCE_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if resource_type:
            self.add_detail("resource_type", resource_type)
        if limit_value is not None:
            self.add_detail("limit_value", limit_value)
        if current_usage is not None:
            self.add_detail("current_usage", current_usage)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Optimize plugin memory usage",
            "Reduce concurrent operations",
            "Clean up unused resources",
            "Request higher resource limits if necessary"
        ])


class PluginCommunicationError(PluginSDKError):
    """
    Raised when plugin communication with system components fails.
    
    Common causes:
    - Network connectivity issues
    - Service unavailable
    - Protocol mismatch
    - Authentication failure
    """
    
    def __init__(
        self, 
        message: str, 
        target_component: Optional[str] = None,
        communication_type: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_COMMUNICATION_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if target_component:
            self.add_detail("target_component", target_component)
        if communication_type:
            self.add_detail("communication_type", communication_type)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Check network connectivity",
            "Verify target service is available",
            "Validate authentication credentials",
            "Retry operation after brief delay"
        ])


class PluginTestError(PluginSDKError):
    """
    Raised when plugin testing fails.
    
    Common causes:
    - Test case failures
    - Mock setup issues
    - Assertion errors
    - Test environment problems
    """
    
    def __init__(
        self, 
        message: str, 
        test_name: Optional[str] = None,
        assertion_error: Optional[str] = None,
        plugin_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            error_code="PLUGIN_TEST_ERROR",
            plugin_id=plugin_id,
            **kwargs
        )
        
        if test_name:
            self.add_detail("test_name", test_name)
        if assertion_error:
            self.add_detail("assertion_error", assertion_error)
        
        # Add common recovery suggestions
        self.recovery_suggestions.extend([
            "Review test case implementation",
            "Check test data and expectations",
            "Verify mock configurations",
            "Run tests in isolation to identify conflicts"
        ])


# Utility functions for error handling

def handle_plugin_error(error: Exception, plugin_id: str = None) -> PluginSDKError:
    """
    Convert generic exceptions to PluginSDKError with context.
    
    Args:
        error: Original exception
        plugin_id: Plugin identifier for context
        
    Returns:
        PluginSDKError: Wrapped error with plugin context
    """
    if isinstance(error, PluginSDKError):
        return error
    
    # Map common exception types to SDK errors
    error_mapping = {
        TimeoutError: PluginTimeoutError,
        PermissionError: PluginSecurityError,
        MemoryError: PluginResourceError,
        ConnectionError: PluginCommunicationError,
        ValueError: PluginValidationError,
        ImportError: PluginDependencyError,
        AttributeError: PluginConfigurationError
    }
    
    error_class = error_mapping.get(type(error), PluginExecutionError)
    
    return error_class(
        message=str(error),
        plugin_id=plugin_id,
        details={"original_error": str(error), "original_type": type(error).__name__}
    )


def create_validation_error(plugin_id: str, validation_errors: List[str]) -> PluginValidationError:
    """
    Create validation error with multiple validation issues.
    
    Args:
        plugin_id: Plugin identifier
        validation_errors: List of validation error messages
        
    Returns:
        PluginValidationError: Validation error with all issues
    """
    message = f"Plugin validation failed with {len(validation_errors)} errors"
    return PluginValidationError(
        message=message,
        validation_errors=validation_errors,
        plugin_id=plugin_id
    )


def create_timeout_error(plugin_id: str, operation: str, timeout_seconds: float) -> PluginTimeoutError:
    """
    Create timeout error with operation context.
    
    Args:
        plugin_id: Plugin identifier
        operation: Operation that timed out
        timeout_seconds: Timeout value that was exceeded
        
    Returns:
        PluginTimeoutError: Timeout error with context
    """
    message = f"Plugin operation '{operation}' timed out after {timeout_seconds} seconds"
    return PluginTimeoutError(
        message=message,
        timeout_seconds=timeout_seconds,
        operation=operation,
        plugin_id=plugin_id
    )


def create_resource_error(plugin_id: str, resource_type: str, limit: Any, current: Any) -> PluginResourceError:
    """
    Create resource error with usage information.
    
    Args:
        plugin_id: Plugin identifier
        resource_type: Type of resource (memory, cpu, etc.)
        limit: Resource limit that was exceeded
        current: Current resource usage
        
    Returns:
        PluginResourceError: Resource error with usage details
    """
    message = f"Plugin exceeded {resource_type} limit: {current} > {limit}"
    return PluginResourceError(
        message=message,
        resource_type=resource_type,
        limit_value=limit,
        current_usage=current,
        plugin_id=plugin_id
    )
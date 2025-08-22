"""
Decorators for LeanVibe Plugin SDK.

Provides powerful decorators for plugin development including:
- Performance tracking
- Error handling
- Caching
- Validation
- Logging
"""

import asyncio
import functools
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, Union, List
import weakref
from collections import defaultdict
import logging

from .exceptions import PluginTimeoutError, PluginExecutionError, PluginResourceError
from .models import PluginEvent, EventSeverity


# Global cache for decorator data
_performance_metrics: Dict[str, List[float]] = defaultdict(list)
_cache_storage: Dict[str, Dict[str, Any]] = {}
_method_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

logger = logging.getLogger("plugin_sdk.decorators")


def plugin_method(
    timeout_seconds: Optional[float] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    log_execution: bool = True
):
    """
    Decorator for plugin methods with built-in error handling and logging.
    
    Args:
        timeout_seconds: Method timeout in seconds
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        log_execution: Whether to log method execution
        
    Example:
        @plugin_method(timeout_seconds=30, max_retries=3)
        def process_data(self, data):
            # Method implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"
            start_time = time.time()
            attempt = 0
            
            while attempt <= max_retries:
                try:
                    if log_execution:
                        logger.debug(f"Executing {method_name} (attempt {attempt + 1})")
                    
                    # Apply timeout if specified
                    if timeout_seconds:
                        import signal
                        
                        def timeout_handler(signum, frame):
                            raise PluginTimeoutError(
                                f"Method {method_name} timed out after {timeout_seconds} seconds",
                                timeout_seconds=timeout_seconds,
                                operation=method_name,
                                plugin_id=getattr(self, 'plugin_id', None)
                            )
                        
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(int(timeout_seconds))
                        
                        try:
                            result = func(self, *args, **kwargs)
                        finally:
                            signal.alarm(0)
                    else:
                        result = func(self, *args, **kwargs)
                    
                    # Log success
                    execution_time = (time.time() - start_time) * 1000
                    if log_execution:
                        logger.debug(f"Completed {method_name} in {execution_time:.2f}ms")
                    
                    return result
                    
                except Exception as e:
                    attempt += 1
                    execution_time = (time.time() - start_time) * 1000
                    
                    if attempt > max_retries:
                        logger.error(f"Failed {method_name} after {attempt} attempts in {execution_time:.2f}ms: {e}")
                        raise PluginExecutionError(
                            f"Method {method_name} failed after {max_retries + 1} attempts: {str(e)}",
                            plugin_id=getattr(self, 'plugin_id', None)
                        )
                    else:
                        logger.warning(f"Retrying {method_name} after failure (attempt {attempt}): {e}")
                        time.sleep(retry_delay)
                        start_time = time.time()  # Reset timer for retry
        
        # Store metadata
        _method_metadata[func.__name__] = {
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "log_execution": log_execution
        }
        
        return wrapper
    
    return decorator


def async_plugin_method(
    timeout_seconds: Optional[float] = None,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    log_execution: bool = True
):
    """
    Decorator for async plugin methods with built-in error handling and logging.
    
    Args:
        timeout_seconds: Method timeout in seconds
        max_retries: Maximum number of retries on failure
        retry_delay: Delay between retries in seconds
        log_execution: Whether to log method execution
        
    Example:
        @async_plugin_method(timeout_seconds=30, max_retries=3)
        async def process_data_async(self, data):
            # Async method implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"
            start_time = time.time()
            attempt = 0
            
            while attempt <= max_retries:
                try:
                    if log_execution:
                        logger.debug(f"Executing async {method_name} (attempt {attempt + 1})")
                    
                    # Apply timeout if specified
                    if timeout_seconds:
                        result = await asyncio.wait_for(
                            func(self, *args, **kwargs),
                            timeout=timeout_seconds
                        )
                    else:
                        result = await func(self, *args, **kwargs)
                    
                    # Log success
                    execution_time = (time.time() - start_time) * 1000
                    if log_execution:
                        logger.debug(f"Completed async {method_name} in {execution_time:.2f}ms")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    raise PluginTimeoutError(
                        f"Async method {method_name} timed out after {timeout_seconds} seconds",
                        timeout_seconds=timeout_seconds,
                        operation=method_name,
                        plugin_id=getattr(self, 'plugin_id', None)
                    )
                    
                except Exception as e:
                    attempt += 1
                    execution_time = (time.time() - start_time) * 1000
                    
                    if attempt > max_retries:
                        logger.error(f"Failed async {method_name} after {attempt} attempts in {execution_time:.2f}ms: {e}")
                        raise PluginExecutionError(
                            f"Async method {method_name} failed after {max_retries + 1} attempts: {str(e)}",
                            plugin_id=getattr(self, 'plugin_id', None)
                        )
                    else:
                        logger.warning(f"Retrying async {method_name} after failure (attempt {attempt}): {e}")
                        await asyncio.sleep(retry_delay)
                        start_time = time.time()  # Reset timer for retry
        
        # Store metadata
        _method_metadata[func.__name__] = {
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "log_execution": log_execution,
            "is_async": True
        }
        
        return wrapper
    
    return decorator


def performance_tracked(
    track_memory: bool = True,
    track_execution_time: bool = True,
    alert_threshold_ms: Optional[float] = None,
    memory_limit_mb: Optional[float] = None
):
    """
    Decorator to track method performance metrics.
    
    Args:
        track_memory: Whether to track memory usage
        track_execution_time: Whether to track execution time
        alert_threshold_ms: Alert if execution time exceeds this threshold
        memory_limit_mb: Raise error if memory usage exceeds this limit
        
    Example:
        @performance_tracked(alert_threshold_ms=1000, memory_limit_mb=100)
        def expensive_operation(self, data):
            # Implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"
            
            # Initial measurements
            start_time = time.time()
            start_memory = _get_memory_usage() if track_memory else 0
            
            try:
                result = func(self, *args, **kwargs)
                
                # Final measurements
                execution_time = (time.time() - start_time) * 1000
                memory_used = _get_memory_usage() - start_memory if track_memory else 0
                
                # Store metrics
                if track_execution_time:
                    _performance_metrics[f"{method_name}_execution_time"].append(execution_time)
                
                if track_memory:
                    _performance_metrics[f"{method_name}_memory_usage"].append(memory_used)
                
                # Check thresholds
                if alert_threshold_ms and execution_time > alert_threshold_ms:
                    logger.warning(f"Performance alert: {method_name} took {execution_time:.2f}ms (threshold: {alert_threshold_ms}ms)")
                    
                    # Emit performance event if plugin has monitoring
                    if hasattr(self, '_monitoring') and self._monitoring:
                        event = PluginEvent(
                            event_type="performance_alert",
                            plugin_id=getattr(self, 'plugin_id', 'unknown'),
                            data={
                                "method": method_name,
                                "execution_time_ms": execution_time,
                                "threshold_ms": alert_threshold_ms
                            },
                            severity=EventSeverity.WARNING
                        )
                        asyncio.create_task(self._monitoring.log_event(event))
                
                if memory_limit_mb and memory_used > memory_limit_mb:
                    raise PluginResourceError(
                        f"Method {method_name} exceeded memory limit: {memory_used:.2f}MB > {memory_limit_mb}MB",
                        resource_type="memory",
                        limit_value=memory_limit_mb,
                        current_usage=memory_used,
                        plugin_id=getattr(self, 'plugin_id', None)
                    )
                
                return result
                
            except Exception as e:
                # Track error metrics
                execution_time = (time.time() - start_time) * 1000
                _performance_metrics[f"{method_name}_error_count"].append(1)
                _performance_metrics[f"{method_name}_error_time"].append(execution_time)
                raise
        
        # Store metadata
        _method_metadata[func.__name__].update({
            "track_memory": track_memory,
            "track_execution_time": track_execution_time,
            "alert_threshold_ms": alert_threshold_ms,
            "memory_limit_mb": memory_limit_mb
        })
        
        return wrapper
    
    return decorator


def error_handled(
    default_return: Any = None,
    log_errors: bool = True,
    suppress_errors: bool = False,
    error_callback: Optional[Callable] = None
):
    """
    Decorator for comprehensive error handling.
    
    Args:
        default_return: Default value to return on error
        log_errors: Whether to log errors
        suppress_errors: Whether to suppress errors (return default instead)
        error_callback: Optional callback function for custom error handling
        
    Example:
        @error_handled(default_return=[], suppress_errors=True)
        def get_data_list(self):
            # Implementation that might fail
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            method_name = f"{self.__class__.__name__}.{func.__name__}"
            
            try:
                return func(self, *args, **kwargs)
                
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {method_name}: {e}")
                
                # Call error callback if provided
                if error_callback:
                    try:
                        error_callback(self, e, method_name, args, kwargs)
                    except Exception as callback_error:
                        logger.error(f"Error in error callback for {method_name}: {callback_error}")
                
                # Emit error event if plugin has monitoring
                if hasattr(self, '_monitoring') and self._monitoring:
                    event = PluginEvent(
                        event_type="method_error",
                        plugin_id=getattr(self, 'plugin_id', 'unknown'),
                        data={
                            "method": method_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        },
                        severity=EventSeverity.ERROR
                    )
                    try:
                        asyncio.create_task(self._monitoring.log_event(event))
                    except:
                        pass  # Don't fail if monitoring unavailable
                
                if suppress_errors:
                    return default_return
                else:
                    raise
        
        # Store metadata
        _method_metadata[func.__name__].update({
            "default_return": default_return,
            "log_errors": log_errors,
            "suppress_errors": suppress_errors,
            "has_error_callback": error_callback is not None
        })
        
        return wrapper
    
    return decorator


def cached_result(
    ttl_seconds: int = 300,
    max_size: int = 100,
    key_func: Optional[Callable] = None,
    ignore_args: Optional[List[str]] = None
):
    """
    Decorator for caching method results.
    
    Args:
        ttl_seconds: Time-to-live for cached results
        max_size: Maximum number of cached results
        key_func: Custom function to generate cache keys
        ignore_args: List of argument names to ignore in cache key
        
    Example:
        @cached_result(ttl_seconds=600, max_size=50)
        def expensive_computation(self, data):
            # Expensive computation
            return result
    """
    
    def decorator(func: Callable) -> Callable:
        cache_key_prefix = f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = f"{cache_key_prefix}:{key_func(self, *args, **kwargs)}"
            else:
                # Default key generation
                filtered_kwargs = kwargs.copy()
                if ignore_args:
                    for arg in ignore_args:
                        filtered_kwargs.pop(arg, None)
                
                cache_key = f"{cache_key_prefix}:{hash((args, tuple(sorted(filtered_kwargs.items()))))}"
            
            # Check cache
            cache_data = _cache_storage.get(cache_key)
            current_time = datetime.utcnow()
            
            if cache_data:
                result, timestamp = cache_data
                if (current_time - timestamp).total_seconds() < ttl_seconds:
                    logger.debug(f"Cache hit for {cache_key}")
                    return result
                else:
                    # Expired
                    del _cache_storage[cache_key]
            
            # Execute function and cache result
            result = func(self, *args, **kwargs)
            
            # Maintain cache size limit
            if len(_cache_storage) >= max_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    _cache_storage.keys(),
                    key=lambda k: _cache_storage[k][1]
                )
                for old_key in sorted_keys[:len(_cache_storage) - max_size + 1]:
                    del _cache_storage[old_key]
            
            _cache_storage[cache_key] = (result, current_time)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        # Store metadata
        _method_metadata[func.__name__].update({
            "cached": True,
            "ttl_seconds": ttl_seconds,
            "max_size": max_size,
            "has_custom_key_func": key_func is not None
        })
        
        return wrapper
    
    return decorator


def validate_params(**param_validators):
    """
    Decorator for parameter validation.
    
    Args:
        **param_validators: Mapping of parameter names to validation functions
        
    Example:
        @validate_params(
            data=lambda x: isinstance(x, dict),
            count=lambda x: isinstance(x, int) and x > 0
        )
        def process_data(self, data, count):
            # Implementation
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            import inspect
            
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            validation_errors = []
            for param_name, validator in param_validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        if not validator(value):
                            validation_errors.append(f"Parameter '{param_name}' failed validation")
                    except Exception as e:
                        validation_errors.append(f"Parameter '{param_name}' validation error: {e}")
            
            if validation_errors:
                from .exceptions import PluginValidationError
                raise PluginValidationError(
                    f"Parameter validation failed for {func.__name__}",
                    validation_errors=validation_errors,
                    plugin_id=getattr(self, 'plugin_id', None)
                )
            
            return func(self, *args, **kwargs)
        
        # Store metadata
        _method_metadata[func.__name__].update({
            "has_param_validation": True,
            "validated_params": list(param_validators.keys())
        })
        
        return wrapper
    
    return decorator


# Utility functions

def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def get_performance_metrics(method_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get performance metrics for decorated methods.
    
    Args:
        method_name: Specific method name to get metrics for (or all if None)
        
    Returns:
        Dict containing performance metrics
    """
    if method_name:
        # Get metrics for specific method
        method_metrics = {}
        for metric_key, values in _performance_metrics.items():
            if metric_key.startswith(method_name):
                if values:
                    method_metrics[metric_key] = {
                        "count": len(values),
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "total": sum(values)
                    }
        return method_metrics
    else:
        # Get all metrics
        all_metrics = {}
        for metric_key, values in _performance_metrics.items():
            if values:
                all_metrics[metric_key] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values)
                }
        return all_metrics


def get_method_metadata(method_name: str) -> Dict[str, Any]:
    """
    Get metadata for a decorated method.
    
    Args:
        method_name: Name of the method
        
    Returns:
        Dict containing method metadata
    """
    return _method_metadata.get(method_name, {})


def clear_cache(method_name: Optional[str] = None) -> None:
    """
    Clear cached results.
    
    Args:
        method_name: Specific method to clear cache for (or all if None)
    """
    if method_name:
        # Clear cache for specific method
        keys_to_remove = [key for key in _cache_storage.keys() if method_name in key]
        for key in keys_to_remove:
            del _cache_storage[key]
    else:
        # Clear all cache
        _cache_storage.clear()


def reset_metrics() -> None:
    """Reset all performance metrics."""
    _performance_metrics.clear()


# Context managers for advanced scenarios

class performance_context:
    """Context manager for tracking performance across multiple operations."""
    
    def __init__(self, operation_name: str, plugin_id: str = None):
        self.operation_name = operation_name
        self.plugin_id = plugin_id
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = _get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (time.time() - self.start_time) * 1000
        memory_used = _get_memory_usage() - self.start_memory
        
        # Store metrics
        _performance_metrics[f"{self.operation_name}_execution_time"].append(execution_time)
        _performance_metrics[f"{self.operation_name}_memory_usage"].append(memory_used)
        
        if exc_type:
            _performance_metrics[f"{self.operation_name}_error_count"].append(1)
        
        logger.debug(f"Operation {self.operation_name} completed in {execution_time:.2f}ms, used {memory_used:.2f}MB")


class error_context:
    """Context manager for error handling across multiple operations."""
    
    def __init__(self, operation_name: str, plugin_id: str = None, suppress_errors: bool = False):
        self.operation_name = operation_name
        self.plugin_id = plugin_id
        self.suppress_errors = suppress_errors
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error in operation {self.operation_name}: {exc_val}")
            return self.suppress_errors  # Suppress exception if configured
        return False
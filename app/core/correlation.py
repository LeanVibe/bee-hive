"""
Correlation ID System for LeanVibe Agent Hive 2.0

Provides comprehensive tracking across all workflows, API calls, Redis messages,
database operations, and multi-agent coordination for enhanced observability
and debugging capabilities.
"""

import uuid
import contextlib
from contextvars import ContextVar
from typing import Optional, Dict, Any
from datetime import datetime
import structlog

# Global correlation context variable
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
correlation_metadata: ContextVar[Optional[Dict[str, Any]]] = ContextVar('correlation_metadata', default=None)

logger = structlog.get_logger(__name__)


class CorrelationManager:
    """
    Manages correlation IDs and metadata throughout the entire system lifecycle.
    
    Provides tracking for:
    - API requests and responses
    - Multi-agent workflows
    - Redis message passing
    - Database operations
    - Custom command execution
    - Error propagation and debugging
    """
    
    @staticmethod
    def generate_id() -> str:
        """Generate new correlation ID for workflow tracking."""
        return str(uuid.uuid4())
    
    @staticmethod
    @contextlib.contextmanager
    def correlation_context(correlation_id_value: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Set correlation ID and metadata for entire workflow context.
        
        Args:
            correlation_id_value: Unique correlation ID
            metadata: Additional context metadata (user_id, workflow_type, etc.)
        
        Usage:
            with CorrelationManager.correlation_context("workflow-123", {"user": "agent-1"}):
                # All operations in this context will have correlation tracking
                result = await execute_workflow()
        """
        # Set correlation ID
        correlation_token = correlation_id.set(correlation_id_value)
        
        # Set metadata
        metadata = metadata or {}
        metadata.update({
            'created_at': datetime.utcnow().isoformat(),
            'correlation_id': correlation_id_value
        })
        metadata_token = correlation_metadata.set(metadata)
        
        try:
            logger.debug(
                "Starting correlation context",
                correlation_id=correlation_id_value,
                metadata=metadata
            )
            yield correlation_id_value
        finally:
            logger.debug(
                "Ending correlation context",
                correlation_id=correlation_id_value
            )
            correlation_id.reset(correlation_token)
            correlation_metadata.reset(metadata_token)
    
    @staticmethod
    def get_current_id() -> str:
        """
        Get current correlation ID or generate new one.
        
        Returns:
            Current correlation ID, or new one if none exists
        """
        current = correlation_id.get()
        if not current:
            current = CorrelationManager.generate_id()
            correlation_id.set(current)
            logger.debug("Generated new correlation ID", correlation_id=current)
        return current
    
    @staticmethod
    def get_current_metadata() -> Dict[str, Any]:
        """
        Get current correlation metadata.
        
        Returns:
            Current correlation metadata dictionary
        """
        metadata = correlation_metadata.get()
        return metadata if metadata is not None else {}
    
    @staticmethod
    def add_metadata(key: str, value: Any) -> None:
        """
        Add metadata to current correlation context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        current_metadata = correlation_metadata.get()
        if current_metadata is None:
            current_metadata = {}
        else:
            # Make a copy to avoid modifying the original
            current_metadata = current_metadata.copy()
        
        current_metadata[key] = value
        correlation_metadata.set(current_metadata)
        
        logger.debug(
            "Added correlation metadata",
            correlation_id=CorrelationManager.get_current_id(),
            key=key,
            value=value
        )
    
    @staticmethod
    def get_correlation_info() -> Dict[str, Any]:
        """
        Get complete correlation information for logging and debugging.
        
        Returns:
            Dictionary with correlation ID and all metadata
        """
        return {
            'correlation_id': CorrelationManager.get_current_id(),
            'metadata': CorrelationManager.get_current_metadata(),
            'timestamp': datetime.utcnow().isoformat()
        }


class EnhancedLogger:
    """
    Enhanced logger with automatic correlation ID and component context.
    
    Provides structured logging with correlation tracking for all system components.
    """
    
    @staticmethod
    def get_logger(component: str) -> structlog.stdlib.BoundLogger:
        """
        Get logger with correlation ID and component context.
        
        Args:
            component: Component name (e.g., "coordination_engine", "redis_manager")
            
        Returns:
            Structured logger with correlation context
        """
        correlation_info = CorrelationManager.get_correlation_info()
        
        return structlog.get_logger().bind(
            component=component,
            correlation_id=correlation_info['correlation_id'],
            correlation_metadata=correlation_info['metadata']
        )
    
    @staticmethod
    def log_operation_start(component: str, operation: str, **kwargs) -> structlog.stdlib.BoundLogger:
        """
        Log the start of an operation with correlation tracking.
        
        Args:
            component: Component name
            operation: Operation being performed
            **kwargs: Additional operation-specific data
            
        Returns:
            Logger for continued use in the operation
        """
        logger = EnhancedLogger.get_logger(component)
        
        operation_data = {
            'operation': operation,
            'operation_id': str(uuid.uuid4()),
            'started_at': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        # Add operation metadata to correlation context
        CorrelationManager.add_metadata(f'operation_{operation}', operation_data)
        
        logger.info(f"Starting {operation}", **operation_data)
        
        return logger
    
    @staticmethod
    def log_operation_complete(
        logger: structlog.stdlib.BoundLogger, 
        operation: str, 
        success: bool = True,
        duration_ms: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Log the completion of an operation with correlation tracking.
        
        Args:
            logger: Logger from log_operation_start
            operation: Operation that was performed
            success: Whether operation succeeded
            duration_ms: Operation duration in milliseconds
            **kwargs: Additional operation-specific data
        """
        completion_data = {
            'operation': operation,
            'success': success,
            'completed_at': datetime.utcnow().isoformat(),
            **kwargs
        }
        
        if duration_ms is not None:
            completion_data['duration_ms'] = duration_ms
        
        # Add completion metadata to correlation context
        CorrelationManager.add_metadata(f'completed_{operation}', completion_data)
        
        if success:
            logger.info(f"Completed {operation}", **completion_data)
        else:
            logger.error(f"Failed {operation}", **completion_data)


class CorrelationMiddleware:
    """
    Middleware for automatic correlation ID management in FastAPI requests.
    
    Automatically generates or extracts correlation IDs from HTTP headers
    and maintains context throughout request processing.
    """
    
    CORRELATION_HEADER = "X-Correlation-ID"
    
    @staticmethod
    def extract_correlation_id_from_request(request) -> str:
        """
        Extract correlation ID from request headers or generate new one.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            Correlation ID for this request
        """
        # Try to get correlation ID from header
        correlation_id_value = request.headers.get(CorrelationMiddleware.CORRELATION_HEADER)
        
        if not correlation_id_value:
            # Generate new correlation ID
            correlation_id_value = CorrelationManager.generate_id()
        
        return correlation_id_value
    
    @staticmethod
    def create_request_metadata(request) -> Dict[str, Any]:
        """
        Create metadata from FastAPI request for correlation context.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            Metadata dictionary with request information
        """
        return {
            'request_method': request.method,
            'request_url': str(request.url),
            'request_path': request.url.path,
            'client_host': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent'),
            'request_time': datetime.utcnow().isoformat()
        }


# Convenience functions for common operations
def with_correlation(correlation_id_value: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for functions that need correlation context.
    
    Usage:
        @with_correlation("workflow-123", {"user": "agent-1"})
        async def execute_workflow():
            # Function executes with correlation context
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with CorrelationManager.correlation_context(correlation_id_value, metadata):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def get_correlation_headers() -> Dict[str, str]:
    """
    Get HTTP headers with current correlation ID for outbound requests.
    
    Returns:
        Dictionary with correlation headers
    """
    return {
        CorrelationMiddleware.CORRELATION_HEADER: CorrelationManager.get_current_id()
    }


def add_correlation_to_redis_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add correlation information to Redis message payload.
    
    Args:
        message: Original message dictionary
        
    Returns:
        Message with correlation information added
    """
    correlation_info = CorrelationManager.get_correlation_info()
    
    return {
        **message,
        'correlation_id': correlation_info['correlation_id'],
        'correlation_metadata': correlation_info['metadata'],
        'message_timestamp': datetime.utcnow().isoformat()
    }


# Export key classes and functions
__all__ = [
    'CorrelationManager',
    'EnhancedLogger', 
    'CorrelationMiddleware',
    'with_correlation',
    'get_correlation_headers',
    'add_correlation_to_redis_message'
]
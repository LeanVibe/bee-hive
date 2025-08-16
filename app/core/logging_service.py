"""
Unified Logging Service for LeanVibe Agent Hive
Consolidates 466+ logger instances into consistent infrastructure

This service provides centralized logging configuration, consistent formatting,
and singleton pattern management for all application logging needs.
"""

import structlog
import logging.config
from typing import Optional, Dict, Any, Union
from enum import Enum
import os


class LogLevel(Enum):
    """Standard log levels for consistent usage across application."""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LoggingService:
    """
    Centralized logging service for consistent application logging.
    
    This singleton service replaces 466+ individual logger instances across
    the codebase with a unified, consistent logging infrastructure.
    
    Key Benefits:
    - Single point of logging configuration
    - Consistent log format across all modules
    - Component-specific context binding
    - Centralized log level management
    - Singleton pattern for efficiency
    """
    
    _instance: Optional['LoggingService'] = None
    _configured: bool = False
    _logger_cache: Dict[str, structlog.stdlib.BoundLogger] = {}
    
    def __new__(cls) -> 'LoggingService':
        """Ensure singleton pattern for logging service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging service with configuration."""
        if not self._configured:
            self._configure_logging()
            self._configured = True
    
    def _configure_logging(self) -> None:
        """
        Configure structured logging with consistent processors.
        
        Uses same configuration as original main.py but centralized here.
        Supports both development and production environments.
        """
        # Determine if we're in development mode
        debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
        
        # Base processors for all environments
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add JSON rendering for structured output
        processors.append(structlog.processors.JSONRenderer())
        
        # Configure structlog with consistent settings
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Set root logging level based on environment
        if debug_mode:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    
    def get_logger(self, name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
        """
        Get configured logger for module.
        
        Args:
            name: Logger name, defaults to calling module if None
            
        Returns:
            Configured structlog.stdlib.BoundLogger instance
        """
        logger_name = name or __name__
        
        # Use cached logger if available for performance
        if logger_name in self._logger_cache:
            return self._logger_cache[logger_name]
        
        # Create new logger and cache it
        logger = structlog.get_logger(logger_name)
        self._logger_cache[logger_name] = logger
        
        return logger
    
    def get_component_logger(self, 
                           component: str, 
                           context: Optional[Dict[str, Any]] = None) -> structlog.stdlib.BoundLogger:
        """
        Get logger with component-specific context.
        
        This method provides consistent component naming and context binding
        for different application modules.
        
        Args:
            component: Component name (e.g., 'orchestrator', 'project_index')
            context: Additional context to bind to logger
            
        Returns:
            Configured logger with component context
        """
        logger_name = f"app.{component}"
        logger = self.get_logger(logger_name)
        
        # Bind component context if provided
        if context:
            logger = logger.bind(**context)
            
        return logger
    
    def get_module_logger(self, module_name: str) -> structlog.stdlib.BoundLogger:
        """
        Get logger for specific module using standard naming convention.
        
        Args:
            module_name: Full module name (e.g., app.core.orchestrator)
            
        Returns:
            Configured logger for the module
        """
        return self.get_logger(module_name)
    
    def bind_context(self, 
                    logger: structlog.stdlib.BoundLogger,
                    **context: Any) -> structlog.stdlib.BoundLogger:
        """
        Bind additional context to existing logger.
        
        Args:
            logger: Existing logger instance
            **context: Context key-value pairs to bind
            
        Returns:
            Logger with additional context bound
        """
        return logger.bind(**context)
    
    def configure_level(self, level: Union[LogLevel, str]) -> None:
        """
        Configure logging level at runtime.
        
        Args:
            level: Log level to set (LogLevel enum or string)
        """
        if isinstance(level, LogLevel):
            level_str = level.value.upper()
        else:
            level_str = level.upper()
            
        numeric_level = getattr(logging, level_str, logging.INFO)
        logging.getLogger().setLevel(numeric_level)
    
    def reset_configuration(self) -> None:
        """
        Reset logging configuration and clear cache.
        
        Useful for testing or runtime reconfiguration.
        """
        self._configured = False
        self._logger_cache.clear()
        self._configure_logging()
    
    @classmethod
    def get_instance(cls) -> 'LoggingService':
        """
        Get singleton instance of logging service.
        
        Returns:
            The singleton LoggingService instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Convenience functions for backward compatibility and ease of use
def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Convenience function to get logger from service.
    
    Args:
        name: Logger name, defaults to calling module
        
    Returns:
        Configured logger instance
    """
    service = LoggingService.get_instance()
    return service.get_logger(name)


def get_component_logger(component: str, 
                        context: Optional[Dict[str, Any]] = None) -> structlog.stdlib.BoundLogger:
    """
    Convenience function to get component logger.
    
    Args:
        component: Component name
        context: Additional context to bind
        
    Returns:
        Configured component logger
    """
    service = LoggingService.get_instance()
    return service.get_component_logger(component, context)


def get_module_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """
    Convenience function to get module logger.
    
    Args:
        module_name: Full module name
        
    Returns:
        Configured module logger
    """
    service = LoggingService.get_instance()
    return service.get_module_logger(module_name)


# Global logging service instance
_logging_service = LoggingService()
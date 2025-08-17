"""
Unified Manager Base Class for LeanVibe Agent Hive 2.0

Provides the foundation for all unified managers with common patterns:
- Dependency injection
- Plugin architecture
- Performance monitoring
- Error handling
- Lifecycle management
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()

T = TypeVar('T')


class ManagerStatus(str, Enum):
    """Manager lifecycle status."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    INACTIVE = "inactive"


class PluginType(str, Enum):
    """Types of plugins supported by managers."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONTEXT = "context"
    COMMUNICATION = "communication"
    STORAGE = "storage"
    WORKFLOW = "workflow"
    MONITORING = "monitoring"


@dataclass
class ManagerMetrics:
    """Performance metrics for a manager."""
    operation_count: int = 0
    total_execution_time: float = 0.0
    error_count: int = 0
    average_response_time: float = 0.0
    peak_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    
    def record_operation(self, execution_time: float, success: bool = True):
        """Record an operation execution."""
        self.operation_count += 1
        self.total_execution_time += execution_time
        if not success:
            self.error_count += 1
        self.average_response_time = self.total_execution_time / self.operation_count

    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.operation_count == 0:
            return 0.0
        return (self.error_count / self.operation_count) * 100


class PluginInterface(ABC):
    """Base interface for all manager plugins."""
    
    @abstractmethod
    async def initialize(self, manager: 'UnifiedManagerBase') -> bool:
        """Initialize the plugin with the manager instance."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Get the plugin type."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the plugin name."""
        pass


class ManagerConfig(BaseModel):
    """Base configuration for unified managers."""
    
    # Core settings
    name: str
    enabled: bool = True
    debug_mode: bool = False
    
    # Performance settings
    max_concurrent_operations: int = 100
    operation_timeout_seconds: float = 30.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # Plugin settings
    plugins_enabled: bool = True
    plugin_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Monitoring settings
    metrics_enabled: bool = True
    health_check_interval_seconds: int = 60
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5


class UnifiedManagerBase(ABC, Generic[T]):
    """
    Base class for all unified managers in the system.
    
    Provides common functionality:
    - Plugin management
    - Performance monitoring
    - Error handling with circuit breaker
    - Dependency injection
    - Lifecycle management
    """
    
    def __init__(
        self,
        config: ManagerConfig,
        dependencies: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.status = ManagerStatus.INITIALIZING
        self.metrics = ManagerMetrics()
        self.dependencies = dependencies or {}
        self.plugins: Dict[str, PluginInterface] = {}
        self.cache: Dict[str, Any] = {}
        self.circuit_breaker_state = {"failures": 0, "last_failure": None, "open": False}
        self._startup_time = datetime.utcnow()
        
        logger.info(f"🚀 {self.config.name} initializing", manager=self.__class__.__name__)
    
    async def initialize(self) -> bool:
        """Initialize the manager and its plugins."""
        try:
            # Load and initialize plugins
            if self.config.plugins_enabled:
                await self._load_plugins()
            
            # Perform manager-specific initialization
            success = await self._initialize_manager()
            
            if success:
                self.status = ManagerStatus.ACTIVE
                logger.info(
                    f"✅ {self.config.name} initialized successfully",
                    manager=self.__class__.__name__,
                    plugins_loaded=len(self.plugins),
                    startup_time_ms=(datetime.utcnow() - self._startup_time).total_seconds() * 1000
                )
                return True
            else:
                self.status = ManagerStatus.DEGRADED
                logger.error(f"❌ {self.config.name} initialization failed")
                return False
                
        except Exception as e:
            self.status = ManagerStatus.INACTIVE
            logger.error(f"❌ {self.config.name} initialization error", error=str(e))
            return False
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown the manager."""
        try:
            self.status = ManagerStatus.SHUTTING_DOWN
            
            # Cleanup plugins
            for plugin in self.plugins.values():
                try:
                    await plugin.cleanup()
                except Exception as e:
                    logger.warning(f"Plugin cleanup error: {plugin.name}", error=str(e))
            
            # Perform manager-specific cleanup
            await self._shutdown_manager()
            
            self.status = ManagerStatus.INACTIVE
            logger.info(f"✅ {self.config.name} shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.config.name} shutdown error", error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        health_status = {
            "status": self.status.value,
            "manager": self.__class__.__name__,
            "uptime_seconds": (datetime.utcnow() - self._startup_time).total_seconds(),
            "metrics": {
                "operations": self.metrics.operation_count,
                "error_rate": self.metrics.get_error_rate(),
                "avg_response_time_ms": self.metrics.average_response_time * 1000,
                "memory_usage_mb": self.metrics.peak_memory_usage
            },
            "plugins": {name: True for name in self.plugins.keys()},
            "circuit_breaker": {
                "open": self.circuit_breaker_state["open"],
                "failures": self.circuit_breaker_state["failures"]
            }
        }
        
        # Add manager-specific health data
        try:
            manager_health = await self._get_manager_health()
            health_status.update(manager_health)
        except Exception as e:
            health_status["manager_health_error"] = str(e)
        
        return health_status
    
    def add_plugin(self, plugin: PluginInterface) -> bool:
        """Add a plugin to the manager."""
        try:
            self.plugins[plugin.name] = plugin
            logger.info(f"Plugin added: {plugin.name}", plugin_type=plugin.plugin_type.value)
            return True
        except Exception as e:
            logger.error(f"Failed to add plugin: {plugin.name}", error=str(e))
            return False
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def inject_dependency(self, name: str, dependency: Any) -> None:
        """Inject a dependency."""
        self.dependencies[name] = dependency
        logger.debug(f"Dependency injected: {name}")
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """Get an injected dependency."""
        return self.dependencies.get(name)
    
    async def execute_with_monitoring(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ) -> Any:
        """Execute an operation with monitoring and error handling."""
        if self.circuit_breaker_state["open"]:
            # Check if circuit breaker should close
            if (datetime.utcnow() - self.circuit_breaker_state["last_failure"]).seconds > 60:
                self.circuit_breaker_state["open"] = False
                self.circuit_breaker_state["failures"] = 0
                logger.info("Circuit breaker closed", manager=self.__class__.__name__)
            else:
                raise Exception("Circuit breaker is open - operation rejected")
        
        start_time = time.time()
        success = True
        
        try:
            # Execute the operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            # Record success metrics
            execution_time = time.time() - start_time
            self.metrics.record_operation(execution_time, success=True)
            
            logger.debug(
                f"Operation completed: {operation_name}",
                execution_time_ms=execution_time * 1000,
                manager=self.__class__.__name__
            )
            
            return result
            
        except Exception as e:
            success = False
            execution_time = time.time() - start_time
            self.metrics.record_operation(execution_time, success=False)
            
            # Update circuit breaker
            self.circuit_breaker_state["failures"] += 1
            self.circuit_breaker_state["last_failure"] = datetime.utcnow()
            
            if (self.circuit_breaker_state["failures"] >= self.config.failure_threshold and
                self.config.circuit_breaker_enabled):
                self.circuit_breaker_state["open"] = True
                logger.warning("Circuit breaker opened", manager=self.__class__.__name__)
            
            logger.error(
                f"Operation failed: {operation_name}",
                error=str(e),
                execution_time_ms=execution_time * 1000,
                manager=self.__class__.__name__
            )
            
            raise
    
    # Abstract methods for manager-specific implementation
    
    @abstractmethod
    async def _initialize_manager(self) -> bool:
        """Manager-specific initialization logic."""
        pass
    
    @abstractmethod
    async def _shutdown_manager(self) -> None:
        """Manager-specific shutdown logic."""
        pass
    
    @abstractmethod
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get manager-specific health information."""
        pass
    
    @abstractmethod
    async def _load_plugins(self) -> None:
        """Load manager-specific plugins."""
        pass
    
    # Optional cache management
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.cache_enabled:
            return None
        
        cached_item = self.cache.get(key)
        if cached_item is None:
            return None
        
        # Check TTL
        if time.time() - cached_item["timestamp"] > self.config.cache_ttl_seconds:
            del self.cache[key]
            return None
        
        self.metrics.cache_hit_rate = len([v for v in self.cache.values() 
                                         if time.time() - v["timestamp"] <= self.config.cache_ttl_seconds]) / max(len(self.cache), 1)
        
        return cached_item["value"]
    
    def cache_set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self.config.cache_enabled:
            return
        
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        
        # Clean old entries if cache is getting large
        if len(self.cache) > 1000:
            current_time = time.time()
            self.cache = {
                k: v for k, v in self.cache.items()
                if current_time - v["timestamp"] <= self.config.cache_ttl_seconds
            }
    
    def cache_clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


# Utility function for creating manager configurations
def create_manager_config(
    name: str,
    **overrides
) -> ManagerConfig:
    """Create a manager configuration with optional overrides."""
    config_data = {"name": name}
    config_data.update(overrides)
    return ManagerConfig(**config_data)
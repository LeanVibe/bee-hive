"""
Base Engine Architecture for LeanVibe Agent Hive 2.0 Consolidated Engines

Provides the foundational interfaces and utilities for all 8 specialized engines.
Performance-first design with async operations, comprehensive metrics, and plugin extensibility.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, TypeVar, Generic
import logging
import traceback

# Core imports with graceful fallbacks
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class EngineStatus(str, Enum):
    """Engine operational status."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


class RequestPriority(str, Enum):
    """Request priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class EngineConfig:
    """Base configuration for all engines."""
    engine_id: str
    name: str
    version: str = "2.0.0"
    max_concurrent_requests: int = 1000
    request_timeout_seconds: int = 30
    health_check_interval_seconds: int = 60
    metrics_collection_enabled: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    plugins_enabled: bool = True
    plugin_configs: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class EngineRequest:
    """Base request for all engines."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: str = ""
    priority: RequestPriority = RequestPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class EngineResponse:
    """Base response for all engines."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    engine_id: str = ""


@dataclass
class HealthStatus:
    """Engine health status information."""
    status: EngineStatus
    last_check: datetime
    uptime_seconds: float
    active_requests: int
    total_requests_processed: int
    error_rate_5min: float
    average_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineMetrics:
    """Comprehensive engine performance metrics."""
    engine_id: str
    requests_per_second: float
    average_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate_percent: float
    success_rate_percent: float
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate_percent: float = 0.0
    plugin_metrics: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed through circuit breaker."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class EnginePlugin(ABC):
    """Base interface for engine plugins."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version."""
        pass
        
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration."""
        pass
    
    @abstractmethod
    async def can_handle(self, request: EngineRequest) -> bool:
        """Check if plugin can handle the request."""
        pass
    
    @abstractmethod
    async def process(self, request: EngineRequest) -> EngineResponse:
        """Process request with plugin."""
        pass
    
    @abstractmethod
    async def get_health(self) -> Dict[str, Any]:
        """Get plugin health information."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown plugin gracefully."""
        pass


class PluginRegistry:
    """Registry for managing engine plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, EnginePlugin] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
    
    async def register_plugin(self, plugin: EnginePlugin, config: Dict[str, Any] = None):
        """Register a plugin with optional configuration."""
        plugin_name = plugin.get_name()
        self._plugins[plugin_name] = plugin
        self._plugin_configs[plugin_name] = config or {}
        
        # Initialize plugin
        await plugin.initialize(self._plugin_configs[plugin_name])
        logger.info(f"Registered plugin: {plugin_name} v{plugin.get_version()}")
    
    async def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin."""
        if plugin_name in self._plugins:
            await self._plugins[plugin_name].shutdown()
            del self._plugins[plugin_name]
            del self._plugin_configs[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
    
    async def find_handler(self, request: EngineRequest) -> Optional[EnginePlugin]:
        """Find the first plugin that can handle the request."""
        for plugin in self._plugins.values():
            try:
                if await plugin.can_handle(request):
                    return plugin
            except Exception as e:
                logger.error(f"Error checking plugin {plugin.get_name()}: {e}")
        return None
    
    def get_plugin(self, name: str) -> Optional[EnginePlugin]:
        """Get plugin by name."""
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())
    
    async def get_plugins_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all plugins."""
        health = {}
        for name, plugin in self._plugins.items():
            try:
                health[name] = await plugin.get_health()
            except Exception as e:
                health[name] = {"status": "error", "error": str(e)}
        return health


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.request_times: List[float] = []
        self.request_timestamps: List[datetime] = []
        self.error_count = 0
        self.success_count = 0
    
    def record_request(self, processing_time_ms: float, success: bool):
        """Record request metrics."""
        now = datetime.utcnow()
        self.request_times.append(processing_time_ms)
        self.request_timestamps.append(now)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Clean old entries
        cutoff = now - timedelta(seconds=self.window_size)
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.pop(0)
            self.request_times.pop(0)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.request_times:
            return {
                "requests_per_second": 0.0,
                "average_response_time_ms": 0.0,
                "p95_response_time_ms": 0.0,
                "p99_response_time_ms": 0.0,
                "error_rate_percent": 0.0,
                "success_rate_percent": 100.0
            }
        
        sorted_times = sorted(self.request_times)
        total_requests = self.success_count + self.error_count
        
        return {
            "requests_per_second": len(self.request_times) / self.window_size,
            "average_response_time_ms": sum(self.request_times) / len(self.request_times),
            "p95_response_time_ms": sorted_times[int(len(sorted_times) * 0.95)],
            "p99_response_time_ms": sorted_times[int(len(sorted_times) * 0.99)],
            "error_rate_percent": (self.error_count / total_requests * 100) if total_requests > 0 else 0.0,
            "success_rate_percent": (self.success_count / total_requests * 100) if total_requests > 0 else 100.0
        }


class BaseEngine(ABC):
    """Base class for all specialized engines."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.status = EngineStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.active_requests: Set[str] = set()
        self.total_requests = 0
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout
        ) if config.circuit_breaker_enabled else None
        self.performance_monitor = PerformanceMonitor()
        self.plugin_registry = PluginRegistry() if config.plugins_enabled else None
        
        # Rate limiting
        self._request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
    async def initialize(self) -> None:
        """Initialize the engine."""
        try:
            logger.info(f"Initializing engine: {self.config.name}")
            await self._engine_initialize()
            self.status = EngineStatus.HEALTHY
            logger.info(f"Engine {self.config.name} initialized successfully")
        except Exception as e:
            self.status = EngineStatus.UNHEALTHY
            logger.error(f"Failed to initialize engine {self.config.name}: {e}")
            raise
    
    @abstractmethod
    async def _engine_initialize(self) -> None:
        """Engine-specific initialization logic."""
        pass
    
    async def process(self, request: EngineRequest) -> EngineResponse:
        """Process a request with full monitoring and error handling."""
        start_time = time.time()
        
        # Apply request timeout if not specified
        if request.timeout_seconds is None:
            request.timeout_seconds = self.config.request_timeout_seconds
        
        try:
            # Circuit breaker check
            if self.circuit_breaker and not self.circuit_breaker.should_allow_request():
                return EngineResponse(
                    request_id=request.request_id,
                    success=False,
                    error="Circuit breaker is open",
                    error_code="CIRCUIT_BREAKER_OPEN",
                    engine_id=self.config.engine_id
                )
            
            # Rate limiting
            async with self._request_semaphore:
                self.active_requests.add(request.request_id)
                self.total_requests += 1
                
                try:
                    # Check if plugin can handle request
                    if self.plugin_registry:
                        plugin = await self.plugin_registry.find_handler(request)
                        if plugin:
                            logger.debug(f"Request {request.request_id} handled by plugin {plugin.get_name()}")
                            response = await asyncio.wait_for(
                                plugin.process(request),
                                timeout=request.timeout_seconds
                            )
                        else:
                            # Process with engine
                            response = await asyncio.wait_for(
                                self._engine_process(request),
                                timeout=request.timeout_seconds
                            )
                    else:
                        response = await asyncio.wait_for(
                            self._engine_process(request),
                            timeout=request.timeout_seconds
                        )
                    
                    response.engine_id = self.config.engine_id
                    
                    # Record success
                    if self.circuit_breaker:
                        self.circuit_breaker.record_success()
                    
                    processing_time = (time.time() - start_time) * 1000
                    response.processing_time_ms = processing_time
                    self.performance_monitor.record_request(processing_time, response.success)
                    
                    return response
                    
                finally:
                    self.active_requests.discard(request.request_id)
                    
        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(processing_time, False)
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=f"Request timeout after {request.timeout_seconds}s",
                error_code="TIMEOUT",
                processing_time_ms=processing_time,
                engine_id=self.config.engine_id
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_request(processing_time, False)
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                
            logger.error(f"Error processing request {request.request_id}: {e}")
            logger.debug(traceback.format_exc())
            
            return EngineResponse(
                request_id=request.request_id,
                success=False,
                error=str(e),
                error_code="PROCESSING_ERROR",
                processing_time_ms=processing_time,
                engine_id=self.config.engine_id
            )
    
    @abstractmethod
    async def _engine_process(self, request: EngineRequest) -> EngineResponse:
        """Engine-specific request processing logic."""
        pass
    
    async def get_health(self) -> HealthStatus:
        """Get engine health status."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        metrics = self.performance_monitor.get_metrics()
        
        # Determine status based on metrics
        status = self.status
        if self.status == EngineStatus.HEALTHY:
            if metrics["error_rate_percent"] > 50:
                status = EngineStatus.UNHEALTHY
            elif metrics["error_rate_percent"] > 10:
                status = EngineStatus.DEGRADED
        
        return HealthStatus(
            status=status,
            last_check=datetime.utcnow(),
            uptime_seconds=uptime,
            active_requests=len(self.active_requests),
            total_requests_processed=self.total_requests,
            error_rate_5min=metrics["error_rate_percent"],
            average_response_time_ms=metrics["average_response_time_ms"],
            memory_usage_mb=0.0,  # To be implemented by specific engines
            cpu_usage_percent=0.0,  # To be implemented by specific engines
            details={
                "circuit_breaker_state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
                "plugin_count": len(self.plugin_registry.list_plugins()) if self.plugin_registry else 0
            }
        )
    
    async def get_metrics(self) -> EngineMetrics:
        """Get comprehensive engine metrics."""
        perf_metrics = self.performance_monitor.get_metrics()
        
        return EngineMetrics(
            engine_id=self.config.engine_id,
            requests_per_second=perf_metrics["requests_per_second"],
            average_response_time_ms=perf_metrics["average_response_time_ms"],
            p95_response_time_ms=perf_metrics["p95_response_time_ms"],
            p99_response_time_ms=perf_metrics["p99_response_time_ms"],
            error_rate_percent=perf_metrics["error_rate_percent"],
            success_rate_percent=perf_metrics["success_rate_percent"],
            active_connections=len(self.active_requests),
            memory_usage_mb=0.0,  # To be implemented by specific engines
            cpu_usage_percent=0.0,  # To be implemented by specific engines
            plugin_metrics=await self.plugin_registry.get_plugins_health() if self.plugin_registry else {}
        )
    
    async def shutdown(self) -> None:
        """Shutdown engine gracefully."""
        logger.info(f"Shutting down engine: {self.config.name}")
        self.status = EngineStatus.SHUTDOWN
        
        # Wait for active requests to complete (with timeout)
        timeout = 30  # 30 seconds
        start_time = time.time()
        while self.active_requests and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self.active_requests)} active requests to complete...")
            await asyncio.sleep(1)
        
        # Shutdown plugins
        if self.plugin_registry:
            for plugin_name in self.plugin_registry.list_plugins():
                await self.plugin_registry.unregister_plugin(plugin_name)
        
        # Engine-specific shutdown
        await self._engine_shutdown()
        logger.info(f"Engine {self.config.name} shut down successfully")
    
    async def _engine_shutdown(self) -> None:
        """Engine-specific shutdown logic."""
        pass
    
    async def register_plugin(self, plugin: EnginePlugin, config: Dict[str, Any] = None):
        """Register a plugin with this engine."""
        if not self.plugin_registry:
            raise RuntimeError("Plugins are not enabled for this engine")
        await self.plugin_registry.register_plugin(plugin, config)
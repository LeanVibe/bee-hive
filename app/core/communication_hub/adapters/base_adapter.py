"""
Base Protocol Adapter Framework for CommunicationHub

This module defines the abstract base class and common functionality
for all communication protocol adapters.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator
from enum import Enum

from ..protocols import (
    UnifiedMessage, MessageResult, SubscriptionResult, 
    ProtocolType, ConnectionConfig, Priority
)


class AdapterStatus(str, Enum):
    """Adapter connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AdapterMetrics:
    """Performance metrics for protocol adapters."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connection_count: int = 0
    active_subscriptions: int = 0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    last_health_check: Optional[datetime] = None


@dataclass
class ConnectionInfo:
    """Information about a protocol connection."""
    connection_id: str
    status: AdapterStatus
    established_at: datetime
    last_activity: datetime
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MessageHandler:
    """Message handler wrapper for subscription callbacks."""
    
    def __init__(
        self,
        handler_id: str,
        callback: Callable[[UnifiedMessage], asyncio.coroutine],
        pattern: str = "*",
        auto_ack: bool = True
    ):
        self.handler_id = handler_id
        self.callback = callback
        self.pattern = pattern
        self.auto_ack = auto_ack
        self.messages_processed = 0
        self.errors = 0
        self.created_at = datetime.utcnow()
    
    async def handle_message(self, message: UnifiedMessage) -> bool:
        """Handle incoming message and return success status."""
        try:
            await self.callback(message)
            self.messages_processed += 1
            return True
        except Exception as e:
            self.errors += 1
            # Log error but don't raise to prevent handler failures
            # from affecting other handlers
            return False


class BaseProtocolAdapter(ABC):
    """
    Abstract base class for all protocol adapters.
    
    Provides common functionality for connection management, metrics collection,
    health monitoring, and error handling that all adapters can use.
    """
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.protocol = config.protocol
        self.status = AdapterStatus.DISCONNECTED
        self.metrics = AdapterMetrics()
        self.connections: Dict[str, ConnectionInfo] = {}
        self.message_handlers: Dict[str, MessageHandler] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._latency_samples: List[float] = []
        self._start_time = time.time()
        
        # Retry and circuit breaker state
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._circuit_breaker_open = False
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[datetime] = None
    
    # === ABSTRACT METHODS ===
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the protocol endpoint.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close all connections and clean up resources."""
        pass
    
    @abstractmethod
    async def send_message(self, message: UnifiedMessage) -> MessageResult:
        """
        Send a message using this protocol.
        
        Args:
            message: The message to send
            
        Returns:
            MessageResult: Result of the send operation
        """
        pass
    
    @abstractmethod
    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[UnifiedMessage], asyncio.coroutine],
        **kwargs
    ) -> SubscriptionResult:
        """
        Subscribe to messages matching a pattern.
        
        Args:
            pattern: Message pattern to match
            handler: Async callback function for messages
            **kwargs: Protocol-specific options
            
        Returns:
            SubscriptionResult: Result of the subscription
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from a message pattern.
        
        Args:
            subscription_id: ID of subscription to cancel
            
        Returns:
            bool: True if unsubscribed successfully
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Perform protocol-specific health check.
        
        Returns:
            HealthStatus: Current health status
        """
        pass
    
    # === COMMON FUNCTIONALITY ===
    
    async def initialize(self) -> bool:
        """
        Initialize the adapter and establish connections.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.status = AdapterStatus.CONNECTING
            
            # Establish connection
            if not await self.connect():
                self.status = AdapterStatus.ERROR
                return False
            
            self.status = AdapterStatus.CONNECTED
            
            # Start background tasks
            await self._start_background_tasks()
            
            return True
            
        except Exception as e:
            self.status = AdapterStatus.ERROR
            await self._record_error(f"Initialization failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the adapter."""
        self.status = AdapterStatus.SHUTDOWN
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close connections
        await self.disconnect()
        
        # Clear state
        self.connections.clear()
        self.message_handlers.clear()
    
    async def get_metrics(self) -> AdapterMetrics:
        """Get current adapter metrics."""
        # Update uptime
        self.metrics.uptime_seconds = time.time() - self._start_time
        
        # Update error rate
        total_messages = self.metrics.messages_sent + self.metrics.messages_received
        if total_messages > 0:
            self.metrics.error_rate = self.metrics.messages_failed / total_messages
        
        # Update average latency
        if self._latency_samples:
            self.metrics.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)
            # Keep only recent samples
            if len(self._latency_samples) > 1000:
                self._latency_samples = self._latency_samples[-500:]
        
        return self.metrics
    
    async def get_connection_info(self) -> List[ConnectionInfo]:
        """Get information about all connections."""
        return list(self.connections.values())
    
    def is_connected(self) -> bool:
        """Check if adapter is connected and operational."""
        return self.status == AdapterStatus.CONNECTED and not self._circuit_breaker_open
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        # Health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._background_tasks.append(self._health_check_task)
        
        # Retry processor task
        retry_task = asyncio.create_task(self._retry_processor())
        self._background_tasks.append(retry_task)
        
        # Circuit breaker monitor
        cb_task = asyncio.create_task(self._circuit_breaker_monitor())
        self._background_tasks.append(cb_task)
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                health_status = await self.health_check()
                self.metrics.last_health_check = datetime.utcnow()
                
                # Update adapter status based on health
                if health_status == HealthStatus.UNHEALTHY:
                    if self.status == AdapterStatus.CONNECTED:
                        self.status = AdapterStatus.ERROR
                        await self._attempt_reconnection()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._record_error(f"Health check failed: {e}")
    
    async def _retry_processor(self) -> None:
        """Process messages in retry queue."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for message or timeout
                try:
                    message = await asyncio.wait_for(
                        self._retry_queue.get(),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if message can still be retried
                if not message.can_retry():
                    continue
                
                # Attempt retry
                result = await self.send_message(message)
                if not result.success:
                    # Re-queue if more retries available
                    message.delivery_attempts += 1
                    if message.can_retry():
                        await self._retry_queue.put(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._record_error(f"Retry processor error: {e}")
    
    async def _circuit_breaker_monitor(self) -> None:
        """Monitor and manage circuit breaker state."""
        failure_threshold = 5
        recovery_timeout = 60  # seconds
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Check if circuit breaker should open
                if (not self._circuit_breaker_open and 
                    self._circuit_breaker_failures >= failure_threshold):
                    self._circuit_breaker_open = True
                    await self._record_error("Circuit breaker opened due to failures")
                
                # Check if circuit breaker should close
                elif (self._circuit_breaker_open and 
                      self._circuit_breaker_last_failure and
                      datetime.utcnow() - self._circuit_breaker_last_failure > 
                      timedelta(seconds=recovery_timeout)):
                    self._circuit_breaker_open = False
                    self._circuit_breaker_failures = 0
                    await self._record_error("Circuit breaker closed - attempting recovery")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self._record_error(f"Circuit breaker monitor error: {e}")
    
    async def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect after connection failure."""
        if self.status == AdapterStatus.RECONNECTING:
            return False  # Already reconnecting
        
        self.status = AdapterStatus.RECONNECTING
        
        for attempt in range(self.config.retry_attempts):
            try:
                await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff
                
                if await self.connect():
                    self.status = AdapterStatus.CONNECTED
                    self._circuit_breaker_failures = 0
                    return True
                
            except Exception as e:
                await self._record_error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        self.status = AdapterStatus.ERROR
        return False
    
    async def _record_error(self, error_message: str) -> None:
        """Record error and update metrics."""
        self.metrics.messages_failed += 1
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.utcnow()
        
        # Log error (would integrate with logging system)
        print(f"[{self.protocol}] Error: {error_message}")
    
    def _record_latency(self, latency_ms: float) -> None:
        """Record message latency for metrics."""
        self._latency_samples.append(latency_ms)
    
    def _record_message_sent(self, message_size: int = 0) -> None:
        """Record successful message send."""
        self.metrics.messages_sent += 1
        self.metrics.bytes_sent += message_size
    
    def _record_message_received(self, message_size: int = 0) -> None:
        """Record successful message receive."""
        self.metrics.messages_received += 1
        self.metrics.bytes_received += message_size
    
    async def _enqueue_for_retry(self, message: UnifiedMessage) -> None:
        """Add message to retry queue."""
        if message.can_retry():
            await self._retry_queue.put(message)
    
    def _create_connection_info(self, connection_id: str, **metadata) -> ConnectionInfo:
        """Create connection info object."""
        return ConnectionInfo(
            connection_id=connection_id,
            status=AdapterStatus.CONNECTED,
            established_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            metadata=metadata
        )
    
    def _update_connection_activity(self, connection_id: str) -> None:
        """Update last activity time for connection."""
        if connection_id in self.connections:
            self.connections[connection_id].last_activity = datetime.utcnow()


class AdapterRegistry:
    """Registry for managing protocol adapters."""
    
    def __init__(self):
        self._adapters: Dict[ProtocolType, BaseProtocolAdapter] = {}
        self._default_configs: Dict[ProtocolType, ConnectionConfig] = {}
    
    def register_adapter(
        self,
        protocol: ProtocolType,
        adapter: BaseProtocolAdapter
    ) -> None:
        """Register a protocol adapter."""
        self._adapters[protocol] = adapter
    
    def get_adapter(self, protocol: ProtocolType) -> Optional[BaseProtocolAdapter]:
        """Get adapter for protocol."""
        return self._adapters.get(protocol)
    
    def get_available_protocols(self) -> List[ProtocolType]:
        """Get list of available protocols."""
        return list(self._adapters.keys())
    
    async def initialize_all(self) -> Dict[ProtocolType, bool]:
        """Initialize all registered adapters."""
        results = {}
        for protocol, adapter in self._adapters.items():
            results[protocol] = await adapter.initialize()
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all adapters."""
        for adapter in self._adapters.values():
            await adapter.shutdown()
    
    async def get_all_metrics(self) -> Dict[ProtocolType, AdapterMetrics]:
        """Get metrics from all adapters."""
        metrics = {}
        for protocol, adapter in self._adapters.items():
            metrics[protocol] = await adapter.get_metrics()
        return metrics
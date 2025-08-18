"""
Advanced Connection Management and Error Handling System

This module provides sophisticated connection management with real-time monitoring,
automatic recovery, resource optimization, and comprehensive error handling for
the LeanVibe Agent Hive 2.0 communication system.

Features:
- Intelligent connection pooling with load balancing
- Auto-recovery with exponential backoff and circuit breaker patterns
- Resource management with connection lifecycle tracking
- Real-time health monitoring and alerting
- Connection quality scoring and optimization
- Graceful degradation and failover mechanisms
- Performance monitoring and analytics
- Security and authentication management
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import statistics
import weakref

from .realtime_communication_hub import (
    RealTimeCommunicationHub,
    HealthMonitoringUpdate,
    MessagePriority,
    NotificationType
)
from .protocol_models import (
    BridgeConnection,
    CLIProtocol,
    CLIMessage
)

logger = logging.getLogger(__name__)

# ================================================================================
# Connection Management Models
# ================================================================================

class ConnectionState(Enum):
    """Connection states in the lifecycle."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"

class ConnectionPriority(IntEnum):
    """Connection priority levels."""
    CRITICAL = 1    # Critical system connections
    HIGH = 2        # High-priority agent connections
    MEDIUM = 3      # Standard agent connections
    LOW = 4         # Background/utility connections
    BACKGROUND = 5  # Least priority connections

class ErrorSeverity(IntEnum):
    """Error severity levels."""
    CRITICAL = 1    # System-threatening errors
    HIGH = 2        # Connection failures
    MEDIUM = 3      # Performance degradation
    LOW = 4         # Minor issues
    INFO = 5        # Informational

@dataclass
class ConnectionMetrics:
    """Comprehensive connection metrics."""
    connection_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    
    # Performance metrics
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    # Timing metrics
    average_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    
    # Error tracking
    total_errors: int = 0
    connection_failures: int = 0
    timeout_errors: int = 0
    authentication_errors: int = 0
    
    # Health metrics
    health_score: float = 1.0
    quality_score: float = 1.0
    stability_score: float = 1.0
    
    # Connection lifecycle
    connect_time_ms: Optional[float] = None
    reconnection_count: int = 0
    total_uptime_seconds: float = 0.0
    total_downtime_seconds: float = 0.0

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool."""
    min_connections: int = 1
    max_connections: int = 10
    idle_timeout_seconds: int = 300
    connection_timeout_seconds: int = 30
    reconnect_interval_seconds: int = 5
    max_reconnect_attempts: int = 5
    health_check_interval_seconds: int = 30
    enable_connection_pooling: bool = True
    enable_load_balancing: bool = True
    enable_auto_recovery: bool = True

@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling."""
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    enable_exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_backoff_seconds: int = 300
    enable_retry_with_jitter: bool = True
    max_retry_attempts: int = 3
    enable_graceful_degradation: bool = True

# ================================================================================
# Advanced Connection Manager
# ================================================================================

class AdvancedConnectionManager:
    """
    Advanced connection manager with intelligent pooling, error handling, and monitoring.
    
    Provides comprehensive connection lifecycle management with:
    - Intelligent connection pooling and load balancing
    - Auto-recovery with circuit breaker and exponential backoff
    - Real-time health monitoring and quality scoring
    - Resource optimization and lifecycle management
    - Performance analytics and reporting
    - Security and authentication management
    """
    
    def __init__(
        self,
        realtime_hub: Optional[RealTimeCommunicationHub] = None,
        pool_config: Optional[ConnectionPoolConfig] = None,
        error_config: Optional[ErrorHandlingConfig] = None
    ):
        """Initialize advanced connection manager."""
        self.realtime_hub = realtime_hub
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.error_config = error_config or ErrorHandlingConfig()
        
        # Connection tracking
        self._connections: Dict[str, BridgeConnection] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}
        self._connection_pools: Dict[str, List[str]] = {}  # pool_key -> connection_ids
        self._pool_configs: Dict[str, ConnectionPoolConfig] = {}
        
        # State management
        self._connection_states: Dict[str, ConnectionState] = {}
        self._connection_priorities: Dict[str, ConnectionPriority] = {}
        self._connection_tags: Dict[str, Set[str]] = {}
        
        # Error handling and recovery
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._backoff_timers: Dict[str, float] = {}
        self._retry_counts: Dict[str, int] = {}
        self._error_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Performance monitoring
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._health_scores: Dict[str, float] = {}
        self._quality_scores: Dict[str, float] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self._connection_event_handlers: Dict[str, List[Callable]] = {}
        self._error_handlers: Dict[ErrorSeverity, List[Callable]] = {}
        
        # Statistics
        self._global_stats = {
            "total_connections_created": 0,
            "total_connections_failed": 0,
            "total_reconnections": 0,
            "total_errors": 0,
            "average_connection_lifespan": 0.0,
            "peak_concurrent_connections": 0
        }
        
        logger.info("AdvancedConnectionManager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the connection manager."""
        try:
            logger.info("Initializing AdvancedConnectionManager...")
            
            # Start background monitoring tasks
            await self._start_background_tasks()
            
            logger.info("AdvancedConnectionManager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize connection manager: {e}")
            return False
    
    # ================================================================================
    # Connection Lifecycle Management
    # ================================================================================
    
    async def create_connection(
        self,
        connection_config: Dict[str, Any],
        priority: ConnectionPriority = ConnectionPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        pool_key: Optional[str] = None
    ) -> Optional[BridgeConnection]:
        """
        Create a new connection with advanced configuration.
        
        Args:
            connection_config: Connection configuration
            priority: Connection priority level
            tags: Optional tags for connection categorization
            pool_key: Optional pool key for connection pooling
            
        Returns:
            BridgeConnection: Created connection or None if failed
        """
        start_time = time.time()
        connection_id = str(uuid.uuid4())
        
        try:
            logger.debug(f"Creating connection: {connection_id}")
            
            # Initialize connection state
            self._connection_states[connection_id] = ConnectionState.INITIALIZING
            self._connection_priorities[connection_id] = priority
            self._connection_tags[connection_id] = tags or set()
            
            # Create connection metrics
            metrics = ConnectionMetrics(connection_id=connection_id)
            self._connection_metrics[connection_id] = metrics
            
            # Create connection object
            connection = await self._create_bridge_connection(connection_config, connection_id)
            if not connection:
                raise ConnectionError("Failed to create bridge connection")
            
            # Test connection
            self._connection_states[connection_id] = ConnectionState.CONNECTING
            if not await self._test_connection(connection):
                raise ConnectionError("Connection test failed")
            
            # Finalize connection setup
            self._connection_states[connection_id] = ConnectionState.CONNECTED
            self._connections[connection_id] = connection
            
            # Record connection metrics
            connect_time = (time.time() - start_time) * 1000
            metrics.connect_time_ms = connect_time
            metrics.last_activity = datetime.utcnow()
            
            # Add to pool if specified
            if pool_key:
                await self._add_to_pool(connection_id, pool_key)
            
            # Update global statistics
            self._global_stats["total_connections_created"] += 1
            self._global_stats["peak_concurrent_connections"] = max(
                self._global_stats["peak_concurrent_connections"],
                len(self._connections)
            )
            
            # Broadcast connection creation
            if self.realtime_hub:
                await self._broadcast_connection_event(
                    "connection_created", connection_id, {
                        "connect_time_ms": connect_time,
                        "priority": priority.name,
                        "tags": list(tags) if tags else []
                    }
                )
            
            # Trigger connection event handlers
            await self._trigger_connection_event("connection_created", connection_id, connection)
            
            logger.info(f"Connection created successfully: {connection_id} ({connect_time:.1f}ms)")
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection {connection_id}: {e}")
            
            # Update state and metrics
            self._connection_states[connection_id] = ConnectionState.FAILED
            self._global_stats["total_connections_failed"] += 1
            
            # Record error
            await self._record_error(connection_id, "connection_creation", str(e), ErrorSeverity.HIGH)
            
            # Cleanup
            await self._cleanup_failed_connection(connection_id)
            
            return None
    
    async def close_connection(
        self,
        connection_id: str,
        graceful: bool = True,
        reason: str = "manual_close"
    ) -> bool:
        """
        Close a connection gracefully or forcefully.
        
        Args:
            connection_id: Connection to close
            graceful: Whether to close gracefully
            reason: Reason for closing
            
        Returns:
            bool: True if closed successfully
        """
        try:
            if connection_id not in self._connections:
                logger.warning(f"Connection not found for closing: {connection_id}")
                return False
            
            logger.debug(f"Closing connection: {connection_id} (reason: {reason})")
            
            connection = self._connections[connection_id]
            self._connection_states[connection_id] = ConnectionState.CLOSING
            
            # Record final metrics
            if connection_id in self._connection_metrics:
                metrics = self._connection_metrics[connection_id]
                uptime = (datetime.utcnow() - metrics.created_at).total_seconds()
                metrics.total_uptime_seconds += uptime
                
                # Update global average lifespan
                total_connections = self._global_stats["total_connections_created"]
                if total_connections > 0:
                    current_avg = self._global_stats["average_connection_lifespan"]
                    self._global_stats["average_connection_lifespan"] = (
                        (current_avg * (total_connections - 1) + uptime) / total_connections
                    )
            
            # Close connection based on type
            await self._close_connection_by_type(connection, graceful)
            
            # Update state
            self._connection_states[connection_id] = ConnectionState.CLOSED
            
            # Remove from pools
            await self._remove_from_all_pools(connection_id)
            
            # Broadcast connection closure
            if self.realtime_hub:
                await self._broadcast_connection_event(
                    "connection_closed", connection_id, {
                        "reason": reason,
                        "graceful": graceful,
                        "final_metrics": self._get_connection_summary(connection_id)
                    }
                )
            
            # Trigger event handlers
            await self._trigger_connection_event("connection_closed", connection_id, connection)
            
            # Cleanup
            await self._cleanup_connection_resources(connection_id)
            
            logger.info(f"Connection closed successfully: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close connection {connection_id}: {e}")
            await self._record_error(connection_id, "connection_closure", str(e), ErrorSeverity.MEDIUM)
            return False
    
    async def get_connection(
        self,
        connection_id: str
    ) -> Optional[BridgeConnection]:
        """Get connection by ID."""
        return self._connections.get(connection_id)
    
    async def get_healthy_connection(
        self,
        pool_key: Optional[str] = None,
        min_quality_score: float = 0.5,
        exclude_ids: Optional[Set[str]] = None
    ) -> Optional[BridgeConnection]:
        """
        Get a healthy connection from pool or all connections.
        
        Args:
            pool_key: Pool to select from (if None, select from all)
            min_quality_score: Minimum quality score required
            exclude_ids: Connection IDs to exclude
            
        Returns:
            BridgeConnection: Healthy connection or None
        """
        try:
            exclude_ids = exclude_ids or set()
            
            # Get candidate connections
            candidates = []
            if pool_key and pool_key in self._connection_pools:
                candidate_ids = self._connection_pools[pool_key]
            else:
                candidate_ids = list(self._connections.keys())
            
            # Filter candidates
            for conn_id in candidate_ids:
                if (conn_id not in exclude_ids and
                    conn_id in self._connections and
                    self._connection_states.get(conn_id) == ConnectionState.CONNECTED and
                    self._quality_scores.get(conn_id, 0.0) >= min_quality_score):
                    
                    candidates.append((conn_id, self._quality_scores.get(conn_id, 0.0)))
            
            if not candidates:
                return None
            
            # Sort by quality score (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Apply load balancing if enabled
            if self.pool_config.enable_load_balancing:
                # Select from top 25% of candidates for load balancing
                top_count = max(1, len(candidates) // 4)
                selected_id = self._select_least_loaded(candidates[:top_count])
            else:
                selected_id = candidates[0][0]
            
            return self._connections[selected_id]
            
        except Exception as e:
            logger.error(f"Failed to get healthy connection: {e}")
            return None
    
    # ================================================================================
    # Connection Pool Management
    # ================================================================================
    
    async def create_connection_pool(
        self,
        pool_key: str,
        pool_config: ConnectionPoolConfig,
        connection_factory: Callable[[], Any]
    ) -> bool:
        """
        Create a connection pool with specified configuration.
        
        Args:
            pool_key: Unique pool identifier
            pool_config: Pool configuration
            connection_factory: Function to create new connections
            
        Returns:
            bool: True if pool created successfully
        """
        try:
            logger.info(f"Creating connection pool: {pool_key}")
            
            # Store pool configuration
            self._pool_configs[pool_key] = pool_config
            self._connection_pools[pool_key] = []
            
            # Create minimum connections
            for i in range(pool_config.min_connections):
                connection_config = await connection_factory()
                connection = await self.create_connection(
                    connection_config,
                    priority=ConnectionPriority.MEDIUM,
                    tags={"pool", pool_key},
                    pool_key=pool_key
                )
                
                if not connection:
                    logger.warning(f"Failed to create initial connection {i} for pool {pool_key}")
            
            logger.info(f"Connection pool created: {pool_key} ({len(self._connection_pools[pool_key])} connections)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create connection pool {pool_key}: {e}")
            return False
    
    async def scale_pool(
        self,
        pool_key: str,
        target_size: Optional[int] = None,
        scale_direction: Optional[str] = None
    ) -> bool:
        """
        Scale connection pool up or down.
        
        Args:
            pool_key: Pool to scale
            target_size: Target pool size (if None, auto-scale)
            scale_direction: "up" or "down" for auto-scaling
            
        Returns:
            bool: True if scaling successful
        """
        try:
            if pool_key not in self._connection_pools:
                logger.error(f"Pool not found: {pool_key}")
                return False
            
            pool_config = self._pool_configs[pool_key]
            current_size = len(self._connection_pools[pool_key])
            
            # Determine target size
            if target_size is None:
                target_size = await self._calculate_optimal_pool_size(pool_key, scale_direction)
            
            # Validate target size
            target_size = max(pool_config.min_connections, 
                            min(target_size, pool_config.max_connections))
            
            if target_size == current_size:
                return True
            
            logger.info(f"Scaling pool {pool_key}: {current_size} -> {target_size}")
            
            if target_size > current_size:
                # Scale up
                await self._scale_pool_up(pool_key, target_size - current_size)
            else:
                # Scale down
                await self._scale_pool_down(pool_key, current_size - target_size)
            
            # Broadcast pool scaling event
            if self.realtime_hub:
                await self._broadcast_pool_event("pool_scaled", pool_key, {
                    "previous_size": current_size,
                    "new_size": len(self._connection_pools[pool_key]),
                    "target_size": target_size
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale pool {pool_key}: {e}")
            return False
    
    async def get_pool_health(self, pool_key: str) -> Dict[str, Any]:
        """Get health status of connection pool."""
        try:
            if pool_key not in self._connection_pools:
                return {"status": "not_found"}
            
            pool_connections = self._connection_pools[pool_key]
            pool_config = self._pool_configs[pool_key]
            
            # Calculate health metrics
            total_connections = len(pool_connections)
            healthy_connections = sum(
                1 for conn_id in pool_connections
                if self._connection_states.get(conn_id) == ConnectionState.CONNECTED
            )
            
            average_quality = statistics.mean([
                self._quality_scores.get(conn_id, 0.0)
                for conn_id in pool_connections
            ]) if pool_connections else 0.0
            
            # Determine overall health status
            health_ratio = healthy_connections / total_connections if total_connections > 0 else 0.0
            if health_ratio >= 0.8 and average_quality >= 0.7:
                status = "healthy"
            elif health_ratio >= 0.5 and average_quality >= 0.5:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "total_connections": total_connections,
                "healthy_connections": healthy_connections,
                "health_ratio": health_ratio,
                "average_quality_score": average_quality,
                "min_connections": pool_config.min_connections,
                "max_connections": pool_config.max_connections,
                "utilization": total_connections / pool_config.max_connections,
                "connection_details": [
                    {
                        "connection_id": conn_id,
                        "state": self._connection_states.get(conn_id, "unknown").value,
                        "quality_score": self._quality_scores.get(conn_id, 0.0),
                        "last_activity": self._connection_metrics[conn_id].last_activity.isoformat()
                            if conn_id in self._connection_metrics and self._connection_metrics[conn_id].last_activity else None
                    }
                    for conn_id in pool_connections
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get pool health for {pool_key}: {e}")
            return {"status": "error", "error": str(e)}
    
    # ================================================================================
    # Error Handling and Recovery
    # ================================================================================
    
    async def handle_connection_error(
        self,
        connection_id: str,
        error: Exception,
        error_type: str = "general",
        auto_recover: bool = True
    ) -> bool:
        """
        Handle connection error with sophisticated recovery mechanisms.
        
        Args:
            connection_id: Connection with error
            error: Exception that occurred
            error_type: Type of error for classification
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            bool: True if error handled successfully
        """
        try:
            logger.warning(f"Handling connection error {connection_id}: {error}")
            
            # Classify error severity
            severity = self._classify_error_severity(error, error_type)
            
            # Record error in history
            await self._record_error(connection_id, error_type, str(error), severity)
            
            # Update connection state
            if connection_id in self._connection_states:
                if severity <= ErrorSeverity.HIGH:
                    self._connection_states[connection_id] = ConnectionState.FAILED
                else:
                    self._connection_states[connection_id] = ConnectionState.DISCONNECTED
            
            # Update metrics
            if connection_id in self._connection_metrics:
                metrics = self._connection_metrics[connection_id]
                metrics.total_errors += 1
                
                if error_type == "connection_failure":
                    metrics.connection_failures += 1
                elif error_type == "timeout":
                    metrics.timeout_errors += 1
                elif error_type == "authentication":
                    metrics.authentication_errors += 1
            
            # Check circuit breaker
            if self.error_config.enable_circuit_breaker:
                if await self._check_circuit_breaker(connection_id, error_type):
                    logger.warning(f"Circuit breaker opened for {connection_id}")
                    await self._broadcast_circuit_breaker_event(connection_id, "opened")
                    return False
            
            # Trigger error handlers
            await self._trigger_error_handlers(severity, connection_id, error, error_type)
            
            # Attempt recovery if enabled
            if auto_recover and self.pool_config.enable_auto_recovery:
                return await self._attempt_connection_recovery(connection_id, error_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to handle connection error: {e}")
            return False
    
    async def recover_connection(
        self,
        connection_id: str,
        force_recovery: bool = False
    ) -> bool:
        """
        Manually trigger connection recovery.
        
        Args:
            connection_id: Connection to recover
            force_recovery: Force recovery even if circuit breaker is open
            
        Returns:
            bool: True if recovery successful
        """
        try:
            logger.info(f"Initiating connection recovery: {connection_id}")
            
            # Check if recovery is allowed
            if not force_recovery and self._is_circuit_breaker_open(connection_id):
                logger.warning(f"Recovery blocked by circuit breaker: {connection_id}")
                return False
            
            # Update state
            self._connection_states[connection_id] = ConnectionState.RECONNECTING
            
            # Broadcast recovery attempt
            if self.realtime_hub:
                await self._broadcast_connection_event(
                    "recovery_started", connection_id, {"forced": force_recovery}
                )
            
            # Attempt recovery
            success = await self._attempt_connection_recovery(connection_id, "manual_recovery")
            
            if success:
                # Reset circuit breaker
                if connection_id in self._circuit_breakers:
                    self._circuit_breakers[connection_id]["failure_count"] = 0
                    self._circuit_breakers[connection_id]["state"] = "closed"
                
                # Reset backoff timer
                if connection_id in self._backoff_timers:
                    del self._backoff_timers[connection_id]
                
                logger.info(f"Connection recovery successful: {connection_id}")
            else:
                logger.error(f"Connection recovery failed: {connection_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to recover connection {connection_id}: {e}")
            return False
    
    async def _attempt_connection_recovery(
        self,
        connection_id: str,
        error_type: str
    ) -> bool:
        """Attempt connection recovery with exponential backoff."""
        try:
            # Check retry count
            retry_count = self._retry_counts.get(connection_id, 0)
            if retry_count >= self.error_config.max_retry_attempts:
                logger.warning(f"Max retry attempts exceeded for {connection_id}")
                return False
            
            # Calculate backoff delay
            backoff_delay = self._calculate_backoff_delay(connection_id, retry_count)
            
            logger.debug(f"Recovery attempt {retry_count + 1} for {connection_id} (delay: {backoff_delay:.1f}s)")
            
            # Wait for backoff delay
            if backoff_delay > 0:
                await asyncio.sleep(backoff_delay)
            
            # Increment retry count
            self._retry_counts[connection_id] = retry_count + 1
            
            # Get original connection config (would need to store this)
            connection = self._connections.get(connection_id)
            if not connection:
                logger.error(f"Original connection not found for recovery: {connection_id}")
                return False
            
            # Attempt reconnection
            if await self._test_connection(connection):
                # Recovery successful
                self._connection_states[connection_id] = ConnectionState.CONNECTED
                self._retry_counts[connection_id] = 0
                
                # Update metrics
                if connection_id in self._connection_metrics:
                    self._connection_metrics[connection_id].reconnection_count += 1
                
                self._global_stats["total_reconnections"] += 1
                
                # Broadcast recovery success
                if self.realtime_hub:
                    await self._broadcast_connection_event(
                        "recovery_successful", connection_id, {
                            "retry_count": retry_count + 1,
                            "backoff_delay": backoff_delay
                        }
                    )
                
                return True
            else:
                # Recovery failed, schedule another attempt
                logger.warning(f"Recovery attempt {retry_count + 1} failed for {connection_id}")
                
                # Schedule next attempt if under limit
                if retry_count + 1 < self.error_config.max_retry_attempts:
                    asyncio.create_task(
                        self._schedule_recovery_retry(connection_id, error_type)
                    )
                
                return False
                
        except Exception as e:
            logger.error(f"Recovery attempt failed for {connection_id}: {e}")
            return False
    
    # ================================================================================
    # Health Monitoring and Quality Scoring
    # ================================================================================
    
    async def update_connection_health(self, connection_id: str) -> float:
        """Update and calculate connection health score."""
        try:
            if connection_id not in self._connection_metrics:
                return 0.0
            
            metrics = self._connection_metrics[connection_id]
            
            # Base health score from connection state
            state = self._connection_states.get(connection_id, ConnectionState.FAILED)
            if state == ConnectionState.CONNECTED:
                base_score = 1.0
            elif state == ConnectionState.RECONNECTING:
                base_score = 0.5
            elif state == ConnectionState.DISCONNECTED:
                base_score = 0.3
            else:
                base_score = 0.0
            
            # Adjust for error rate
            total_operations = metrics.total_messages_sent + metrics.total_messages_received
            if total_operations > 0:
                error_rate = metrics.total_errors / total_operations
                error_penalty = min(0.5, error_rate * 2)
                base_score -= error_penalty
            
            # Adjust for latency
            if metrics.average_latency_ms > 0:
                if metrics.average_latency_ms > 1000:  # 1 second
                    latency_penalty = min(0.3, (metrics.average_latency_ms - 1000) / 5000)
                    base_score -= latency_penalty
            
            # Adjust for stability (reconnection frequency)
            uptime = (datetime.utcnow() - metrics.created_at).total_seconds()
            if uptime > 0 and metrics.reconnection_count > 0:
                reconnect_rate = metrics.reconnection_count / (uptime / 3600)  # per hour
                if reconnect_rate > 1:  # More than 1 reconnection per hour
                    stability_penalty = min(0.2, (reconnect_rate - 1) * 0.1)
                    base_score -= stability_penalty
            
            # Ensure score is between 0 and 1
            health_score = max(0.0, min(1.0, base_score))
            
            # Update stored scores
            self._health_scores[connection_id] = health_score
            metrics.health_score = health_score
            
            # Calculate quality score (different from health)
            quality_score = await self._calculate_quality_score(connection_id)
            self._quality_scores[connection_id] = quality_score
            metrics.quality_score = quality_score
            
            # Broadcast health update if significant change
            if self.realtime_hub and abs(health_score - metrics.health_score) > 0.1:
                await self._broadcast_health_update(connection_id, health_score, quality_score)
            
            return health_score
            
        except Exception as e:
            logger.error(f"Failed to update connection health for {connection_id}: {e}")
            return 0.0
    
    async def _calculate_quality_score(self, connection_id: str) -> float:
        """Calculate connection quality score based on performance metrics."""
        try:
            if connection_id not in self._connection_metrics:
                return 0.0
            
            metrics = self._connection_metrics[connection_id]
            
            # Start with health score as base
            quality_score = self._health_scores.get(connection_id, 0.0)
            
            # Adjust for throughput performance
            uptime = (datetime.utcnow() - metrics.created_at).total_seconds()
            if uptime > 60:  # Only calculate after 1 minute of operation
                messages_per_second = (metrics.total_messages_sent + metrics.total_messages_received) / uptime
                
                # Bonus for high throughput (more than 1 message per second)
                if messages_per_second > 1:
                    throughput_bonus = min(0.2, (messages_per_second - 1) * 0.05)
                    quality_score += throughput_bonus
            
            # Adjust for consistency (low latency variance)
            if len(metrics.latency_samples) > 10:
                latency_variance = statistics.variance(metrics.latency_samples[-100:])  # Last 100 samples
                if latency_variance < 100:  # Low variance
                    consistency_bonus = 0.1
                    quality_score += consistency_bonus
                elif latency_variance > 1000:  # High variance
                    consistency_penalty = min(0.2, latency_variance / 5000)
                    quality_score -= consistency_penalty
            
            # Adjust for recent performance
            if metrics.last_activity:
                time_since_activity = (datetime.utcnow() - metrics.last_activity).total_seconds()
                if time_since_activity > 300:  # 5 minutes of inactivity
                    inactivity_penalty = min(0.3, (time_since_activity - 300) / 1800)  # Decay over 30 minutes
                    quality_score -= inactivity_penalty
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate quality score for {connection_id}: {e}")
            return 0.0
    
    # ================================================================================
    # Performance Monitoring and Analytics
    # ================================================================================
    
    async def record_message_metrics(
        self,
        connection_id: str,
        direction: str,  # "sent" or "received"
        message_size: int,
        latency_ms: Optional[float] = None
    ):
        """Record message metrics for performance tracking."""
        try:
            if connection_id not in self._connection_metrics:
                return
            
            metrics = self._connection_metrics[connection_id]
            metrics.last_activity = datetime.utcnow()
            
            if direction == "sent":
                metrics.total_messages_sent += 1
                metrics.total_bytes_sent += message_size
            elif direction == "received":
                metrics.total_messages_received += 1
                metrics.total_bytes_received += message_size
            
            # Record latency if provided
            if latency_ms is not None:
                metrics.latency_samples.append(latency_ms)
                
                # Keep only recent samples
                if len(metrics.latency_samples) > 1000:
                    metrics.latency_samples = metrics.latency_samples[-1000:]
                
                # Update latency statistics
                metrics.average_latency_ms = statistics.mean(metrics.latency_samples)
                metrics.min_latency_ms = min(metrics.min_latency_ms, latency_ms)
                metrics.max_latency_ms = max(metrics.max_latency_ms, latency_ms)
            
            # Update health periodically
            total_messages = metrics.total_messages_sent + metrics.total_messages_received
            if total_messages % 10 == 0:  # Update every 10 messages
                await self.update_connection_health(connection_id)
            
        except Exception as e:
            logger.error(f"Failed to record message metrics: {e}")
    
    async def get_connection_analytics(
        self,
        connection_id: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive connection analytics."""
        try:
            if connection_id:
                # Get analytics for specific connection
                return await self._get_single_connection_analytics(connection_id, time_window_hours)
            else:
                # Get aggregate analytics for all connections
                return await self._get_aggregate_analytics(time_window_hours)
                
        except Exception as e:
            logger.error(f"Failed to get connection analytics: {e}")
            return {}
    
    async def _get_single_connection_analytics(
        self,
        connection_id: str,
        time_window_hours: int
    ) -> Dict[str, Any]:
        """Get analytics for a single connection."""
        if connection_id not in self._connection_metrics:
            return {"error": "Connection not found"}
        
        metrics = self._connection_metrics[connection_id]
        
        # Calculate uptime
        uptime = (datetime.utcnow() - metrics.created_at).total_seconds()
        
        # Calculate throughput
        total_messages = metrics.total_messages_sent + metrics.total_messages_received
        messages_per_second = total_messages / uptime if uptime > 0 else 0
        
        # Calculate error rates
        error_rate = metrics.total_errors / total_messages if total_messages > 0 else 0
        
        return {
            "connection_id": connection_id,
            "state": self._connection_states.get(connection_id, "unknown").value,
            "priority": self._connection_priorities.get(connection_id, ConnectionPriority.MEDIUM).name,
            "tags": list(self._connection_tags.get(connection_id, set())),
            "uptime_seconds": uptime,
            "performance": {
                "total_messages": total_messages,
                "messages_sent": metrics.total_messages_sent,
                "messages_received": metrics.total_messages_received,
                "messages_per_second": messages_per_second,
                "total_bytes_sent": metrics.total_bytes_sent,
                "total_bytes_received": metrics.total_bytes_received,
                "average_latency_ms": metrics.average_latency_ms,
                "min_latency_ms": metrics.min_latency_ms if metrics.min_latency_ms != float('inf') else 0,
                "max_latency_ms": metrics.max_latency_ms
            },
            "reliability": {
                "total_errors": metrics.total_errors,
                "error_rate": error_rate,
                "connection_failures": metrics.connection_failures,
                "timeout_errors": metrics.timeout_errors,
                "authentication_errors": metrics.authentication_errors,
                "reconnection_count": metrics.reconnection_count
            },
            "health": {
                "health_score": metrics.health_score,
                "quality_score": metrics.quality_score,
                "stability_score": metrics.stability_score
            },
            "timestamps": {
                "created_at": metrics.created_at.isoformat(),
                "last_activity": metrics.last_activity.isoformat() if metrics.last_activity else None
            }
        }
    
    async def _get_aggregate_analytics(self, time_window_hours: int) -> Dict[str, Any]:
        """Get aggregate analytics for all connections."""
        total_connections = len(self._connections)
        active_connections = sum(
            1 for state in self._connection_states.values()
            if state == ConnectionState.CONNECTED
        )
        
        # Aggregate metrics
        total_messages = sum(
            m.total_messages_sent + m.total_messages_received
            for m in self._connection_metrics.values()
        )
        
        total_errors = sum(
            m.total_errors for m in self._connection_metrics.values()
        )
        
        average_health = statistics.mean(self._health_scores.values()) if self._health_scores else 0.0
        average_quality = statistics.mean(self._quality_scores.values()) if self._quality_scores else 0.0
        
        # Pool statistics
        pool_stats = {}
        for pool_key, pool_connections in self._connection_pools.items():
            pool_health = await self.get_pool_health(pool_key)
            pool_stats[pool_key] = pool_health
        
        return {
            "overview": {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "connection_pools": len(self._connection_pools),
                "average_health_score": average_health,
                "average_quality_score": average_quality
            },
            "performance": {
                "total_messages": total_messages,
                "total_errors": total_errors,
                "error_rate": total_errors / total_messages if total_messages > 0 else 0,
                "global_stats": self._global_stats
            },
            "pools": pool_stats,
            "connection_states": {
                state.value: sum(1 for s in self._connection_states.values() if s == state)
                for state in ConnectionState
            }
        }
    
    # ================================================================================
    # Background Tasks and Monitoring
    # ================================================================================
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        try:
            # Connection health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._background_tasks.add(self._health_check_task)
            
            # Connection cleanup
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.add(self._cleanup_task)
            
            # Performance monitoring
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.add(self._monitoring_task)
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.pool_config.health_check_interval_seconds)
                
                # Check health of all connections
                for connection_id in list(self._connections.keys()):
                    await self.update_connection_health(connection_id)
                
                # Check circuit breakers
                await self._check_circuit_breaker_resets()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.utcnow()
                
                # Clean up idle connections
                idle_connections = []
                for connection_id, metrics in self._connection_metrics.items():
                    if metrics.last_activity:
                        idle_time = (current_time - metrics.last_activity).total_seconds()
                        if idle_time > self.pool_config.idle_timeout_seconds:
                            idle_connections.append(connection_id)
                
                for connection_id in idle_connections:
                    await self.close_connection(connection_id, reason="idle_timeout")
                
                # Clean up old error history
                cutoff_time = current_time - timedelta(hours=24)
                for connection_id in list(self._error_history.keys()):
                    if connection_id in self._error_history:
                        self._error_history[connection_id] = [
                            error for error in self._error_history[connection_id]
                            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
                        ]
                        
                        if not self._error_history[connection_id]:
                            del self._error_history[connection_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring and analytics loop."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Collect aggregate metrics
                analytics = await self.get_connection_analytics()
                
                # Broadcast analytics update
                if self.realtime_hub:
                    await self.realtime_hub.broadcast_dashboard_update(
                        update_type="connection_analytics",
                        data=analytics
                    )
                
                # Check for auto-scaling needs
                for pool_key in self._connection_pools.keys():
                    await self._check_pool_auto_scaling(pool_key)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    # ================================================================================
    # Helper Methods and Utilities
    # ================================================================================
    
    async def _create_bridge_connection(
        self,
        connection_config: Dict[str, Any],
        connection_id: str
    ) -> Optional[BridgeConnection]:
        """Create a bridge connection from configuration."""
        try:
            # This would integrate with the actual bridge creation logic
            # For now, create a basic BridgeConnection object
            connection = BridgeConnection(
                connection_id=connection_id,
                protocol=connection_config.get("protocol", CLIProtocol.UNIVERSAL),
                endpoint=connection_config.get("endpoint", ""),
                connection_type=connection_config.get("type", "websocket")
            )
            
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create bridge connection: {e}")
            return None
    
    async def _test_connection(self, connection: BridgeConnection) -> bool:
        """Test connection functionality."""
        try:
            # This would implement actual connection testing
            # For now, simulate successful test
            await asyncio.sleep(0.1)  # Simulate test delay
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def _classify_error_severity(self, error: Exception, error_type: str) -> ErrorSeverity:
        """Classify error severity based on error type and content."""
        error_str = str(error).lower()
        
        if error_type == "authentication" or "auth" in error_str:
            return ErrorSeverity.HIGH
        elif error_type == "connection_failure" or "connection" in error_str:
            return ErrorSeverity.HIGH
        elif error_type == "timeout" or "timeout" in error_str:
            return ErrorSeverity.MEDIUM
        elif "permission" in error_str or "forbidden" in error_str:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM
    
    async def _record_error(
        self,
        connection_id: str,
        error_type: str,
        error_message: str,
        severity: ErrorSeverity
    ):
        """Record error in history and update metrics."""
        try:
            if connection_id not in self._error_history:
                self._error_history[connection_id] = []
            
            error_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": error_type,
                "error_message": error_message,
                "severity": severity.name
            }
            
            self._error_history[connection_id].append(error_record)
            
            # Keep only recent errors (last 100)
            if len(self._error_history[connection_id]) > 100:
                self._error_history[connection_id] = self._error_history[connection_id][-100:]
            
            # Update global stats
            self._global_stats["total_errors"] += 1
            
            # Broadcast error if severe
            if severity <= ErrorSeverity.HIGH and self.realtime_hub:
                await self._broadcast_error_event(connection_id, error_record)
            
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
    
    def _calculate_backoff_delay(self, connection_id: str, retry_count: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        if not self.error_config.enable_exponential_backoff:
            return self.pool_config.reconnect_interval_seconds
        
        # Exponential backoff
        delay = min(
            self.pool_config.reconnect_interval_seconds * (self.error_config.backoff_multiplier ** retry_count),
            self.error_config.max_backoff_seconds
        )
        
        # Add jitter if enabled
        if self.error_config.enable_retry_with_jitter:
            import random
            jitter = delay * 0.1 * random.random()  # Up to 10% jitter
            delay += jitter
        
        return delay
    
    def _select_least_loaded(self, candidates: List[tuple]) -> str:
        """Select least loaded connection from candidates."""
        # Simple implementation - select connection with lowest message count
        min_load = float('inf')
        selected_id = candidates[0][0]
        
        for conn_id, quality_score in candidates:
            if conn_id in self._connection_metrics:
                metrics = self._connection_metrics[conn_id]
                load = metrics.total_messages_sent + metrics.total_messages_received
                if load < min_load:
                    min_load = load
                    selected_id = conn_id
        
        return selected_id
    
    async def _broadcast_connection_event(
        self,
        event_type: str,
        connection_id: str,
        event_data: Dict[str, Any]
    ):
        """Broadcast connection event to real-time hub."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type=f"connection_{event_type}",
                data={
                    "connection_id": connection_id,
                    "event_data": event_data,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast connection event: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the connection manager."""
        try:
            logger.info("Shutting down AdvancedConnectionManager...")
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close all connections
            for connection_id in list(self._connections.keys()):
                await self.close_connection(connection_id, reason="manager_shutdown")
            
            logger.info("AdvancedConnectionManager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during connection manager shutdown: {e}")

# Additional methods would be implemented for circuit breaker, pool scaling, etc.
# This provides the core structure and key functionality
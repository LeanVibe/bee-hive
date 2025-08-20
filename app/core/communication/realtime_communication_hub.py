"""
Real-Time Communication Hub for LeanVibe Agent Hive 2.0

This module extends the existing communication infrastructure with comprehensive
real-time capabilities for agent coordination, status broadcasting, and client updates.

Features:
- Real-time agent status broadcasting via Redis pub/sub
- Task execution notifications and updates
- Agent health monitoring with live dashboard updates
- WebSocket client communication for real-time UI updates
- Message routing and delivery guarantees
- Connection pooling and resource management
- Comprehensive monitoring and analytics

Integration Points:
- Builds upon existing redis_websocket_bridge.py
- Integrates with multi_cli_protocol.py
- Enhances communication_bridge.py capabilities
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .redis_websocket_bridge import (
    RedisConfig,
    WebSocketConfig,
    RedisMessageBroker,
    WebSocketMessageBridge,
    UnifiedCommunicationBridge
)
from .protocol_models import (
    BridgeConnection,
    CLIProtocol,
    UniversalMessage,
    CLIMessage
)

logger = logging.getLogger(__name__)

# ================================================================================
# Real-Time Communication Models
# ================================================================================

class MessagePriority(IntEnum):
    """Message priority levels for real-time communication."""
    CRITICAL = 1    # System alerts, failures
    HIGH = 2       # Task completions, errors
    MEDIUM = 3     # Status updates, progress
    LOW = 4        # Metrics, diagnostics
    DEBUG = 5      # Debug information

class NotificationType(Enum):
    """Types of real-time notifications."""
    AGENT_STATUS = "agent_status"
    TASK_EXECUTION = "task_execution"
    HEALTH_MONITORING = "health_monitoring"
    SYSTEM_ALERT = "system_alert"
    DASHBOARD_UPDATE = "dashboard_update"
    COORDINATION_EVENT = "coordination_event"

class AgentStatus(Enum):
    """Agent status states."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TaskExecutionStatus(Enum):
    """Task execution status states."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class RealTimeMessage:
    """Real-time message structure for agent communication."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notification_type: NotificationType = NotificationType.DASHBOARD_UPDATE
    priority: MessagePriority = MessagePriority.MEDIUM
    source_agent_id: str = ""
    target_agents: List[str] = field(default_factory=list)
    channel: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    requires_acknowledgment: bool = False
    delivery_mode: str = "fire_and_forget"  # fire_and_forget, at_least_once, exactly_once

@dataclass
class AgentStatusUpdate:
    """Agent status update structure."""
    agent_id: str
    status: AgentStatus
    capabilities: List[str] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    health_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.utcnow)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecutionUpdate:
    """Task execution update structure."""
    task_id: str
    agent_id: str
    status: TaskExecutionStatus
    progress_percentage: float = 0.0
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    next_actions: List[str] = field(default_factory=list)

@dataclass
class HealthMonitoringUpdate:
    """Health monitoring update structure."""
    component_id: str
    component_type: str  # agent, service, connection, etc.
    health_score: float
    status: str  # healthy, degraded, unhealthy, error
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.utcnow)

# ================================================================================
# Real-Time Communication Hub
# ================================================================================

class RealTimeCommunicationHub:
    """
    Advanced real-time communication hub for LeanVibe Agent Hive 2.0.
    
    Provides comprehensive real-time communication capabilities including:
    - Agent status broadcasting and monitoring
    - Task execution notifications and tracking
    - Health monitoring with alerts and recommendations
    - Client dashboard real-time updates
    - Message routing with delivery guarantees
    - Connection management and resource optimization
    """
    
    def __init__(
        self,
        redis_config: Optional[RedisConfig] = None,
        websocket_config: Optional[WebSocketConfig] = None,
        hub_config: Optional[Dict[str, Any]] = None
    ):
        self.redis_config = redis_config or RedisConfig()
        self.websocket_config = websocket_config or WebSocketConfig()
        self.hub_config = hub_config or {}
        
        # Core communication bridge
        self.communication_bridge = UnifiedCommunicationBridge(
            redis_config=self.redis_config,
            websocket_config=self.websocket_config
        )
        
        # Real-time state management
        self.agent_statuses: Dict[str, AgentStatusUpdate] = {}
        self.active_tasks: Dict[str, TaskExecutionUpdate] = {}
        self.health_states: Dict[str, HealthMonitoringUpdate] = {}
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> channels
        
        # Message handlers and routing
        self.message_handlers: Dict[NotificationType, List[Callable]] = {}
        self.channel_subscribers: Dict[str, Set[str]] = {}  # channel -> client_ids
        self.message_filters: Dict[str, List[Callable]] = {}
        
        # Performance tracking
        self.metrics = {
            "messages_processed": 0,
            "notifications_sent": 0,
            "client_connections": 0,
            "agent_updates": 0,
            "task_updates": 0,
            "health_updates": 0,
            "average_latency_ms": 0.0,
            "error_count": 0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.config = {
            "agent_status_ttl_seconds": 300,  # 5 minutes
            "task_update_ttl_seconds": 3600,  # 1 hour
            "health_check_interval": 30,  # 30 seconds
            "metrics_collection_interval": 60,  # 1 minute
            "max_message_queue_size": 10000,
            "message_batch_size": 100,
            "enable_message_persistence": True,
            "cleanup_interval_seconds": 300,  # 5 minutes
            **self.hub_config
        }
        
        logger.info("RealTimeCommunicationHub initialized")
    
    async def initialize(self) -> bool:
        """Initialize the real-time communication hub."""
        try:
            logger.info("Initializing RealTimeCommunicationHub...")
            
            # Initialize underlying communication bridge
            bridge_success = await self.communication_bridge.initialize(
                enable_redis=True,
                enable_websocket=True,
                start_websocket_server=True
            )
            
            if not bridge_success:
                logger.error("Failed to initialize communication bridge")
                return False
            
            # Register default message handlers
            await self._register_default_handlers()
            
            # Start background services
            await self._start_background_services()
            
            # Set up Redis channels for real-time communication
            await self._setup_redis_channels()
            
            logger.info("RealTimeCommunicationHub initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RealTimeCommunicationHub: {e}")
            return False
    
    # ================================================================================
    # Agent Status Broadcasting
    # ================================================================================
    
    async def broadcast_agent_status(
        self,
        agent_status: AgentStatusUpdate,
        target_clients: Optional[List[str]] = None
    ) -> bool:
        """
        Broadcast agent status update to subscribed clients.
        
        Args:
            agent_status: Agent status update to broadcast
            target_clients: Specific clients to target (if None, broadcast to all)
            
        Returns:
            bool: True if broadcast successful
        """
        try:
            # Update internal state
            self.agent_statuses[agent_status.agent_id] = agent_status
            
            # Create real-time message
            message = RealTimeMessage(
                notification_type=NotificationType.AGENT_STATUS,
                priority=MessagePriority.MEDIUM,
                source_agent_id=agent_status.agent_id,
                channel="agent_status",
                payload={
                    "agent_id": agent_status.agent_id,
                    "status": agent_status.status.value,
                    "capabilities": agent_status.capabilities,
                    "current_tasks": agent_status.current_tasks,
                    "performance_metrics": agent_status.performance_metrics,
                    "health_score": agent_status.health_score,
                    "last_seen": agent_status.last_seen.isoformat(),
                    "additional_info": agent_status.additional_info
                },
                metadata={
                    "broadcast_type": "agent_status",
                    "target_clients": target_clients or []
                }
            )
            
            # Broadcast via Redis pub/sub
            redis_success = await self._broadcast_via_redis(message)
            
            # Send to WebSocket clients
            websocket_success = await self._broadcast_via_websocket(message, target_clients)
            
            # Update metrics
            if redis_success or websocket_success:
                self.metrics["agent_updates"] += 1
                self.metrics["notifications_sent"] += 1
                logger.debug(f"Agent status broadcast sent: {agent_status.agent_id}")
                return True
            else:
                self.metrics["error_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to broadcast agent status: {e}")
            self.metrics["error_count"] += 1
            return False
    
    async def subscribe_to_agent_status(
        self,
        client_id: str,
        agent_ids: Optional[List[str]] = None
    ) -> AsyncGenerator[AgentStatusUpdate, None]:
        """
        Subscribe to agent status updates.
        
        Args:
            client_id: Unique client identifier
            agent_ids: Specific agent IDs to subscribe to (if None, subscribe to all)
            
        Yields:
            AgentStatusUpdate: Real-time agent status updates
        """
        try:
            # Add to client subscriptions
            if client_id not in self.client_subscriptions:
                self.client_subscriptions[client_id] = set()
            self.client_subscriptions[client_id].add("agent_status")
            
            # Create message filter for specific agents if provided
            def agent_filter(message: RealTimeMessage) -> bool:
                if agent_ids is None:
                    return True
                return message.payload.get("agent_id") in agent_ids
            
            # Listen for Redis messages
            async for message in self.communication_bridge.listen_redis_channel("agent_status"):
                try:
                    # Convert CLI message to RealTimeMessage
                    rt_message = self._cli_to_realtime_message(message)
                    
                    if rt_message.notification_type == NotificationType.AGENT_STATUS:
                        if agent_filter(rt_message):
                            # Convert to AgentStatusUpdate
                            agent_update = AgentStatusUpdate(
                                agent_id=rt_message.payload["agent_id"],
                                status=AgentStatus(rt_message.payload["status"]),
                                capabilities=rt_message.payload.get("capabilities", []),
                                current_tasks=rt_message.payload.get("current_tasks", []),
                                performance_metrics=rt_message.payload.get("performance_metrics", {}),
                                health_score=rt_message.payload.get("health_score", 1.0),
                                last_seen=datetime.fromisoformat(rt_message.payload["last_seen"]),
                                additional_info=rt_message.payload.get("additional_info", {})
                            )
                            
                            yield agent_update
                            
                except Exception as e:
                    logger.error(f"Error processing agent status message: {e}")
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Agent status subscription cancelled for client: {client_id}")
        except Exception as e:
            logger.error(f"Error in agent status subscription: {e}")
        finally:
            # Cleanup subscription
            if client_id in self.client_subscriptions:
                self.client_subscriptions[client_id].discard("agent_status")
    
    # ================================================================================
    # Task Execution Notifications
    # ================================================================================
    
    async def broadcast_task_execution_update(
        self,
        task_update: TaskExecutionUpdate,
        target_clients: Optional[List[str]] = None
    ) -> bool:
        """
        Broadcast task execution update to subscribed clients.
        
        Args:
            task_update: Task execution update to broadcast
            target_clients: Specific clients to target (if None, broadcast to all)
            
        Returns:
            bool: True if broadcast successful
        """
        try:
            # Update internal state
            self.active_tasks[task_update.task_id] = task_update
            
            # Determine priority based on status
            priority = MessagePriority.MEDIUM
            if task_update.status in [TaskExecutionStatus.FAILED, TaskExecutionStatus.TIMEOUT]:
                priority = MessagePriority.HIGH
            elif task_update.status == TaskExecutionStatus.COMPLETED:
                priority = MessagePriority.MEDIUM
            
            # Create real-time message
            message = RealTimeMessage(
                notification_type=NotificationType.TASK_EXECUTION,
                priority=priority,
                source_agent_id=task_update.agent_id,
                channel="task_execution",
                payload={
                    "task_id": task_update.task_id,
                    "agent_id": task_update.agent_id,
                    "status": task_update.status.value,
                    "progress_percentage": task_update.progress_percentage,
                    "execution_time_ms": task_update.execution_time_ms,
                    "error_message": task_update.error_message,
                    "result_data": task_update.result_data,
                    "resource_usage": task_update.resource_usage,
                    "next_actions": task_update.next_actions
                },
                metadata={
                    "broadcast_type": "task_execution",
                    "target_clients": target_clients or []
                }
            )
            
            # Broadcast via Redis pub/sub
            redis_success = await self._broadcast_via_redis(message)
            
            # Send to WebSocket clients
            websocket_success = await self._broadcast_via_websocket(message, target_clients)
            
            # Update metrics
            if redis_success or websocket_success:
                self.metrics["task_updates"] += 1
                self.metrics["notifications_sent"] += 1
                logger.debug(f"Task execution update broadcast: {task_update.task_id}")
                return True
            else:
                self.metrics["error_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to broadcast task execution update: {e}")
            self.metrics["error_count"] += 1
            return False
    
    async def subscribe_to_task_execution(
        self,
        client_id: str,
        task_ids: Optional[List[str]] = None,
        agent_ids: Optional[List[str]] = None
    ) -> AsyncGenerator[TaskExecutionUpdate, None]:
        """
        Subscribe to task execution updates.
        
        Args:
            client_id: Unique client identifier
            task_ids: Specific task IDs to subscribe to
            agent_ids: Specific agent IDs to subscribe to
            
        Yields:
            TaskExecutionUpdate: Real-time task execution updates
        """
        try:
            # Add to client subscriptions
            if client_id not in self.client_subscriptions:
                self.client_subscriptions[client_id] = set()
            self.client_subscriptions[client_id].add("task_execution")
            
            # Create message filter
            def task_filter(message: RealTimeMessage) -> bool:
                if task_ids and message.payload.get("task_id") not in task_ids:
                    return False
                if agent_ids and message.payload.get("agent_id") not in agent_ids:
                    return False
                return True
            
            # Listen for Redis messages
            async for message in self.communication_bridge.listen_redis_channel("task_execution"):
                try:
                    # Convert CLI message to RealTimeMessage
                    rt_message = self._cli_to_realtime_message(message)
                    
                    if rt_message.notification_type == NotificationType.TASK_EXECUTION:
                        if task_filter(rt_message):
                            # Convert to TaskExecutionUpdate
                            task_update = TaskExecutionUpdate(
                                task_id=rt_message.payload["task_id"],
                                agent_id=rt_message.payload["agent_id"],
                                status=TaskExecutionStatus(rt_message.payload["status"]),
                                progress_percentage=rt_message.payload.get("progress_percentage", 0.0),
                                execution_time_ms=rt_message.payload.get("execution_time_ms"),
                                error_message=rt_message.payload.get("error_message"),
                                result_data=rt_message.payload.get("result_data", {}),
                                resource_usage=rt_message.payload.get("resource_usage", {}),
                                next_actions=rt_message.payload.get("next_actions", [])
                            )
                            
                            yield task_update
                            
                except Exception as e:
                    logger.error(f"Error processing task execution message: {e}")
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Task execution subscription cancelled for client: {client_id}")
        except Exception as e:
            logger.error(f"Error in task execution subscription: {e}")
        finally:
            # Cleanup subscription
            if client_id in self.client_subscriptions:
                self.client_subscriptions[client_id].discard("task_execution")
    
    # ================================================================================
    # Health Monitoring Broadcasts
    # ================================================================================
    
    async def broadcast_health_monitoring_update(
        self,
        health_update: HealthMonitoringUpdate,
        target_clients: Optional[List[str]] = None
    ) -> bool:
        """
        Broadcast health monitoring update to subscribed clients.
        
        Args:
            health_update: Health monitoring update to broadcast
            target_clients: Specific clients to target (if None, broadcast to all)
            
        Returns:
            bool: True if broadcast successful
        """
        try:
            # Update internal state
            self.health_states[health_update.component_id] = health_update
            
            # Determine priority based on health status
            priority = MessagePriority.LOW
            if health_update.status == "error":
                priority = MessagePriority.CRITICAL
            elif health_update.status == "unhealthy":
                priority = MessagePriority.HIGH
            elif health_update.status == "degraded":
                priority = MessagePriority.MEDIUM
            
            # Create real-time message
            message = RealTimeMessage(
                notification_type=NotificationType.HEALTH_MONITORING,
                priority=priority,
                source_agent_id=health_update.component_id,
                channel="health_monitoring",
                payload={
                    "component_id": health_update.component_id,
                    "component_type": health_update.component_type,
                    "health_score": health_update.health_score,
                    "status": health_update.status,
                    "metrics": health_update.metrics,
                    "alerts": health_update.alerts,
                    "recommendations": health_update.recommendations,
                    "last_check": health_update.last_check.isoformat()
                },
                metadata={
                    "broadcast_type": "health_monitoring",
                    "target_clients": target_clients or []
                }
            )
            
            # Broadcast via Redis pub/sub
            redis_success = await self._broadcast_via_redis(message)
            
            # Send to WebSocket clients
            websocket_success = await self._broadcast_via_websocket(message, target_clients)
            
            # Update metrics
            if redis_success or websocket_success:
                self.metrics["health_updates"] += 1
                self.metrics["notifications_sent"] += 1
                logger.debug(f"Health monitoring update broadcast: {health_update.component_id}")
                return True
            else:
                self.metrics["error_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to broadcast health monitoring update: {e}")
            self.metrics["error_count"] += 1
            return False
    
    async def subscribe_to_health_monitoring(
        self,
        client_id: str,
        component_types: Optional[List[str]] = None,
        min_priority: MessagePriority = MessagePriority.LOW
    ) -> AsyncGenerator[HealthMonitoringUpdate, None]:
        """
        Subscribe to health monitoring updates.
        
        Args:
            client_id: Unique client identifier
            component_types: Specific component types to subscribe to
            min_priority: Minimum priority level for updates
            
        Yields:
            HealthMonitoringUpdate: Real-time health monitoring updates
        """
        try:
            # Add to client subscriptions
            if client_id not in self.client_subscriptions:
                self.client_subscriptions[client_id] = set()
            self.client_subscriptions[client_id].add("health_monitoring")
            
            # Create message filter
            def health_filter(message: RealTimeMessage) -> bool:
                if component_types and message.payload.get("component_type") not in component_types:
                    return False
                if message.priority.value > min_priority.value:
                    return False
                return True
            
            # Listen for Redis messages
            async for message in self.communication_bridge.listen_redis_channel("health_monitoring"):
                try:
                    # Convert CLI message to RealTimeMessage
                    rt_message = self._cli_to_realtime_message(message)
                    
                    if rt_message.notification_type == NotificationType.HEALTH_MONITORING:
                        if health_filter(rt_message):
                            # Convert to HealthMonitoringUpdate
                            health_update = HealthMonitoringUpdate(
                                component_id=rt_message.payload["component_id"],
                                component_type=rt_message.payload["component_type"],
                                health_score=rt_message.payload["health_score"],
                                status=rt_message.payload["status"],
                                metrics=rt_message.payload.get("metrics", {}),
                                alerts=rt_message.payload.get("alerts", []),
                                recommendations=rt_message.payload.get("recommendations", []),
                                last_check=datetime.fromisoformat(rt_message.payload["last_check"])
                            )
                            
                            yield health_update
                            
                except Exception as e:
                    logger.error(f"Error processing health monitoring message: {e}")
                    continue
                    
        except asyncio.CancelledError:
            logger.info(f"Health monitoring subscription cancelled for client: {client_id}")
        except Exception as e:
            logger.error(f"Error in health monitoring subscription: {e}")
        finally:
            # Cleanup subscription
            if client_id in self.client_subscriptions:
                self.client_subscriptions[client_id].discard("health_monitoring")
    
    # ================================================================================
    # Dashboard Real-Time Updates
    # ================================================================================
    
    async def broadcast_dashboard_update(
        self,
        update_type: str,
        data: Dict[str, Any],
        target_clients: Optional[List[str]] = None
    ) -> bool:
        """
        Broadcast dashboard update to connected clients.
        
        Args:
            update_type: Type of dashboard update
            data: Update data
            target_clients: Specific clients to target
            
        Returns:
            bool: True if broadcast successful
        """
        try:
            message = RealTimeMessage(
                notification_type=NotificationType.DASHBOARD_UPDATE,
                priority=MessagePriority.MEDIUM,
                channel="dashboard_updates",
                payload={
                    "update_type": update_type,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                },
                metadata={
                    "broadcast_type": "dashboard_update",
                    "target_clients": target_clients or []
                }
            )
            
            # Broadcast via WebSocket (primary for dashboard updates)
            websocket_success = await self._broadcast_via_websocket(message, target_clients)
            
            # Also broadcast via Redis for persistence
            redis_success = await self._broadcast_via_redis(message)
            
            if websocket_success or redis_success:
                self.metrics["notifications_sent"] += 1
                return True
            else:
                self.metrics["error_count"] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to broadcast dashboard update: {e}")
            self.metrics["error_count"] += 1
            return False
    
    async def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get current real-time dashboard data."""
        try:
            # Get health status from communication bridge
            bridge_health = await self.communication_bridge.get_health_status()
            
            return {
                "agent_statuses": {
                    agent_id: {
                        "status": status.status.value,
                        "health_score": status.health_score,
                        "current_tasks": len(status.current_tasks),
                        "last_seen": status.last_seen.isoformat()
                    }
                    for agent_id, status in self.agent_statuses.items()
                },
                "active_tasks": {
                    task_id: {
                        "status": task.status.value,
                        "progress": task.progress_percentage,
                        "agent_id": task.agent_id
                    }
                    for task_id, task in self.active_tasks.items()
                },
                "health_overview": {
                    "healthy_components": len([h for h in self.health_states.values() if h.status == "healthy"]),
                    "degraded_components": len([h for h in self.health_states.values() if h.status == "degraded"]),
                    "unhealthy_components": len([h for h in self.health_states.values() if h.status in ["unhealthy", "error"]]),
                    "total_components": len(self.health_states)
                },
                "communication_metrics": {
                    "redis_status": bridge_health.get("redis", {}).get("status", "unknown"),
                    "websocket_status": bridge_health.get("websocket", {}).get("status", "unknown"),
                    "client_connections": self.metrics["client_connections"],
                    "messages_processed": self.metrics["messages_processed"],
                    "average_latency_ms": self.metrics["average_latency_ms"]
                },
                "system_overview": {
                    "total_agents": len(self.agent_statuses),
                    "online_agents": len([a for a in self.agent_statuses.values() if a.status == AgentStatus.ONLINE]),
                    "active_tasks": len([t for t in self.active_tasks.values() if t.status == TaskExecutionStatus.IN_PROGRESS]),
                    "error_count": self.metrics["error_count"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    # ================================================================================
    # Connection Management and Message Routing
    # ================================================================================
    
    async def register_websocket_client(self, client_id: str, websocket_connection_id: str) -> bool:
        """Register a WebSocket client for real-time updates."""
        try:
            # Add client tracking
            if client_id not in self.client_subscriptions:
                self.client_subscriptions[client_id] = set()
            
            # Update metrics
            self.metrics["client_connections"] += 1
            
            logger.info(f"WebSocket client registered: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register WebSocket client: {e}")
            return False
    
    async def unregister_websocket_client(self, client_id: str) -> bool:
        """Unregister a WebSocket client."""
        try:
            # Remove client subscriptions
            if client_id in self.client_subscriptions:
                del self.client_subscriptions[client_id]
            
            # Update metrics
            self.metrics["client_connections"] = max(0, self.metrics["client_connections"] - 1)
            
            logger.info(f"WebSocket client unregistered: {client_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister WebSocket client: {e}")
            return False
    
    async def add_message_handler(
        self,
        notification_type: NotificationType,
        handler: Callable[[RealTimeMessage], Any]
    ) -> bool:
        """Add a custom message handler for specific notification types."""
        try:
            if notification_type not in self.message_handlers:
                self.message_handlers[notification_type] = []
            
            self.message_handlers[notification_type].append(handler)
            logger.info(f"Message handler added for {notification_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message handler: {e}")
            return False
    
    # ================================================================================
    # Health and Monitoring
    # ================================================================================
    
    async def get_hub_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of the communication hub."""
        try:
            # Get bridge health
            bridge_health = await self.communication_bridge.get_health_status()
            
            # Calculate hub-specific metrics
            total_subscriptions = sum(len(subs) for subs in self.client_subscriptions.values())
            active_agents = len([a for a in self.agent_statuses.values() if a.status == AgentStatus.ONLINE])
            
            return {
                "hub_status": "healthy",
                "bridge_health": bridge_health,
                "metrics": self.metrics,
                "statistics": {
                    "total_clients": len(self.client_subscriptions),
                    "total_subscriptions": total_subscriptions,
                    "active_agents": active_agents,
                    "tracked_tasks": len(self.active_tasks),
                    "monitored_components": len(self.health_states)
                },
                "configuration": self.config,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting hub health status: {e}")
            return {
                "hub_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            return {
                "message_throughput": {
                    "messages_per_second": self.metrics["messages_processed"] / max(1, time.time()),
                    "notifications_per_second": self.metrics["notifications_sent"] / max(1, time.time())
                },
                "latency_metrics": {
                    "average_latency_ms": self.metrics["average_latency_ms"],
                    "latency_percentiles": {}  # Would need to track more detailed metrics
                },
                "resource_usage": {
                    "active_connections": self.metrics["client_connections"],
                    "background_tasks": len(self.background_tasks),
                    "memory_usage_mb": 0  # Would need psutil integration
                },
                "error_metrics": {
                    "total_errors": self.metrics["error_count"],
                    "error_rate": self.metrics["error_count"] / max(1, self.metrics["messages_processed"])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    # ================================================================================
    # Internal Helper Methods
    # ================================================================================
    
    async def _broadcast_via_redis(self, message: RealTimeMessage) -> bool:
        """Broadcast message via Redis pub/sub."""
        try:
            if not self.communication_bridge.redis_broker:
                return False
            
            # Convert to CLI message format
            cli_message = self._realtime_to_cli_message(message)
            
            # Publish to Redis channel
            success = await self.communication_bridge.send_message_redis(
                channel=message.channel,
                message=cli_message,
                persistent=self.config["enable_message_persistence"]
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to broadcast via Redis: {e}")
            return False
    
    async def _broadcast_via_websocket(
        self,
        message: RealTimeMessage,
        target_clients: Optional[List[str]] = None
    ) -> bool:
        """Broadcast message via WebSocket connections."""
        try:
            if not self.communication_bridge.websocket_bridge:
                return False
            
            # Determine target clients
            clients = target_clients or list(self.client_subscriptions.keys())
            
            success_count = 0
            for client_id in clients:
                # Check if client is subscribed to this channel
                if message.channel in self.client_subscriptions.get(client_id, set()):
                    # Convert to CLI message format
                    cli_message = self._realtime_to_cli_message(message)
                    
                    # Send to WebSocket client
                    if await self.communication_bridge.send_message_websocket(
                        connection_id=client_id,
                        message=cli_message,
                        require_ack=message.requires_acknowledgment
                    ):
                        success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast via WebSocket: {e}")
            return False
    
    def _realtime_to_cli_message(self, rt_message: RealTimeMessage) -> CLIMessage:
        """Convert RealTimeMessage to CLIMessage format."""
        return CLIMessage(
            universal_message_id=rt_message.message_id,
            cli_protocol=CLIProtocol.UNIVERSAL,
            cli_command="realtime_notification",
            cli_args=[rt_message.notification_type.value],
            cli_options={
                "priority": rt_message.priority.value,
                "channel": rt_message.channel,
                "requires_ack": rt_message.requires_acknowledgment
            },
            input_data={
                "payload": rt_message.payload,
                "metadata": rt_message.metadata,
                "timestamp": rt_message.timestamp.isoformat(),
                "expires_at": rt_message.expires_at.isoformat() if rt_message.expires_at else None
            },
            timeout_seconds=300,
            priority=rt_message.priority.value
        )
    
    def _cli_to_realtime_message(self, cli_message: CLIMessage) -> RealTimeMessage:
        """Convert CLIMessage to RealTimeMessage format."""
        input_data = cli_message.input_data or {}
        
        return RealTimeMessage(
            message_id=cli_message.universal_message_id,
            notification_type=NotificationType(cli_message.cli_args[0] if cli_message.cli_args else "dashboard_update"),
            priority=MessagePriority(cli_message.cli_options.get("priority", MessagePriority.MEDIUM.value)),
            channel=cli_message.cli_options.get("channel", "default"),
            payload=input_data.get("payload", {}),
            metadata=input_data.get("metadata", {}),
            timestamp=datetime.fromisoformat(input_data["timestamp"]) if "timestamp" in input_data else datetime.utcnow(),
            expires_at=datetime.fromisoformat(input_data["expires_at"]) if input_data.get("expires_at") else None,
            requires_acknowledgment=cli_message.cli_options.get("requires_ack", False)
        )
    
    async def _register_default_handlers(self):
        """Register default message handlers."""
        try:
            # Register handler for metrics collection
            async def metrics_handler(message: RealTimeMessage):
                self.metrics["messages_processed"] += 1
                
                # Update latency metrics
                message_age = (datetime.utcnow() - message.timestamp).total_seconds() * 1000
                if self.metrics["average_latency_ms"] == 0:
                    self.metrics["average_latency_ms"] = message_age
                else:
                    self.metrics["average_latency_ms"] = (
                        self.metrics["average_latency_ms"] * 0.9 + message_age * 0.1
                    )
            
            # Add handler for all notification types
            for notification_type in NotificationType:
                await self.add_message_handler(notification_type, metrics_handler)
            
            logger.info("Default message handlers registered")
            
        except Exception as e:
            logger.error(f"Failed to register default handlers: {e}")
    
    async def _setup_redis_channels(self):
        """Set up Redis channels for real-time communication."""
        try:
            # Define standard channels
            channels = [
                "agent_status",
                "task_execution", 
                "health_monitoring",
                "dashboard_updates",
                "system_alerts",
                "coordination_events"
            ]
            
            # No need to explicitly create channels in Redis - they're created on first publish
            logger.info(f"Redis channels configured: {channels}")
            
        except Exception as e:
            logger.error(f"Failed to setup Redis channels: {e}")
    
    async def _start_background_services(self):
        """Start background monitoring and maintenance tasks."""
        try:
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.background_tasks.add(self.cleanup_task)
            
            # Start metrics collection task
            self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
            self.background_tasks.add(self.metrics_task)
            
            logger.info("Background services started")
            
        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of expired data and inactive connections."""
        while True:
            try:
                await asyncio.sleep(self.config["cleanup_interval_seconds"])
                
                current_time = datetime.utcnow()
                
                # Clean up expired agent statuses
                expired_agents = [
                    agent_id for agent_id, status in self.agent_statuses.items()
                    if (current_time - status.last_seen).total_seconds() > self.config["agent_status_ttl_seconds"]
                ]
                
                for agent_id in expired_agents:
                    del self.agent_statuses[agent_id]
                    logger.debug(f"Cleaned up expired agent status: {agent_id}")
                
                # Clean up old task updates
                expired_tasks = [
                    task_id for task_id, task in self.active_tasks.items()
                    if task.status in [TaskExecutionStatus.COMPLETED, TaskExecutionStatus.FAILED, TaskExecutionStatus.CANCELLED]
                    and (current_time - datetime.utcnow()).total_seconds() > self.config["task_update_ttl_seconds"]
                ]
                
                for task_id in expired_tasks:
                    del self.active_tasks[task_id]
                    logger.debug(f"Cleaned up completed task: {task_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _metrics_collection_loop(self):
        """Background metrics collection and reporting."""
        while True:
            try:
                await asyncio.sleep(self.config["metrics_collection_interval"])
                
                # Collect and report hub metrics
                metrics_data = await self.get_performance_metrics()
                
                # Broadcast metrics update to dashboard
                await self.broadcast_dashboard_update(
                    update_type="metrics_update",
                    data=metrics_data
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the real-time communication hub."""
        try:
            logger.info("Shutting down RealTimeCommunicationHub...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Shutdown communication bridge
            await self.communication_bridge.shutdown()
            
            logger.info("RealTimeCommunicationHub shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# ================================================================================
# Factory and Configuration Functions
# ================================================================================

def create_realtime_hub(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    websocket_host: str = "localhost", 
    websocket_port: int = 8765,
    **kwargs
) -> RealTimeCommunicationHub:
    """Create a configured real-time communication hub."""
    
    redis_config = RedisConfig(
        host=redis_host,
        port=redis_port,
        **{k: v for k, v in kwargs.items() if k.startswith('redis_')}
    )
    
    websocket_config = WebSocketConfig(
        host=websocket_host,
        port=websocket_port,
        **{k: v for k, v in kwargs.items() if k.startswith('websocket_')}
    )
    
    hub_config = {k: v for k, v in kwargs.items() if not k.startswith(('redis_', 'websocket_'))}
    
    return RealTimeCommunicationHub(
        redis_config=redis_config,
        websocket_config=websocket_config,
        hub_config=hub_config
    )

async def main():
    """Example usage of the RealTimeCommunicationHub."""
    # Create and initialize hub
    hub = create_realtime_hub()
    
    try:
        success = await hub.initialize()
        if not success:
            logger.error("Failed to initialize communication hub")
            return
        
        # Example: Broadcast an agent status update
        agent_status = AgentStatusUpdate(
            agent_id="agent_001",
            status=AgentStatus.ONLINE,
            capabilities=["code_generation", "debugging"],
            current_tasks=["task_123"],
            health_score=0.95
        )
        
        await hub.broadcast_agent_status(agent_status)
        
        # Example: Broadcast a task execution update
        task_update = TaskExecutionUpdate(
            task_id="task_123",
            agent_id="agent_001",
            status=TaskExecutionStatus.IN_PROGRESS,
            progress_percentage=75.0
        )
        
        await hub.broadcast_task_execution_update(task_update)
        
        # Get dashboard data
        dashboard_data = await hub.get_real_time_dashboard_data()
        logger.info(f"Dashboard data: {dashboard_data}")
        
        # Keep running for demo
        await asyncio.sleep(60)
        
    finally:
        await hub.shutdown()

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class RealtimeCommunicationHubScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(RealtimeCommunicationHubScript)
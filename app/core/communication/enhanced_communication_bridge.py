"""
Enhanced Communication Bridge with Real-Time Integration

This module extends the existing communication bridge with real-time capabilities
for seamless integration with the RealTimeCommunicationHub.

Features:
- Integration with RealTimeCommunicationHub for agent coordination
- Enhanced message routing with real-time notifications
- Connection management with real-time status broadcasting
- Performance monitoring with live dashboard updates
- Auto-recovery with health monitoring broadcasts
- Resource management with real-time metrics
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Set, Callable

from .communication_bridge import ProductionCommunicationBridge, CommunicationBridge
from .realtime_communication_hub import (
    RealTimeCommunicationHub,
    AgentStatusUpdate,
    TaskExecutionUpdate,
    HealthMonitoringUpdate,
    AgentStatus,
    TaskExecutionStatus,
    MessagePriority,
    NotificationType
)
from .protocol_models import (
    BridgeConnection,
    CLIProtocol,
    UniversalMessage,
    CLIMessage
)

logger = logging.getLogger(__name__)

# ================================================================================
# Enhanced Communication Bridge
# ================================================================================

class EnhancedCommunicationBridge(ProductionCommunicationBridge):
    """
    Enhanced communication bridge with real-time integration.
    
    Extends ProductionCommunicationBridge with:
    - Real-time agent status broadcasting
    - Task execution notifications
    - Health monitoring broadcasts
    - Performance metrics streaming
    - Dashboard updates integration
    - Enhanced connection management
    """
    
    def __init__(
        self,
        realtime_hub: Optional[RealTimeCommunicationHub] = None,
        enable_realtime: bool = True
    ):
        """Initialize enhanced communication bridge."""
        super().__init__()
        
        # Real-time integration
        self.realtime_hub = realtime_hub
        self.enable_realtime = enable_realtime
        
        # Enhanced tracking
        self._agent_connections: Dict[str, str] = {}  # agent_id -> connection_id
        self._connection_agents: Dict[str, str] = {}  # connection_id -> agent_id
        self._task_executions: Dict[str, str] = {}  # task_id -> agent_id
        self._connection_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Real-time callbacks
        self._status_callbacks: List[Callable] = []
        self._health_callbacks: List[Callable] = []
        self._task_callbacks: List[Callable] = []
        
        # Enhanced configuration
        self._config.update({
            "realtime_status_interval": 10,  # seconds
            "health_broadcast_interval": 30,  # seconds
            "performance_metrics_interval": 60,  # seconds
            "enable_connection_broadcasting": True,
            "enable_task_notifications": True,
            "enable_health_monitoring": True
        })
        
        logger.info("EnhancedCommunicationBridge initialized")
    
    async def initialize_with_realtime(self, realtime_hub: RealTimeCommunicationHub) -> bool:
        """Initialize bridge with real-time hub integration."""
        try:
            self.realtime_hub = realtime_hub
            
            # Start real-time background tasks
            if self.enable_realtime:
                await self._start_realtime_services()
            
            logger.info("Enhanced bridge initialized with real-time hub")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced bridge: {e}")
            return False
    
    # ================================================================================
    # Enhanced Bridge Operations with Real-Time Integration
    # ================================================================================
    
    async def establish_bridge(
        self,
        source_protocol: CLIProtocol,
        target_protocol: CLIProtocol,
        connection_config: Dict[str, Any]
    ) -> BridgeConnection:
        """
        Establish bridge with real-time status broadcasting.
        """
        start_time = time.time()
        
        try:
            # Call parent implementation
            connection = await super().establish_bridge(
                source_protocol, target_protocol, connection_config
            )
            
            # Enhanced tracking
            agent_id = connection_config.get("agent_id", f"agent_{connection.connection_id[:8]}")
            self._agent_connections[agent_id] = connection.connection_id
            self._connection_agents[connection.connection_id] = agent_id
            
            # Initialize connection metrics
            self._connection_metrics[connection.connection_id] = {
                "established_at": datetime.utcnow(),
                "total_messages": 0,
                "error_count": 0,
                "last_activity": datetime.utcnow(),
                "performance_score": 1.0
            }
            
            # Broadcast agent status if real-time enabled
            if self.enable_realtime and self.realtime_hub:
                await self._broadcast_agent_connection_status(agent_id, connection, True)
            
            # Broadcast health update
            if self._config["enable_health_monitoring"] and self.realtime_hub:
                await self._broadcast_connection_health_update(connection, "healthy")
            
            establishment_time = (time.time() - start_time) * 1000
            logger.info(f"Enhanced bridge established: {connection.connection_id} ({establishment_time:.1f}ms)")
            
            return connection
            
        except Exception as e:
            logger.error(f"Failed to establish enhanced bridge: {e}")
            
            # Broadcast failure if real-time enabled
            if self.enable_realtime and self.realtime_hub:
                await self._broadcast_connection_failure(
                    connection_config.get("agent_id", "unknown"),
                    str(e)
                )
            
            raise
    
    async def send_message_through_bridge(
        self,
        connection_id: str,
        message: CLIMessage
    ) -> bool:
        """
        Send message with real-time task execution tracking.
        """
        start_time = time.time()
        
        try:
            # Update connection metrics
            if connection_id in self._connection_metrics:
                self._connection_metrics[connection_id]["total_messages"] += 1
                self._connection_metrics[connection_id]["last_activity"] = datetime.utcnow()
            
            # Extract task information if present
            task_id = message.input_data.get("task_id") if message.input_data else None
            agent_id = self._connection_agents.get(connection_id)
            
            # Broadcast task start if applicable
            if task_id and agent_id and self._config["enable_task_notifications"] and self.realtime_hub:
                await self._broadcast_task_execution_start(task_id, agent_id, message)
            
            # Call parent implementation
            success = await super().send_message_through_bridge(connection_id, message)
            
            # Update performance metrics
            delivery_time = (time.time() - start_time) * 1000
            if connection_id in self._connection_metrics:
                if success:
                    self._connection_metrics[connection_id]["performance_score"] = min(
                        1.0,
                        self._connection_metrics[connection_id]["performance_score"] * 0.95 + 0.05
                    )
                else:
                    self._connection_metrics[connection_id]["error_count"] += 1
                    self._connection_metrics[connection_id]["performance_score"] *= 0.9
            
            # Broadcast performance update
            if self.enable_realtime and self.realtime_hub:
                await self._broadcast_performance_update(connection_id, delivery_time, success)
            
            return success
            
        except Exception as e:
            logger.error(f"Enhanced message send failed: {e}")
            
            # Update error metrics
            if connection_id in self._connection_metrics:
                self._connection_metrics[connection_id]["error_count"] += 1
                self._connection_metrics[connection_id]["performance_score"] *= 0.8
            
            # Broadcast error if real-time enabled
            if self.enable_realtime and self.realtime_hub:
                await self._broadcast_message_error(connection_id, message, str(e))
            
            return False
    
    async def listen_for_messages(
        self,
        connection_id: str
    ) -> AsyncGenerator[CLIMessage, None]:
        """
        Listen for messages with real-time task completion tracking.
        """
        agent_id = self._connection_agents.get(connection_id)
        
        try:
            async for message in super().listen_for_messages(connection_id):
                # Update connection activity
                if connection_id in self._connection_metrics:
                    self._connection_metrics[connection_id]["last_activity"] = datetime.utcnow()
                
                # Check for task completion
                if message.input_data and self._config["enable_task_notifications"]:
                    await self._handle_message_task_updates(message, agent_id)
                
                # Broadcast message received
                if self.enable_realtime and self.realtime_hub:
                    await self._broadcast_message_received(connection_id, message)
                
                yield message
                
        except Exception as e:
            logger.error(f"Enhanced message listening failed: {e}")
            
            # Broadcast listening error
            if self.enable_realtime and self.realtime_hub:
                await self._broadcast_listening_error(connection_id, str(e))
    
    async def monitor_bridge_health(
        self,
        connection_id: str
    ) -> Dict[str, Any]:
        """
        Monitor bridge health with real-time broadcasting.
        """
        try:
            # Call parent implementation
            health_data = await super().monitor_bridge_health(connection_id)
            
            # Add enhanced metrics
            if connection_id in self._connection_metrics:
                metrics = self._connection_metrics[connection_id]
                health_data.update({
                    "enhanced_metrics": {
                        "total_messages": metrics["total_messages"],
                        "error_count": metrics["error_count"],
                        "error_rate": metrics["error_count"] / max(1, metrics["total_messages"]),
                        "performance_score": metrics["performance_score"],
                        "uptime_seconds": (datetime.utcnow() - metrics["established_at"]).total_seconds()
                    }
                })
            
            # Broadcast health update
            if self._config["enable_health_monitoring"] and self.realtime_hub:
                await self._broadcast_detailed_health_update(connection_id, health_data)
            
            return health_data
            
        except Exception as e:
            logger.error(f"Enhanced health monitoring failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # ================================================================================
    # Real-Time Broadcasting Methods
    # ================================================================================
    
    async def _broadcast_agent_connection_status(
        self,
        agent_id: str,
        connection: BridgeConnection,
        is_connected: bool
    ):
        """Broadcast agent connection status update."""
        try:
            if not self.realtime_hub:
                return
            
            status = AgentStatus.ONLINE if is_connected else AgentStatus.OFFLINE
            capabilities = self._get_agent_capabilities(agent_id, connection)
            
            agent_status = AgentStatusUpdate(
                agent_id=agent_id,
                status=status,
                capabilities=capabilities,
                current_tasks=list(self._get_agent_tasks(agent_id)),
                performance_metrics=self._get_agent_performance_metrics(agent_id),
                health_score=self._calculate_agent_health_score(agent_id),
                last_seen=datetime.utcnow(),
                additional_info={
                    "connection_id": connection.connection_id,
                    "connection_type": connection.connection_type,
                    "protocol": connection.protocol.value if connection.protocol else "unknown"
                }
            )
            
            await self.realtime_hub.broadcast_agent_status(agent_status)
            logger.debug(f"Agent status broadcast: {agent_id} -> {status.value}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast agent status: {e}")
    
    async def _broadcast_task_execution_start(
        self,
        task_id: str,
        agent_id: str,
        message: CLIMessage
    ):
        """Broadcast task execution start."""
        try:
            if not self.realtime_hub:
                return
            
            self._task_executions[task_id] = agent_id
            
            task_update = TaskExecutionUpdate(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskExecutionStatus.STARTED,
                progress_percentage=0.0,
                resource_usage=self._get_resource_usage_estimate(message),
                next_actions=self._extract_next_actions(message)
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            logger.debug(f"Task execution start broadcast: {task_id}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast task start: {e}")
    
    async def _broadcast_connection_health_update(
        self,
        connection: BridgeConnection,
        status: str
    ):
        """Broadcast connection health update."""
        try:
            if not self.realtime_hub:
                return
            
            health_update = HealthMonitoringUpdate(
                component_id=connection.connection_id,
                component_type="connection",
                health_score=connection.connection_quality if hasattr(connection, 'connection_quality') else 1.0,
                status=status,
                metrics=self._get_connection_health_metrics(connection),
                alerts=self._get_connection_alerts(connection),
                recommendations=self._get_connection_recommendations(connection),
                last_check=datetime.utcnow()
            )
            
            await self.realtime_hub.broadcast_health_monitoring_update(health_update)
            logger.debug(f"Connection health broadcast: {connection.connection_id} -> {status}")
            
        except Exception as e:
            logger.error(f"Failed to broadcast connection health: {e}")
    
    async def _broadcast_performance_update(
        self,
        connection_id: str,
        delivery_time_ms: float,
        success: bool
    ):
        """Broadcast performance metrics update."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="performance_metrics",
                data={
                    "connection_id": connection_id,
                    "delivery_time_ms": delivery_time_ms,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat(),
                    "performance_score": self._connection_metrics.get(connection_id, {}).get("performance_score", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast performance update: {e}")
    
    async def _broadcast_connection_failure(self, agent_id: str, error_message: str):
        """Broadcast connection failure."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="connection_failure",
                data={
                    "agent_id": agent_id,
                    "error_message": error_message,
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast connection failure: {e}")
    
    async def _handle_message_task_updates(self, message: CLIMessage, agent_id: str):
        """Handle task updates from received messages."""
        try:
            if not message.input_data or not self.realtime_hub:
                return
            
            task_id = message.input_data.get("task_id")
            if not task_id:
                return
            
            # Check for task completion indicators
            if message.input_data.get("task_completed"):
                await self._broadcast_task_completion(task_id, agent_id, message)
            elif message.input_data.get("task_progress"):
                await self._broadcast_task_progress(task_id, agent_id, message)
            elif message.input_data.get("task_error"):
                await self._broadcast_task_error(task_id, agent_id, message)
            
        except Exception as e:
            logger.error(f"Failed to handle task updates: {e}")
    
    async def _broadcast_task_completion(self, task_id: str, agent_id: str, message: CLIMessage):
        """Broadcast task completion."""
        try:
            task_update = TaskExecutionUpdate(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskExecutionStatus.COMPLETED,
                progress_percentage=100.0,
                execution_time_ms=message.input_data.get("execution_time_ms"),
                result_data=message.input_data.get("result_data", {}),
                resource_usage=message.input_data.get("resource_usage", {})
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
            # Clean up task tracking
            if task_id in self._task_executions:
                del self._task_executions[task_id]
            
        except Exception as e:
            logger.error(f"Failed to broadcast task completion: {e}")
    
    async def _broadcast_task_progress(self, task_id: str, agent_id: str, message: CLIMessage):
        """Broadcast task progress update."""
        try:
            task_update = TaskExecutionUpdate(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskExecutionStatus.IN_PROGRESS,
                progress_percentage=message.input_data.get("progress_percentage", 0.0),
                resource_usage=message.input_data.get("resource_usage", {}),
                next_actions=message.input_data.get("next_actions", [])
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast task progress: {e}")
    
    async def _broadcast_task_error(self, task_id: str, agent_id: str, message: CLIMessage):
        """Broadcast task error."""
        try:
            task_update = TaskExecutionUpdate(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskExecutionStatus.FAILED,
                progress_percentage=message.input_data.get("progress_percentage", 0.0),
                error_message=message.input_data.get("error_message", "Unknown error"),
                resource_usage=message.input_data.get("resource_usage", {})
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
            # Clean up task tracking
            if task_id in self._task_executions:
                del self._task_executions[task_id]
            
        except Exception as e:
            logger.error(f"Failed to broadcast task error: {e}")
    
    # ================================================================================
    # Enhanced Monitoring and Analytics
    # ================================================================================
    
    async def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced bridge metrics with real-time data."""
        try:
            base_metrics = self._metrics.copy()
            
            # Add enhanced metrics
            enhanced_metrics = {
                "agent_connections": {
                    "total_agents": len(self._agent_connections),
                    "online_agents": len([
                        aid for aid, cid in self._agent_connections.items()
                        if cid in self._connections and self._connections[cid].is_connected
                    ]),
                    "agent_details": {
                        agent_id: {
                            "connection_id": connection_id,
                            "status": "online" if connection_id in self._connections and 
                                     self._connections[connection_id].is_connected else "offline",
                            "metrics": self._connection_metrics.get(connection_id, {})
                        }
                        for agent_id, connection_id in self._agent_connections.items()
                    }
                },
                "task_executions": {
                    "active_tasks": len(self._task_executions),
                    "task_details": {
                        task_id: {
                            "agent_id": agent_id,
                            "status": "in_progress"
                        }
                        for task_id, agent_id in self._task_executions.items()
                    }
                },
                "performance_summary": {
                    "average_performance_score": self._calculate_average_performance_score(),
                    "total_errors": sum(
                        metrics.get("error_count", 0) 
                        for metrics in self._connection_metrics.values()
                    ),
                    "total_messages": sum(
                        metrics.get("total_messages", 0) 
                        for metrics in self._connection_metrics.values()
                    )
                }
            }
            
            return {**base_metrics, **enhanced_metrics}
            
        except Exception as e:
            logger.error(f"Failed to get enhanced metrics: {e}")
            return self._metrics.copy()
    
    async def get_agent_status_summary(self) -> Dict[str, Any]:
        """Get summary of all agent statuses."""
        try:
            agent_summary = {}
            
            for agent_id, connection_id in self._agent_connections.items():
                connection = self._connections.get(connection_id)
                metrics = self._connection_metrics.get(connection_id, {})
                
                agent_summary[agent_id] = {
                    "status": "online" if connection and connection.is_connected else "offline",
                    "connection_id": connection_id,
                    "connection_type": connection.connection_type if connection else "unknown",
                    "health_score": self._calculate_agent_health_score(agent_id),
                    "performance_score": metrics.get("performance_score", 0.0),
                    "total_messages": metrics.get("total_messages", 0),
                    "error_count": metrics.get("error_count", 0),
                    "last_activity": metrics.get("last_activity"),
                    "uptime_seconds": (
                        datetime.utcnow() - metrics["established_at"]
                    ).total_seconds() if metrics.get("established_at") else 0
                }
            
            return agent_summary
            
        except Exception as e:
            logger.error(f"Failed to get agent status summary: {e}")
            return {}
    
    # ================================================================================
    # Real-Time Background Services
    # ================================================================================
    
    async def _start_realtime_services(self):
        """Start real-time background services."""
        try:
            # Start status broadcasting service
            status_task = asyncio.create_task(self._status_broadcasting_loop())
            self._background_tasks.add(status_task)
            
            # Start health monitoring service
            health_task = asyncio.create_task(self._health_monitoring_loop())
            self._background_tasks.add(health_task)
            
            # Start performance metrics service
            metrics_task = asyncio.create_task(self._performance_metrics_loop())
            self._background_tasks.add(metrics_task)
            
            logger.info("Real-time services started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time services: {e}")
    
    async def _status_broadcasting_loop(self):
        """Background service for periodic status broadcasting."""
        while True:
            try:
                await asyncio.sleep(self._config["realtime_status_interval"])
                
                if self.realtime_hub:
                    # Broadcast status for all connected agents
                    for agent_id, connection_id in self._agent_connections.items():
                        connection = self._connections.get(connection_id)
                        if connection:
                            await self._broadcast_agent_connection_status(
                                agent_id, connection, connection.is_connected
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status broadcasting loop: {e}")
    
    async def _health_monitoring_loop(self):
        """Background service for health monitoring and broadcasting."""
        while True:
            try:
                await asyncio.sleep(self._config["health_broadcast_interval"])
                
                if self.realtime_hub:
                    # Monitor and broadcast health for all connections
                    for connection_id, connection in self._connections.items():
                        health_status = await self._assess_connection_health(connection)
                        await self._broadcast_connection_health_update(connection, health_status)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _performance_metrics_loop(self):
        """Background service for performance metrics collection and broadcasting."""
        while True:
            try:
                await asyncio.sleep(self._config["performance_metrics_interval"])
                
                if self.realtime_hub:
                    # Collect and broadcast performance metrics
                    enhanced_metrics = await self.get_enhanced_metrics()
                    
                    await self.realtime_hub.broadcast_dashboard_update(
                        update_type="bridge_metrics",
                        data=enhanced_metrics
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance metrics loop: {e}")
    
    # ================================================================================
    # Helper Methods for Real-Time Integration
    # ================================================================================
    
    def _get_agent_capabilities(self, agent_id: str, connection: BridgeConnection) -> List[str]:
        """Get agent capabilities based on connection and configuration."""
        capabilities = []
        
        # Add capabilities based on connection type
        if connection.connection_type == "websocket":
            capabilities.extend(["real_time_communication", "bidirectional_messaging"])
        elif connection.connection_type == "redis":
            capabilities.extend(["message_queuing", "pub_sub_messaging"])
        elif connection.connection_type == "http":
            capabilities.extend(["request_response", "rest_api"])
        
        # Add protocol-specific capabilities
        if hasattr(connection, 'protocol') and connection.protocol:
            if connection.protocol == CLIProtocol.CLAUDE_CODE:
                capabilities.extend(["code_generation", "analysis", "documentation"])
            elif connection.protocol == CLIProtocol.CURSOR:
                capabilities.extend(["code_editing", "file_management"])
            elif connection.protocol == CLIProtocol.GEMINI_CLI:
                capabilities.extend(["ai_assistance", "text_generation"])
        
        return capabilities
    
    def _get_agent_tasks(self, agent_id: str) -> Set[str]:
        """Get current tasks for an agent."""
        return {
            task_id for task_id, task_agent_id in self._task_executions.items()
            if task_agent_id == agent_id
        }
    
    def _get_agent_performance_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent."""
        connection_id = self._agent_connections.get(agent_id)
        if not connection_id or connection_id not in self._connection_metrics:
            return {}
        
        metrics = self._connection_metrics[connection_id]
        return {
            "total_messages": metrics.get("total_messages", 0),
            "error_count": metrics.get("error_count", 0),
            "error_rate": metrics.get("error_count", 0) / max(1, metrics.get("total_messages", 1)),
            "performance_score": metrics.get("performance_score", 0.0),
            "uptime_seconds": (
                datetime.utcnow() - metrics["established_at"]
            ).total_seconds() if metrics.get("established_at") else 0
        }
    
    def _calculate_agent_health_score(self, agent_id: str) -> float:
        """Calculate health score for an agent."""
        connection_id = self._agent_connections.get(agent_id)
        if not connection_id or connection_id not in self._connection_metrics:
            return 0.0
        
        metrics = self._connection_metrics[connection_id]
        connection = self._connections.get(connection_id)
        
        # Base score from connection status
        base_score = 1.0 if connection and connection.is_connected else 0.0
        
        # Adjust for performance
        performance_score = metrics.get("performance_score", 1.0)
        
        # Adjust for error rate
        error_rate = metrics.get("error_count", 0) / max(1, metrics.get("total_messages", 1))
        error_penalty = min(0.5, error_rate * 2)
        
        # Adjust for activity recency
        last_activity = metrics.get("last_activity")
        activity_score = 1.0
        if last_activity:
            inactive_time = (datetime.utcnow() - last_activity).total_seconds()
            if inactive_time > 300:  # 5 minutes
                activity_score = max(0.5, 1.0 - (inactive_time - 300) / 3600)  # Decay over 1 hour
        
        final_score = base_score * performance_score * activity_score - error_penalty
        return max(0.0, min(1.0, final_score))
    
    def _get_resource_usage_estimate(self, message: CLIMessage) -> Dict[str, Any]:
        """Estimate resource usage for a message/task."""
        # Basic estimation based on message size and complexity
        message_size = len(str(message.input_data)) if message.input_data else 0
        arg_count = len(message.cli_args) if message.cli_args else 0
        
        return {
            "estimated_memory_mb": max(1, message_size / 1024 / 1024),
            "estimated_cpu_percentage": min(100, max(1, arg_count * 5)),
            "estimated_duration_seconds": max(1, message_size / 10000)  # Very rough estimate
        }
    
    def _extract_next_actions(self, message: CLIMessage) -> List[str]:
        """Extract potential next actions from a message."""
        actions = []
        
        if message.cli_command:
            actions.append(f"Execute {message.cli_command}")
        
        if message.cli_args:
            actions.extend([f"Process argument: {arg}" for arg in message.cli_args[:3]])
        
        if message.input_data:
            if "files" in message.input_data:
                actions.append("Process input files")
            if "dependencies" in message.input_data:
                actions.append("Resolve dependencies")
        
        return actions[:5]  # Limit to 5 actions
    
    def _get_connection_health_metrics(self, connection: BridgeConnection) -> Dict[str, Any]:
        """Get health metrics for a connection."""
        connection_id = connection.connection_id
        metrics = self._connection_metrics.get(connection_id, {})
        
        return {
            "is_connected": connection.is_connected if hasattr(connection, 'is_connected') else False,
            "connection_quality": getattr(connection, 'connection_quality', 1.0),
            "total_messages": metrics.get("total_messages", 0),
            "error_count": metrics.get("error_count", 0),
            "performance_score": metrics.get("performance_score", 1.0),
            "last_activity": metrics.get("last_activity"),
            "uptime_seconds": (
                datetime.utcnow() - metrics["established_at"]
            ).total_seconds() if metrics.get("established_at") else 0
        }
    
    def _get_connection_alerts(self, connection: BridgeConnection) -> List[str]:
        """Get alerts for a connection."""
        alerts = []
        connection_id = connection.connection_id
        metrics = self._connection_metrics.get(connection_id, {})
        
        # Check for high error rate
        error_rate = metrics.get("error_count", 0) / max(1, metrics.get("total_messages", 1))
        if error_rate > 0.1:
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        # Check for low performance
        performance_score = metrics.get("performance_score", 1.0)
        if performance_score < 0.5:
            alerts.append(f"Low performance score: {performance_score:.2f}")
        
        # Check for connection issues
        if hasattr(connection, 'is_connected') and not connection.is_connected:
            alerts.append("Connection is down")
        
        # Check for inactivity
        last_activity = metrics.get("last_activity")
        if last_activity:
            inactive_time = (datetime.utcnow() - last_activity).total_seconds()
            if inactive_time > 600:  # 10 minutes
                alerts.append(f"Inactive for {inactive_time/60:.1f} minutes")
        
        return alerts
    
    def _get_connection_recommendations(self, connection: BridgeConnection) -> List[str]:
        """Get recommendations for a connection."""
        recommendations = []
        alerts = self._get_connection_alerts(connection)
        
        if "High error rate" in str(alerts):
            recommendations.append("Consider restarting the connection")
            recommendations.append("Check network connectivity")
        
        if "Low performance score" in str(alerts):
            recommendations.append("Monitor resource usage")
            recommendations.append("Consider connection pooling optimization")
        
        if "Connection is down" in str(alerts):
            recommendations.append("Enable auto-reconnection")
            recommendations.append("Check agent availability")
        
        if "Inactive" in str(alerts):
            recommendations.append("Send heartbeat message")
            recommendations.append("Verify agent responsiveness")
        
        return recommendations
    
    def _calculate_average_performance_score(self) -> float:
        """Calculate average performance score across all connections."""
        if not self._connection_metrics:
            return 0.0
        
        scores = [
            metrics.get("performance_score", 0.0) 
            for metrics in self._connection_metrics.values()
        ]
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def _assess_connection_health(self, connection: BridgeConnection) -> str:
        """Assess overall health status of a connection."""
        connection_id = connection.connection_id
        metrics = self._connection_metrics.get(connection_id, {})
        
        # Check connection status
        if not (hasattr(connection, 'is_connected') and connection.is_connected):
            return "error"
        
        # Check error rate
        error_rate = metrics.get("error_count", 0) / max(1, metrics.get("total_messages", 1))
        if error_rate > 0.2:
            return "unhealthy"
        elif error_rate > 0.1:
            return "degraded"
        
        # Check performance score
        performance_score = metrics.get("performance_score", 1.0)
        if performance_score < 0.3:
            return "unhealthy"
        elif performance_score < 0.7:
            return "degraded"
        
        # Check activity
        last_activity = metrics.get("last_activity")
        if last_activity:
            inactive_time = (datetime.utcnow() - last_activity).total_seconds()
            if inactive_time > 1800:  # 30 minutes
                return "degraded"
        
        return "healthy"
    
    async def _broadcast_detailed_health_update(self, connection_id: str, health_data: Dict[str, Any]):
        """Broadcast detailed health update."""
        try:
            if not self.realtime_hub:
                return
            
            health_update = HealthMonitoringUpdate(
                component_id=connection_id,
                component_type="enhanced_connection",
                health_score=health_data.get("quality_score", 0.0),
                status=health_data.get("status", "unknown"),
                metrics=health_data,
                alerts=self._get_connection_alerts(self._connections.get(connection_id)) if connection_id in self._connections else [],
                recommendations=self._get_connection_recommendations(self._connections.get(connection_id)) if connection_id in self._connections else [],
                last_check=datetime.utcnow()
            )
            
            await self.realtime_hub.broadcast_health_monitoring_update(health_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast detailed health update: {e}")
    
    async def _broadcast_message_received(self, connection_id: str, message: CLIMessage):
        """Broadcast message received event."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="message_received",
                data={
                    "connection_id": connection_id,
                    "message_id": message.cli_message_id,
                    "command": message.cli_command,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast message received: {e}")
    
    async def _broadcast_message_error(self, connection_id: str, message: CLIMessage, error: str):
        """Broadcast message error event."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="message_error",
                data={
                    "connection_id": connection_id,
                    "message_id": message.cli_message_id,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "high"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast message error: {e}")
    
    async def _broadcast_listening_error(self, connection_id: str, error: str):
        """Broadcast listening error event."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="listening_error",
                data={
                    "connection_id": connection_id,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "medium"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast listening error: {e}")
    
    async def shutdown(self):
        """Enhanced shutdown with real-time cleanup."""
        try:
            logger.info("Shutting down EnhancedCommunicationBridge...")
            
            # Broadcast shutdown notifications
            if self.enable_realtime and self.realtime_hub:
                for agent_id in self._agent_connections.keys():
                    await self._broadcast_agent_connection_status(
                        agent_id, 
                        BridgeConnection(connection_id="shutdown", protocol=CLIProtocol.UNIVERSAL, endpoint=""), 
                        False
                    )
            
            # Cancel real-time background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Call parent shutdown
            if hasattr(super(), 'shutdown'):
                await super().shutdown()
            
            logger.info("EnhancedCommunicationBridge shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during enhanced bridge shutdown: {e}")

# ================================================================================
# Factory Functions
# ================================================================================

def create_enhanced_bridge(
    realtime_hub: Optional[RealTimeCommunicationHub] = None,
    enable_realtime: bool = True
) -> EnhancedCommunicationBridge:
    """Create an enhanced communication bridge with real-time capabilities."""
    return EnhancedCommunicationBridge(
        realtime_hub=realtime_hub,
        enable_realtime=enable_realtime
    )

async def create_integrated_communication_system(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    websocket_host: str = "localhost",
    websocket_port: int = 8765,
    **kwargs
) -> tuple[RealTimeCommunicationHub, EnhancedCommunicationBridge]:
    """
    Create a fully integrated communication system with real-time hub and enhanced bridge.
    
    Returns:
        tuple: (RealTimeCommunicationHub, EnhancedCommunicationBridge)
    """
    from .realtime_communication_hub import create_realtime_hub
    
    # Create real-time hub
    hub = create_realtime_hub(
        redis_host=redis_host,
        redis_port=redis_port,
        websocket_host=websocket_host,
        websocket_port=websocket_port,
        **kwargs
    )
    
    # Initialize hub
    await hub.initialize()
    
    # Create enhanced bridge
    bridge = create_enhanced_bridge(realtime_hub=hub, enable_realtime=True)
    
    # Initialize bridge with real-time integration
    await bridge.initialize_with_realtime(hub)
    
    return hub, bridge
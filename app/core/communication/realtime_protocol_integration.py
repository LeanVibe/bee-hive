"""
Real-Time Protocol Integration for Multi-CLI Coordination

This module provides seamless integration between the existing multi_cli_protocol.py
and the new real-time communication capabilities, enabling sophisticated
heterogeneous agent coordination with real-time status updates.

Features:
- Real-time multi-CLI protocol coordination
- Agent handoff with live status updates
- Context preservation with real-time progress tracking
- Message translation with delivery notifications
- Performance monitoring across all CLI protocols
- Health monitoring with real-time alerts
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable

from .multi_cli_protocol import MultiCLIProtocol, ProductionMultiCLIProtocol
from .realtime_communication_hub import (
    RealTimeCommunicationHub,
    AgentStatusUpdate,
    TaskExecutionUpdate,
    HealthMonitoringUpdate,
    AgentStatus,
    TaskExecutionStatus,
    MessagePriority,
    NotificationType,
    RealTimeMessage
)
from .enhanced_communication_bridge import EnhancedCommunicationBridge
from .protocol_models import (
    UniversalMessage,
    CLIMessage,
    ContextPackage,
    HandoffRequest,
    ProtocolConfig,
    MessageRoute,
    BridgeConnection,
    CLIProtocol,
    MessageType,
    HandoffStatus
)
from ..agents.universal_agent_interface import AgentType

logger = logging.getLogger(__name__)

# ================================================================================
# Real-Time Multi-CLI Protocol
# ================================================================================

class RealTimeMultiCLIProtocol(ProductionMultiCLIProtocol):
    """
    Real-time enhanced Multi-CLI Protocol with live coordination capabilities.
    
    Extends ProductionMultiCLIProtocol with:
    - Real-time agent status broadcasting during handoffs
    - Live task execution tracking across CLI protocols
    - Real-time health monitoring and alerts
    - Performance metrics streaming
    - Context transfer progress updates
    - Message delivery confirmations with notifications
    """
    
    def __init__(
        self, 
        protocol_id: str,
        realtime_hub: Optional[RealTimeCommunicationHub] = None,
        enhanced_bridge: Optional[EnhancedCommunicationBridge] = None
    ):
        """Initialize real-time multi-CLI protocol."""
        super().__init__(protocol_id)
        
        # Real-time integration
        self.realtime_hub = realtime_hub
        self.enhanced_bridge = enhanced_bridge
        
        # Enhanced tracking
        self._active_handoffs_rt: Dict[str, Dict[str, Any]] = {}
        self._message_delivery_tracking: Dict[str, Dict[str, Any]] = {}
        self._protocol_performance_rt: Dict[CLIProtocol, Dict[str, Any]] = {}
        self._agent_coordination_state: Dict[str, Dict[str, Any]] = {}
        
        # Real-time callbacks
        self._handoff_progress_callbacks: List[Callable] = []
        self._message_delivery_callbacks: List[Callable] = []
        self._protocol_health_callbacks: List[Callable] = []
        
        # Enhanced configuration
        self._realtime_config = {
            "enable_handoff_progress": True,
            "enable_message_tracking": True,
            "enable_protocol_health_monitoring": True,
            "handoff_progress_interval": 5,  # seconds
            "message_delivery_timeout": 30,  # seconds
            "protocol_health_check_interval": 60,  # seconds
            "coordination_state_sync_interval": 30,  # seconds
        }
        
        logger.info(f"RealTimeMultiCLIProtocol initialized: {protocol_id}")
    
    async def initialize_realtime(
        self,
        realtime_hub: RealTimeCommunicationHub,
        enhanced_bridge: Optional[EnhancedCommunicationBridge] = None
    ) -> bool:
        """Initialize real-time capabilities."""
        try:
            self.realtime_hub = realtime_hub
            self.enhanced_bridge = enhanced_bridge
            
            # Initialize parent protocol
            await super().initialize({})
            
            # Start real-time background services
            await self._start_realtime_background_services()
            
            logger.info("Real-time multi-CLI protocol initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize real-time protocol: {e}")
            return False
    
    # ================================================================================
    # Enhanced Core Communication Methods
    # ================================================================================
    
    async def send_message(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol,
        route_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send message with real-time delivery tracking and notifications.
        """
        start_time = time.time()
        delivery_id = str(uuid.uuid4())
        
        try:
            # Initialize delivery tracking
            self._message_delivery_tracking[message.message_id] = {
                "delivery_id": delivery_id,
                "start_time": start_time,
                "target_protocol": target_protocol,
                "status": "sending",
                "attempts": 0,
                "last_attempt": datetime.utcnow()
            }
            
            # Broadcast message sending notification
            if self.realtime_hub:
                await self._broadcast_message_sending(message, target_protocol, delivery_id)
            
            # Call parent implementation with enhanced error handling
            success = await super().send_message(message, target_protocol, route_config)
            
            # Update delivery tracking
            delivery_time = (time.time() - start_time) * 1000
            tracking = self._message_delivery_tracking[message.message_id]
            tracking.update({
                "status": "delivered" if success else "failed",
                "delivery_time_ms": delivery_time,
                "completed_at": datetime.utcnow()
            })
            
            # Broadcast delivery result
            if self.realtime_hub:
                await self._broadcast_message_delivery_result(
                    message, target_protocol, success, delivery_time, delivery_id
                )
            
            # Update protocol performance metrics
            await self._update_protocol_performance_realtime(
                target_protocol, success, delivery_time
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Real-time message send failed: {e}")
            
            # Update tracking with error
            if message.message_id in self._message_delivery_tracking:
                self._message_delivery_tracking[message.message_id].update({
                    "status": "error",
                    "error": str(e),
                    "completed_at": datetime.utcnow()
                })
            
            # Broadcast error
            if self.realtime_hub:
                await self._broadcast_message_error(message, target_protocol, str(e), delivery_id)
            
            return False
        finally:
            # Cleanup old tracking data
            await self._cleanup_delivery_tracking()
    
    async def receive_message(
        self,
        source_protocol: CLIProtocol,
        timeout_seconds: float = 30.0
    ) -> Optional[UniversalMessage]:
        """
        Receive message with real-time processing notifications.
        """
        start_time = time.time()
        
        try:
            # Call parent implementation
            message = await super().receive_message(source_protocol, timeout_seconds)
            
            if message:
                # Broadcast message received
                if self.realtime_hub:
                    await self._broadcast_message_received(message, source_protocol)
                
                # Update protocol performance
                processing_time = (time.time() - start_time) * 1000
                await self._update_protocol_performance_realtime(
                    source_protocol, True, processing_time, "receive"
                )
            
            return message
            
        except Exception as e:
            logger.error(f"Real-time message receive failed: {e}")
            
            # Broadcast receive error
            if self.realtime_hub:
                await self._broadcast_receive_error(source_protocol, str(e))
            
            return None
    
    async def initiate_handoff(
        self,
        handoff_request: HandoffRequest
    ) -> HandoffStatus:
        """
        Initiate agent handoff with real-time progress tracking.
        """
        handoff_id = handoff_request.handoff_id
        start_time = time.time()
        
        try:
            logger.info(f"Initiating real-time handoff: {handoff_id}")
            
            # Initialize real-time handoff tracking
            self._active_handoffs_rt[handoff_id] = {
                "start_time": start_time,
                "status": "initializing",
                "progress_percentage": 0.0,
                "current_phase": "initialization",
                "source_agent": handoff_request.source_agent_id,
                "target_agent_type": handoff_request.target_agent_type,
                "last_update": datetime.utcnow()
            }
            
            # Broadcast handoff initiation
            if self.realtime_hub:
                await self._broadcast_handoff_initiation(handoff_request)
            
            # Phase 1: Context Packaging (0-25%)
            await self._update_handoff_progress(handoff_id, 10.0, "context_packaging")
            
            if not handoff_request.context_package:
                context_package = await self.package_context(
                    handoff_request.context_package.execution_context if handoff_request.context_package else {},
                    handoff_request.target_agent_type
                )
                handoff_request.context_package = context_package
            
            await self._update_handoff_progress(handoff_id, 25.0, "context_packaged")
            handoff_request.status = HandoffStatus.CONTEXT_PACKAGED
            
            # Phase 2: Agent Selection (25-50%)
            await self._update_handoff_progress(handoff_id, 30.0, "agent_selection")
            
            target_agent = await self._select_optimal_agent(
                handoff_request.target_agent_type,
                handoff_request.required_capabilities,
                handoff_request.preferred_agents,
                handoff_request.excluded_agents
            )
            
            if not target_agent:
                await self._update_handoff_progress(handoff_id, 100.0, "failed", "No suitable agent found")
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
                return HandoffStatus.HANDOFF_FAILED
            
            handoff_request.target_agent_id = target_agent
            await self._update_handoff_progress(handoff_id, 50.0, "agent_selected")
            handoff_request.status = HandoffStatus.AGENT_SELECTED
            
            # Phase 3: Context Transfer (50-75%)
            await self._update_handoff_progress(handoff_id, 60.0, "context_transfer")
            
            transfer_success = await self._transfer_context_with_progress(
                handoff_request.context_package,
                target_agent,
                handoff_request.target_agent_type,
                handoff_id
            )
            
            if not transfer_success:
                await self._update_handoff_progress(handoff_id, 100.0, "failed", "Context transfer failed")
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
                return HandoffStatus.HANDOFF_FAILED
            
            await self._update_handoff_progress(handoff_id, 75.0, "context_transferred")
            handoff_request.status = HandoffStatus.CONTEXT_TRANSFERRED
            
            # Phase 4: Handoff Confirmation (75-100%)
            await self._update_handoff_progress(handoff_id, 85.0, "confirmation")
            
            confirmation = await self._confirm_handoff_with_monitoring(
                target_agent,
                handoff_request.context_package,
                handoff_request.target_agent_type,
                handoff_id
            )
            
            if confirmation:
                await self._update_handoff_progress(handoff_id, 100.0, "completed")
                handoff_request.status = HandoffStatus.HANDOFF_COMPLETED
                handoff_request.completed_at = datetime.utcnow()
                
                # Broadcast handoff completion
                if self.realtime_hub:
                    await self._broadcast_handoff_completion(handoff_request)
                
                logger.info(f"Real-time handoff completed: {handoff_id} -> {target_agent}")
            else:
                await self._update_handoff_progress(handoff_id, 100.0, "failed", "Confirmation failed")
                handoff_request.status = HandoffStatus.HANDOFF_FAILED
            
            return handoff_request.status
            
        except Exception as e:
            logger.error(f"Real-time handoff error {handoff_id}: {e}")
            await self._update_handoff_progress(handoff_id, 100.0, "failed", str(e))
            handoff_request.status = HandoffStatus.HANDOFF_FAILED
            return HandoffStatus.HANDOFF_FAILED
        finally:
            # Cleanup handoff tracking
            if handoff_id in self._active_handoffs_rt:
                # Keep completed handoffs for a while for metrics
                if self._active_handoffs_rt[handoff_id]["status"] != "completed":
                    del self._active_handoffs_rt[handoff_id]
    
    # ================================================================================
    # Real-Time Broadcasting Methods
    # ================================================================================
    
    async def _broadcast_message_sending(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol,
        delivery_id: str
    ):
        """Broadcast message sending notification."""
        try:
            if not self.realtime_hub:
                return
            
            task_update = TaskExecutionUpdate(
                task_id=f"msg_{message.message_id}",
                agent_id=message.source_agent_id,
                status=TaskExecutionStatus.STARTED,
                progress_percentage=0.0,
                resource_usage={
                    "target_protocol": target_protocol.value,
                    "message_size": len(str(message.payload)) if message.payload else 0
                },
                next_actions=[f"Send to {target_protocol.value}"]
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast message sending: {e}")
    
    async def _broadcast_message_delivery_result(
        self,
        message: UniversalMessage,
        target_protocol: CLIProtocol,
        success: bool,
        delivery_time_ms: float,
        delivery_id: str
    ):
        """Broadcast message delivery result."""
        try:
            if not self.realtime_hub:
                return
            
            status = TaskExecutionStatus.COMPLETED if success else TaskExecutionStatus.FAILED
            
            task_update = TaskExecutionUpdate(
                task_id=f"msg_{message.message_id}",
                agent_id=message.source_agent_id,
                status=status,
                progress_percentage=100.0,
                execution_time_ms=delivery_time_ms,
                result_data={
                    "delivery_success": success,
                    "target_protocol": target_protocol.value,
                    "delivery_id": delivery_id
                },
                resource_usage={
                    "delivery_time_ms": delivery_time_ms,
                    "protocol_overhead": 0  # Would calculate actual overhead
                }
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
            # Also broadcast dashboard update
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="message_delivery",
                data={
                    "message_id": message.message_id,
                    "target_protocol": target_protocol.value,
                    "success": success,
                    "delivery_time_ms": delivery_time_ms,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast delivery result: {e}")
    
    async def _broadcast_handoff_initiation(self, handoff_request: HandoffRequest):
        """Broadcast handoff initiation."""
        try:
            if not self.realtime_hub:
                return
            
            task_update = TaskExecutionUpdate(
                task_id=f"handoff_{handoff_request.handoff_id}",
                agent_id=handoff_request.source_agent_id,
                status=TaskExecutionStatus.STARTED,
                progress_percentage=0.0,
                resource_usage={
                    "target_agent_type": handoff_request.target_agent_type.value if handoff_request.target_agent_type else "unknown",
                    "required_capabilities": handoff_request.required_capabilities
                },
                next_actions=["Package context", "Select target agent", "Transfer context", "Confirm handoff"]
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast handoff initiation: {e}")
    
    async def _broadcast_handoff_completion(self, handoff_request: HandoffRequest):
        """Broadcast handoff completion."""
        try:
            if not self.realtime_hub:
                return
            
            task_update = TaskExecutionUpdate(
                task_id=f"handoff_{handoff_request.handoff_id}",
                agent_id=handoff_request.source_agent_id,
                status=TaskExecutionStatus.COMPLETED,
                progress_percentage=100.0,
                execution_time_ms=(
                    (handoff_request.completed_at - handoff_request.created_at).total_seconds() * 1000
                    if handoff_request.completed_at and handoff_request.created_at else None
                ),
                result_data={
                    "target_agent_id": handoff_request.target_agent_id,
                    "context_package_size": handoff_request.context_package.package_size_bytes if handoff_request.context_package else 0
                }
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
            # Broadcast agent status updates
            if handoff_request.target_agent_id:
                # Update target agent status
                target_status = AgentStatusUpdate(
                    agent_id=handoff_request.target_agent_id,
                    status=AgentStatus.BUSY,
                    current_tasks=[handoff_request.handoff_id],
                    health_score=1.0,
                    last_seen=datetime.utcnow(),
                    additional_info={
                        "handoff_received": True,
                        "source_agent": handoff_request.source_agent_id
                    }
                )
                await self.realtime_hub.broadcast_agent_status(target_status)
            
            # Update source agent status
            source_status = AgentStatusUpdate(
                agent_id=handoff_request.source_agent_id,
                status=AgentStatus.IDLE,
                current_tasks=[],
                health_score=1.0,
                last_seen=datetime.utcnow(),
                additional_info={
                    "handoff_completed": True,
                    "target_agent": handoff_request.target_agent_id
                }
            )
            await self.realtime_hub.broadcast_agent_status(source_status)
            
        except Exception as e:
            logger.error(f"Failed to broadcast handoff completion: {e}")
    
    async def _update_handoff_progress(
        self,
        handoff_id: str,
        progress_percentage: float,
        phase: str,
        error_message: Optional[str] = None
    ):
        """Update and broadcast handoff progress."""
        try:
            if handoff_id not in self._active_handoffs_rt:
                return
            
            # Update tracking
            tracking = self._active_handoffs_rt[handoff_id]
            tracking.update({
                "progress_percentage": progress_percentage,
                "current_phase": phase,
                "last_update": datetime.utcnow()
            })
            
            if error_message:
                tracking["error_message"] = error_message
                tracking["status"] = "failed"
            elif progress_percentage >= 100.0:
                tracking["status"] = "completed"
            else:
                tracking["status"] = "in_progress"
            
            # Broadcast progress update
            if self.realtime_hub:
                task_update = TaskExecutionUpdate(
                    task_id=f"handoff_{handoff_id}",
                    agent_id=tracking["source_agent"],
                    status=TaskExecutionStatus.IN_PROGRESS if progress_percentage < 100 else (
                        TaskExecutionStatus.COMPLETED if not error_message else TaskExecutionStatus.FAILED
                    ),
                    progress_percentage=progress_percentage,
                    error_message=error_message,
                    resource_usage={
                        "current_phase": phase,
                        "elapsed_time_ms": (time.time() - tracking["start_time"]) * 1000
                    },
                    next_actions=self._get_next_handoff_actions(phase, progress_percentage)
                )
                
                await self.realtime_hub.broadcast_task_execution_update(task_update)
            
        except Exception as e:
            logger.error(f"Failed to update handoff progress: {e}")
    
    # ================================================================================
    # Enhanced Protocol Performance Monitoring
    # ================================================================================
    
    async def _update_protocol_performance_realtime(
        self,
        protocol: CLIProtocol,
        success: bool,
        operation_time_ms: float,
        operation_type: str = "send"
    ):
        """Update protocol performance metrics with real-time broadcasting."""
        try:
            if protocol not in self._protocol_performance_rt:
                self._protocol_performance_rt[protocol] = {
                    "total_operations": 0,
                    "successful_operations": 0,
                    "total_time_ms": 0.0,
                    "error_count": 0,
                    "last_operation": datetime.utcnow(),
                    "average_time_ms": 0.0,
                    "success_rate": 1.0
                }
            
            perf = self._protocol_performance_rt[protocol]
            perf["total_operations"] += 1
            perf["total_time_ms"] += operation_time_ms
            perf["last_operation"] = datetime.utcnow()
            
            if success:
                perf["successful_operations"] += 1
            else:
                perf["error_count"] += 1
            
            # Calculate derived metrics
            perf["average_time_ms"] = perf["total_time_ms"] / perf["total_operations"]
            perf["success_rate"] = perf["successful_operations"] / perf["total_operations"]
            
            # Broadcast performance update if significant change
            if (perf["total_operations"] % 10 == 0 or not success) and self.realtime_hub:
                await self._broadcast_protocol_performance_update(protocol, perf, operation_type)
            
        except Exception as e:
            logger.error(f"Failed to update protocol performance: {e}")
    
    async def _broadcast_protocol_performance_update(
        self,
        protocol: CLIProtocol,
        performance_data: Dict[str, Any],
        operation_type: str
    ):
        """Broadcast protocol performance update."""
        try:
            if not self.realtime_hub:
                return
            
            # Determine health status based on performance
            health_status = "healthy"
            if performance_data["success_rate"] < 0.8:
                health_status = "degraded"
            if performance_data["success_rate"] < 0.5:
                health_status = "unhealthy"
            
            health_update = HealthMonitoringUpdate(
                component_id=f"protocol_{protocol.value}",
                component_type="cli_protocol",
                health_score=performance_data["success_rate"],
                status=health_status,
                metrics={
                    "operation_type": operation_type,
                    "total_operations": performance_data["total_operations"],
                    "success_rate": performance_data["success_rate"],
                    "average_time_ms": performance_data["average_time_ms"],
                    "error_count": performance_data["error_count"]
                },
                alerts=self._get_protocol_alerts(protocol, performance_data),
                recommendations=self._get_protocol_recommendations(protocol, performance_data),
                last_check=datetime.utcnow()
            )
            
            await self.realtime_hub.broadcast_health_monitoring_update(health_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast protocol performance: {e}")
    
    # ================================================================================
    # Enhanced Context Transfer with Progress Tracking
    # ================================================================================
    
    async def _transfer_context_with_progress(
        self,
        context_package: ContextPackage,
        target_agent: str,
        agent_type: Optional[AgentType],
        handoff_id: str
    ) -> bool:
        """Transfer context with real-time progress updates."""
        try:
            # Simulate context transfer with progress updates
            transfer_phases = [
                ("validation", 60.0),
                ("compression", 65.0),
                ("transmission", 70.0),
                ("verification", 75.0)
            ]
            
            for phase, progress in transfer_phases:
                await self._update_handoff_progress(handoff_id, progress, f"context_transfer_{phase}")
                await asyncio.sleep(0.1)  # Simulate processing time
            
            # Call parent implementation (or simulate for now)
            success = await self._transfer_context(context_package, target_agent, agent_type)
            
            return success
            
        except Exception as e:
            logger.error(f"Context transfer with progress failed: {e}")
            return False
    
    async def _confirm_handoff_with_monitoring(
        self,
        target_agent: str,
        context_package: ContextPackage,
        agent_type: Optional[AgentType],
        handoff_id: str
    ) -> bool:
        """Confirm handoff with real-time monitoring."""
        try:
            # Update progress for confirmation phases
            confirmation_phases = [
                ("agent_ping", 85.0),
                ("context_verification", 90.0),
                ("readiness_check", 95.0),
                ("final_confirmation", 100.0)
            ]
            
            for phase, progress in confirmation_phases:
                await self._update_handoff_progress(handoff_id, progress, f"confirmation_{phase}")
                await asyncio.sleep(0.1)  # Simulate processing time
            
            # Call parent implementation (or simulate for now)
            success = await self._confirm_handoff(target_agent, context_package, agent_type)
            
            return success
            
        except Exception as e:
            logger.error(f"Handoff confirmation with monitoring failed: {e}")
            return False
    
    # ================================================================================
    # Real-Time Background Services
    # ================================================================================
    
    async def _start_realtime_background_services(self):
        """Start real-time background monitoring services."""
        try:
            # Start handoff progress monitoring
            if self._realtime_config["enable_handoff_progress"]:
                handoff_task = asyncio.create_task(self._handoff_progress_monitoring_loop())
                self._background_tasks.add(handoff_task)
            
            # Start protocol health monitoring
            if self._realtime_config["enable_protocol_health_monitoring"]:
                health_task = asyncio.create_task(self._protocol_health_monitoring_loop())
                self._background_tasks.add(health_task)
            
            # Start coordination state synchronization
            coordination_task = asyncio.create_task(self._coordination_state_sync_loop())
            self._background_tasks.add(coordination_task)
            
            logger.info("Real-time background services started")
            
        except Exception as e:
            logger.error(f"Failed to start real-time background services: {e}")
    
    async def _handoff_progress_monitoring_loop(self):
        """Background monitoring for handoff progress."""
        while True:
            try:
                await asyncio.sleep(self._realtime_config["handoff_progress_interval"])
                
                current_time = datetime.utcnow()
                
                # Check for stalled handoffs
                for handoff_id, tracking in list(self._active_handoffs_rt.items()):
                    last_update = tracking.get("last_update", current_time)
                    time_since_update = (current_time - last_update).total_seconds()
                    
                    # Alert on stalled handoffs (no progress for 30 seconds)
                    if time_since_update > 30 and tracking["status"] == "in_progress":
                        await self._alert_stalled_handoff(handoff_id, tracking, time_since_update)
                    
                    # Cleanup completed handoffs after 5 minutes
                    if (tracking["status"] in ["completed", "failed"] and 
                        time_since_update > 300):
                        del self._active_handoffs_rt[handoff_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in handoff progress monitoring: {e}")
    
    async def _protocol_health_monitoring_loop(self):
        """Background monitoring for protocol health."""
        while True:
            try:
                await asyncio.sleep(self._realtime_config["protocol_health_check_interval"])
                
                # Check health of each protocol
                for protocol, perf_data in self._protocol_performance_rt.items():
                    await self._check_protocol_health(protocol, perf_data)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in protocol health monitoring: {e}")
    
    async def _coordination_state_sync_loop(self):
        """Background synchronization of coordination state."""
        while True:
            try:
                await asyncio.sleep(self._realtime_config["coordination_state_sync_interval"])
                
                if self.realtime_hub:
                    # Sync coordination state with dashboard
                    coordination_data = await self._get_coordination_state_summary()
                    await self.realtime_hub.broadcast_dashboard_update(
                        update_type="coordination_state",
                        data=coordination_data
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in coordination state sync: {e}")
    
    # ================================================================================
    # Helper Methods
    # ================================================================================
    
    async def _cleanup_delivery_tracking(self):
        """Clean up old delivery tracking data."""
        try:
            current_time = datetime.utcnow()
            expired_messages = []
            
            for message_id, tracking in self._message_delivery_tracking.items():
                last_attempt = tracking.get("last_attempt", current_time)
                if (current_time - last_attempt).total_seconds() > self._realtime_config["message_delivery_timeout"]:
                    expired_messages.append(message_id)
            
            for message_id in expired_messages:
                del self._message_delivery_tracking[message_id]
            
        except Exception as e:
            logger.error(f"Failed to cleanup delivery tracking: {e}")
    
    def _get_next_handoff_actions(self, current_phase: str, progress: float) -> List[str]:
        """Get next actions for handoff based on current phase."""
        if current_phase == "context_packaging":
            return ["Complete context packaging", "Validate context integrity"]
        elif current_phase == "agent_selection":
            return ["Evaluate agent capabilities", "Check agent availability"]
        elif current_phase == "context_transfer":
            return ["Transfer context data", "Verify transfer integrity"]
        elif current_phase == "confirmation":
            return ["Confirm agent readiness", "Validate context reception"]
        else:
            return []
    
    def _get_protocol_alerts(self, protocol: CLIProtocol, perf_data: Dict[str, Any]) -> List[str]:
        """Get alerts for protocol performance."""
        alerts = []
        
        if perf_data["success_rate"] < 0.8:
            alerts.append(f"Low success rate: {perf_data['success_rate']:.1%}")
        
        if perf_data["average_time_ms"] > 5000:
            alerts.append(f"High latency: {perf_data['average_time_ms']:.0f}ms")
        
        if perf_data["error_count"] > 10:
            alerts.append(f"High error count: {perf_data['error_count']}")
        
        return alerts
    
    def _get_protocol_recommendations(self, protocol: CLIProtocol, perf_data: Dict[str, Any]) -> List[str]:
        """Get recommendations for protocol optimization."""
        recommendations = []
        
        if perf_data["success_rate"] < 0.5:
            recommendations.append("Consider protocol restart or reconfiguration")
        
        if perf_data["average_time_ms"] > 10000:
            recommendations.append("Investigate network connectivity and agent responsiveness")
        
        if perf_data["error_count"] > 20:
            recommendations.append("Review error logs and implement retry mechanisms")
        
        return recommendations
    
    async def _alert_stalled_handoff(self, handoff_id: str, tracking: Dict[str, Any], stall_time: float):
        """Alert about stalled handoff."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="handoff_stalled",
                data={
                    "handoff_id": handoff_id,
                    "stall_time_seconds": stall_time,
                    "current_phase": tracking["current_phase"],
                    "progress_percentage": tracking["progress_percentage"],
                    "source_agent": tracking["source_agent"],
                    "severity": "high"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to alert stalled handoff: {e}")
    
    async def _check_protocol_health(self, protocol: CLIProtocol, perf_data: Dict[str, Any]):
        """Check and broadcast protocol health status."""
        try:
            # Calculate health score
            health_score = perf_data["success_rate"]
            
            # Adjust for latency
            if perf_data["average_time_ms"] > 1000:
                health_score *= 0.8
            
            # Determine status
            if health_score >= 0.8:
                status = "healthy"
            elif health_score >= 0.5:
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Broadcast if status changed or unhealthy
            if status != "healthy" and self.realtime_hub:
                await self._broadcast_protocol_performance_update(protocol, perf_data, "health_check")
            
        except Exception as e:
            logger.error(f"Failed to check protocol health: {e}")
    
    async def _get_coordination_state_summary(self) -> Dict[str, Any]:
        """Get summary of current coordination state."""
        try:
            return {
                "active_handoffs": len(self._active_handoffs_rt),
                "tracked_messages": len(self._message_delivery_tracking),
                "monitored_protocols": len(self._protocol_performance_rt),
                "handoff_details": {
                    handoff_id: {
                        "progress": tracking["progress_percentage"],
                        "phase": tracking["current_phase"],
                        "status": tracking["status"]
                    }
                    for handoff_id, tracking in self._active_handoffs_rt.items()
                },
                "protocol_health": {
                    protocol.value: {
                        "success_rate": perf["success_rate"],
                        "average_time_ms": perf["average_time_ms"],
                        "total_operations": perf["total_operations"]
                    }
                    for protocol, perf in self._protocol_performance_rt.items()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get coordination state summary: {e}")
            return {}
    
    async def _broadcast_message_received(self, message: UniversalMessage, source_protocol: CLIProtocol):
        """Broadcast message received notification."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="message_received",
                data={
                    "message_id": message.message_id,
                    "source_protocol": source_protocol.value,
                    "source_agent": message.source_agent_id,
                    "message_type": message.message_type.value if message.message_type else "unknown",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast message received: {e}")
    
    async def _broadcast_message_error(
        self, 
        message: UniversalMessage, 
        target_protocol: CLIProtocol, 
        error: str, 
        delivery_id: str
    ):
        """Broadcast message error notification."""
        try:
            if not self.realtime_hub:
                return
            
            # Broadcast task failure
            task_update = TaskExecutionUpdate(
                task_id=f"msg_{message.message_id}",
                agent_id=message.source_agent_id,
                status=TaskExecutionStatus.FAILED,
                progress_percentage=100.0,
                error_message=error,
                result_data={
                    "target_protocol": target_protocol.value,
                    "delivery_id": delivery_id,
                    "error_type": "delivery_failure"
                }
            )
            
            await self.realtime_hub.broadcast_task_execution_update(task_update)
            
        except Exception as e:
            logger.error(f"Failed to broadcast message error: {e}")
    
    async def _broadcast_receive_error(self, source_protocol: CLIProtocol, error: str):
        """Broadcast receive error notification."""
        try:
            if not self.realtime_hub:
                return
            
            await self.realtime_hub.broadcast_dashboard_update(
                update_type="receive_error",
                data={
                    "source_protocol": source_protocol.value,
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                    "severity": "medium"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to broadcast receive error: {e}")
    
    async def get_realtime_protocol_metrics(self) -> Dict[str, Any]:
        """Get comprehensive real-time protocol metrics."""
        try:
            base_metrics = await self.get_communication_metrics()
            
            return {
                **base_metrics,
                "realtime_metrics": {
                    "active_handoffs": len(self._active_handoffs_rt),
                    "tracked_messages": len(self._message_delivery_tracking),
                    "protocol_performance": self._protocol_performance_rt,
                    "coordination_state": await self._get_coordination_state_summary()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get real-time protocol metrics: {e}")
            return {}
    
    async def shutdown(self):
        """Enhanced shutdown with real-time cleanup."""
        try:
            logger.info("Shutting down RealTimeMultiCLIProtocol...")
            
            # Broadcast shutdown notifications for active handoffs
            if self.realtime_hub:
                for handoff_id in self._active_handoffs_rt.keys():
                    await self._update_handoff_progress(
                        handoff_id, 100.0, "shutdown", "Protocol shutdown initiated"
                    )
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Call parent shutdown
            await super().shutdown()
            
            logger.info("RealTimeMultiCLIProtocol shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during real-time protocol shutdown: {e}")

# ================================================================================
# Factory Functions
# ================================================================================

def create_realtime_multi_cli_protocol(
    protocol_id: str,
    realtime_hub: Optional[RealTimeCommunicationHub] = None,
    enhanced_bridge: Optional[EnhancedCommunicationBridge] = None
) -> RealTimeMultiCLIProtocol:
    """Create a real-time multi-CLI protocol instance."""
    return RealTimeMultiCLIProtocol(
        protocol_id=protocol_id,
        realtime_hub=realtime_hub,
        enhanced_bridge=enhanced_bridge
    )

async def create_integrated_realtime_system(
    protocol_id: str,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    websocket_host: str = "localhost",
    websocket_port: int = 8765,
    **kwargs
) -> tuple[RealTimeCommunicationHub, EnhancedCommunicationBridge, RealTimeMultiCLIProtocol]:
    """
    Create a complete integrated real-time system.
    
    Returns:
        tuple: (RealTimeCommunicationHub, EnhancedCommunicationBridge, RealTimeMultiCLIProtocol)
    """
    from .enhanced_communication_bridge import create_integrated_communication_system
    
    # Create hub and bridge
    hub, bridge = await create_integrated_communication_system(
        redis_host=redis_host,
        redis_port=redis_port,
        websocket_host=websocket_host,
        websocket_port=websocket_port,
        **kwargs
    )
    
    # Create real-time protocol
    protocol = create_realtime_multi_cli_protocol(
        protocol_id=protocol_id,
        realtime_hub=hub,
        enhanced_bridge=bridge
    )
    
    # Initialize protocol
    await protocol.initialize_realtime(hub, bridge)
    
    return hub, bridge, protocol
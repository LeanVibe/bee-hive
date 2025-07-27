"""
Enhanced Lifecycle Hook System with Real-Time Monitoring.

Extends existing hook infrastructure with:
- Advanced agent lifecycle tracking
- Context-aware event processing  
- Real-time WebSocket streaming
- Performance analytics and alerting
- Redis Streams integration for scalability
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.observability import EventType, AgentEvent
from ..observability.hooks import EventCapture, EventProcessor
from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings

logger = structlog.get_logger()


class EnhancedEventType(str, Enum):
    """
    Enhanced event types for comprehensive agent lifecycle tracking.
    
    Extends the base EventType enum with additional lifecycle events
    as recommended by the production readiness analysis.
    """
    # Existing events from base system
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    
    # Enhanced lifecycle events
    AGENT_LIFECYCLE_START = "AgentLifecycleStart"
    AGENT_LIFECYCLE_PAUSE = "AgentLifecyclePause"
    AGENT_LIFECYCLE_RESUME = "AgentLifecycleResume"
    AGENT_LIFECYCLE_COMPLETE = "AgentLifecycleComplete"
    
    # Context and performance events
    CONTEXT_THRESHOLD_REACHED = "ContextThresholdReached"
    PERFORMANCE_DEGRADATION = "PerformanceDegradation"
    MEMORY_PRESSURE = "MemoryPressure"
    ERROR_PATTERN_DETECTED = "ErrorPatternDetected"
    
    # Task and coordination events
    TASK_ASSIGNMENT = "TaskAssignment"
    TASK_PROGRESS_UPDATE = "TaskProgressUpdate"
    TASK_COMPLETION = "TaskCompletion"
    AGENT_COORDINATION = "AgentCoordination"
    
    # System health events
    HEALTH_CHECK = "HealthCheck"
    RESOURCE_UTILIZATION = "ResourceUtilization"
    PERFORMANCE_METRIC = "PerformanceMetric"


@dataclass
class LifecycleEventData:
    """Structured data for lifecycle events."""
    session_id: str
    agent_id: str
    event_type: EnhancedEventType
    timestamp: str
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical
    tags: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LifecycleEventFilter:
    """Filter and routing logic for lifecycle events."""
    
    def __init__(self):
        self.high_priority_events = {
            EnhancedEventType.CONTEXT_THRESHOLD_REACHED,
            EnhancedEventType.PERFORMANCE_DEGRADATION,
            EnhancedEventType.MEMORY_PRESSURE,
            EnhancedEventType.ERROR_PATTERN_DETECTED
        }
        
        self.real_time_events = {
            EnhancedEventType.AGENT_LIFECYCLE_START,
            EnhancedEventType.AGENT_LIFECYCLE_PAUSE,
            EnhancedEventType.AGENT_LIFECYCLE_RESUME,
            EnhancedEventType.AGENT_LIFECYCLE_COMPLETE,
            EnhancedEventType.TASK_PROGRESS_UPDATE,
            EnhancedEventType.PRE_TOOL_USE,
            EnhancedEventType.POST_TOOL_USE
        }
        
        self.analytics_events = {
            EnhancedEventType.PERFORMANCE_METRIC,
            EnhancedEventType.RESOURCE_UTILIZATION,
            EnhancedEventType.HEALTH_CHECK
        }
    
    def should_stream_real_time(self, event_type: EnhancedEventType) -> bool:
        """Determine if event should be streamed in real-time."""
        return event_type in self.real_time_events
    
    def get_priority(self, event_type: EnhancedEventType) -> str:
        """Get event priority for processing."""
        if event_type in self.high_priority_events:
            return "high"
        elif event_type in self.real_time_events:
            return "medium"
        else:
            return "low"
    
    def should_trigger_alert(self, event_data: LifecycleEventData) -> bool:
        """Determine if event should trigger an alert."""
        return (
            event_data.severity in ["error", "critical"] or
            event_data.event_type in self.high_priority_events
        )


class EnhancedLifecycleHookProcessor:
    """
    Enhanced processor for agent lifecycle hooks with real-time capabilities.
    
    Provides comprehensive lifecycle tracking including:
    - Real-time event streaming via Redis Streams
    - WebSocket broadcasting for dashboard updates
    - Performance analytics and pattern detection
    - Automated alerting for critical events
    - Historical event analysis and reporting
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.event_filter = LifecycleEventFilter()
        
        # Redis streams configuration
        self.lifecycle_stream_name = "agent_lifecycle_events"
        self.analytics_stream_name = "agent_analytics_events"
        self.alert_stream_name = "agent_alert_events"
        
        # Processing state
        self.active_sessions: Set[str] = set()
        self.event_processors: List[Callable] = []
        self.websocket_clients: Set[Any] = set()
        
        # Performance tracking
        self.performance_metrics = {
            "events_processed": 0,
            "events_streamed": 0,
            "alerts_triggered": 0,
            "avg_processing_time_ms": 0,
            "last_performance_check": datetime.utcnow()
        }
        
        # Event pattern detection
        self.error_patterns = {}
        self.performance_baseline = {}
        
        logger.info("🔄 Enhanced Lifecycle Hook Processor initialized")
    
    async def process_enhanced_event(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        event_type: EnhancedEventType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        severity: str = "info",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Process enhanced lifecycle event with comprehensive handling.
        
        Args:
            session_id: Session identifier
            agent_id: Agent identifier
            event_type: Type of lifecycle event
            payload: Event-specific payload data
            correlation_id: Optional correlation ID for event chaining
            severity: Event severity level
            tags: Optional metadata tags
            
        Returns:
            Event ID for tracking
        """
        try:
            start_time = datetime.utcnow()
            event_id = str(uuid.uuid4())
            
            # Create structured event data
            event_data = LifecycleEventData(
                session_id=str(session_id),
                agent_id=str(agent_id),
                event_type=event_type,
                timestamp=start_time.isoformat(),
                payload=payload,
                correlation_id=correlation_id,
                severity=severity,
                tags=tags or {}
            )
            
            # Add processing metadata
            event_data.payload["event_id"] = event_id
            event_data.payload["processing_metadata"] = {
                "processor_version": "2.0",
                "priority": self.event_filter.get_priority(event_type),
                "real_time_eligible": self.event_filter.should_stream_real_time(event_type)
            }
            
            # Store in database
            await self._store_event_in_database(event_data)
            
            # Stream to Redis for real-time processing
            await self._stream_to_redis(event_data)
            
            # Real-time WebSocket broadcasting
            if self.event_filter.should_stream_real_time(event_type):
                await self._broadcast_to_websockets(event_data)
            
            # Pattern detection and analysis
            await self._analyze_event_patterns(event_data)
            
            # Alert processing
            if self.event_filter.should_trigger_alert(event_data):
                await self._process_alert(event_data)
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_performance_metrics(processing_time)
            
            logger.debug(
                f"📊 Enhanced lifecycle event processed",
                event_id=event_id,
                event_type=event_type.value,
                agent_id=str(agent_id),
                processing_time_ms=processing_time,
                severity=severity
            )
            
            return event_id
            
        except Exception as e:
            logger.error(
                f"❌ Enhanced lifecycle event processing failed: {e}",
                event_type=event_type.value if event_type else "unknown",
                agent_id=str(agent_id),
                session_id=str(session_id)
            )
            raise
    
    # Enhanced event creation methods
    
    async def capture_agent_lifecycle_start(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        agent_type: str,
        capabilities: List[str],
        initial_context: Dict[str, Any]
    ) -> str:
        """Capture agent lifecycle start event."""
        payload = {
            "agent_type": agent_type,
            "capabilities": capabilities,
            "initial_context": initial_context,
            "startup_timestamp": datetime.utcnow().isoformat(),
            "context_size": len(json.dumps(initial_context, default=str)),
            "capability_count": len(capabilities)
        }
        
        return await self.process_enhanced_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            payload=payload,
            severity="info"
        )
    
    async def capture_context_threshold_reached(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        current_usage_percent: float,
        threshold_type: str,
        recommended_action: str,
        context_stats: Dict[str, Any]
    ) -> str:
        """Capture context threshold reached event."""
        payload = {
            "current_usage_percent": current_usage_percent,
            "threshold_type": threshold_type,
            "recommended_action": recommended_action,
            "context_stats": context_stats,
            "urgency_level": "critical" if current_usage_percent > 95 else "high",
            "estimated_tokens_remaining": context_stats.get("tokens_remaining", 0)
        }
        
        severity = "critical" if current_usage_percent > 95 else "warning"
        
        return await self.process_enhanced_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.CONTEXT_THRESHOLD_REACHED,
            payload=payload,
            severity=severity,
            tags={"threshold_type": threshold_type, "urgency": payload["urgency_level"]}
        )
    
    async def capture_performance_degradation(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        performance_metrics: Dict[str, Any],
        degradation_details: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> str:
        """Capture performance degradation event."""
        payload = {
            "current_metrics": performance_metrics,
            "baseline_metrics": baseline_metrics,
            "degradation_details": degradation_details,
            "degradation_percentage": degradation_details.get("percentage_degradation", 0),
            "affected_operations": degradation_details.get("affected_operations", []),
            "detection_timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.process_enhanced_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.PERFORMANCE_DEGRADATION,
            payload=payload,
            severity="warning",
            tags={"degradation_type": degradation_details.get("type", "general")}
        )
    
    async def capture_task_progress_update(
        self,
        session_id: uuid.UUID,
        agent_id: uuid.UUID,
        task_id: str,
        progress_percentage: float,
        current_stage: str,
        performance_data: Dict[str, Any]
    ) -> str:
        """Capture task progress update event."""
        payload = {
            "task_id": task_id,
            "progress_percentage": progress_percentage,
            "current_stage": current_stage,
            "performance_data": performance_data,
            "estimated_completion": performance_data.get("estimated_completion"),
            "stages_completed": performance_data.get("stages_completed", []),
            "next_stage": performance_data.get("next_stage")
        }
        
        return await self.process_enhanced_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.TASK_PROGRESS_UPDATE,
            payload=payload,
            correlation_id=task_id,
            severity="info"
        )
    
    async def capture_agent_coordination(
        self,
        session_id: uuid.UUID,
        source_agent_id: uuid.UUID,
        target_agent_id: uuid.UUID,
        coordination_type: str,
        coordination_data: Dict[str, Any]
    ) -> str:
        """Capture agent coordination event."""
        payload = {
            "source_agent_id": str(source_agent_id),
            "target_agent_id": str(target_agent_id),
            "coordination_type": coordination_type,
            "coordination_data": coordination_data,
            "coordination_timestamp": datetime.utcnow().isoformat(),
            "message_size": len(json.dumps(coordination_data, default=str))
        }
        
        return await self.process_enhanced_event(
            session_id=session_id,
            agent_id=source_agent_id,
            event_type=EnhancedEventType.AGENT_COORDINATION,
            payload=payload,
            severity="info",
            tags={"coordination_type": coordination_type, "target_agent": str(target_agent_id)}
        )
    
    # WebSocket management
    
    def register_websocket_client(self, websocket_client: Any) -> None:
        """Register WebSocket client for real-time updates."""
        self.websocket_clients.add(websocket_client)
        logger.debug(f"📡 WebSocket client registered. Total clients: {len(self.websocket_clients)}")
    
    def unregister_websocket_client(self, websocket_client: Any) -> None:
        """Unregister WebSocket client."""
        self.websocket_clients.discard(websocket_client)
        logger.debug(f"📡 WebSocket client unregistered. Total clients: {len(self.websocket_clients)}")
    
    async def get_event_analytics(
        self,
        session_id: Optional[uuid.UUID] = None,
        agent_id: Optional[uuid.UUID] = None,
        time_range: Optional[int] = 3600,  # 1 hour in seconds
        event_types: Optional[List[EnhancedEventType]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive event analytics and insights.
        
        Args:
            session_id: Optional session filter
            agent_id: Optional agent filter  
            time_range: Time range in seconds (default: 1 hour)
            event_types: Optional event type filters
            
        Returns:
            Analytics data including trends, patterns, and insights
        """
        try:
            analytics = {
                "summary": {},
                "event_distribution": {},
                "performance_trends": {},
                "error_patterns": {},
                "agent_activity": {},
                "recommendations": []
            }
            
            # Get event statistics
            event_stats = await self._get_event_statistics(
                session_id, agent_id, time_range, event_types
            )
            analytics["summary"] = event_stats
            
            # Event distribution analysis
            distribution = await self._analyze_event_distribution(
                session_id, agent_id, time_range
            )
            analytics["event_distribution"] = distribution
            
            # Performance trend analysis
            performance_trends = await self._analyze_performance_trends(
                session_id, agent_id, time_range
            )
            analytics["performance_trends"] = performance_trends
            
            # Error pattern detection
            error_patterns = await self._detect_error_patterns(
                session_id, agent_id, time_range
            )
            analytics["error_patterns"] = error_patterns
            
            # Agent activity analysis
            if agent_id:
                agent_activity = await self._analyze_agent_activity(agent_id, time_range)
                analytics["agent_activity"] = agent_activity
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(analytics)
            analytics["recommendations"] = recommendations
            
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Event analytics generation failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _store_event_in_database(self, event_data: LifecycleEventData) -> None:
        """Store event in database for persistence."""
        try:
            async with get_async_session() as session:
                agent_event = AgentEvent(
                    session_id=uuid.UUID(event_data.session_id),
                    agent_id=uuid.UUID(event_data.agent_id),
                    event_type=EventType(event_data.event_type.value),
                    payload=event_data.payload,
                    timestamp=datetime.fromisoformat(event_data.timestamp),
                    latency_ms=event_data.payload.get("processing_metadata", {}).get("processing_time_ms")
                )
                
                session.add(agent_event)
                await session.commit()
                
        except Exception as e:
            logger.error(f"❌ Database event storage failed: {e}")
    
    async def _stream_to_redis(self, event_data: LifecycleEventData) -> None:
        """Stream event to Redis for real-time processing."""
        try:
            redis_client = get_redis()
            
            # Determine target stream
            if event_data.event_type in self.event_filter.analytics_events:
                stream_name = self.analytics_stream_name
            elif event_data.severity in ["error", "critical"]:
                stream_name = self.alert_stream_name
            else:
                stream_name = self.lifecycle_stream_name
            
            # Stream event data
            await redis_client.xadd(
                stream_name,
                event_data.to_dict(),
                maxlen=10000  # Keep last 10k events
            )
            
            self.performance_metrics["events_streamed"] += 1
            
        except Exception as e:
            logger.error(f"❌ Redis streaming failed: {e}")
    
    async def _broadcast_to_websockets(self, event_data: LifecycleEventData) -> None:
        """Broadcast event to WebSocket clients."""
        if not self.websocket_clients:
            return
        
        try:
            # Prepare WebSocket message
            ws_message = {
                "type": "lifecycle_event",
                "data": event_data.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Broadcast to all connected clients
            disconnect_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send_text(json.dumps(ws_message))
                except Exception:
                    disconnect_clients.add(client)
            
            # Remove disconnected clients
            for client in disconnect_clients:
                self.websocket_clients.discard(client)
            
        except Exception as e:
            logger.error(f"❌ WebSocket broadcasting failed: {e}")
    
    async def _analyze_event_patterns(self, event_data: LifecycleEventData) -> None:
        """Analyze event for patterns and anomalies."""
        try:
            # Error pattern detection
            if event_data.severity in ["error", "critical"]:
                error_key = f"{event_data.agent_id}:{event_data.event_type.value}"
                
                if error_key not in self.error_patterns:
                    self.error_patterns[error_key] = {
                        "count": 0,
                        "first_seen": event_data.timestamp,
                        "last_seen": event_data.timestamp
                    }
                
                self.error_patterns[error_key]["count"] += 1
                self.error_patterns[error_key]["last_seen"] = event_data.timestamp
                
                # Alert if error pattern exceeds threshold
                if self.error_patterns[error_key]["count"] > 5:
                    await self._trigger_pattern_alert(error_key, self.error_patterns[error_key])
            
            # Performance pattern detection for tool usage
            if event_data.event_type == EnhancedEventType.POST_TOOL_USE:
                await self._analyze_tool_performance_pattern(event_data)
            
        except Exception as e:
            logger.error(f"❌ Event pattern analysis failed: {e}")
    
    async def _process_alert(self, event_data: LifecycleEventData) -> None:
        """Process alert for critical events."""
        try:
            alert_data = {
                "alert_id": str(uuid.uuid4()),
                "event_id": event_data.payload.get("event_id"),
                "alert_type": "lifecycle_event",
                "severity": event_data.severity,
                "agent_id": event_data.agent_id,
                "session_id": event_data.session_id,
                "event_type": event_data.event_type.value,
                "message": f"{event_data.event_type.value} alert for agent {event_data.agent_id}",
                "details": event_data.payload,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stream alert to alert processing system
            redis_client = get_redis()
            await redis_client.xadd(self.alert_stream_name, alert_data)
            
            self.performance_metrics["alerts_triggered"] += 1
            
            logger.warning(
                f"🚨 Alert triggered",
                alert_id=alert_data["alert_id"],
                event_type=event_data.event_type.value,
                severity=event_data.severity,
                agent_id=event_data.agent_id
            )
            
        except Exception as e:
            logger.error(f"❌ Alert processing failed: {e}")
    
    async def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance metrics."""
        self.performance_metrics["events_processed"] += 1
        
        # Update average processing time (exponential moving average)
        current_avg = self.performance_metrics["avg_processing_time_ms"]
        new_avg = (current_avg * 0.9) + (processing_time_ms * 0.1)
        self.performance_metrics["avg_processing_time_ms"] = new_avg
        
        # Check for performance degradation
        if processing_time_ms > 100:  # 100ms threshold
            logger.warning(
                f"⚠️ Slow event processing detected",
                processing_time_ms=processing_time_ms,
                average_time_ms=new_avg
            )
    
    # Analytics helper methods (simplified implementations)
    
    async def _get_event_statistics(
        self, session_id, agent_id, time_range, event_types
    ) -> Dict[str, Any]:
        """Get basic event statistics."""
        return {
            "total_events": self.performance_metrics["events_processed"],
            "events_streamed": self.performance_metrics["events_streamed"],
            "alerts_triggered": self.performance_metrics["alerts_triggered"],
            "avg_processing_time_ms": self.performance_metrics["avg_processing_time_ms"]
        }
    
    async def _analyze_event_distribution(
        self, session_id, agent_id, time_range
    ) -> Dict[str, Any]:
        """Analyze event type distribution."""
        return {"distribution": "placeholder"}
    
    async def _analyze_performance_trends(
        self, session_id, agent_id, time_range
    ) -> Dict[str, Any]:
        """Analyze performance trends."""
        return {"trends": "placeholder"}
    
    async def _detect_error_patterns(
        self, session_id, agent_id, time_range
    ) -> Dict[str, Any]:
        """Detect error patterns."""
        return {"patterns": self.error_patterns}
    
    async def _analyze_agent_activity(
        self, agent_id, time_range
    ) -> Dict[str, Any]:
        """Analyze agent activity patterns."""
        return {"activity": "placeholder"}
    
    async def _generate_recommendations(self, analytics) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if analytics["summary"]["avg_processing_time_ms"] > 50:
            recommendations.append("Consider optimizing event processing pipeline")
        
        if analytics["summary"]["alerts_triggered"] > 10:
            recommendations.append("Review alert thresholds to reduce noise")
        
        return recommendations
    
    async def _trigger_pattern_alert(self, pattern_key: str, pattern_data: Dict) -> None:
        """Trigger alert for detected error patterns."""
        logger.warning(
            f"🔍 Error pattern detected",
            pattern_key=pattern_key,
            error_count=pattern_data["count"],
            duration=pattern_data["last_seen"]
        )
    
    async def _analyze_tool_performance_pattern(self, event_data: LifecycleEventData) -> None:
        """Analyze tool performance patterns."""
        execution_time = event_data.payload.get("execution_time_ms", 0)
        tool_name = event_data.payload.get("tool_name", "unknown")
        
        if execution_time > 5000:  # 5 second threshold
            logger.warning(
                f"🐌 Slow tool execution detected",
                tool_name=tool_name,
                execution_time_ms=execution_time,
                agent_id=event_data.agent_id
            )


# Factory function
def get_enhanced_lifecycle_hook_processor() -> EnhancedLifecycleHookProcessor:
    """Get the enhanced lifecycle hook processor instance."""
    return EnhancedLifecycleHookProcessor()


# Convenience functions for common event captures
async def capture_agent_start(
    session_id: uuid.UUID,
    agent_id: uuid.UUID,
    agent_type: str,
    capabilities: List[str],
    initial_context: Dict[str, Any]
) -> str:
    """Convenience function to capture agent start event."""
    processor = get_enhanced_lifecycle_hook_processor()
    return await processor.capture_agent_lifecycle_start(
        session_id, agent_id, agent_type, capabilities, initial_context
    )


async def capture_context_threshold(
    session_id: uuid.UUID,
    agent_id: uuid.UUID,
    usage_percent: float,
    threshold_type: str,
    action: str,
    stats: Dict[str, Any]
) -> str:
    """Convenience function to capture context threshold event."""
    processor = get_enhanced_lifecycle_hook_processor()
    return await processor.capture_context_threshold_reached(
        session_id, agent_id, usage_percent, threshold_type, action, stats
    )
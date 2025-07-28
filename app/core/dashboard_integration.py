"""
Dashboard Integration Layer for LeanVibe Agent Hive 2.0

Seamless integration between chat transcript analysis tools and the visual
coordination dashboard with real-time updates and interactive debugging.

Features:
- Real-time transcript visualization in coordination dashboard
- Interactive conversation debugging interface
- Pattern detection visualization and alerts
- Performance metrics integration with dashboard charts
- Agent communication flow visualization
- Error cascade visualization and analysis
- Dashboard event streaming and synchronization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from pydantic import BaseModel, Field

from .chat_transcript_manager import ChatTranscriptManager, ConversationEvent
from .communication_analyzer import CommunicationAnalyzer, CommunicationInsight
from .transcript_streaming import TranscriptStreamingManager, StreamEventType
from .conversation_debugging_tools import ConversationDebugger
from .coordination_dashboard import (
    CoordinationDashboard, AgentGraphNode, AgentGraphEdge,
    AgentNodeType, AgentNodeStatus, AgentEdgeType
)

logger = structlog.get_logger()


class DashboardEventType(Enum):
    """Types of dashboard integration events."""
    TRANSCRIPT_UPDATE = "transcript_update"
    CONVERSATION_HIGHLIGHT = "conversation_highlight"
    PATTERN_VISUALIZATION = "pattern_visualization"
    ERROR_CASCADE_ALERT = "error_cascade_alert"
    PERFORMANCE_METRIC_UPDATE = "performance_metric_update"
    DEBUG_SESSION_ACTIVE = "debug_session_active"
    AGENT_INTERACTION_FLOW = "agent_interaction_flow"
    COMMUNICATION_BOTTLENECK = "communication_bottleneck"


class VisualizationStyle(Enum):
    """Visualization styles for dashboard elements."""
    NORMAL = "normal"
    HIGHLIGHTED = "highlighted"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    DEBUG = "debug"


@dataclass
class DashboardUpdate:
    """Dashboard update with visualization data."""
    update_type: DashboardEventType
    target_elements: List[str]  # Node/edge IDs to update
    visualization_data: Dict[str, Any]
    style: VisualizationStyle
    duration_ms: Optional[int] = None  # Animation duration
    priority: int = 1  # 1=low, 2=medium, 3=high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'update_type': self.update_type.value,
            'style': self.style.value
        }


@dataclass
class ConversationVisualization:
    """Visualization data for conversation flows."""
    conversation_thread_id: str
    participating_agents: List[str]
    message_flow: List[Dict[str, Any]]
    timeline_data: List[Dict[str, Any]]
    pattern_highlights: List[Dict[str, Any]]
    performance_indicators: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DashboardIntegrationManager:
    """
    Manages integration between transcript analysis and visual dashboard.
    
    Provides real-time visualization updates, interactive debugging integration,
    and seamless synchronization between backend analysis and frontend display.
    """
    
    def __init__(
        self,
        transcript_manager: ChatTranscriptManager,
        analyzer: CommunicationAnalyzer,
        streaming_manager: TranscriptStreamingManager,
        debugger: ConversationDebugger,
        coordination_dashboard: CoordinationDashboard
    ):
        self.transcript_manager = transcript_manager
        self.analyzer = analyzer
        self.streaming_manager = streaming_manager
        self.debugger = debugger
        self.coordination_dashboard = coordination_dashboard
        
        # Dashboard state management
        self.active_visualizations: Dict[str, ConversationVisualization] = {}
        self.dashboard_subscriptions: Dict[str, Set[str]] = {}  # event_type -> connection_ids
        self.update_queue: asyncio.Queue = asyncio.Queue()
        
        # Visualization cache
        self.conversation_cache: Dict[str, Dict[str, Any]] = {}
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Integration state
        self.integration_active = False
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("DashboardIntegrationManager initialized")
    
    async def start_integration(self) -> None:
        """Start dashboard integration with real-time updates."""
        if self.integration_active:
            return
        
        self.integration_active = True
        
        # Start background processing tasks
        tasks = [
            self._process_transcript_updates(),
            self._process_dashboard_updates(),
            self._monitor_conversation_patterns(),
            self._sync_performance_metrics()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Dashboard integration started")
    
    async def stop_integration(self) -> None:
        """Stop dashboard integration and cleanup resources."""
        self.integration_active = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        
        logger.info("Dashboard integration stopped")
    
    async def visualize_conversation_thread(
        self,
        session_id: str,
        thread_participants: List[str],
        highlight_patterns: bool = True
    ) -> ConversationVisualization:
        """
        Create comprehensive visualization for a conversation thread.
        
        Args:
            session_id: Session containing the conversation
            thread_participants: Agents participating in the thread
            highlight_patterns: Whether to highlight detected patterns
            
        Returns:
            ConversationVisualization with all visualization data
        """
        try:
            # Get conversation events
            events = await self.transcript_manager.get_conversation_transcript(
                session_id=session_id,
                agent_filter=thread_participants
            )
            
            if not events:
                raise ValueError(f"No conversation events found for session {session_id}")
            
            # Build message flow data
            message_flow = []
            for i, event in enumerate(events):
                flow_item = {
                    'sequence': i,
                    'timestamp': event.timestamp.isoformat(),
                    'source_agent': event.source_agent_id,
                    'target_agent': event.target_agent_id,
                    'message_type': event.event_type.value,
                    'content_preview': event.message_content[:100] + "..." if len(event.message_content) > 100 else event.message_content,
                    'response_time_ms': event.response_time_ms,
                    'tool_calls': len(event.tool_calls),
                    'context_refs': len(event.context_references)
                }
                message_flow.append(flow_item)
            
            # Build timeline data
            timeline_data = await self._build_timeline_data(events)
            
            # Detect and highlight patterns
            pattern_highlights = []
            if highlight_patterns:
                insights = await self.analyzer.analyze_conversation_events(events)
                pattern_highlights = await self._build_pattern_highlights(insights, events)
            
            # Calculate performance indicators
            performance_indicators = await self._calculate_performance_indicators(events)
            
            # Create visualization
            thread_id = f"thread_{session_id}_{hash(tuple(thread_participants))}"
            visualization = ConversationVisualization(
                conversation_thread_id=thread_id,
                participating_agents=thread_participants,
                message_flow=message_flow,
                timeline_data=timeline_data,
                pattern_highlights=pattern_highlights,
                performance_indicators=performance_indicators
            )
            
            # Cache visualization
            self.active_visualizations[thread_id] = visualization
            
            # Send dashboard update
            await self._send_dashboard_update(
                DashboardUpdate(
                    update_type=DashboardEventType.TRANSCRIPT_UPDATE,
                    target_elements=thread_participants,
                    visualization_data=visualization.to_dict(),
                    style=VisualizationStyle.NORMAL,
                    priority=2
                )
            )
            
            logger.info(
                "Conversation thread visualized",
                thread_id=thread_id,
                participants=len(thread_participants),
                events=len(events)
            )
            
            return visualization
            
        except Exception as e:
            logger.error(f"Failed to visualize conversation thread: {e}")
            raise
    
    async def highlight_conversation_patterns(
        self,
        session_id: str,
        pattern_types: List[str] = None
    ) -> List[DashboardUpdate]:
        """
        Highlight detected conversation patterns in the dashboard.
        
        Args:
            session_id: Session to analyze for patterns
            pattern_types: Specific pattern types to highlight
            
        Returns:
            List of dashboard updates for pattern visualization
        """
        try:
            # Get conversation events
            events = await self.transcript_manager.get_conversation_transcript(session_id)
            
            # Analyze patterns
            insights = await self.analyzer.analyze_conversation_events(events)
            
            # Filter insights by pattern types if specified
            if pattern_types:
                insights = [
                    insight for insight in insights
                    if any(pattern in insight.title.lower() for pattern in pattern_types)
                ]
            
            dashboard_updates = []
            
            for insight in insights:
                # Determine visualization style based on severity
                style = VisualizationStyle.NORMAL
                if insight.severity.value == "critical":
                    style = VisualizationStyle.ERROR
                elif insight.severity.value == "error":
                    style = VisualizationStyle.WARNING
                elif insight.severity.value == "warning":
                    style = VisualizationStyle.WARNING
                
                # Create dashboard update
                update = DashboardUpdate(
                    update_type=DashboardEventType.PATTERN_VISUALIZATION,
                    target_elements=insight.affected_agents,
                    visualization_data={
                        'pattern_type': insight.analysis_type.value,
                        'title': insight.title,
                        'description': insight.description,
                        'metrics': insight.metrics,
                        'recommendations': insight.recommendations,
                        'timestamp': insight.timestamp.isoformat()
                    },
                    style=style,
                    duration_ms=5000,  # 5 second highlight
                    priority=3 if style == VisualizationStyle.ERROR else 2
                )
                
                dashboard_updates.append(update)
                
                # Send update immediately for high priority patterns
                if update.priority >= 3:
                    await self._send_dashboard_update(update)
            
            # Cache pattern highlights
            self.pattern_cache[session_id] = [update.to_dict() for update in dashboard_updates]
            
            logger.info(
                "Conversation patterns highlighted",
                session_id=session_id,
                pattern_count=len(dashboard_updates)
            )
            
            return dashboard_updates
            
        except Exception as e:
            logger.error(f"Failed to highlight conversation patterns: {e}")
            raise
    
    async def visualize_error_cascade(
        self,
        error_analysis_results: List[Dict[str, Any]]
    ) -> List[DashboardUpdate]:
        """
        Visualize error cascades in the dashboard with flow analysis.
        
        Args:
            error_analysis_results: Results from error analysis
            
        Returns:
            List of dashboard updates for error visualization
        """
        try:
            dashboard_updates = []
            
            for error_result in error_analysis_results:
                affected_agents = error_result.get('affected_agents', [])
                cascade_events = error_result.get('cascade_events', [])
                
                # Build error flow visualization
                error_flow = []
                for event in cascade_events:
                    flow_item = {
                        'timestamp': event.get('timestamp'),
                        'agent': event.get('source_agent_id'),
                        'error_type': event.get('event_type'),
                        'severity': 'high' if 'critical' in event.get('message_content', '').lower() else 'medium'
                    }
                    error_flow.append(flow_item)
                
                # Create dashboard update
                update = DashboardUpdate(
                    update_type=DashboardEventType.ERROR_CASCADE_ALERT,
                    target_elements=affected_agents,
                    visualization_data={
                        'error_id': error_result.get('error_id'),
                        'root_cause': error_result.get('root_cause'),
                        'severity': error_result.get('severity'),
                        'error_flow': error_flow,
                        'resolution_steps': error_result.get('resolution_steps', []),
                        'cascade_size': len(cascade_events)
                    },
                    style=VisualizationStyle.ERROR,
                    duration_ms=10000,  # 10 second alert
                    priority=3
                )
                
                dashboard_updates.append(update)
                
                # Send critical updates immediately
                await self._send_dashboard_update(update)
            
            logger.info(
                "Error cascades visualized",
                cascade_count=len(dashboard_updates)
            )
            
            return dashboard_updates
            
        except Exception as e:
            logger.error(f"Failed to visualize error cascades: {e}")
            raise
    
    async def integrate_debug_session(
        self,
        debug_session_id: str,
        target_session_id: str
    ) -> Dict[str, Any]:
        """
        Integrate debug session with dashboard visualization.
        
        Args:
            debug_session_id: ID of the debug session
            target_session_id: Session being debugged
            
        Returns:
            Integration status and visualization setup
        """
        try:
            # Get debug session status
            debug_status = await self.debugger.get_debug_session_status(debug_session_id)
            
            # Create debug visualization overlay
            debug_overlay = {
                'debug_session_id': debug_session_id,
                'target_session': target_session_id,
                'progress': debug_status['progress'],
                'breakpoints': debug_status['breakpoints'],
                'current_event': None,
                'debug_controls': {
                    'step_enabled': not debug_status['progress']['at_end'],
                    'pause_enabled': True,
                    'resume_enabled': debug_status['progress']['is_paused'],
                    'stop_enabled': True
                }
            }
            
            # Send dashboard update for debug mode
            await self._send_dashboard_update(
                DashboardUpdate(
                    update_type=DashboardEventType.DEBUG_SESSION_ACTIVE,
                    target_elements=[target_session_id],
                    visualization_data=debug_overlay,
                    style=VisualizationStyle.DEBUG,
                    priority=2
                )
            )
            
            # Subscribe to debug session updates
            await self._subscribe_to_debug_updates(debug_session_id)
            
            logger.info(
                "Debug session integrated with dashboard",
                debug_session_id=debug_session_id,
                target_session=target_session_id
            )
            
            return {
                'success': True,
                'debug_session_id': debug_session_id,
                'debug_overlay': debug_overlay,
                'integration_active': True
            }
            
        except Exception as e:
            logger.error(f"Failed to integrate debug session: {e}")
            raise
    
    async def update_performance_metrics(
        self,
        session_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Update dashboard with latest performance metrics.
        
        Args:
            session_id: Session ID for metrics
            metrics: Performance metrics data
        """
        try:
            # Process metrics for visualization
            visualization_metrics = {
                'session_id': session_id,
                'response_times': metrics.get('response_times', {}),
                'message_frequency': metrics.get('message_frequency', 0),
                'error_rate': metrics.get('error_rate', 0),
                'tool_usage': metrics.get('tool_usage', 0),
                'context_sharing': metrics.get('context_sharing', 0),
                'agent_activity': metrics.get('agent_activity', {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Determine if metrics indicate issues
            style = VisualizationStyle.NORMAL
            if metrics.get('error_rate', 0) > 0.1:  # More than 10% errors
                style = VisualizationStyle.ERROR
            elif metrics.get('response_times', {}).get('average', 0) > 5000:  # Slow responses
                style = VisualizationStyle.WARNING
            
            # Send dashboard update
            await self._send_dashboard_update(
                DashboardUpdate(
                    update_type=DashboardEventType.PERFORMANCE_METRIC_UPDATE,
                    target_elements=[session_id],
                    visualization_data=visualization_metrics,
                    style=style,
                    priority=1 if style == VisualizationStyle.NORMAL else 2
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def visualize_agent_interactions(
        self,
        session_id: str,
        time_window_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Visualize agent interaction flows and communication patterns.
        
        Args:
            session_id: Session to analyze
            time_window_minutes: Time window for analysis
            
        Returns:
            Agent interaction visualization data
        """
        try:
            # Get recent conversation events
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            events = await self.transcript_manager.get_conversation_transcript(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time
            )
            
            # Build interaction graph
            interaction_flows = {}
            agent_stats = {}
            
            for event in events:
                source = event.source_agent_id
                target = event.target_agent_id
                
                # Track agent statistics
                if source not in agent_stats:
                    agent_stats[source] = {
                        'messages_sent': 0,
                        'messages_received': 0,
                        'tool_calls': 0,
                        'context_shares': 0,
                        'avg_response_time': 0,
                        'error_count': 0
                    }
                
                agent_stats[source]['messages_sent'] += 1
                agent_stats[source]['tool_calls'] += len(event.tool_calls)
                agent_stats[source]['context_shares'] += len(event.context_references)
                
                if event.response_time_ms:
                    current_avg = agent_stats[source]['avg_response_time']
                    count = agent_stats[source]['messages_sent']
                    agent_stats[source]['avg_response_time'] = (
                        (current_avg * (count - 1) + event.response_time_ms) / count
                    )
                
                # Track interactions
                if target:
                    flow_key = (source, target)
                    if flow_key not in interaction_flows:
                        interaction_flows[flow_key] = {
                            'source': source,
                            'target': target,
                            'message_count': 0,
                            'avg_response_time': 0,
                            'last_interaction': event.timestamp.isoformat()
                        }
                    
                    interaction_flows[flow_key]['message_count'] += 1
                    interaction_flows[flow_key]['last_interaction'] = event.timestamp.isoformat()
                    
                    # Track target statistics
                    if target not in agent_stats:
                        agent_stats[target] = {
                            'messages_sent': 0,
                            'messages_received': 0,
                            'tool_calls': 0,
                            'context_shares': 0,
                            'avg_response_time': 0,
                            'error_count': 0
                        }
                    agent_stats[target]['messages_received'] += 1
            
            # Build visualization data
            visualization_data = {
                'session_id': session_id,
                'time_window_minutes': time_window_minutes,
                'agent_statistics': agent_stats,
                'interaction_flows': list(interaction_flows.values()),
                'total_interactions': len(events),
                'active_agents': len(agent_stats),
                'generated_at': datetime.utcnow().isoformat()
            }
            
            # Send dashboard update
            await self._send_dashboard_update(
                DashboardUpdate(
                    update_type=DashboardEventType.AGENT_INTERACTION_FLOW,
                    target_elements=list(agent_stats.keys()),
                    visualization_data=visualization_data,
                    style=VisualizationStyle.NORMAL,
                    priority=1
                )
            )
            
            logger.info(
                "Agent interactions visualized",
                session_id=session_id,
                active_agents=len(agent_stats),
                interactions=len(events)
            )
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Failed to visualize agent interactions: {e}")
            raise
    
    # Private helper methods
    
    async def _process_transcript_updates(self) -> None:
        """Background task to process transcript updates."""
        while self.integration_active:
            try:
                # Process real-time transcript events
                await asyncio.sleep(1)  # Process every second
                
                # This would integrate with streaming manager
                # to get real-time events and create visualizations
                
            except Exception as e:
                logger.error(f"Transcript update processing error: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_dashboard_updates(self) -> None:
        """Background task to process queued dashboard updates."""
        while self.integration_active:
            try:
                # Process updates from queue
                update = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                
                # Send update to coordination dashboard
                await self._apply_dashboard_update(update)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Dashboard update processing error: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_conversation_patterns(self) -> None:
        """Background task to monitor for conversation patterns."""
        while self.integration_active:
            try:
                # Monitor for patterns every 30 seconds
                await asyncio.sleep(30)
                
                # This would check for new patterns and create alerts
                
            except Exception as e:
                logger.error(f"Pattern monitoring error: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def _sync_performance_metrics(self) -> None:
        """Background task to sync performance metrics."""
        while self.integration_active:
            try:
                # Sync metrics every 10 seconds
                await asyncio.sleep(10)
                
                # Get performance insights
                performance_insights = await self.transcript_manager.get_performance_insights()
                
                # Update dashboard with metrics
                # This would send performance data to coordination dashboard
                
            except Exception as e:
                logger.error(f"Performance metrics sync error: {e}")
                await asyncio.sleep(30)  # Back off on error
    
    async def _send_dashboard_update(self, update: DashboardUpdate) -> None:
        """Send update to dashboard via queue."""
        await self.update_queue.put(update)
    
    async def _apply_dashboard_update(self, update: DashboardUpdate) -> None:
        """Apply update to coordination dashboard."""
        try:
            # This would integrate with the coordination dashboard
            # to update visualizations based on the update data
            
            # For now, just log the update
            logger.info(
                "Dashboard update applied",
                update_type=update.update_type.value,
                target_elements=len(update.target_elements),
                style=update.style.value
            )
            
        except Exception as e:
            logger.error(f"Failed to apply dashboard update: {e}")
    
    async def _build_timeline_data(
        self,
        events: List[ConversationEvent]
    ) -> List[Dict[str, Any]]:
        """Build timeline data for conversation visualization."""
        timeline_data = []
        
        for event in events:
            timeline_item = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'agent': event.source_agent_id,
                'duration_ms': event.response_time_ms,
                'has_tools': len(event.tool_calls) > 0,
                'has_context': len(event.context_references) > 0,
                'is_error': event.event_type.value == 'error_occurred'
            }
            timeline_data.append(timeline_item)
        
        return timeline_data
    
    async def _build_pattern_highlights(
        self,
        insights: List,
        events: List[ConversationEvent]
    ) -> List[Dict[str, Any]]:
        """Build pattern highlight data from insights."""
        pattern_highlights = []
        
        for insight in insights:
            highlight = {
                'pattern_type': insight.analysis_type.value,
                'severity': insight.severity.value,
                'title': insight.title,
                'affected_agents': insight.affected_agents,
                'timestamp': insight.timestamp.isoformat(),
                'highlight_color': self._get_severity_color(insight.severity.value)
            }
            pattern_highlights.append(highlight)
        
        return pattern_highlights
    
    async def _calculate_performance_indicators(
        self,
        events: List[ConversationEvent]
    ) -> Dict[str, Any]:
        """Calculate performance indicators for visualization."""
        if not events:
            return {}
        
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        error_events = [e for e in events if 'error' in e.event_type.value]
        
        indicators = {
            'total_events': len(events),
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'error_rate': len(error_events) / len(events) if events else 0,
            'tool_usage_rate': sum(len(e.tool_calls) for e in events) / len(events) if events else 0,
            'context_sharing_rate': sum(len(e.context_references) for e in events) / len(events) if events else 0,
            'time_span_minutes': (
                (events[-1].timestamp - events[0].timestamp).total_seconds() / 60
                if len(events) > 1 else 0
            )
        }
        
        return indicators
    
    async def _subscribe_to_debug_updates(self, debug_session_id: str) -> None:
        """Subscribe to debug session updates."""
        # This would set up subscription to debug session events
        # and forward them to the dashboard
        pass
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color code for severity level."""
        color_map = {
            'info': '#4A90E2',
            'warning': '#FFD93D',
            'error': '#FF6B6B',
            'critical': '#FF4444'
        }
        return color_map.get(severity, '#4A90E2')
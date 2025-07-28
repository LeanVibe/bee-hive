"""
Conversation Debugging Tools for LeanVibe Agent Hive 2.0

Advanced debugging capabilities for multi-agent conversations including
conversation replay, error analysis, bottleneck detection, and performance profiling.

Features:
- Interactive conversation replay with step-by-step debugging
- Error cascade analysis and root cause detection
- Communication bottleneck identification and resolution
- Performance profiling and optimization recommendations
- Agent behavior anomaly detection
- Conversation flow visualization and analysis
- Debug session management and recording
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

import structlog
from pydantic import BaseModel, Field

from .chat_transcript_manager import (
    ChatTranscriptManager, ConversationEvent, ConversationEventType,
    ConversationThread, ConversationMetrics
)
from .communication_analyzer import (
    CommunicationAnalyzer, CommunicationInsight, AnalysisType,
    AlertSeverity, AgentBehaviorProfile
)
from .transcript_streaming import TranscriptStreamingManager

logger = structlog.get_logger()


class DebugLevel(Enum):
    """Debug output levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ReplayMode(Enum):
    """Conversation replay modes."""
    STEP_BY_STEP = "step_by_step"
    CONTINUOUS = "continuous"
    BREAKPOINT = "breakpoint"
    FAST_FORWARD = "fast_forward"


class AnalysisScope(Enum):
    """Scope of debugging analysis."""
    SINGLE_EVENT = "single_event"
    CONVERSATION_THREAD = "conversation_thread"
    SESSION_WIDE = "session_wide"
    CROSS_SESSION = "cross_session"
    SYSTEM_WIDE = "system_wide"


@dataclass
class DebugBreakpoint:
    """Debugging breakpoint configuration."""
    breakpoint_id: str
    condition_type: str  # event_type, agent_id, error, pattern, etc.
    condition_value: Any
    action: str  # pause, log, alert, analyze
    enabled: bool = True
    hit_count: int = 0
    
    def matches_event(self, event: ConversationEvent) -> bool:
        """Check if event matches breakpoint condition."""
        if not self.enabled:
            return False
        
        if self.condition_type == "event_type":
            return event.event_type == self.condition_value
        elif self.condition_type == "agent_id":
            return event.source_agent_id == self.condition_value
        elif self.condition_type == "error":
            return event.event_type == ConversationEventType.ERROR_OCCURRED
        elif self.condition_type == "response_time":
            return event.response_time_ms and event.response_time_ms > self.condition_value
        elif self.condition_type == "content_contains":
            return self.condition_value.lower() in event.message_content.lower()
        
        return False


@dataclass
class DebugSession:
    """Debug session state and configuration."""
    session_id: str
    target_conversation_session: str
    replay_mode: ReplayMode
    debug_level: DebugLevel
    breakpoints: List[DebugBreakpoint]
    start_time: datetime
    current_event_index: int
    total_events: int
    is_paused: bool
    analysis_scope: AnalysisScope
    auto_analyze: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'breakpoints': [asdict(bp) for bp in self.breakpoints],
            'replay_mode': self.replay_mode.value,
            'debug_level': self.debug_level.value,
            'analysis_scope': self.analysis_scope.value
        }


@dataclass
class ErrorAnalysisResult:
    """Result of error analysis with recommendations."""
    error_id: str
    error_type: str
    root_cause: str
    affected_agents: List[str]
    error_timeline: List[ConversationEvent]
    cascade_events: List[ConversationEvent]
    severity: AlertSeverity
    resolution_steps: List[str]
    prevention_recommendations: List[str]
    similar_errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'error_timeline': [e.to_dict() for e in self.error_timeline],
            'cascade_events': [e.to_dict() for e in self.cascade_events]
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of communication bottlenecks."""
    bottleneck_id: str
    bottleneck_type: str
    location: str  # agent_id, connection, or system component
    severity_score: float
    impact_metrics: Dict[str, float]
    affected_flows: List[Tuple[str, str]]  # (source_agent, target_agent) pairs
    recommended_actions: List[str]
    estimated_improvement: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'affected_flows': [{'source': src, 'target': tgt} for src, tgt in self.affected_flows]
        }


class ConversationDebugger:
    """
    Advanced conversation debugging system for multi-agent communications.
    
    Provides interactive debugging, error analysis, and performance profiling
    with sophisticated replay capabilities and automated issue detection.
    """
    
    def __init__(
        self,
        transcript_manager: ChatTranscriptManager,
        analyzer: CommunicationAnalyzer,
        streaming_manager: TranscriptStreamingManager
    ):
        self.transcript_manager = transcript_manager
        self.analyzer = analyzer
        self.streaming_manager = streaming_manager
        
        # Active debug sessions
        self.active_sessions: Dict[str, DebugSession] = {}
        self.session_events: Dict[str, List[ConversationEvent]] = {}
        
        # Debug output and logging
        self.debug_logs: Dict[str, deque] = {}
        self.analysis_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.debug_metrics: Dict[str, Any] = {
            'sessions_created': 0,
            'breakpoints_hit': 0,
            'errors_analyzed': 0,
            'bottlenecks_detected': 0,
            'average_analysis_time_ms': 0
        }
        
        logger.info("ConversationDebugger initialized")
    
    async def create_debug_session(
        self,
        target_session_id: str,
        replay_mode: ReplayMode = ReplayMode.STEP_BY_STEP,
        debug_level: DebugLevel = DebugLevel.INFO,
        analysis_scope: AnalysisScope = AnalysisScope.SESSION_WIDE,
        auto_analyze: bool = True
    ) -> str:
        """
        Create a new debugging session for conversation analysis.
        
        Args:
            target_session_id: Session to debug
            replay_mode: How to replay conversations
            debug_level: Level of debug output
            analysis_scope: Scope of analysis
            auto_analyze: Whether to automatically analyze events
            
        Returns:
            Debug session ID
        """
        try:
            debug_session_id = f"debug_{target_session_id}_{int(time.time())}"
            
            # Load conversation events for the target session
            events = await self.transcript_manager.get_conversation_transcript(
                session_id=target_session_id
            )
            
            if not events:
                raise ValueError(f"No conversation events found for session {target_session_id}")
            
            # Create debug session
            debug_session = DebugSession(
                session_id=debug_session_id,
                target_conversation_session=target_session_id,
                replay_mode=replay_mode,
                debug_level=debug_level,
                breakpoints=[],
                start_time=datetime.utcnow(),
                current_event_index=0,
                total_events=len(events),
                is_paused=False,
                analysis_scope=analysis_scope,
                auto_analyze=auto_analyze
            )
            
            self.active_sessions[debug_session_id] = debug_session
            self.session_events[debug_session_id] = events
            self.debug_logs[debug_session_id] = deque(maxlen=1000)
            
            # Log session creation
            await self._log_debug(
                debug_session_id,
                DebugLevel.INFO,
                f"Debug session created for {target_session_id}",
                {
                    'total_events': len(events),
                    'replay_mode': replay_mode.value,
                    'analysis_scope': analysis_scope.value
                }
            )
            
            # Perform initial analysis if enabled
            if auto_analyze:
                await self._perform_initial_analysis(debug_session_id)
            
            self.debug_metrics['sessions_created'] += 1
            
            logger.info(
                "Debug session created",
                debug_session_id=debug_session_id,
                target_session=target_session_id,
                event_count=len(events)
            )
            
            return debug_session_id
            
        except Exception as e:
            logger.error(f"Failed to create debug session: {e}")
            raise
    
    async def add_breakpoint(
        self,
        debug_session_id: str,
        condition_type: str,
        condition_value: Any,
        action: str = "pause"
    ) -> str:
        """Add a debugging breakpoint to a session."""
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        breakpoint_id = f"bp_{debug_session_id}_{len(self.active_sessions[debug_session_id].breakpoints)}"
        
        breakpoint = DebugBreakpoint(
            breakpoint_id=breakpoint_id,
            condition_type=condition_type,
            condition_value=condition_value,
            action=action
        )
        
        self.active_sessions[debug_session_id].breakpoints.append(breakpoint)
        
        await self._log_debug(
            debug_session_id,
            DebugLevel.INFO,
            f"Breakpoint added: {condition_type}={condition_value}",
            {'breakpoint_id': breakpoint_id, 'action': action}
        )
        
        return breakpoint_id
    
    async def step_debug_session(
        self,
        debug_session_id: str,
        steps: int = 1
    ) -> Dict[str, Any]:
        """
        Step through debug session by specified number of events.
        
        Args:
            debug_session_id: Debug session ID
            steps: Number of events to step through
            
        Returns:
            Debug step result with current state
        """
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        session = self.active_sessions[debug_session_id]
        events = self.session_events[debug_session_id]
        
        try:
            step_results = []
            
            for _ in range(steps):
                if session.current_event_index >= len(events):
                    break
                
                current_event = events[session.current_event_index]
                
                # Check breakpoints
                breakpoint_hit = await self._check_breakpoints(debug_session_id, current_event)
                
                # Analyze current event if auto-analysis is enabled
                analysis_result = None
                if session.auto_analyze:
                    analysis_result = await self._analyze_debug_event(
                        debug_session_id, current_event
                    )
                
                # Log debug step
                await self._log_debug(
                    debug_session_id,
                    DebugLevel.DEBUG,
                    f"Step {session.current_event_index + 1}/{len(events)}",
                    {
                        'event_id': current_event.id,
                        'event_type': current_event.event_type.value,
                        'source_agent': current_event.source_agent_id,
                        'breakpoint_hit': breakpoint_hit is not None
                    }
                )
                
                step_result = {
                    'step_index': session.current_event_index,
                    'event': current_event.to_dict(),
                    'breakpoint_hit': breakpoint_hit.to_dict() if breakpoint_hit else None,
                    'analysis': analysis_result,
                    'session_state': session.to_dict()
                }
                step_results.append(step_result)
                
                session.current_event_index += 1
                
                # Pause if breakpoint hit
                if breakpoint_hit and breakpoint_hit.action == "pause":
                    session.is_paused = True
                    break
            
            return {
                'debug_session_id': debug_session_id,
                'steps_executed': len(step_results),
                'step_results': step_results,
                'session_state': session.to_dict(),
                'at_end': session.current_event_index >= len(events)
            }
            
        except Exception as e:
            logger.error(f"Debug step failed: {e}")
            raise
    
    async def analyze_error_patterns(
        self,
        debug_session_id: str,
        error_analysis_depth: str = "deep"
    ) -> List[ErrorAnalysisResult]:
        """
        Analyze error patterns in the debugging session.
        
        Args:
            debug_session_id: Debug session ID
            error_analysis_depth: Depth of analysis (quick, standard, deep)
            
        Returns:
            List of error analysis results
        """
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        events = self.session_events[debug_session_id]
        error_events = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        
        if not error_events:
            return []
        
        try:
            analysis_results = []
            
            for error_event in error_events:
                # Analyze error cascade
                cascade_events = await self._find_error_cascade(events, error_event)
                
                # Determine root cause
                root_cause = await self._determine_root_cause(error_event, cascade_events)
                
                # Find affected agents
                affected_agents = list(set([error_event.source_agent_id] + 
                                         [e.source_agent_id for e in cascade_events]))
                
                # Generate resolution steps
                resolution_steps = await self._generate_resolution_steps(error_event, root_cause)
                
                # Find similar errors
                similar_errors = await self._find_similar_errors(debug_session_id, error_event)
                
                analysis_result = ErrorAnalysisResult(
                    error_id=error_event.id,
                    error_type=error_event.metadata.get('error_type', 'unknown'),
                    root_cause=root_cause,
                    affected_agents=affected_agents,
                    error_timeline=[error_event] + cascade_events,
                    cascade_events=cascade_events,
                    severity=self._assess_error_severity(error_event, cascade_events),
                    resolution_steps=resolution_steps,
                    prevention_recommendations=await self._generate_prevention_recommendations(
                        error_event, root_cause
                    ),
                    similar_errors=similar_errors
                )
                
                analysis_results.append(analysis_result)
                
                # Log analysis
                await self._log_debug(
                    debug_session_id,
                    DebugLevel.INFO,
                    f"Error analyzed: {error_event.id}",
                    {
                        'root_cause': root_cause,
                        'cascade_size': len(cascade_events),
                        'severity': analysis_result.severity.value
                    }
                )
            
            self.debug_metrics['errors_analyzed'] += len(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")
            raise
    
    async def detect_bottlenecks(
        self,
        debug_session_id: str,
        analysis_window_minutes: int = 30
    ) -> List[BottleneckAnalysis]:
        """
        Detect communication bottlenecks in the debugging session.
        
        Args:
            debug_session_id: Debug session ID
            analysis_window_minutes: Time window for analysis
            
        Returns:
            List of bottleneck analyses
        """
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        events = self.session_events[debug_session_id]
        
        try:
            bottlenecks = []
            
            # Analyze response time bottlenecks
            response_time_bottlenecks = await self._analyze_response_time_bottlenecks(events)
            bottlenecks.extend(response_time_bottlenecks)
            
            # Analyze agent overload bottlenecks
            agent_overload_bottlenecks = await self._analyze_agent_overload_bottlenecks(events)
            bottlenecks.extend(agent_overload_bottlenecks)
            
            # Analyze message queue bottlenecks
            queue_bottlenecks = await self._analyze_queue_bottlenecks(events)
            bottlenecks.extend(queue_bottlenecks)
            
            # Log bottleneck detection
            await self._log_debug(
                debug_session_id,
                DebugLevel.INFO,
                f"Bottleneck analysis completed: {len(bottlenecks)} found",
                {
                    'response_time_bottlenecks': len(response_time_bottlenecks),
                    'agent_overload_bottlenecks': len(agent_overload_bottlenecks),
                    'queue_bottlenecks': len(queue_bottlenecks)
                }
            )
            
            self.debug_metrics['bottlenecks_detected'] += len(bottlenecks)
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Bottleneck detection failed: {e}")
            raise
    
    async def generate_debug_report(
        self,
        debug_session_id: str,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive debug report for a session.
        
        Args:
            debug_session_id: Debug session ID
            include_recommendations: Whether to include optimization recommendations
            
        Returns:
            Comprehensive debug report
        """
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        session = self.active_sessions[debug_session_id]
        events = self.session_events[debug_session_id]
        
        try:
            # Gather analysis results
            error_analyses = await self.analyze_error_patterns(debug_session_id)
            bottleneck_analyses = await self.detect_bottlenecks(debug_session_id)
            
            # Calculate session metrics
            session_metrics = await self._calculate_debug_session_metrics(events)
            
            # Generate insights
            insights = await self.analyzer.analyze_conversation_events(
                events, [AnalysisType.PATTERN_DETECTION, AnalysisType.PERFORMANCE_ANALYSIS]
            )
            
            # Get debug logs
            debug_logs = list(self.debug_logs.get(debug_session_id, []))
            
            report = {
                'debug_session': session.to_dict(),
                'summary': {
                    'total_events': len(events),
                    'errors_found': len(error_analyses),
                    'bottlenecks_detected': len(bottleneck_analyses),
                    'insights_generated': len(insights),
                    'session_duration': (datetime.utcnow() - session.start_time).total_seconds(),
                    'debug_efficiency': session.current_event_index / len(events) if events else 0
                },
                'metrics': session_metrics,
                'error_analyses': [ea.to_dict() for ea in error_analyses],
                'bottleneck_analyses': [ba.to_dict() for ba in bottleneck_analyses],
                'insights': [insight.to_dict() for insight in insights],
                'debug_logs': debug_logs[-100:],  # Last 100 log entries
                'breakpoint_statistics': {
                    'total_breakpoints': len(session.breakpoints),
                    'breakpoints_hit': sum(bp.hit_count for bp in session.breakpoints),
                    'enabled_breakpoints': len([bp for bp in session.breakpoints if bp.enabled])
                }
            }
            
            # Add recommendations if requested
            if include_recommendations:
                report['recommendations'] = await self._generate_debug_recommendations(
                    debug_session_id, error_analyses, bottleneck_analyses, insights
                )
            
            # Add conversation flow analysis
            report['conversation_flow'] = await self._analyze_conversation_flow(events)
            
            return report
            
        except Exception as e:
            logger.error(f"Debug report generation failed: {e}")
            raise
    
    async def get_debug_session_status(self, debug_session_id: str) -> Dict[str, Any]:
        """Get current status of a debug session."""
        if debug_session_id not in self.active_sessions:
            raise ValueError(f"Debug session {debug_session_id} not found")
        
        session = self.active_sessions[debug_session_id]
        events = self.session_events[debug_session_id]
        
        return {
            'session_id': debug_session_id,
            'target_session': session.target_conversation_session,
            'progress': {
                'current_index': session.current_event_index,
                'total_events': len(events),
                'percentage': (session.current_event_index / len(events) * 100) if events else 0,
                'is_paused': session.is_paused,
                'at_end': session.current_event_index >= len(events)
            },
            'breakpoints': {
                'total': len(session.breakpoints),
                'enabled': len([bp for bp in session.breakpoints if bp.enabled]),
                'hit_count': sum(bp.hit_count for bp in session.breakpoints)
            },
            'runtime_stats': {
                'duration_seconds': (datetime.utcnow() - session.start_time).total_seconds(),
                'debug_logs': len(self.debug_logs.get(debug_session_id, [])),
                'replay_mode': session.replay_mode.value,
                'debug_level': session.debug_level.value
            }
        }
    
    # Private helper methods
    
    async def _perform_initial_analysis(self, debug_session_id: str) -> None:
        """Perform initial analysis of the debug session."""
        events = self.session_events[debug_session_id]
        
        # Quick pattern analysis
        insights = await self.analyzer.analyze_conversation_events(
            events, [AnalysisType.PATTERN_DETECTION]
        )
        
        if insights:
            await self._log_debug(
                debug_session_id,
                DebugLevel.INFO,
                f"Initial analysis found {len(insights)} insights",
                {'insight_types': [i.analysis_type.value for i in insights]}
            )
    
    async def _check_breakpoints(
        self,
        debug_session_id: str,
        event: ConversationEvent
    ) -> Optional[DebugBreakpoint]:
        """Check if event matches any breakpoints."""
        session = self.active_sessions[debug_session_id]
        
        for breakpoint in session.breakpoints:
            if breakpoint.matches_event(event):
                breakpoint.hit_count += 1
                self.debug_metrics['breakpoints_hit'] += 1
                
                await self._log_debug(
                    debug_session_id,
                    DebugLevel.WARNING,
                    f"Breakpoint hit: {breakpoint.breakpoint_id}",
                    {
                        'condition': f"{breakpoint.condition_type}={breakpoint.condition_value}",
                        'action': breakpoint.action,
                        'event_id': event.id
                    }
                )
                
                return breakpoint
        
        return None
    
    async def _analyze_debug_event(
        self,
        debug_session_id: str,
        event: ConversationEvent
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single event during debugging."""
        analysis = {}
        
        # Response time analysis
        if event.response_time_ms:
            if event.response_time_ms > 5000:
                analysis['slow_response'] = {
                    'response_time_ms': event.response_time_ms,
                    'threshold_exceeded': True,
                    'severity': 'warning' if event.response_time_ms < 10000 else 'error'
                }
        
        # Error analysis
        if event.event_type == ConversationEventType.ERROR_OCCURRED:
            analysis['error_detected'] = {
                'error_type': event.metadata.get('error_type', 'unknown'),
                'error_message': event.message_content,
                'requires_investigation': True
            }
        
        # Tool usage analysis
        if event.tool_calls:
            analysis['tool_usage'] = {
                'tool_count': len(event.tool_calls),
                'tools': [tool.get('name', 'unknown') for tool in event.tool_calls],
                'heavy_usage': len(event.tool_calls) > 3
            }
        
        return analysis if analysis else None
    
    async def _find_error_cascade(
        self,
        events: List[ConversationEvent],
        error_event: ConversationEvent
    ) -> List[ConversationEvent]:
        """Find events that are part of an error cascade."""
        cascade_events = []
        error_time = error_event.timestamp
        
        # Look for events within 5 minutes after the error
        for event in events:
            if (event.timestamp > error_time and 
                (event.timestamp - error_time).total_seconds() < 300):
                
                # Check if event is likely related to the error
                if (event.event_type == ConversationEventType.ERROR_OCCURRED or
                    event.source_agent_id == error_event.source_agent_id or
                    event.target_agent_id == error_event.source_agent_id):
                    cascade_events.append(event)
        
        return cascade_events
    
    async def _determine_root_cause(
        self,
        error_event: ConversationEvent,
        cascade_events: List[ConversationEvent]
    ) -> str:
        """Determine the root cause of an error."""
        # Simple root cause analysis - can be enhanced with ML
        error_content = error_event.message_content.lower()
        
        if 'timeout' in error_content:
            return "Communication timeout"
        elif 'connection' in error_content:
            return "Connection failure"
        elif 'resource' in error_content or 'memory' in error_content:
            return "Resource exhaustion"
        elif 'permission' in error_content or 'access' in error_content:
            return "Permission/access issue"
        elif 'validation' in error_content:
            return "Input validation failure"
        elif len(cascade_events) > 3:
            return "Cascade failure"
        else:
            return "Unknown error condition"
    
    async def _generate_resolution_steps(
        self,
        error_event: ConversationEvent,
        root_cause: str
    ) -> List[str]:
        """Generate resolution steps for an error."""
        steps = []
        
        if root_cause == "Communication timeout":
            steps.extend([
                "Check network connectivity",
                "Verify agent response times",
                "Consider increasing timeout values",
                "Review agent workload"
            ])
        elif root_cause == "Connection failure":
            steps.extend([
                "Verify service availability",
                "Check network configuration",
                "Restart affected agents",
                "Review connection pooling"
            ])
        elif root_cause == "Resource exhaustion":
            steps.extend([
                "Check system resource usage",
                "Scale up affected services",
                "Optimize resource allocation",
                "Implement resource monitoring"
            ])
        else:
            steps.extend([
                "Review error logs in detail",
                "Check agent configuration",
                "Verify system dependencies",
                "Consider manual intervention"
            ])
        
        return steps
    
    async def _generate_prevention_recommendations(
        self,
        error_event: ConversationEvent,
        root_cause: str
    ) -> List[str]:
        """Generate recommendations to prevent similar errors."""
        recommendations = []
        
        if root_cause == "Communication timeout":
            recommendations.extend([
                "Implement circuit breakers",
                "Add retry mechanisms with backoff",
                "Monitor response time trends",
                "Set up proactive alerting"
            ])
        elif root_cause == "Resource exhaustion":
            recommendations.extend([
                "Implement resource monitoring",
                "Set up auto-scaling",
                "Add resource usage alerts",
                "Optimize resource consumption"
            ])
        else:
            recommendations.extend([
                "Add comprehensive error handling",
                "Implement health checks",
                "Set up monitoring and alerting",
                "Regular system maintenance"
            ])
        
        return recommendations
    
    async def _find_similar_errors(
        self,
        debug_session_id: str,
        error_event: ConversationEvent
    ) -> List[str]:
        """Find similar errors in the session."""
        events = self.session_events[debug_session_id]
        error_events = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        
        similar = []
        error_content = error_event.message_content.lower()
        
        for event in error_events:
            if event.id != error_event.id:
                # Simple similarity check - can be enhanced
                if (event.source_agent_id == error_event.source_agent_id or
                    any(word in event.message_content.lower() for word in error_content.split()[:3])):
                    similar.append(event.id)
        
        return similar[:5]  # Limit to 5 similar errors
    
    def _assess_error_severity(
        self,
        error_event: ConversationEvent,
        cascade_events: List[ConversationEvent]
    ) -> AlertSeverity:
        """Assess the severity of an error."""
        if len(cascade_events) > 5:
            return AlertSeverity.CRITICAL
        elif len(cascade_events) > 2:
            return AlertSeverity.ERROR
        elif 'critical' in error_event.message_content.lower():
            return AlertSeverity.ERROR
        else:
            return AlertSeverity.WARNING
    
    async def _analyze_response_time_bottlenecks(
        self,
        events: List[ConversationEvent]
    ) -> List[BottleneckAnalysis]:
        """Analyze response time bottlenecks."""
        bottlenecks = []
        
        # Group events by agent
        agent_response_times = defaultdict(list)
        for event in events:
            if event.response_time_ms:
                agent_response_times[event.source_agent_id].append(event.response_time_ms)
        
        # Find agents with consistently slow response times
        for agent_id, response_times in agent_response_times.items():
            if len(response_times) > 5:  # Need sufficient samples
                avg_response_time = sum(response_times) / len(response_times)
                if avg_response_time > 3000:  # More than 3 seconds
                    bottleneck = BottleneckAnalysis(
                        bottleneck_id=f"response_time_{agent_id}",
                        bottleneck_type="response_time",
                        location=agent_id,
                        severity_score=min(avg_response_time / 10000, 1.0),
                        impact_metrics={
                            'average_response_time_ms': avg_response_time,
                            'max_response_time_ms': max(response_times),
                            'affected_requests': len(response_times)
                        },
                        affected_flows=[(agent_id, "all")],
                        recommended_actions=[
                            f"Investigate {agent_id} performance",
                            "Check resource allocation",
                            "Consider agent optimization",
                            "Review workload distribution"
                        ],
                        estimated_improvement="30-50% faster responses"
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _analyze_agent_overload_bottlenecks(
        self,
        events: List[ConversationEvent]
    ) -> List[BottleneckAnalysis]:
        """Analyze agent overload bottlenecks."""
        bottlenecks = []
        
        # Count messages per agent per minute
        agent_message_counts = defaultdict(int)
        for event in events:
            agent_message_counts[event.source_agent_id] += 1
        
        # Calculate message rate (assuming events span reasonable time)
        if events:
            time_span = (events[-1].timestamp - events[0].timestamp).total_seconds() / 60
            if time_span > 0:
                for agent_id, count in agent_message_counts.items():
                    message_rate = count / time_span
                    if message_rate > 10:  # More than 10 messages per minute
                        bottleneck = BottleneckAnalysis(
                            bottleneck_id=f"overload_{agent_id}",
                            bottleneck_type="agent_overload",
                            location=agent_id,
                            severity_score=min(message_rate / 20, 1.0),
                            impact_metrics={
                                'message_rate_per_minute': message_rate,
                                'total_messages': count,
                                'time_span_minutes': time_span
                            },
                            affected_flows=[(agent_id, "all")],
                            recommended_actions=[
                                f"Distribute load from {agent_id}",
                                "Implement rate limiting",
                                "Scale agent horizontally",
                                "Optimize message processing"
                            ],
                            estimated_improvement="Reduced message queue backlog"
                        )
                        bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _analyze_queue_bottlenecks(
        self,
        events: List[ConversationEvent]
    ) -> List[BottleneckAnalysis]:
        """Analyze message queue bottlenecks."""
        # Queue bottleneck analysis would require more detailed queue metrics
        # This is a placeholder implementation
        return []
    
    async def _calculate_debug_session_metrics(
        self,
        events: List[ConversationEvent]
    ) -> Dict[str, Any]:
        """Calculate metrics for debug session."""
        if not events:
            return {}
        
        # Basic metrics
        total_events = len(events)
        error_events = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        
        # Response time metrics
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        
        # Agent activity
        agent_activity = defaultdict(int)
        for event in events:
            agent_activity[event.source_agent_id] += 1
        
        return {
            'total_events': total_events,
            'error_count': len(error_events),
            'error_rate': len(error_events) / total_events if total_events > 0 else 0,
            'unique_agents': len(set(e.source_agent_id for e in events)),
            'response_time_stats': {
                'average_ms': sum(response_times) / len(response_times) if response_times else 0,
                'max_ms': max(response_times) if response_times else 0,
                'min_ms': min(response_times) if response_times else 0,
                'samples': len(response_times)
            },
            'agent_activity': dict(agent_activity),
            'time_span_minutes': (
                (events[-1].timestamp - events[0].timestamp).total_seconds() / 60
                if len(events) > 1 else 0
            )
        }
    
    async def _generate_debug_recommendations(
        self,
        debug_session_id: str,
        error_analyses: List[ErrorAnalysisResult],
        bottleneck_analyses: List[BottleneckAnalysis],
        insights: List[CommunicationInsight]
    ) -> List[Dict[str, Any]]:
        """Generate debugging recommendations."""
        recommendations = []
        
        # Error-based recommendations
        if error_analyses:
            high_severity_errors = [ea for ea in error_analyses if ea.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]]
            if high_severity_errors:
                recommendations.append({
                    'type': 'error_resolution',
                    'priority': 'high',
                    'title': 'Critical Errors Require Immediate Attention',
                    'description': f'{len(high_severity_errors)} critical errors detected',
                    'actions': [
                        'Review error analysis results',
                        'Implement recommended resolution steps',
                        'Add error prevention measures',
                        'Set up monitoring for similar issues'
                    ]
                })
        
        # Bottleneck-based recommendations
        if bottleneck_analyses:
            high_impact_bottlenecks = [ba for ba in bottleneck_analyses if ba.severity_score > 0.7]
            if high_impact_bottlenecks:
                recommendations.append({
                    'type': 'performance_optimization',
                    'priority': 'medium',
                    'title': 'Performance Bottlenecks Detected',
                    'description': f'{len(high_impact_bottlenecks)} high-impact bottlenecks found',
                    'actions': [
                        'Address response time issues',
                        'Optimize agent workload distribution',
                        'Scale resources as needed',
                        'Implement performance monitoring'
                    ]
                })
        
        # Insight-based recommendations
        critical_insights = [i for i in insights if i.severity == AlertSeverity.CRITICAL]
        if critical_insights:
            recommendations.append({
                'type': 'system_improvement',
                'priority': 'medium',
                'title': 'System Improvements Recommended',
                'description': f'{len(critical_insights)} critical insights generated',
                'actions': [
                    'Review system architecture',
                    'Implement suggested improvements',
                    'Add monitoring and alerting',
                    'Schedule regular system reviews'
                ]
            })
        
        return recommendations
    
    async def _analyze_conversation_flow(
        self,
        events: List[ConversationEvent]
    ) -> Dict[str, Any]:
        """Analyze conversation flow patterns."""
        if not events:
            return {}
        
        # Build conversation flow graph
        flows = defaultdict(int)
        for event in events:
            if event.target_agent_id:
                flow_key = (event.source_agent_id, event.target_agent_id)
                flows[flow_key] += 1
        
        # Find most active flows
        top_flows = sorted(flows.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Detect circular flows
        circular_flows = []
        for (source, target), count in flows.items():
            reverse_flow = (target, source)
            if reverse_flow in flows:
                circular_flows.append({
                    'agents': [source, target],
                    'forward_count': count,
                    'reverse_count': flows[reverse_flow]
                })
        
        return {
            'total_flows': len(flows),
            'top_flows': [
                {
                    'source': source,
                    'target': target,
                    'message_count': count,
                    'percentage': count / sum(flows.values()) * 100
                }
                for (source, target), count in top_flows
            ],
            'circular_flows': circular_flows[:5],  # Top 5 circular flows
            'flow_diversity': len(flows) / len(events) if events else 0
        }
    
    async def _log_debug(
        self,
        debug_session_id: str,
        level: DebugLevel,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log debug message to session log."""
        if debug_session_id not in self.debug_logs:
            self.debug_logs[debug_session_id] = deque(maxlen=1000)
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.value,
            'message': message,
            'metadata': metadata or {}
        }
        
        self.debug_logs[debug_session_id].append(log_entry)
        
        # Also log to main logger
        logger.info(
            f"Debug[{debug_session_id}]: {message}",
            level=level.value,
            metadata=metadata
        )
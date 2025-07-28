"""
Communication Analyzer for LeanVibe Agent Hive 2.0

Advanced pattern detection, debugging insights, and communication analysis
for multi-agent systems with ML-enhanced pattern recognition capabilities.

Features:
- Real-time communication pattern detection
- Bottleneck and performance analysis
- Error cascade detection and prevention
- Agent behavior profiling and anomaly detection
- Communication flow optimization recommendations
- Advanced debugging insights and visualizations
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

import structlog
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from .chat_transcript_manager import (
    ConversationEvent, ConversationEventType, ConversationPattern,
    ConversationThread, ConversationMetrics
)

logger = structlog.get_logger()


class AnalysisType(Enum):
    """Types of communication analysis."""
    PATTERN_DETECTION = "pattern_detection"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    BEHAVIOR_PROFILING = "behavior_profiling"
    BOTTLENECK_DETECTION = "bottleneck_detection"
    ERROR_ANALYSIS = "error_analysis"
    FLOW_OPTIMIZATION = "flow_optimization"
    ANOMALY_DETECTION = "anomaly_detection"


class AlertSeverity(Enum):
    """Severity levels for communication alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CommunicationInsight:
    """Represents an insight discovered from communication analysis."""
    id: str
    analysis_type: AnalysisType
    severity: AlertSeverity
    title: str
    description: str
    affected_agents: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'analysis_type': self.analysis_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentBehaviorProfile:
    """Profile of an agent's communication behavior."""
    agent_id: str
    session_id: str
    message_frequency: float  # messages per minute
    average_response_time: float  # milliseconds
    tool_usage_rate: float  # tools per message
    context_sharing_rate: float  # context shares per message
    error_rate: float  # errors per message
    collaboration_score: float  # 0-1 based on interactions with other agents
    activity_pattern: Dict[str, int]  # hourly activity distribution
    preferred_communication_types: Dict[str, int]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class CommunicationFlow:
    """Represents communication flow between agents."""
    source_agent: str
    target_agent: str
    message_count: int
    average_response_time: float
    success_rate: float
    flow_efficiency: float  # 0-1 score
    bottleneck_indicators: List[str]
    last_interaction: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'last_interaction': self.last_interaction.isoformat()
        }


@dataclass
class BottleneckAnalysis:
    """Analysis of communication bottlenecks."""
    bottleneck_type: str
    affected_agents: List[str]
    severity_score: float  # 0-1
    impact_metrics: Dict[str, float]
    root_cause: str
    resolution_suggestions: List[str]
    estimated_performance_gain: float  # percentage improvement
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommunicationAnalyzer:
    """
    Advanced communication analyzer for multi-agent system debugging.
    
    Provides real-time pattern detection, performance analysis, and
    optimization recommendations with ML-enhanced insights.
    """
    
    def __init__(self):
        # Analysis state
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        self.communication_flows: Dict[Tuple[str, str], CommunicationFlow] = {}
        self.recent_insights: deque = deque(maxlen=1000)
        self.pattern_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.analysis_metrics: Dict[str, List[float]] = defaultdict(list)
        self.bottleneck_history: List[BottleneckAnalysis] = []
        
        # Configuration
        self.analysis_window_minutes = 30
        self.pattern_detection_threshold = 0.7
        self.anomaly_detection_enabled = True
        self.real_time_analysis_enabled = True
        
        # Statistical models for anomaly detection
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        logger.info("CommunicationAnalyzer initialized")
    
    async def analyze_conversation_events(
        self,
        events: List[ConversationEvent],
        analysis_types: List[AnalysisType] = None
    ) -> List[CommunicationInsight]:
        """
        Comprehensive analysis of conversation events.
        
        Args:
            events: List of conversation events to analyze
            analysis_types: Specific types of analysis to perform
            
        Returns:
            List of communication insights and recommendations
        """
        if not events:
            return []
        
        if analysis_types is None:
            analysis_types = list(AnalysisType)
        
        insights = []
        
        try:
            # Update agent profiles
            await self._update_agent_profiles(events)
            
            # Update communication flows
            await self._update_communication_flows(events)
            
            # Perform requested analyses
            for analysis_type in analysis_types:
                if analysis_type == AnalysisType.PATTERN_DETECTION:
                    insights.extend(await self._detect_communication_patterns(events))
                elif analysis_type == AnalysisType.PERFORMANCE_ANALYSIS:
                    insights.extend(await self._analyze_performance(events))
                elif analysis_type == AnalysisType.BEHAVIOR_PROFILING:
                    insights.extend(await self._analyze_agent_behavior(events))
                elif analysis_type == AnalysisType.BOTTLENECK_DETECTION:
                    insights.extend(await self._detect_bottlenecks(events))
                elif analysis_type == AnalysisType.ERROR_ANALYSIS:
                    insights.extend(await self._analyze_errors(events))
                elif analysis_type == AnalysisType.FLOW_OPTIMIZATION:
                    insights.extend(await self._optimize_communication_flow(events))
                elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                    insights.extend(await self._detect_anomalies(events))
            
            # Store insights
            for insight in insights:
                self.recent_insights.append(insight)
            
            logger.info(
                "Communication analysis completed",
                event_count=len(events),
                insight_count=len(insights),
                analysis_types=[at.value for at in analysis_types]
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Communication analysis failed: {e}")
            raise
    
    async def get_agent_behavior_profile(self, agent_id: str) -> Optional[AgentBehaviorProfile]:
        """Get behavior profile for a specific agent."""
        return self.agent_profiles.get(agent_id)
    
    async def get_communication_flow_analysis(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive communication flow analysis for a session.
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Flow analysis with visualizations and recommendations
        """
        try:
            # Filter flows for this session
            session_flows = [
                flow for flow in self.communication_flows.values()
                if any(agent.startswith(session_id) for agent in [flow.source_agent, flow.target_agent])
            ]
            
            if not session_flows:
                return {
                    "session_id": session_id,
                    "flows": [],
                    "insights": [],
                    "recommendations": []
                }
            
            # Calculate flow metrics
            total_messages = sum(flow.message_count for flow in session_flows)
            average_response_time = statistics.mean([flow.average_response_time for flow in session_flows])
            overall_success_rate = statistics.mean([flow.success_rate for flow in session_flows])
            
            # Identify bottlenecks
            bottlenecks = [
                flow for flow in session_flows
                if flow.flow_efficiency < 0.5 or flow.average_response_time > 5000
            ]
            
            # Generate recommendations
            recommendations = []
            if bottlenecks:
                recommendations.append({
                    "type": "bottleneck_resolution",
                    "priority": "high",
                    "description": f"Detected {len(bottlenecks)} communication bottlenecks",
                    "actions": [
                        "Review agent load balancing",
                        "Optimize message routing",
                        "Consider agent scaling",
                        "Implement message prioritization"
                    ]
                })
            
            if overall_success_rate < 0.9:
                recommendations.append({
                    "type": "reliability_improvement",
                    "priority": "medium",
                    "description": f"Success rate is {overall_success_rate:.1%}",
                    "actions": [
                        "Implement retry mechanisms",
                        "Add error handling",
                        "Review message validation",
                        "Monitor agent health"
                    ]
                })
            
            return {
                "session_id": session_id,
                "metrics": {
                    "total_flows": len(session_flows),
                    "total_messages": total_messages,
                    "average_response_time": average_response_time,
                    "success_rate": overall_success_rate,
                    "bottleneck_count": len(bottlenecks)
                },
                "flows": [flow.to_dict() for flow in session_flows],
                "bottlenecks": [flow.to_dict() for flow in bottlenecks],
                "recommendations": recommendations,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Flow analysis failed for session {session_id}: {e}")
            raise
    
    async def detect_real_time_issues(
        self,
        recent_events: List[ConversationEvent],
        time_window_minutes: int = 5
    ) -> List[CommunicationInsight]:
        """
        Real-time detection of communication issues.
        
        Args:
            recent_events: Recent conversation events
            time_window_minutes: Time window for analysis
            
        Returns:
            List of immediate issues requiring attention
        """
        if not self.real_time_analysis_enabled:
            return []
        
        critical_insights = []
        
        try:
            # Filter events to time window
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            recent = [e for e in recent_events if e.timestamp >= cutoff_time]
            
            if not recent:
                return []
            
            # Detect error spikes
            error_events = [
                e for e in recent
                if e.event_type == ConversationEventType.ERROR_OCCURRED
            ]
            
            if len(error_events) > len(recent) * 0.2:  # More than 20% errors
                critical_insights.append(CommunicationInsight(
                    id=f"error_spike_{datetime.utcnow().timestamp()}",
                    analysis_type=AnalysisType.ERROR_ANALYSIS,
                    severity=AlertSeverity.CRITICAL,
                    title="Error Spike Detected",
                    description=f"High error rate detected: {len(error_events)}/{len(recent)} events",
                    affected_agents=list(set(e.source_agent_id for e in error_events)),
                    recommendations=[
                        "Investigate error patterns",
                        "Check agent health status",
                        "Review recent configuration changes",
                        "Consider agent restart if persistent"
                    ],
                    metrics={
                        "error_rate": len(error_events) / len(recent),
                        "total_errors": len(error_events),
                        "time_window_minutes": time_window_minutes
                    },
                    timestamp=datetime.utcnow(),
                    session_id=recent[0].session_id if recent else "unknown"
                ))
            
            # Detect response time degradation
            response_times = [e.response_time_ms for e in recent if e.response_time_ms]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                if avg_response_time > 10000:  # More than 10 seconds
                    critical_insights.append(CommunicationInsight(
                        id=f"slow_response_{datetime.utcnow().timestamp()}",
                        analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                        severity=AlertSeverity.WARNING,
                        title="Slow Response Times Detected",
                        description=f"Average response time: {avg_response_time:.0f}ms",
                        affected_agents=list(set(e.source_agent_id for e in recent if e.response_time_ms and e.response_time_ms > 5000)),
                        recommendations=[
                            "Check system resources",
                            "Review agent workload",
                            "Consider load balancing",
                            "Optimize message processing"
                        ],
                        metrics={
                            "average_response_time": avg_response_time,
                            "max_response_time": max(response_times),
                            "slow_responses": len([t for t in response_times if t > 5000])
                        },
                        timestamp=datetime.utcnow(),
                        session_id=recent[0].session_id if recent else "unknown"
                    ))
            
            # Detect infinite loops
            agent_message_counts = defaultdict(int)
            for event in recent:
                agent_message_counts[event.source_agent_id] += 1
            
            max_messages = max(agent_message_counts.values()) if agent_message_counts else 0
            if max_messages > 20:  # More than 20 messages from one agent in 5 minutes
                busiest_agent = max(agent_message_counts, key=agent_message_counts.get)
                critical_insights.append(CommunicationInsight(
                    id=f"potential_loop_{datetime.utcnow().timestamp()}",
                    analysis_type=AnalysisType.PATTERN_DETECTION,
                    severity=AlertSeverity.ERROR,
                    title="Potential Infinite Loop Detected",
                    description=f"Agent {busiest_agent} sent {max_messages} messages in {time_window_minutes} minutes",
                    affected_agents=[busiest_agent],
                    recommendations=[
                        "Review agent logic for loops",
                        "Check termination conditions",
                        "Implement message rate limiting",
                        "Consider manual intervention"
                    ],
                    metrics={
                        "message_count": max_messages,
                        "time_window_minutes": time_window_minutes,
                        "agent_activity": dict(agent_message_counts)
                    },
                    timestamp=datetime.utcnow(),
                    session_id=recent[0].session_id if recent else "unknown"
                ))
            
            return critical_insights
            
        except Exception as e:
            logger.error(f"Real-time issue detection failed: {e}")
            return []
    
    async def generate_optimization_report(
        self,
        session_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report for communication patterns.
        
        Args:
            session_id: Session to analyze
            time_window_hours: Time window for analysis
            
        Returns:
            Optimization report with actionable recommendations
        """
        try:
            # Gather metrics
            session_profiles = {
                agent_id: profile for agent_id, profile in self.agent_profiles.items()
                if profile.session_id == session_id
            }
            
            session_flows = [
                flow for flow in self.communication_flows.values()
                if session_id in flow.source_agent or session_id in flow.target_agent
            ]
            
            # Calculate optimization opportunities
            optimization_opportunities = []
            
            # 1. Bottleneck resolution
            bottlenecks = [flow for flow in session_flows if flow.flow_efficiency < 0.6]
            if bottlenecks:
                potential_improvement = sum(
                    (0.9 - flow.flow_efficiency) * flow.message_count
                    for flow in bottlenecks
                ) / sum(flow.message_count for flow in session_flows) * 100
                
                optimization_opportunities.append({
                    "type": "bottleneck_resolution",
                    "priority": "high",
                    "description": f"{len(bottlenecks)} communication bottlenecks identified",
                    "potential_improvement": f"{potential_improvement:.1f}% throughput increase",
                    "effort": "medium",
                    "affected_flows": len(bottlenecks)
                })
            
            # 2. Load balancing
            if session_profiles:
                message_frequencies = [p.message_frequency for p in session_profiles.values()]
                if len(message_frequencies) > 1:
                    freq_std = statistics.stdev(message_frequencies)
                    freq_mean = statistics.mean(message_frequencies)
                    if freq_std / freq_mean > 0.5:  # High variation
                        optimization_opportunities.append({
                            "type": "load_balancing",
                            "priority": "medium",
                            "description": "Uneven message distribution across agents",
                            "potential_improvement": "20-30% better resource utilization",
                            "effort": "high",
                            "metrics": {
                                "coefficient_of_variation": freq_std / freq_mean,
                                "message_frequency_range": [min(message_frequencies), max(message_frequencies)]
                            }
                        })
            
            # 3. Error reduction
            error_rates = [p.error_rate for p in session_profiles.values()]
            if error_rates:
                avg_error_rate = statistics.mean(error_rates)
                if avg_error_rate > 0.05:  # More than 5% error rate
                    optimization_opportunities.append({
                        "type": "error_reduction",
                        "priority": "high",
                        "description": f"High error rate: {avg_error_rate:.1%}",
                        "potential_improvement": "Improved reliability and performance",
                        "effort": "medium",
                        "target_error_rate": "< 2%"
                    })
            
            # 4. Response time optimization
            response_times = [p.average_response_time for p in session_profiles.values()]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                if avg_response_time > 2000:  # More than 2 seconds
                    optimization_opportunities.append({
                        "type": "response_time_optimization",
                        "priority": "medium",
                        "description": f"Slow average response time: {avg_response_time:.0f}ms",
                        "potential_improvement": "40-60% faster responses",
                        "effort": "medium",
                        "target_response_time": "< 1000ms"
                    })
            
            # Generate summary
            total_messages = sum(flow.message_count for flow in session_flows)
            overall_efficiency = statistics.mean([flow.flow_efficiency for flow in session_flows]) if session_flows else 0
            
            return {
                "session_id": session_id,
                "analysis_period": f"{time_window_hours} hours",
                "summary": {
                    "total_agents": len(session_profiles),
                    "total_flows": len(session_flows),
                    "total_messages": total_messages,
                    "overall_efficiency": overall_efficiency,
                    "optimization_opportunities": len(optimization_opportunities)
                },
                "current_performance": {
                    "agent_profiles": {aid: profile.to_dict() for aid, profile in session_profiles.items()},
                    "flow_efficiency": overall_efficiency,
                    "bottleneck_count": len(bottlenecks) if 'bottlenecks' in locals() else 0
                },
                "optimization_opportunities": optimization_opportunities,
                "implementation_roadmap": [
                    {
                        "phase": "Immediate (0-1 week)",
                        "actions": [opp for opp in optimization_opportunities if opp["priority"] == "high" and opp["effort"] != "high"]
                    },
                    {
                        "phase": "Short-term (1-4 weeks)",
                        "actions": [opp for opp in optimization_opportunities if opp["priority"] == "medium" or opp["effort"] == "medium"]
                    },
                    {
                        "phase": "Long-term (1-3 months)",
                        "actions": [opp for opp in optimization_opportunities if opp["effort"] == "high"]
                    }
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Optimization report generation failed: {e}")
            raise
    
    # Private helper methods
    
    async def _update_agent_profiles(self, events: List[ConversationEvent]) -> None:
        """Update agent behavior profiles based on events."""
        agent_event_groups = defaultdict(list)
        for event in events:
            agent_event_groups[event.source_agent_id].append(event)
        
        for agent_id, agent_events in agent_event_groups.items():
            if agent_id not in self.agent_profiles:
                self.agent_profiles[agent_id] = AgentBehaviorProfile(
                    agent_id=agent_id,
                    session_id=agent_events[0].session_id,
                    message_frequency=0,
                    average_response_time=0,
                    tool_usage_rate=0,
                    context_sharing_rate=0,
                    error_rate=0,
                    collaboration_score=0,
                    activity_pattern={},
                    preferred_communication_types={},
                    last_updated=datetime.utcnow()
                )
            
            profile = self.agent_profiles[agent_id]
            
            # Calculate metrics
            time_span = (max(e.timestamp for e in agent_events) - min(e.timestamp for e in agent_events)).total_seconds() / 60
            if time_span > 0:
                profile.message_frequency = len(agent_events) / time_span
            
            response_times = [e.response_time_ms for e in agent_events if e.response_time_ms]
            if response_times:
                profile.average_response_time = statistics.mean(response_times)
            
            total_tool_calls = sum(len(e.tool_calls) for e in agent_events)
            profile.tool_usage_rate = total_tool_calls / len(agent_events) if agent_events else 0
            
            total_context_shares = sum(len(e.context_references) for e in agent_events)
            profile.context_sharing_rate = total_context_shares / len(agent_events) if agent_events else 0
            
            error_events = [e for e in agent_events if e.event_type == ConversationEventType.ERROR_OCCURRED]
            profile.error_rate = len(error_events) / len(agent_events) if agent_events else 0
            
            # Calculate collaboration score
            unique_targets = set(e.target_agent_id for e in agent_events if e.target_agent_id)
            profile.collaboration_score = min(len(unique_targets) / 5, 1.0)  # Normalize to 0-1
            
            profile.last_updated = datetime.utcnow()
    
    async def _update_communication_flows(self, events: List[ConversationEvent]) -> None:
        """Update communication flow metrics."""
        flow_events = defaultdict(list)
        
        for event in events:
            if event.target_agent_id:  # Skip broadcasts
                flow_key = (event.source_agent_id, event.target_agent_id)
                flow_events[flow_key].append(event)
        
        for flow_key, flow_event_list in flow_events.items():
            source, target = flow_key
            
            if flow_key not in self.communication_flows:
                self.communication_flows[flow_key] = CommunicationFlow(
                    source_agent=source,
                    target_agent=target,
                    message_count=0,
                    average_response_time=0,
                    success_rate=1.0,
                    flow_efficiency=1.0,
                    bottleneck_indicators=[],
                    last_interaction=datetime.utcnow()
                )
            
            flow = self.communication_flows[flow_key]
            
            # Update metrics
            flow.message_count += len(flow_event_list)
            
            response_times = [e.response_time_ms for e in flow_event_list if e.response_time_ms]
            if response_times:
                flow.average_response_time = statistics.mean(response_times)
            
            error_events = [e for e in flow_event_list if e.event_type == ConversationEventType.ERROR_OCCURRED]
            flow.success_rate = 1.0 - (len(error_events) / len(flow_event_list))
            
            # Calculate flow efficiency (inverse of response time, normalized)
            if flow.average_response_time > 0:
                flow.flow_efficiency = min(1000 / flow.average_response_time, 1.0)
            
            # Identify bottleneck indicators
            flow.bottleneck_indicators = []
            if flow.average_response_time > 5000:
                flow.bottleneck_indicators.append("high_response_time")
            if flow.success_rate < 0.9:
                flow.bottleneck_indicators.append("low_success_rate")
            if len(flow_event_list) > 100:  # High volume
                flow.bottleneck_indicators.append("high_volume")
            
            flow.last_interaction = max(e.timestamp for e in flow_event_list)
    
    async def _detect_communication_patterns(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Detect communication patterns and anomalies."""
        insights = []
        
        # Pattern: Request-Response chains
        chains = self._identify_request_response_chains(events)
        long_chains = [chain for chain in chains if len(chain) > 10]
        
        if long_chains:
            insights.append(CommunicationInsight(
                id=f"long_chain_{datetime.utcnow().timestamp()}",
                analysis_type=AnalysisType.PATTERN_DETECTION,
                severity=AlertSeverity.WARNING,
                title="Long Communication Chains Detected",
                description=f"Found {len(long_chains)} chains longer than 10 messages",
                affected_agents=list(set(
                    agent for chain in long_chains for event in chain for agent in [event.source_agent_id, event.target_agent_id] if agent
                )),
                recommendations=[
                    "Review chain termination conditions",
                    "Consider breaking long chains into smaller tasks",
                    "Implement chain length monitoring",
                    "Add intermediate checkpoints"
                ],
                metrics={
                    "longest_chain": max(len(chain) for chain in long_chains),
                    "total_long_chains": len(long_chains),
                    "average_chain_length": statistics.mean([len(chain) for chain in long_chains])
                },
                timestamp=datetime.utcnow(),
                session_id=events[0].session_id if events else "unknown"
            ))
        
        return insights
    
    async def _analyze_performance(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Analyze performance metrics and identify issues."""
        insights = []
        
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        if not response_times:
            return insights
        
        # Analyze response time distribution
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        
        if p95_response_time > 5000:  # 5 seconds
            insights.append(CommunicationInsight(
                id=f"slow_p95_{datetime.utcnow().timestamp()}",
                analysis_type=AnalysisType.PERFORMANCE_ANALYSIS,
                severity=AlertSeverity.WARNING,
                title="Slow P95 Response Times",
                description=f"95th percentile response time: {p95_response_time:.0f}ms",
                affected_agents=list(set(
                    e.source_agent_id for e in events
                    if e.response_time_ms and e.response_time_ms > p95_response_time * 0.8
                )),
                recommendations=[
                    "Profile slow agents",
                    "Optimize message processing",
                    "Check resource utilization",
                    "Consider horizontal scaling"
                ],
                metrics={
                    "p95_response_time": p95_response_time,
                    "p99_response_time": p99_response_time,
                    "slow_message_count": len([t for t in response_times if t > 5000])
                },
                timestamp=datetime.utcnow(),
                session_id=events[0].session_id if events else "unknown"
            ))
        
        return insights
    
    async def _analyze_agent_behavior(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Analyze agent behavior patterns."""
        insights = []
        
        # Analyze message frequency per agent
        agent_message_counts = defaultdict(int)
        for event in events:
            agent_message_counts[event.source_agent_id] += 1
        
        if len(agent_message_counts) > 1:
            frequencies = list(agent_message_counts.values())
            if statistics.stdev(frequencies) / statistics.mean(frequencies) > 1.0:  # High variation
                most_active = max(agent_message_counts, key=agent_message_counts.get)
                least_active = min(agent_message_counts, key=agent_message_counts.get)
                
                insights.append(CommunicationInsight(
                    id=f"uneven_activity_{datetime.utcnow().timestamp()}",
                    analysis_type=AnalysisType.BEHAVIOR_PROFILING,
                    severity=AlertSeverity.INFO,
                    title="Uneven Agent Activity",
                    description=f"High variation in agent activity levels",
                    affected_agents=[most_active, least_active],
                    recommendations=[
                        "Review task distribution",
                        "Consider load balancing",
                        "Monitor agent capacity",
                        "Investigate inactive agents"
                    ],
                    metrics={
                        "most_active_agent": most_active,
                        "most_active_count": agent_message_counts[most_active],
                        "least_active_agent": least_active,
                        "least_active_count": agent_message_counts[least_active],
                        "activity_coefficient_of_variation": statistics.stdev(frequencies) / statistics.mean(frequencies)
                    },
                    timestamp=datetime.utcnow(),
                    session_id=events[0].session_id if events else "unknown"
                ))
        
        return insights
    
    async def _detect_bottlenecks(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Detect communication bottlenecks."""
        insights = []
        
        # Find agents that are frequently targets but slow to respond
        target_response_times = defaultdict(list)
        for event in events:
            if event.target_agent_id and event.response_time_ms:
                target_response_times[event.target_agent_id].append(event.response_time_ms)
        
        for agent_id, response_times in target_response_times.items():
            if len(response_times) > 5:  # Enough samples
                avg_response_time = statistics.mean(response_times)
                if avg_response_time > 3000:  # More than 3 seconds
                    insights.append(CommunicationInsight(
                        id=f"bottleneck_{agent_id}_{datetime.utcnow().timestamp()}",
                        analysis_type=AnalysisType.BOTTLENECK_DETECTION,
                        severity=AlertSeverity.WARNING,
                        title=f"Bottleneck Detected: {agent_id}",
                        description=f"Agent shows slow response times: {avg_response_time:.0f}ms average",
                        affected_agents=[agent_id],
                        recommendations=[
                            f"Investigate {agent_id} performance",
                            "Check resource allocation",
                            "Consider agent optimization",
                            "Review message queue length"
                        ],
                        metrics={
                            "average_response_time": avg_response_time,
                            "max_response_time": max(response_times),
                            "message_count": len(response_times),
                            "p95_response_time": np.percentile(response_times, 95)
                        },
                        timestamp=datetime.utcnow(),
                        session_id=events[0].session_id if events else "unknown"
                    ))
        
        return insights
    
    async def _analyze_errors(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Analyze error patterns and cascades."""
        insights = []
        
        error_events = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        if not error_events:
            return insights
        
        # Detect error cascades (multiple errors in short time)
        error_times = [e.timestamp for e in error_events]
        error_times.sort()
        
        cascades = []
        current_cascade = [error_times[0]] if error_times else []
        
        for i in range(1, len(error_times)):
            if (error_times[i] - error_times[i-1]).total_seconds() < 300:  # Within 5 minutes
                current_cascade.append(error_times[i])
            else:
                if len(current_cascade) > 2:  # At least 3 errors
                    cascades.append(current_cascade)
                current_cascade = [error_times[i]]
        
        if len(current_cascade) > 2:
            cascades.append(current_cascade)
        
        if cascades:
            insights.append(CommunicationInsight(
                id=f"error_cascade_{datetime.utcnow().timestamp()}",
                analysis_type=AnalysisType.ERROR_ANALYSIS,
                severity=AlertSeverity.ERROR,
                title="Error Cascades Detected",
                description=f"Found {len(cascades)} error cascades",
                affected_agents=list(set(e.source_agent_id for e in error_events)),
                recommendations=[
                    "Implement circuit breakers",
                    "Add error isolation",
                    "Review error handling",
                    "Investigate root causes"
                ],
                metrics={
                    "cascade_count": len(cascades),
                    "largest_cascade": max(len(cascade) for cascade in cascades),
                    "total_errors": len(error_events),
                    "error_rate": len(error_events) / len(events)
                },
                timestamp=datetime.utcnow(),
                session_id=events[0].session_id if events else "unknown"
            ))
        
        return insights
    
    async def _optimize_communication_flow(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Generate flow optimization recommendations."""
        insights = []
        
        # Analyze message routing efficiency
        broadcast_events = [e for e in events if not e.target_agent_id]
        direct_events = [e for e in events if e.target_agent_id]
        
        if len(broadcast_events) > len(direct_events) * 0.5:  # Too many broadcasts
            insights.append(CommunicationInsight(
                id=f"broadcast_optimization_{datetime.utcnow().timestamp()}",
                analysis_type=AnalysisType.FLOW_OPTIMIZATION,
                severity=AlertSeverity.INFO,
                title="Broadcast Message Optimization",
                description=f"High broadcast ratio: {len(broadcast_events)}/{len(events)} messages",
                affected_agents=list(set(e.source_agent_id for e in broadcast_events)),
                recommendations=[
                    "Convert broadcasts to direct messages",
                    "Implement message routing",
                    "Use topic-based messaging",
                    "Optimize recipient lists"
                ],
                metrics={
                    "broadcast_count": len(broadcast_events),
                    "direct_count": len(direct_events),
                    "broadcast_ratio": len(broadcast_events) / len(events)
                },
                timestamp=datetime.utcnow(),
                session_id=events[0].session_id if events else "unknown"
            ))
        
        return insights
    
    async def _detect_anomalies(self, events: List[ConversationEvent]) -> List[CommunicationInsight]:
        """Detect anomalous communication patterns."""
        if not self.anomaly_detection_enabled:
            return []
        
        insights = []
        
        # Statistical anomaly detection for response times
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        if len(response_times) > 10:
            # Use z-score for anomaly detection
            mean_rt = statistics.mean(response_times)
            std_rt = statistics.stdev(response_times) if len(response_times) > 1 else 0
            
            if std_rt > 0:
                anomalous_events = []
                for event in events:
                    if event.response_time_ms:
                        z_score = abs((event.response_time_ms - mean_rt) / std_rt)
                        if z_score > 3:  # 3 standard deviations
                            anomalous_events.append(event)
                
                if anomalous_events:
                    insights.append(CommunicationInsight(
                        id=f"response_anomaly_{datetime.utcnow().timestamp()}",
                        analysis_type=AnalysisType.ANOMALY_DETECTION,
                        severity=AlertSeverity.WARNING,
                        title="Response Time Anomalies",
                        description=f"Detected {len(anomalous_events)} response time anomalies",
                        affected_agents=list(set(e.source_agent_id for e in anomalous_events)),
                        recommendations=[
                            "Investigate anomalous agents",
                            "Check for resource constraints",
                            "Review unusual workloads",
                            "Monitor system health"
                        ],
                        metrics={
                            "anomaly_count": len(anomalous_events),
                            "mean_response_time": mean_rt,
                            "std_response_time": std_rt,
                            "max_z_score": max(
                                abs((e.response_time_ms - mean_rt) / std_rt)
                                for e in anomalous_events
                            )
                        },
                        timestamp=datetime.utcnow(),
                        session_id=events[0].session_id if events else "unknown"
                    ))
        
        return insights
    
    def _identify_request_response_chains(self, events: List[ConversationEvent]) -> List[List[ConversationEvent]]:
        """Identify request-response chains in communication."""
        chains = []
        
        # Simple chain detection - can be enhanced with more sophisticated algorithms
        events_by_time = sorted(events, key=lambda e: e.timestamp)
        
        i = 0
        while i < len(events_by_time):
            current_chain = [events_by_time[i]]
            j = i + 1
            
            # Look for responses within a reasonable time window
            while j < len(events_by_time) and j < i + 20:  # Limit chain detection
                time_diff = (events_by_time[j].timestamp - events_by_time[j-1].timestamp).total_seconds()
                
                if time_diff < 300:  # Within 5 minutes
                    # Check if it's a response (target becomes source)
                    prev_target = events_by_time[j-1].target_agent_id
                    current_source = events_by_time[j].source_agent_id
                    
                    if prev_target == current_source:
                        current_chain.append(events_by_time[j])
                    else:
                        break
                else:
                    break
                j += 1
            
            if len(current_chain) > 1:
                chains.append(current_chain)
            
            i += 1
        
        return chains
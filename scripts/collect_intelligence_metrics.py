#!/usr/bin/env python3
"""
Intelligence Metrics Collector - LeanVibe Agent Hive 2.0

Comprehensive metrics collection system for tracking and analyzing intelligence
enhancement, agent collaboration effectiveness, and semantic memory utilization
across Phase 3 implementation.

Metrics Categories:
1. Workflow Intelligence Enhancement
2. Cross-Agent Collaboration Effectiveness  
3. Semantic Memory Utilization
4. Context Compression and Efficiency
5. Knowledge Sharing and Learning Progression
6. Performance Impact Analysis
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import statistics
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# INTELLIGENCE METRICS DATA STRUCTURES
# =============================================================================

@dataclass
class IntelligenceMetric:
    """Individual intelligence metric measurement."""
    metric_id: str
    metric_type: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_agent: Optional[str] = None
    workflow_id: Optional[str] = None

@dataclass
class CollaborationEvent:
    """Agent collaboration event record."""
    event_id: str
    event_type: str  # knowledge_sharing, consensus_building, task_coordination
    participants: List[str]
    timestamp: datetime
    duration_ms: float
    effectiveness_score: float
    knowledge_exchanged: int
    outcome: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextUtilizationMetric:
    """Context utilization and compression metrics."""
    utilization_id: str
    workflow_id: str
    agent_id: str
    timestamp: datetime
    original_context_size: int
    compressed_context_size: int
    compression_ratio: float
    semantic_preservation_score: float
    retrieval_latency_ms: float
    relevance_score: float
    context_sources: List[str] = field(default_factory=list)

@dataclass
class LearningProgressionMetric:
    """Learning and knowledge evolution metrics."""
    learning_id: str
    agent_id: str
    timestamp: datetime
    knowledge_base_size: int
    patterns_discovered: int
    accuracy_improvement: float
    decision_quality_score: float
    adaptation_speed: float
    knowledge_retention_score: float
    cross_domain_transfer: float

@dataclass
class IntelligenceReport:
    """Comprehensive intelligence analysis report."""
    report_id: str
    collection_period: str
    timestamp: datetime
    workflow_intelligence: Dict[str, float]
    collaboration_analytics: Dict[str, float]
    context_efficiency: Dict[str, float]
    learning_progression: Dict[str, float]
    performance_impact: Dict[str, float]
    intelligence_trends: Dict[str, List[float]]
    recommendations: List[str]
    summary: str

# =============================================================================
# INTELLIGENCE METRICS COLLECTOR
# =============================================================================

class IntelligenceMetricsCollector:
    """Collects and analyzes intelligence metrics across the system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_storage: List[IntelligenceMetric] = []
        self.collaboration_events: List[CollaborationEvent] = []
        self.context_utilization: List[ContextUtilizationMetric] = []
        self.learning_progression: List[LearningProgressionMetric] = []
        
        # Real-time tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.collection_start_time = datetime.utcnow()
        
        # HTTP clients for API interactions
        self.semantic_client = httpx.AsyncClient(
            base_url=self.config.get("semantic_memory_url", "http://localhost:8001/api/v1"),
            timeout=30
        )
        self.orchestrator_client = httpx.AsyncClient(
            base_url=self.config.get("orchestrator_url", "http://localhost:8000/api/v1"),
            timeout=30
        )
    
    async def start_collection(self) -> None:
        """Start intelligence metrics collection."""
        self.collection_start_time = datetime.utcnow()
        logger.info(f"🧠 Starting Intelligence Metrics Collection at {self.collection_start_time}")
        
        # Initialize baseline measurements
        await self._collect_baseline_intelligence_metrics()
        await self._initialize_agent_intelligence_profiles()
        
        logger.info("✅ Intelligence metrics collection initialized")
    
    async def _collect_baseline_intelligence_metrics(self) -> None:
        """Collect baseline intelligence metrics for comparison."""
        logger.info("📊 Collecting baseline intelligence metrics...")
        
        try:
            # Get system health and current performance
            health_response = await self.semantic_client.get("/memory/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                # Extract baseline performance metrics
                perf_metrics = health_data.get("performance_metrics", {})
                
                baseline_metrics = [
                    IntelligenceMetric(
                        metric_id=f"baseline_{uuid.uuid4().hex[:8]}",
                        metric_type="baseline_search_latency",
                        value=perf_metrics.get("avg_search_time_ms", 100.0),
                        unit="milliseconds",
                        timestamp=datetime.utcnow(),
                        context={"component": "semantic_memory", "measurement": "baseline"}
                    ),
                    IntelligenceMetric(
                        metric_id=f"baseline_{uuid.uuid4().hex[:8]}",
                        metric_type="baseline_ingestion_throughput",
                        value=perf_metrics.get("throughput_docs_per_sec", 500.0),
                        unit="docs_per_second",
                        timestamp=datetime.utcnow(),
                        context={"component": "semantic_memory", "measurement": "baseline"}
                    ),
                    IntelligenceMetric(
                        metric_id=f"baseline_{uuid.uuid4().hex[:8]}",
                        metric_type="baseline_system_intelligence",
                        value=0.6,  # Assumed baseline intelligence without semantic enhancement
                        unit="intelligence_score",
                        timestamp=datetime.utcnow(),
                        context={"component": "system", "measurement": "baseline"}
                    )
                ]
                
                self.metrics_storage.extend(baseline_metrics)
                logger.info(f"📈 Collected {len(baseline_metrics)} baseline metrics")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not collect baseline metrics: {e}")
    
    async def _initialize_agent_intelligence_profiles(self) -> None:
        """Initialize intelligence profiles for all agents."""
        logger.info("🤖 Initializing agent intelligence profiles...")
        
        # Mock agent list for demonstration
        agents = [
            "workflow-orchestrator",
            "semantic-memory-agent", 
            "context-optimizer",
            "knowledge-manager",
            "collaboration-coordinator",
            "performance-analyzer"
        ]
        
        for agent_id in agents:
            try:
                # Get agent knowledge base if available
                knowledge_response = await self.semantic_client.get(f"/memory/agent-knowledge/{agent_id}")
                
                if knowledge_response.status_code == 200:
                    knowledge_data = knowledge_response.json()
                    knowledge_stats = knowledge_data.get("knowledge_stats", {})
                    
                    self.agent_states[agent_id] = {
                        "knowledge_base_size": knowledge_stats.get("total_documents", 0),
                        "intelligence_score": knowledge_stats.get("knowledge_confidence", 0.5),
                        "collaboration_readiness": 0.8,  # Default readiness
                        "last_updated": datetime.utcnow(),
                        "specialization_areas": knowledge_data.get("knowledge_base", {}).get("consolidated_knowledge", {}).get("expertise_areas", [])
                    }
                else:
                    # Default profile for agents without knowledge base
                    self.agent_states[agent_id] = {
                        "knowledge_base_size": 0,
                        "intelligence_score": 0.5,
                        "collaboration_readiness": 0.7,
                        "last_updated": datetime.utcnow(),
                        "specialization_areas": []
                    }
                    
            except Exception as e:
                logger.warning(f"Could not initialize profile for {agent_id}: {e}")
                # Fallback profile
                self.agent_states[agent_id] = {
                    "knowledge_base_size": 0,
                    "intelligence_score": 0.5,
                    "collaboration_readiness": 0.7,
                    "last_updated": datetime.utcnow(),
                    "specialization_areas": []
                }
        
        logger.info(f"🧠 Initialized intelligence profiles for {len(self.agent_states)} agents")
    
    # =============================================================================
    # WORKFLOW INTELLIGENCE TRACKING
    # =============================================================================
    
    async def track_workflow_intelligence(self, workflow_id: str, workflow_type: str,
                                        baseline_performance: float, enhanced_performance: float,
                                        context_injections: int, agents_involved: List[str]) -> None:
        """Track intelligence enhancement for a specific workflow."""
        
        intelligence_gain = (enhanced_performance - baseline_performance) / baseline_performance
        
        # Record intelligence metrics
        intelligence_metrics = [
            IntelligenceMetric(
                metric_id=f"workflow_intel_{uuid.uuid4().hex[:8]}",
                metric_type="workflow_intelligence_gain",
                value=intelligence_gain,
                unit="improvement_ratio",
                timestamp=datetime.utcnow(),
                context={
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "baseline_performance": baseline_performance,
                    "enhanced_performance": enhanced_performance,
                    "agents_involved": agents_involved
                },
                workflow_id=workflow_id
            ),
            IntelligenceMetric(
                metric_id=f"context_util_{uuid.uuid4().hex[:8]}",
                metric_type="context_injection_effectiveness",
                value=context_injections / len(agents_involved) if agents_involved else 0,
                unit="injections_per_agent",
                timestamp=datetime.utcnow(),
                context={"workflow_id": workflow_id, "total_injections": context_injections},
                workflow_id=workflow_id
            ),
            IntelligenceMetric(
                metric_id=f"multi_agent_{uuid.uuid4().hex[:8]}",
                metric_type="multi_agent_intelligence_amplification",
                value=enhanced_performance * len(agents_involved) / max(len(agents_involved), 1),
                unit="amplification_score",
                timestamp=datetime.utcnow(),
                context={"workflow_id": workflow_id, "agent_count": len(agents_involved)},
                workflow_id=workflow_id
            )
        ]
        
        self.metrics_storage.extend(intelligence_metrics)
        
        # Update active workflow tracking
        self.active_workflows[workflow_id] = {
            "type": workflow_type,
            "start_time": datetime.utcnow(),
            "baseline_performance": baseline_performance,
            "enhanced_performance": enhanced_performance,
            "intelligence_gain": intelligence_gain,
            "agents_involved": agents_involved,
            "status": "tracked"
        }
        
        logger.info(f"📊 Tracked workflow intelligence: {workflow_id} (gain: {intelligence_gain:.3f})")
    
    async def track_context_utilization(self, workflow_id: str, agent_id: str,
                                      original_size: int, compressed_size: int,
                                      semantic_preservation: float, retrieval_latency: float,
                                      relevance_score: float, context_sources: List[str]) -> None:
        """Track context utilization and compression effectiveness."""
        
        compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0
        
        context_metric = ContextUtilizationMetric(
            utilization_id=f"context_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_id,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            original_context_size=original_size,
            compressed_context_size=compressed_size,
            compression_ratio=compression_ratio,
            semantic_preservation_score=semantic_preservation,
            retrieval_latency_ms=retrieval_latency,
            relevance_score=relevance_score,
            context_sources=context_sources
        )
        
        self.context_utilization.append(context_metric)
        
        # Record efficiency metrics
        efficiency_metrics = [
            IntelligenceMetric(
                metric_id=f"compression_{uuid.uuid4().hex[:8]}",
                metric_type="context_compression_efficiency",
                value=compression_ratio,
                unit="compression_ratio",
                timestamp=datetime.utcnow(),
                context={"workflow_id": workflow_id, "semantic_preservation": semantic_preservation},
                source_agent=agent_id,
                workflow_id=workflow_id
            ),
            IntelligenceMetric(
                metric_id=f"relevance_{uuid.uuid4().hex[:8]}",
                metric_type="context_relevance_score",
                value=relevance_score,
                unit="relevance_score",
                timestamp=datetime.utcnow(),
                context={"workflow_id": workflow_id, "sources_count": len(context_sources)},
                source_agent=agent_id,
                workflow_id=workflow_id
            )
        ]
        
        self.metrics_storage.extend(efficiency_metrics)
        
        logger.info(f"📦 Tracked context utilization: {agent_id} (compression: {compression_ratio:.3f}, relevance: {relevance_score:.3f})")
    
    # =============================================================================
    # COLLABORATION EFFECTIVENESS TRACKING
    # =============================================================================
    
    async def track_collaboration_event(self, event_type: str, participants: List[str],
                                      duration_ms: float, effectiveness_score: float,
                                      knowledge_exchanged: int, outcome: str,
                                      context: Dict[str, Any] = None) -> None:
        """Track agent collaboration events and effectiveness."""
        
        collaboration_event = CollaborationEvent(
            event_id=f"collab_{uuid.uuid4().hex[:8]}",
            event_type=event_type,
            participants=participants,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            effectiveness_score=effectiveness_score,
            knowledge_exchanged=knowledge_exchanged,
            outcome=outcome,
            context=context or {}
        )
        
        self.collaboration_events.append(collaboration_event)
        
        # Record collaboration metrics
        collaboration_metrics = [
            IntelligenceMetric(
                metric_id=f"collab_eff_{uuid.uuid4().hex[:8]}",
                metric_type="collaboration_effectiveness",
                value=effectiveness_score,
                unit="effectiveness_score",
                timestamp=datetime.utcnow(),
                context={
                    "event_type": event_type,
                    "participants_count": len(participants),
                    "knowledge_exchanged": knowledge_exchanged
                }
            ),
            IntelligenceMetric(
                metric_id=f"knowledge_flow_{uuid.uuid4().hex[:8]}",
                metric_type="knowledge_flow_rate",
                value=knowledge_exchanged / (duration_ms / 1000) if duration_ms > 0 else 0,
                unit="knowledge_items_per_second",
                timestamp=datetime.utcnow(),
                context={"event_type": event_type, "duration_ms": duration_ms}
            ),
            IntelligenceMetric(
                metric_id=f"collab_speed_{uuid.uuid4().hex[:8]}",
                metric_type="collaboration_speed",
                value=len(participants) / (duration_ms / 1000) if duration_ms > 0 else 0,
                unit="agents_per_second",
                timestamp=datetime.utcnow(),
                context={"participants": participants, "outcome": outcome}
            )
        ]
        
        self.metrics_storage.extend(collaboration_metrics)
        
        # Update agent collaboration readiness
        for agent_id in participants:
            if agent_id in self.agent_states:
                current_readiness = self.agent_states[agent_id].get("collaboration_readiness", 0.7)
                # Adjust readiness based on collaboration effectiveness
                adjustment = (effectiveness_score - 0.7) * 0.1  # Small adjustment
                new_readiness = max(0.0, min(1.0, current_readiness + adjustment))
                self.agent_states[agent_id]["collaboration_readiness"] = new_readiness
        
        logger.info(f"🤝 Tracked collaboration: {event_type} with {len(participants)} agents (effectiveness: {effectiveness_score:.3f})")
    
    # =============================================================================
    # LEARNING PROGRESSION TRACKING
    # =============================================================================
    
    async def track_learning_progression(self, agent_id: str, knowledge_base_size: int,
                                       patterns_discovered: int, accuracy_improvement: float,
                                       decision_quality: float, adaptation_speed: float,
                                       knowledge_retention: float, cross_domain_transfer: float) -> None:
        """Track agent learning progression and knowledge evolution."""
        
        learning_metric = LearningProgressionMetric(
            learning_id=f"learning_{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            knowledge_base_size=knowledge_base_size,
            patterns_discovered=patterns_discovered,
            accuracy_improvement=accuracy_improvement,
            decision_quality_score=decision_quality,
            adaptation_speed=adaptation_speed,
            knowledge_retention_score=knowledge_retention,
            cross_domain_transfer=cross_domain_transfer
        )
        
        self.learning_progression.append(learning_metric)
        
        # Record learning intelligence metrics
        learning_intelligence_metrics = [
            IntelligenceMetric(
                metric_id=f"learning_rate_{uuid.uuid4().hex[:8]}",
                metric_type="learning_rate",
                value=patterns_discovered / max(knowledge_base_size, 1) * 1000,  # Patterns per 1000 knowledge items
                unit="patterns_per_1000_items",
                timestamp=datetime.utcnow(),
                context={"knowledge_base_size": knowledge_base_size},
                source_agent=agent_id
            ),
            IntelligenceMetric(
                metric_id=f"adaptation_{uuid.uuid4().hex[:8]}",
                metric_type="adaptation_intelligence",
                value=adaptation_speed * decision_quality,
                unit="adaptation_score",
                timestamp=datetime.utcnow(),
                context={"decision_quality": decision_quality, "adaptation_speed": adaptation_speed},
                source_agent=agent_id
            ),
            IntelligenceMetric(
                metric_id=f"knowledge_growth_{uuid.uuid4().hex[:8]}",
                metric_type="knowledge_growth_rate",
                value=accuracy_improvement * knowledge_retention,
                unit="growth_score",
                timestamp=datetime.utcnow(),
                context={"retention_score": knowledge_retention},
                source_agent=agent_id
            )
        ]
        
        self.metrics_storage.extend(learning_intelligence_metrics)
        
        # Update agent intelligence profile
        if agent_id in self.agent_states:
            self.agent_states[agent_id].update({
                "knowledge_base_size": knowledge_base_size,
                "intelligence_score": min(1.0, decision_quality * 0.7 + cross_domain_transfer * 0.3),
                "last_updated": datetime.utcnow()
            })
        
        logger.info(f"📚 Tracked learning progression: {agent_id} (patterns: {patterns_discovered}, quality: {decision_quality:.3f})")
    
    # =============================================================================
    # INTELLIGENCE ANALYSIS AND REPORTING
    # =============================================================================
    
    async def generate_intelligence_report(self, period_hours: int = 24) -> IntelligenceReport:
        """Generate comprehensive intelligence analysis report."""
        logger.info(f"📋 Generating intelligence report for last {period_hours} hours...")
        
        cutoff_time = datetime.utcnow() - timedelta(hours=period_hours)
        
        # Filter metrics by time period
        recent_metrics = [m for m in self.metrics_storage if m.timestamp >= cutoff_time]
        recent_collaborations = [c for c in self.collaboration_events if c.timestamp >= cutoff_time]
        recent_context_usage = [c for c in self.context_utilization if c.timestamp >= cutoff_time]
        recent_learning = [l for l in self.learning_progression if l.timestamp >= cutoff_time]
        
        # Analyze workflow intelligence
        workflow_intelligence = await self._analyze_workflow_intelligence(recent_metrics)
        
        # Analyze collaboration effectiveness
        collaboration_analytics = await self._analyze_collaboration_effectiveness(recent_collaborations)
        
        # Analyze context efficiency
        context_efficiency = await self._analyze_context_efficiency(recent_context_usage)
        
        # Analyze learning progression
        learning_progression = await self._analyze_learning_progression(recent_learning)
        
        # Analyze performance impact
        performance_impact = await self._analyze_performance_impact(recent_metrics)
        
        # Generate intelligence trends
        intelligence_trends = await self._calculate_intelligence_trends(recent_metrics)
        
        # Generate recommendations
        recommendations = await self._generate_intelligence_recommendations(
            workflow_intelligence, collaboration_analytics, context_efficiency, learning_progression
        )
        
        # Create summary
        summary = await self._create_intelligence_summary(
            workflow_intelligence, collaboration_analytics, context_efficiency, learning_progression, performance_impact
        )
        
        report = IntelligenceReport(
            report_id=f"intel_report_{uuid.uuid4().hex[:8]}",
            collection_period=f"{period_hours}h",
            timestamp=datetime.utcnow(),
            workflow_intelligence=workflow_intelligence,
            collaboration_analytics=collaboration_analytics,
            context_efficiency=context_efficiency,
            learning_progression=learning_progression,
            performance_impact=performance_impact,
            intelligence_trends=intelligence_trends,
            recommendations=recommendations,
            summary=summary
        )
        
        logger.info(f"✅ Intelligence report generated: {report.report_id}")
        return report
    
    async def _analyze_workflow_intelligence(self, metrics: List[IntelligenceMetric]) -> Dict[str, float]:
        """Analyze workflow intelligence enhancement metrics."""
        workflow_gains = [m.value for m in metrics if m.metric_type == "workflow_intelligence_gain"]
        context_effectiveness = [m.value for m in metrics if m.metric_type == "context_injection_effectiveness"]
        amplification_scores = [m.value for m in metrics if m.metric_type == "multi_agent_intelligence_amplification"]
        
        return {
            "average_intelligence_gain": statistics.mean(workflow_gains) if workflow_gains else 0.0,
            "max_intelligence_gain": max(workflow_gains) if workflow_gains else 0.0,
            "intelligence_consistency": 1.0 - (statistics.stdev(workflow_gains) / statistics.mean(workflow_gains)) if len(workflow_gains) > 1 and statistics.mean(workflow_gains) > 0 else 1.0,
            "context_injection_rate": statistics.mean(context_effectiveness) if context_effectiveness else 0.0,
            "multi_agent_amplification": statistics.mean(amplification_scores) if amplification_scores else 0.0,
            "workflow_count": len(set(m.workflow_id for m in metrics if m.workflow_id))
        }
    
    async def _analyze_collaboration_effectiveness(self, collaborations: List[CollaborationEvent]) -> Dict[str, float]:
        """Analyze collaboration effectiveness metrics."""
        if not collaborations:
            return {"collaboration_events": 0, "average_effectiveness": 0.0}
        
        effectiveness_scores = [c.effectiveness_score for c in collaborations]
        knowledge_exchanges = [c.knowledge_exchanged for c in collaborations]
        durations = [c.duration_ms for c in collaborations]
        
        unique_agents = set()
        for c in collaborations:
            unique_agents.update(c.participants)
        
        return {
            "collaboration_events": len(collaborations),
            "average_effectiveness": statistics.mean(effectiveness_scores),
            "peak_effectiveness": max(effectiveness_scores),
            "total_knowledge_exchanged": sum(knowledge_exchanges),
            "average_collaboration_speed": statistics.mean([len(c.participants) / (c.duration_ms / 1000) for c in collaborations if c.duration_ms > 0]),
            "unique_agents_involved": len(unique_agents),
            "collaboration_frequency": len(collaborations) / 24 if collaborations else 0  # Events per hour
        }
    
    async def _analyze_context_efficiency(self, context_usage: List[ContextUtilizationMetric]) -> Dict[str, float]:
        """Analyze context efficiency and utilization metrics."""
        if not context_usage:
            return {"context_operations": 0, "average_compression": 0.0}
        
        compression_ratios = [c.compression_ratio for c in context_usage]
        preservation_scores = [c.semantic_preservation_score for c in context_usage]
        relevance_scores = [c.relevance_score for c in context_usage]
        retrieval_latencies = [c.retrieval_latency_ms for c in context_usage]
        
        return {
            "context_operations": len(context_usage),
            "average_compression_ratio": statistics.mean(compression_ratios),
            "average_semantic_preservation": statistics.mean(preservation_scores),
            "average_relevance_score": statistics.mean(relevance_scores),
            "average_retrieval_latency_ms": statistics.mean(retrieval_latencies),
            "compression_efficiency": statistics.mean(compression_ratios) * statistics.mean(preservation_scores),
            "peak_compression": max(compression_ratios),
            "context_quality_score": statistics.mean(relevance_scores) * statistics.mean(preservation_scores)
        }
    
    async def _analyze_learning_progression(self, learning_data: List[LearningProgressionMetric]) -> Dict[str, float]:
        """Analyze learning progression and knowledge evolution."""
        if not learning_data:
            return {"learning_events": 0, "average_learning_rate": 0.0}
        
        accuracy_improvements = [l.accuracy_improvement for l in learning_data]
        decision_qualities = [l.decision_quality_score for l in learning_data]
        adaptation_speeds = [l.adaptation_speed for l in learning_data]
        retention_scores = [l.knowledge_retention_score for l in learning_data]
        transfer_scores = [l.cross_domain_transfer for l in learning_data]
        
        unique_agents = set(l.agent_id for l in learning_data)
        
        return {
            "learning_events": len(learning_data),
            "average_accuracy_improvement": statistics.mean(accuracy_improvements),
            "average_decision_quality": statistics.mean(decision_qualities),
            "average_adaptation_speed": statistics.mean(adaptation_speeds),
            "average_knowledge_retention": statistics.mean(retention_scores),
            "average_cross_domain_transfer": statistics.mean(transfer_scores),
            "learning_agents": len(unique_agents),
            "overall_learning_score": statistics.mean([
                sum([acc, qual, adapt, ret, trans]) / 5
                for acc, qual, adapt, ret, trans in zip(
                    accuracy_improvements, decision_qualities, adaptation_speeds, retention_scores, transfer_scores
                )
            ])
        }
    
    async def _analyze_performance_impact(self, metrics: List[IntelligenceMetric]) -> Dict[str, float]:
        """Analyze performance impact of intelligence enhancements."""
        compression_metrics = [m.value for m in metrics if m.metric_type == "context_compression_efficiency"]
        relevance_metrics = [m.value for m in metrics if m.metric_type == "context_relevance_score"]
        learning_rates = [m.value for m in metrics if m.metric_type == "learning_rate"]
        
        # Calculate estimated performance improvements
        compression_improvement = statistics.mean(compression_metrics) * 0.3 if compression_metrics else 0  # 30% weight
        relevance_improvement = statistics.mean(relevance_metrics) * 0.4 if relevance_metrics else 0  # 40% weight
        learning_improvement = statistics.mean(learning_rates) * 0.01 if learning_rates else 0  # Small but significant
        
        overall_performance_impact = compression_improvement + relevance_improvement + learning_improvement
        
        return {
            "compression_performance_gain": compression_improvement,
            "relevance_performance_gain": relevance_improvement,
            "learning_performance_gain": learning_improvement,
            "overall_performance_impact": overall_performance_impact,
            "estimated_efficiency_improvement": overall_performance_impact * 1.2,  # Amplification factor
            "cost_savings_ratio": statistics.mean(compression_metrics) if compression_metrics else 0
        }
    
    async def _calculate_intelligence_trends(self, metrics: List[IntelligenceMetric]) -> Dict[str, List[float]]:
        """Calculate intelligence trends over time."""
        trend_metrics = {}
        
        # Group metrics by type and calculate trends
        metric_types = set(m.metric_type for m in metrics)
        
        for metric_type in metric_types:
            type_metrics = [m for m in metrics if m.metric_type == metric_type]
            type_metrics.sort(key=lambda x: x.timestamp)
            
            # Calculate trend (simple moving average)
            values = [m.value for m in type_metrics]
            if len(values) >= 3:
                trend = []
                window_size = min(3, len(values))
                for i in range(window_size - 1, len(values)):
                    window_avg = statistics.mean(values[max(0, i-window_size+1):i+1])
                    trend.append(window_avg)
                trend_metrics[metric_type] = trend
            else:
                trend_metrics[metric_type] = values
        
        return trend_metrics
    
    async def _generate_intelligence_recommendations(self, workflow_intel: Dict[str, float],
                                                   collab_analytics: Dict[str, float],
                                                   context_efficiency: Dict[str, float],
                                                   learning_progression: Dict[str, float]) -> List[str]:
        """Generate intelligence improvement recommendations."""
        recommendations = []
        
        # Workflow intelligence recommendations
        if workflow_intel.get("average_intelligence_gain", 0) < 0.2:
            recommendations.append("Increase semantic context injection to improve workflow intelligence")
        
        if workflow_intel.get("intelligence_consistency", 1.0) < 0.8:
            recommendations.append("Optimize context relevance scoring to improve intelligence consistency")
        
        # Collaboration recommendations
        if collab_analytics.get("average_effectiveness", 0) < 0.8:
            recommendations.append("Enhance cross-agent knowledge sharing protocols")
        
        if collab_analytics.get("collaboration_frequency", 0) < 1.0:
            recommendations.append("Increase collaborative workflow frequency for better knowledge transfer")
        
        # Context efficiency recommendations
        if context_efficiency.get("average_compression_ratio", 0) < 0.6:
            recommendations.append("Improve context compression algorithms for better efficiency")
        
        if context_efficiency.get("context_quality_score", 0) < 0.8:
            recommendations.append("Enhance semantic preservation during context compression")
        
        # Learning progression recommendations
        if learning_progression.get("average_learning_rate", 0) < 0.5:
            recommendations.append("Increase pattern discovery algorithms effectiveness")
        
        if learning_progression.get("average_cross_domain_transfer", 0) < 0.7:
            recommendations.append("Improve cross-domain knowledge transfer mechanisms")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Intelligence metrics are performing well - continue current optimization strategies")
        
        return recommendations
    
    async def _create_intelligence_summary(self, workflow_intel: Dict[str, float],
                                         collab_analytics: Dict[str, float],
                                         context_efficiency: Dict[str, float],
                                         learning_progression: Dict[str, float],
                                         performance_impact: Dict[str, float]) -> str:
        """Create executive summary of intelligence metrics."""
        
        # Calculate overall intelligence score
        intel_score = workflow_intel.get("average_intelligence_gain", 0) * 0.3
        collab_score = collab_analytics.get("average_effectiveness", 0) * 0.25  
        context_score = context_efficiency.get("context_quality_score", 0) * 0.25
        learning_score = learning_progression.get("overall_learning_score", 0) * 0.2
        
        overall_score = intel_score + collab_score + context_score + learning_score
        
        summary_lines = [
            f"Intelligence Enhancement Summary:",
            f"- Overall Intelligence Score: {overall_score:.3f}",
            f"- Workflow Intelligence Gain: {workflow_intel.get('average_intelligence_gain', 0):.1%}",
            f"- Collaboration Effectiveness: {collab_analytics.get('average_effectiveness', 0):.3f}",
            f"- Context Efficiency: {context_efficiency.get('context_quality_score', 0):.3f}",
            f"- Learning Progression: {learning_progression.get('overall_learning_score', 0):.3f}",
            f"- Performance Impact: {performance_impact.get('overall_performance_impact', 0):.3f}",
            f"- Active Workflows: {workflow_intel.get('workflow_count', 0)}",
            f"- Collaboration Events: {collab_analytics.get('collaboration_events', 0)}",
            f"- Learning Agents: {learning_progression.get('learning_agents', 0)}"
        ]
        
        return "\n".join(summary_lines)
    
    # =============================================================================
    # DATA EXPORT AND PERSISTENCE
    # =============================================================================
    
    async def export_metrics_data(self, output_path: str) -> Dict[str, str]:
        """Export all collected metrics data to files."""
        logger.info(f"💾 Exporting metrics data to {output_path}")
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        files_created = {}
        
        # Export intelligence metrics
        intelligence_file = output_dir / f"intelligence_metrics_{timestamp}.json"
        with open(intelligence_file, 'w') as f:
            json.dump([
                {
                    "metric_id": m.metric_id,
                    "metric_type": m.metric_type,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat(),
                    "context": m.context,
                    "confidence": m.confidence,
                    "source_agent": m.source_agent,
                    "workflow_id": m.workflow_id
                }
                for m in self.metrics_storage
            ], f, indent=2)
        files_created["intelligence_metrics"] = str(intelligence_file)
        
        # Export collaboration events
        collaboration_file = output_dir / f"collaboration_events_{timestamp}.json"
        with open(collaboration_file, 'w') as f:
            json.dump([
                {
                    "event_id": c.event_id,
                    "event_type": c.event_type,
                    "participants": c.participants,
                    "timestamp": c.timestamp.isoformat(),
                    "duration_ms": c.duration_ms,
                    "effectiveness_score": c.effectiveness_score,
                    "knowledge_exchanged": c.knowledge_exchanged,
                    "outcome": c.outcome,
                    "context": c.context
                }
                for c in self.collaboration_events
            ], f, indent=2)
        files_created["collaboration_events"] = str(collaboration_file)
        
        # Export context utilization
        context_file = output_dir / f"context_utilization_{timestamp}.json"
        with open(context_file, 'w') as f:
            json.dump([
                {
                    "utilization_id": c.utilization_id,
                    "workflow_id": c.workflow_id,
                    "agent_id": c.agent_id,
                    "timestamp": c.timestamp.isoformat(),
                    "original_context_size": c.original_context_size,
                    "compressed_context_size": c.compressed_context_size,
                    "compression_ratio": c.compression_ratio,
                    "semantic_preservation_score": c.semantic_preservation_score,
                    "retrieval_latency_ms": c.retrieval_latency_ms,
                    "relevance_score": c.relevance_score,
                    "context_sources": c.context_sources
                }
                for c in self.context_utilization
            ], f, indent=2)
        files_created["context_utilization"] = str(context_file)
        
        # Export learning progression
        learning_file = output_dir / f"learning_progression_{timestamp}.json"
        with open(learning_file, 'w') as f:
            json.dump([
                {
                    "learning_id": l.learning_id,
                    "agent_id": l.agent_id,
                    "timestamp": l.timestamp.isoformat(),
                    "knowledge_base_size": l.knowledge_base_size,
                    "patterns_discovered": l.patterns_discovered,
                    "accuracy_improvement": l.accuracy_improvement,
                    "decision_quality_score": l.decision_quality_score,
                    "adaptation_speed": l.adaptation_speed,
                    "knowledge_retention_score": l.knowledge_retention_score,
                    "cross_domain_transfer": l.cross_domain_transfer
                }
                for l in self.learning_progression
            ], f, indent=2)
        files_created["learning_progression"] = str(learning_file)
        
        # Export agent states
        agent_states_file = output_dir / f"agent_states_{timestamp}.json"
        with open(agent_states_file, 'w') as f:
            json.dump(self.agent_states, f, indent=2, default=str)
        files_created["agent_states"] = str(agent_states_file)
        
        logger.info(f"✅ Exported {len(files_created)} metrics data files")
        return files_created
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.semantic_client.aclose()
        await self.orchestrator_client.aclose()
        logger.info("🧹 Intelligence metrics collector cleaned up")

# =============================================================================
# MAIN EXECUTION AND DEMONSTRATION
# =============================================================================

async def main():
    """Demonstrate intelligence metrics collection."""
    print("=" * 80)
    print("🧠 LeanVibe Agent Hive 2.0 - Intelligence Metrics Collection")
    print("Comprehensive intelligence analysis and tracking system")
    print("=" * 80)
    
    # Initialize collector
    config = {
        "semantic_memory_url": "http://localhost:8001/api/v1",
        "orchestrator_url": "http://localhost:8000/api/v1"
    }
    
    collector = IntelligenceMetricsCollector(config)
    
    try:
        # Start collection
        await collector.start_collection()
        
        # Simulate intelligence tracking for demonstration
        print("\n📊 Simulating Intelligence Metrics Collection...")
        
        # Track workflow intelligence
        await collector.track_workflow_intelligence(
            workflow_id="demo-workflow-001",
            workflow_type="intelligent_development",
            baseline_performance=0.6,
            enhanced_performance=0.85,
            context_injections=4,
            agents_involved=["architect", "developer", "reviewer"]
        )
        
        # Track context utilization
        await collector.track_context_utilization(
            workflow_id="demo-workflow-001",
            agent_id="architect",
            original_size=3000,
            compressed_size=2100,
            semantic_preservation=0.92,
            retrieval_latency=45.0,
            relevance_score=0.87,
            context_sources=["requirements", "patterns", "best_practices"]
        )
        
        # Track collaboration event
        await collector.track_collaboration_event(
            event_type="knowledge_sharing",
            participants=["architect", "developer", "reviewer"],
            duration_ms=1500,
            effectiveness_score=0.89,
            knowledge_exchanged=12,
            outcome="consensus_reached",
            context={"topic": "system_design", "complexity": "high"}
        )
        
        # Track learning progression
        await collector.track_learning_progression(
            agent_id="architect",
            knowledge_base_size=450,
            patterns_discovered=8,
            accuracy_improvement=0.15,
            decision_quality=0.88,
            adaptation_speed=0.82,
            knowledge_retention=0.91,
            cross_domain_transfer=0.76
        )
        
        # Generate intelligence report
        print("\n📋 Generating Intelligence Report...")
        report = await collector.generate_intelligence_report(period_hours=1)
        
        # Display report summary
        print("\n" + "=" * 80)
        print("📈 INTELLIGENCE METRICS REPORT")
        print("=" * 80)
        print(f"Report ID: {report.report_id}")
        print(f"Collection Period: {report.collection_period}")
        print(f"Generated: {report.timestamp}")
        print()
        print("📊 Workflow Intelligence:")
        for key, value in report.workflow_intelligence.items():
            print(f"  - {key}: {value:.3f}")
        
        print("\n🤝 Collaboration Analytics:")
        for key, value in report.collaboration_analytics.items():
            print(f"  - {key}: {value:.3f}")
        
        print("\n📦 Context Efficiency:")
        for key, value in report.context_efficiency.items():
            print(f"  - {key}: {value:.3f}")
        
        print("\n📚 Learning Progression:")
        for key, value in report.learning_progression.items():
            print(f"  - {key}: {value:.3f}")
        
        print("\n⚡ Performance Impact:")
        for key, value in report.performance_impact.items():
            print(f"  - {key}: {value:.3f}")
        
        print(f"\n📝 Summary:")
        print(report.summary)
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Export metrics data
        print("\n💾 Exporting Metrics Data...")
        output_dir = "intelligence_metrics_export"
        exported_files = await collector.export_metrics_data(output_dir)
        
        print(f"✅ Exported {len(exported_files)} data files:")
        for data_type, file_path in exported_files.items():
            print(f"  - {data_type}: {file_path}")
        
        # Final statistics
        print(f"\n📊 Collection Statistics:")
        print(f"  - Intelligence Metrics: {len(collector.metrics_storage)}")
        print(f"  - Collaboration Events: {len(collector.collaboration_events)}")
        print(f"  - Context Operations: {len(collector.context_utilization)}")
        print(f"  - Learning Events: {len(collector.learning_progression)}")
        print(f"  - Agent Profiles: {len(collector.agent_states)}")
        print(f"  - Active Workflows: {len(collector.active_workflows)}")
        
        print(f"\n🚀 Intelligence metrics collection demonstration completed!")
        
    except Exception as e:
        print(f"\n❌ METRICS COLLECTION FAILED: {e}")
        logger.exception("Intelligence metrics collection failed")
        raise
    
    finally:
        await collector.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
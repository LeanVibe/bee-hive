"""
Self-Optimization Agent - Performance-Based Learning and Improvement System

This agent specializes in continuous self-improvement, performance analysis,
and cross-agent learning optimization. Part of the AI Enhancement Team for 
LeanVibe Agent Hive 2.0.
"""

import asyncio
import json
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import math

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .redis import get_message_broker
from .intelligence_framework import (
    IntelligenceModelInterface,
    IntelligencePrediction,
    DataPoint,
    DataType
)
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class OptimizationType(Enum):
    """Types of optimizations that can be performed."""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    RESOURCE_USAGE = "resource_usage"
    COLLABORATION = "collaboration"
    LEARNING_RATE = "learning_rate"
    DECISION_QUALITY = "decision_quality"


class LearningStrategy(Enum):
    """Learning strategies for optimization."""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"


class PerformanceCategory(Enum):
    """Categories of performance metrics."""
    TASK_EXECUTION = "task_execution"
    CODE_QUALITY = "code_quality"
    COLLABORATION = "collaboration"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    LEARNING_EFFECTIVENESS = "learning_effectiveness"


@dataclass
class PerformanceSnapshot:
    """Snapshot of agent performance at a point in time."""
    agent_id: str
    timestamp: datetime
    task_success_rate: float
    average_task_time: float
    code_quality_score: float
    collaboration_rating: float
    resource_utilization: float
    learning_velocity: float
    decision_accuracy: float
    context_efficiency: float
    user_satisfaction: float
    error_rate: float
    
    def overall_score(self) -> float:
        """Calculate weighted overall performance score."""
        weights = {
            'task_success_rate': 0.25,
            'code_quality_score': 0.20,
            'collaboration_rating': 0.15,
            'resource_utilization': 0.10,
            'learning_velocity': 0.10,
            'decision_accuracy': 0.10,
            'user_satisfaction': 0.10
        }
        
        # Convert error rate to positive metric (lower error = higher score)
        error_score = max(0.0, 1.0 - self.error_rate)
        
        return (
            weights['task_success_rate'] * self.task_success_rate +
            weights['code_quality_score'] * self.code_quality_score +
            weights['collaboration_rating'] * self.collaboration_rating +
            weights['resource_utilization'] * (1.0 - self.resource_utilization) +  # Lower is better
            weights['learning_velocity'] * self.learning_velocity +
            weights['decision_accuracy'] * self.decision_accuracy +
            weights['user_satisfaction'] * self.user_satisfaction
        )
    
    def get_improvement_areas(self) -> List[str]:
        """Identify areas needing improvement."""
        areas = []
        threshold = 0.7
        
        if self.task_success_rate < threshold:
            areas.append("task_execution")
        if self.code_quality_score < threshold:
            areas.append("code_quality")
        if self.collaboration_rating < threshold:
            areas.append("collaboration")
        if self.resource_utilization > 0.8:  # High usage is bad
            areas.append("resource_efficiency")
        if self.learning_velocity < threshold:
            areas.append("learning_rate")
        if self.decision_accuracy < threshold:
            areas.append("decision_making")
        if self.error_rate > 0.3:
            areas.append("error_reduction")
            
        return areas


@dataclass
class OptimizationExperiment:
    """Represents an optimization experiment."""
    experiment_id: str
    agent_id: str
    optimization_type: OptimizationType
    strategy: LearningStrategy
    hypothesis: str
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    performance_before: PerformanceSnapshot
    performance_after: Optional[PerformanceSnapshot]
    success: Optional[bool]
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    started_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[timedelta]
    
    def is_successful(self) -> bool:
        """Determine if the experiment was successful."""
        if not self.performance_after:
            return False
        
        improvement = (self.performance_after.overall_score() - 
                      self.performance_before.overall_score())
        
        return improvement > 0.05 and self.statistical_significance > 0.95  # 5% improvement, 95% confidence
    
    def get_improvement_metrics(self) -> Dict[str, float]:
        """Calculate improvement metrics."""
        if not self.performance_after:
            return {}
        
        return {
            'overall_improvement': (self.performance_after.overall_score() - 
                                  self.performance_before.overall_score()),
            'task_success_improvement': (self.performance_after.task_success_rate - 
                                       self.performance_before.task_success_rate),
            'quality_improvement': (self.performance_after.code_quality_score - 
                                  self.performance_before.code_quality_score),
            'efficiency_improvement': (self.performance_before.resource_utilization - 
                                     self.performance_after.resource_utilization),
            'error_reduction': (self.performance_before.error_rate - 
                              self.performance_after.error_rate)
        }


@dataclass
class LearningInsight:
    """Represents a learning insight derived from experiments."""
    insight_id: str
    category: str
    title: str
    description: str
    supporting_experiments: List[str]
    confidence: float
    impact_score: float
    applicability: List[str]  # Which agents/contexts this applies to
    created_at: datetime
    validated: bool = False
    
    def to_knowledge_item(self) -> Dict[str, Any]:
        """Convert insight to knowledge base item."""
        return {
            'id': self.insight_id,
            'type': 'optimization_insight',
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'impact': self.impact_score,
            'evidence': {
                'experiments': self.supporting_experiments,
                'validation_status': self.validated
            },
            'application_guidelines': {
                'applicable_contexts': self.applicability,
                'recommendations': self._generate_recommendations()
            },
            'metadata': {
                'created_at': self.created_at.isoformat(),
                'category': self.category
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations from the insight."""
        recommendations = []
        
        if self.impact_score > 0.8:
            recommendations.append("High-impact optimization - implement across all applicable agents")
        elif self.impact_score > 0.6:
            recommendations.append("Moderate impact - implement selectively based on agent context")
        else:
            recommendations.append("Low impact - consider for specific use cases only")
            
        if self.confidence > 0.9:
            recommendations.append("High confidence - safe for production implementation")
        elif self.confidence > 0.7:
            recommendations.append("Good confidence - implement with monitoring")
        else:
            recommendations.append("Low confidence - requires further validation")
            
        return recommendations


class PerformanceAnalyzer:
    """Advanced performance analysis and trend detection."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[PerformanceSnapshot]] = defaultdict(list)
        self.baseline_metrics: Dict[str, PerformanceSnapshot] = {}
        self.trend_window = 10  # Number of snapshots to analyze for trends
    
    async def add_performance_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Add a new performance snapshot."""
        agent_history = self.performance_history[snapshot.agent_id]
        agent_history.append(snapshot)
        
        # Keep only recent snapshots
        max_history = 100
        if len(agent_history) > max_history:
            self.performance_history[snapshot.agent_id] = agent_history[-max_history:]
        
        # Update baseline if this is the first snapshot
        if snapshot.agent_id not in self.baseline_metrics:
            self.baseline_metrics[snapshot.agent_id] = snapshot
    
    async def analyze_trends(self, agent_id: str) -> Dict[str, Any]:
        """Analyze performance trends for an agent."""
        history = self.performance_history.get(agent_id, [])
        if len(history) < 3:
            return {"status": "insufficient_data", "snapshots": len(history)}
        
        recent_snapshots = history[-self.trend_window:]
        
        # Calculate trends for each metric
        trends = {}
        metrics = [
            'task_success_rate', 'code_quality_score', 'collaboration_rating',
            'resource_utilization', 'learning_velocity', 'decision_accuracy',
            'error_rate'
        ]
        
        for metric in metrics:
            values = [getattr(snapshot, metric) for snapshot in recent_snapshots]
            trends[metric] = self._calculate_trend(values)
        
        # Calculate overall trend
        overall_scores = [snapshot.overall_score() for snapshot in recent_snapshots]
        overall_trend = self._calculate_trend(overall_scores)
        
        # Identify performance patterns
        patterns = await self._identify_patterns(recent_snapshots)
        
        # Generate trend summary
        summary = self._generate_trend_summary(trends, overall_trend, patterns)
        
        return {
            'agent_id': agent_id,
            'analysis_window': len(recent_snapshots),
            'overall_trend': overall_trend,
            'metric_trends': trends,
            'patterns': patterns,
            'summary': summary,
            'recommendations': await self._generate_trend_recommendations(agent_id, trends, patterns)
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a series of values."""
        if len(values) < 2:
            return {'direction': 'stable', 'slope': 0.0, 'confidence': 0.0}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'improving'
        else:
            direction = 'declining'
        
        # Calculate confidence based on correlation
        if len(values) > 2:
            variance_x = sum((x[i] - x_mean) ** 2 for i in range(n)) / n
            variance_y = sum((values[i] - y_mean) ** 2 for i in range(n)) / n
            covariance = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n)) / n
            
            if variance_x > 0 and variance_y > 0:
                correlation = covariance / (variance_x * variance_y) ** 0.5
                confidence = abs(correlation)
            else:
                confidence = 0.0
        else:
            confidence = 0.5
        
        return {
            'direction': direction,
            'slope': slope,
            'confidence': confidence,
            'recent_value': values[-1],
            'change_from_start': values[-1] - values[0]
        }
    
    async def _identify_patterns(self, snapshots: List[PerformanceSnapshot]) -> List[Dict[str, Any]]:
        """Identify performance patterns in the snapshots."""
        patterns = []
        
        # Check for cyclical patterns
        if len(snapshots) >= 6:
            overall_scores = [s.overall_score() for s in snapshots]
            cyclical = self._detect_cyclical_pattern(overall_scores)
            if cyclical:
                patterns.append({
                    'type': 'cyclical',
                    'description': 'Performance shows cyclical variation',
                    'period': cyclical['period'],
                    'amplitude': cyclical['amplitude']
                })
        
        # Check for performance drops
        recent_drops = self._detect_performance_drops(snapshots)
        if recent_drops:
            patterns.append({
                'type': 'performance_drop',
                'description': 'Significant performance drop detected',
                'drop_points': recent_drops
            })
        
        # Check for improvement plateaus
        plateau = self._detect_improvement_plateau(snapshots)
        if plateau:
            patterns.append({
                'type': 'improvement_plateau',
                'description': 'Performance has plateaued',
                'plateau_duration': plateau['duration']
            })
        
        return patterns
    
    def _detect_cyclical_pattern(self, values: List[float]) -> Optional[Dict[str, Any]]:
        """Detect cyclical patterns in performance data."""
        # Simple autocorrelation-based detection
        n = len(values)
        best_period = None
        best_correlation = 0.0
        
        for period in range(2, min(n // 2, 10)):  # Check periods up to 10
            correlation = 0.0
            count = 0
            
            for i in range(n - period):
                correlation += values[i] * values[i + period]
                count += 1
            
            if count > 0:
                correlation /= count
                
                if correlation > best_correlation and correlation > 0.7:
                    best_correlation = correlation
                    best_period = period
        
        if best_period:
            amplitude = statistics.stdev(values) if len(values) > 1 else 0.0
            return {'period': best_period, 'amplitude': amplitude, 'correlation': best_correlation}
        
        return None
    
    def _detect_performance_drops(self, snapshots: List[PerformanceSnapshot]) -> List[Dict[str, Any]]:
        """Detect significant performance drops."""
        drops = []
        
        if len(snapshots) < 3:
            return drops
        
        for i in range(1, len(snapshots)):
            current_score = snapshots[i].overall_score()
            previous_score = snapshots[i-1].overall_score()
            
            # Significant drop threshold (more than 10% decrease)
            if previous_score > 0 and (previous_score - current_score) / previous_score > 0.1:
                drops.append({
                    'timestamp': snapshots[i].timestamp,
                    'drop_magnitude': previous_score - current_score,
                    'drop_percentage': (previous_score - current_score) / previous_score
                })
        
        return drops
    
    def _detect_improvement_plateau(self, snapshots: List[PerformanceSnapshot]) -> Optional[Dict[str, Any]]:
        """Detect if performance has plateaued."""
        if len(snapshots) < 5:
            return None
        
        recent_scores = [s.overall_score() for s in snapshots[-5:]]
        
        # Check if recent scores are very similar (low variance)
        if len(recent_scores) > 1:
            variance = statistics.variance(recent_scores)
            
            # Low variance indicates plateau
            if variance < 0.01:  # Very low variance threshold
                return {
                    'duration': len(recent_scores),
                    'variance': variance,
                    'average_score': statistics.mean(recent_scores)
                }
        
        return None
    
    def _generate_trend_summary(self, trends: Dict[str, Any], overall_trend: Dict[str, Any], 
                               patterns: List[Dict[str, Any]]) -> str:
        """Generate human-readable trend summary."""
        summary_parts = []
        
        # Overall trend
        if overall_trend['direction'] == 'improving':
            summary_parts.append(f"✅ Overall performance is improving (slope: {overall_trend['slope']:.3f})")
        elif overall_trend['direction'] == 'declining':
            summary_parts.append(f"⚠️ Overall performance is declining (slope: {overall_trend['slope']:.3f})")
        else:
            summary_parts.append(f"➡️ Overall performance is stable")
        
        # Key metric improvements
        improving_metrics = [metric for metric, trend in trends.items() 
                           if trend['direction'] == 'improving' and trend['confidence'] > 0.6]
        if improving_metrics:
            summary_parts.append(f"📈 Improving: {', '.join(improving_metrics)}")
        
        # Key metric declines
        declining_metrics = [metric for metric, trend in trends.items() 
                           if trend['direction'] == 'declining' and trend['confidence'] > 0.6]
        if declining_metrics:
            summary_parts.append(f"📉 Declining: {', '.join(declining_metrics)}")
        
        # Patterns
        if patterns:
            pattern_descriptions = [p['description'] for p in patterns]
            summary_parts.append(f"🔍 Patterns: {'; '.join(pattern_descriptions)}")
        
        return ' | '.join(summary_parts)
    
    async def _generate_trend_recommendations(self, agent_id: str, trends: Dict[str, Any], 
                                            patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        # Recommendations based on declining metrics
        for metric, trend in trends.items():
            if trend['direction'] == 'declining' and trend['confidence'] > 0.7:
                if metric == 'task_success_rate':
                    recommendations.append("Focus on improving task execution strategies")
                elif metric == 'code_quality_score':
                    recommendations.append("Review and enhance code quality practices")
                elif metric == 'collaboration_rating':
                    recommendations.append("Improve inter-agent communication and collaboration")
                elif metric == 'resource_utilization':
                    recommendations.append("Optimize resource usage and efficiency")
                elif metric == 'learning_velocity':
                    recommendations.append("Enhance learning algorithms and knowledge integration")
        
        # Recommendations based on patterns
        for pattern in patterns:
            if pattern['type'] == 'performance_drop':
                recommendations.append("Investigate causes of recent performance drops")
            elif pattern['type'] == 'improvement_plateau':
                recommendations.append("Consider new optimization strategies to break through performance plateau")
            elif pattern['type'] == 'cyclical':
                recommendations.append("Analyze cyclical performance patterns for optimization opportunities")
        
        # General recommendations if no specific issues found
        if not recommendations:
            recommendations.append("Continue current strategies - performance is stable")
        
        return recommendations
    
    async def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple agents."""
        if len(agent_ids) < 2:
            return {"error": "Need at least 2 agents for comparison"}
        
        agent_stats = {}
        
        for agent_id in agent_ids:
            history = self.performance_history.get(agent_id, [])
            if history:
                recent_snapshot = history[-1]
                agent_stats[agent_id] = {
                    'latest_score': recent_snapshot.overall_score(),
                    'snapshots_count': len(history),
                    'improvement_areas': recent_snapshot.get_improvement_areas(),
                    'strengths': self._identify_agent_strengths(recent_snapshot)
                }
        
        # Rank agents by performance
        ranked_agents = sorted(agent_stats.keys(), 
                             key=lambda x: agent_stats[x]['latest_score'], 
                             reverse=True)
        
        # Identify best practices from top performers
        best_practices = await self._extract_best_practices(ranked_agents, agent_stats)
        
        return {
            'comparison_summary': agent_stats,
            'rankings': ranked_agents,
            'best_practices': best_practices,
            'improvement_opportunities': self._identify_cross_agent_improvements(agent_stats)
        }
    
    def _identify_agent_strengths(self, snapshot: PerformanceSnapshot) -> List[str]:
        """Identify an agent's key strengths."""
        strengths = []
        threshold = 0.8
        
        if snapshot.task_success_rate > threshold:
            strengths.append("task_execution")
        if snapshot.code_quality_score > threshold:
            strengths.append("code_quality")
        if snapshot.collaboration_rating > threshold:
            strengths.append("collaboration")
        if snapshot.learning_velocity > threshold:
            strengths.append("learning_ability")
        if snapshot.decision_accuracy > threshold:
            strengths.append("decision_making")
        if snapshot.error_rate < 0.1:
            strengths.append("reliability")
            
        return strengths
    
    async def _extract_best_practices(self, ranked_agents: List[str], 
                                    agent_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract best practices from top-performing agents."""
        best_practices = []
        
        if not ranked_agents:
            return best_practices
        
        top_agent = ranked_agents[0]
        top_strengths = agent_stats[top_agent]['strengths']
        
        for strength in top_strengths:
            best_practices.append({
                'practice': f"Excel in {strength}",
                'source_agent': top_agent,
                'performance_area': strength,
                'recommendation': f"Study {top_agent}'s approach to {strength} for improvement"
            })
        
        return best_practices
    
    def _identify_cross_agent_improvements(self, agent_stats: Dict[str, Any]) -> List[str]:
        """Identify improvement opportunities by comparing agents."""
        improvements = []
        
        # Find common improvement areas
        all_improvement_areas = []
        for stats in agent_stats.values():
            all_improvement_areas.extend(stats['improvement_areas'])
        
        if all_improvement_areas:
            from collections import Counter
            common_areas = Counter(all_improvement_areas).most_common(3)
            
            for area, count in common_areas:
                if count > 1:
                    improvements.append(f"Multiple agents need improvement in: {area}")
        
        return improvements


class ExperimentManager:
    """Manages optimization experiments across agents."""
    
    def __init__(self):
        self.active_experiments: Dict[str, OptimizationExperiment] = {}
        self.completed_experiments: List[OptimizationExperiment] = []
        self.experiment_queue: List[Dict[str, Any]] = []
        self.success_threshold = 0.05  # 5% improvement threshold
    
    async def design_experiment(self, agent_id: str, performance_snapshot: PerformanceSnapshot,
                               improvement_areas: List[str]) -> Optional[OptimizationExperiment]:
        """Design an optimization experiment for an agent."""
        if not improvement_areas:
            return None
        
        # Select the most critical improvement area
        primary_area = improvement_areas[0]
        optimization_type = self._map_area_to_optimization_type(primary_area)
        
        # Select appropriate strategy
        strategy = self._select_optimization_strategy(optimization_type, performance_snapshot)
        
        # Generate hypothesis
        hypothesis = self._generate_hypothesis(agent_id, optimization_type, performance_snapshot)
        
        # Design parameter changes
        parameters_before = await self._get_current_parameters(agent_id)
        parameters_after = self._design_parameter_changes(optimization_type, parameters_before)
        
        experiment = OptimizationExperiment(
            experiment_id=str(uuid.uuid4()),
            agent_id=agent_id,
            optimization_type=optimization_type,
            strategy=strategy,
            hypothesis=hypothesis,
            parameters_before=parameters_before,
            parameters_after=parameters_after,
            performance_before=performance_snapshot,
            performance_after=None,
            success=None,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            started_at=datetime.now(),
            completed_at=None,
            duration=None
        )
        
        return experiment
    
    async def start_experiment(self, experiment: OptimizationExperiment) -> bool:
        """Start an optimization experiment."""
        try:
            self.active_experiments[experiment.experiment_id] = experiment
            
            # Apply parameter changes to the agent
            await self._apply_parameter_changes(experiment.agent_id, experiment.parameters_after)
            
            logger.info(f"Started optimization experiment {experiment.experiment_id} for agent {experiment.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {experiment.experiment_id}: {e}")
            return False
    
    async def evaluate_experiment(self, experiment_id: str, 
                                post_performance: PerformanceSnapshot) -> Optional[OptimizationExperiment]:
        """Evaluate a completed experiment."""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        experiment.performance_after = post_performance
        experiment.completed_at = datetime.now()
        experiment.duration = experiment.completed_at - experiment.started_at
        
        # Calculate statistical significance
        experiment.statistical_significance = self._calculate_statistical_significance(
            experiment.performance_before, post_performance
        )
        
        # Determine success
        experiment.success = experiment.is_successful()
        
        # Calculate confidence interval
        experiment.confidence_interval = self._calculate_confidence_interval(
            experiment.performance_before, post_performance
        )
        
        # Move to completed experiments
        del self.active_experiments[experiment_id]
        self.completed_experiments.append(experiment)
        
        # If experiment failed, revert parameters
        if not experiment.success:
            await self._revert_parameter_changes(experiment.agent_id, experiment.parameters_before)
            logger.info(f"Experiment {experiment_id} failed - reverted parameters")
        else:
            logger.info(f"Experiment {experiment_id} succeeded - keeping parameters")
        
        return experiment
    
    def _map_area_to_optimization_type(self, improvement_area: str) -> OptimizationType:
        """Map improvement area to optimization type."""
        mapping = {
            'task_execution': OptimizationType.PERFORMANCE,
            'code_quality': OptimizationType.ACCURACY,
            'collaboration': OptimizationType.COLLABORATION,
            'resource_efficiency': OptimizationType.RESOURCE_USAGE,
            'learning_rate': OptimizationType.LEARNING_RATE,
            'decision_making': OptimizationType.DECISION_QUALITY,
            'error_reduction': OptimizationType.ACCURACY
        }
        
        return mapping.get(improvement_area, OptimizationType.EFFICIENCY)
    
    def _select_optimization_strategy(self, optimization_type: OptimizationType, 
                                    performance: PerformanceSnapshot) -> LearningStrategy:
        """Select appropriate optimization strategy."""
        # Simple strategy selection logic
        if optimization_type == OptimizationType.LEARNING_RATE:
            return LearningStrategy.GRADIENT_DESCENT
        elif optimization_type == OptimizationType.COLLABORATION:
            return LearningStrategy.REINFORCEMENT
        elif optimization_type == OptimizationType.DECISION_QUALITY:
            return LearningStrategy.BAYESIAN
        else:
            return LearningStrategy.EVOLUTIONARY
    
    def _generate_hypothesis(self, agent_id: str, optimization_type: OptimizationType, 
                           performance: PerformanceSnapshot) -> str:
        """Generate experiment hypothesis."""
        hypotheses = {
            OptimizationType.PERFORMANCE: f"Adjusting task execution parameters will improve success rate above {performance.task_success_rate:.2f}",
            OptimizationType.ACCURACY: f"Modifying decision-making thresholds will improve accuracy above {performance.decision_accuracy:.2f}",
            OptimizationType.EFFICIENCY: f"Optimizing resource allocation will reduce utilization below {performance.resource_utilization:.2f}",
            OptimizationType.COLLABORATION: f"Enhancing communication protocols will improve collaboration rating above {performance.collaboration_rating:.2f}",
            OptimizationType.LEARNING_RATE: f"Adjusting learning parameters will increase velocity above {performance.learning_velocity:.2f}",
            OptimizationType.DECISION_QUALITY: f"Refining decision algorithms will improve accuracy above {performance.decision_accuracy:.2f}"
        }
        
        return hypotheses.get(optimization_type, "Parameter optimization will improve overall performance")
    
    async def _get_current_parameters(self, agent_id: str) -> Dict[str, Any]:
        """Get current agent parameters."""
        # This would interface with the actual agent configuration
        # For now, return mock parameters
        return {
            'learning_rate': 0.1,
            'confidence_threshold': 0.7,
            'context_window_limit': 0.8,
            'collaboration_frequency': 5,
            'decision_timeout': 30,
            'resource_limit': 0.8
        }
    
    def _design_parameter_changes(self, optimization_type: OptimizationType, 
                                current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Design parameter changes for the experiment."""
        new_params = current_params.copy()
        
        if optimization_type == OptimizationType.LEARNING_RATE:
            # Adjust learning rate
            current_lr = new_params.get('learning_rate', 0.1)
            new_params['learning_rate'] = min(0.2, current_lr * 1.2)  # 20% increase
        
        elif optimization_type == OptimizationType.ACCURACY:
            # Adjust confidence threshold
            current_threshold = new_params.get('confidence_threshold', 0.7)
            new_params['confidence_threshold'] = min(0.9, current_threshold + 0.1)
        
        elif optimization_type == OptimizationType.RESOURCE_USAGE:
            # Reduce resource limits
            current_limit = new_params.get('resource_limit', 0.8)
            new_params['resource_limit'] = max(0.5, current_limit - 0.1)
        
        elif optimization_type == OptimizationType.COLLABORATION:
            # Increase collaboration frequency
            current_freq = new_params.get('collaboration_frequency', 5)
            new_params['collaboration_frequency'] = min(10, current_freq + 2)
        
        return new_params
    
    async def _apply_parameter_changes(self, agent_id: str, new_parameters: Dict[str, Any]) -> None:
        """Apply parameter changes to an agent."""
        # This would interface with the actual agent configuration system
        logger.info(f"Applied parameter changes to agent {agent_id}: {new_parameters}")
    
    async def _revert_parameter_changes(self, agent_id: str, original_parameters: Dict[str, Any]) -> None:
        """Revert parameter changes for a failed experiment."""
        # This would interface with the actual agent configuration system
        logger.info(f"Reverted parameters for agent {agent_id}: {original_parameters}")
    
    def _calculate_statistical_significance(self, before: PerformanceSnapshot, 
                                          after: PerformanceSnapshot) -> float:
        """Calculate statistical significance of the improvement."""
        # Simplified significance calculation
        # In practice, this would use proper statistical tests
        
        before_score = before.overall_score()
        after_score = after.overall_score()
        
        improvement = after_score - before_score
        
        # Mock significance calculation (would use t-test or similar)
        if abs(improvement) > 0.1:
            return 0.95
        elif abs(improvement) > 0.05:
            return 0.8
        else:
            return 0.5
    
    def _calculate_confidence_interval(self, before: PerformanceSnapshot, 
                                     after: PerformanceSnapshot) -> Tuple[float, float]:
        """Calculate confidence interval for the improvement."""
        # Simplified confidence interval calculation
        improvement = after.overall_score() - before.overall_score()
        margin_of_error = 0.02  # Mock margin of error
        
        return (improvement - margin_of_error, improvement + margin_of_error)


class SelfOptimizationAgent(IntelligenceModelInterface):
    """
    Self-Optimization Agent with performance-based learning and continuous improvement.
    
    This agent specializes in:
    - Analyzing agent performance patterns
    - Designing and running optimization experiments
    - Learning from successful optimizations
    - Sharing optimization insights across agents
    """
    
    def __init__(self, agent_id: str, anthropic_client: Optional[AsyncAnthropic] = None):
        self.agent_id = agent_id
        self.client = anthropic_client
        self.performance_analyzer = PerformanceAnalyzer()
        self.experiment_manager = ExperimentManager()
        self.learning_insights: List[LearningInsight] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Optimization settings
        self.min_experiment_duration = timedelta(hours=1)
        self.max_concurrent_experiments = 3
        self.learning_rate = 0.1
    
    async def predict(self, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Make performance optimization predictions."""
        request_type = input_data.get('type', 'performance_analysis')
        agent_id = input_data.get('agent_id', '')
        
        if request_type == 'performance_analysis':
            return await self._analyze_agent_performance(agent_id, input_data)
        elif request_type == 'optimization_recommendation':
            return await self._recommend_optimizations(agent_id, input_data)
        elif request_type == 'experiment_design':
            return await self._design_optimization_experiment(agent_id, input_data)
        elif request_type == 'cross_agent_insights':
            return await self._provide_cross_agent_insights(input_data)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _analyze_agent_performance(self, agent_id: str, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Analyze an agent's performance and identify optimization opportunities."""
        
        # Get performance snapshot from input data
        performance_data = input_data.get('performance_data', {})
        snapshot = PerformanceSnapshot(
            agent_id=agent_id,
            timestamp=datetime.now(),
            task_success_rate=performance_data.get('task_success_rate', 0.7),
            average_task_time=performance_data.get('average_task_time', 120.0),
            code_quality_score=performance_data.get('code_quality_score', 0.8),
            collaboration_rating=performance_data.get('collaboration_rating', 0.7),
            resource_utilization=performance_data.get('resource_utilization', 0.6),
            learning_velocity=performance_data.get('learning_velocity', 0.6),
            decision_accuracy=performance_data.get('decision_accuracy', 0.8),
            context_efficiency=performance_data.get('context_efficiency', 0.7),
            user_satisfaction=performance_data.get('user_satisfaction', 0.8),
            error_rate=performance_data.get('error_rate', 0.1)
        )
        
        # Add to performance history
        await self.performance_analyzer.add_performance_snapshot(snapshot)
        
        # Analyze trends
        trend_analysis = await self.performance_analyzer.analyze_trends(agent_id)
        
        # Identify improvement areas
        improvement_areas = snapshot.get_improvement_areas()
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(snapshot, trend_analysis)
        
        analysis_result = {
            'performance_snapshot': asdict(snapshot),
            'overall_score': snapshot.overall_score(),
            'improvement_areas': improvement_areas,
            'trend_analysis': trend_analysis,
            'optimization_potential': optimization_potential,
            'recommendations': self._generate_performance_recommendations(snapshot, trend_analysis)
        }
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"agent_id": agent_id},
            prediction=analysis_result,
            confidence=0.9,
            explanation=f"Performance analysis for {agent_id}: {snapshot.overall_score():.2f}/1.0 overall score",
            timestamp=datetime.now()
        )
    
    async def _recommend_optimizations(self, agent_id: str, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Recommend specific optimizations for an agent."""
        
        # Get recent performance data
        performance_history = self.performance_analyzer.performance_history.get(agent_id, [])
        if not performance_history:
            return IntelligencePrediction(
                model_id=self.agent_id,
                prediction_id=str(uuid.uuid4()),
                input_data={"agent_id": agent_id},
                prediction={"error": "No performance history available"},
                confidence=0.0,
                explanation="Cannot provide recommendations without performance history",
                timestamp=datetime.now()
            )
        
        latest_snapshot = performance_history[-1]
        improvement_areas = latest_snapshot.get_improvement_areas()
        
        # Generate specific optimization recommendations
        recommendations = []
        
        for area in improvement_areas:
            area_recommendations = await self._generate_area_specific_recommendations(
                agent_id, area, latest_snapshot
            )
            recommendations.extend(area_recommendations)
        
        # Prioritize recommendations by impact
        prioritized_recommendations = sorted(
            recommendations, 
            key=lambda x: x.get('estimated_impact', 0), 
            reverse=True
        )
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"agent_id": agent_id, "improvement_areas": improvement_areas},
            prediction={
                "recommendations": prioritized_recommendations[:10],  # Top 10 recommendations
                "total_recommendations": len(recommendations),
                "improvement_potential": sum(r.get('estimated_impact', 0) for r in recommendations),
                "implementation_timeline": self._estimate_implementation_timeline(prioritized_recommendations)
            },
            confidence=0.8,
            explanation=f"Generated {len(recommendations)} optimization recommendations for {agent_id}",
            timestamp=datetime.now()
        )
    
    async def _design_optimization_experiment(self, agent_id: str, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Design an optimization experiment for an agent."""
        
        # Get current performance
        performance_history = self.performance_analyzer.performance_history.get(agent_id, [])
        if not performance_history:
            return IntelligencePrediction(
                model_id=self.agent_id,
                prediction_id=str(uuid.uuid4()),
                input_data={"agent_id": agent_id},
                prediction={"error": "No performance history to base experiment on"},
                confidence=0.0,
                explanation="Need performance history to design experiment",
                timestamp=datetime.now()
            )
        
        latest_snapshot = performance_history[-1]
        improvement_areas = input_data.get('focus_areas', latest_snapshot.get_improvement_areas())
        
        # Design experiment
        experiment = await self.experiment_manager.design_experiment(
            agent_id, latest_snapshot, improvement_areas
        )
        
        if not experiment:
            return IntelligencePrediction(
                model_id=self.agent_id,
                prediction_id=str(uuid.uuid4()),
                input_data={"agent_id": agent_id},
                prediction={"error": "Could not design experiment"},
                confidence=0.0,
                explanation="No suitable improvement areas found for experimentation",
                timestamp=datetime.now()
            )
        
        # Create experiment plan
        experiment_plan = {
            "experiment": asdict(experiment),
            "execution_plan": {
                "duration": "1-2 hours",
                "monitoring_metrics": self._get_monitoring_metrics(experiment.optimization_type),
                "success_criteria": self._define_success_criteria(experiment),
                "rollback_plan": "Automatic parameter reversion if experiment fails"
            },
            "risk_assessment": self._assess_experiment_risk(experiment),
            "expected_outcomes": self._predict_experiment_outcomes(experiment)
        }
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=experiment.experiment_id,
            input_data={"agent_id": agent_id, "focus_areas": improvement_areas},
            prediction=experiment_plan,
            confidence=0.7,
            explanation=f"Designed {experiment.optimization_type.value} optimization experiment for {agent_id}",
            timestamp=datetime.now()
        )
    
    async def _provide_cross_agent_insights(self, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Provide insights from cross-agent performance analysis."""
        
        agent_ids = input_data.get('agent_ids', [])
        if len(agent_ids) < 2:
            return IntelligencePrediction(
                model_id=self.agent_id,
                prediction_id=str(uuid.uuid4()),
                input_data=input_data,
                prediction={"error": "Need at least 2 agents for cross-agent analysis"},
                confidence=0.0,
                explanation="Cross-agent analysis requires multiple agents",
                timestamp=datetime.now()
            )
        
        # Compare agent performance
        comparison = await self.performance_analyzer.compare_agents(agent_ids)
        
        # Extract insights
        insights = []
        
        # Performance ranking insights
        if len(comparison['rankings']) > 1:
            top_performer = comparison['rankings'][0]
            insights.append({
                'type': 'performance_leader',
                'title': f"Agent {top_performer} is the top performer",
                'description': f"Consider studying {top_performer}'s strategies",
                'impact': 'high'
            })
        
        # Best practices insights
        for practice in comparison.get('best_practices', []):
            insights.append({
                'type': 'best_practice',
                'title': f"Best practice: {practice['practice']}",
                'description': f"Source: {practice['source_agent']}",
                'recommendation': practice['recommendation'],
                'impact': 'medium'
            })
        
        # Common improvement areas
        for improvement in comparison.get('improvement_opportunities', []):
            insights.append({
                'type': 'common_improvement',
                'title': 'Common improvement opportunity',
                'description': improvement,
                'impact': 'medium'
            })
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data=input_data,
            prediction={
                "agent_comparison": comparison,
                "insights": insights,
                "recommendations": self._generate_cross_agent_recommendations(comparison, insights)
            },
            confidence=0.8,
            explanation=f"Cross-agent analysis of {len(agent_ids)} agents with {len(insights)} insights",
            timestamp=datetime.now()
        )
    
    def _calculate_optimization_potential(self, snapshot: PerformanceSnapshot, 
                                        trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the potential for optimization."""
        current_score = snapshot.overall_score()
        
        # Calculate theoretical maximum improvement
        improvement_areas = snapshot.get_improvement_areas()
        max_potential = 1.0 - current_score
        
        # Adjust based on trends
        trend_direction = trend_analysis.get('overall_trend', {}).get('direction', 'stable')
        if trend_direction == 'improving':
            achievable_potential = max_potential * 0.7  # Already improving, harder to optimize further
        elif trend_direction == 'declining':
            achievable_potential = max_potential * 0.9  # Declining, high potential for improvement
        else:
            achievable_potential = max_potential * 0.8  # Stable, moderate potential
        
        return {
            'current_score': current_score,
            'maximum_potential': max_potential,
            'achievable_potential': achievable_potential,
            'improvement_areas_count': len(improvement_areas),
            'difficulty_level': 'high' if achievable_potential < 0.1 else 'medium' if achievable_potential < 0.2 else 'low'
        }
    
    def _generate_performance_recommendations(self, snapshot: PerformanceSnapshot, 
                                            trend_analysis: Dict[str, Any]) -> List[str]:
        """Generate performance-based recommendations."""
        recommendations = []
        
        # Based on improvement areas
        improvement_areas = snapshot.get_improvement_areas()
        for area in improvement_areas:
            if area == 'task_execution':
                recommendations.append("🎯 Focus on improving task success rates through better planning")
            elif area == 'code_quality':
                recommendations.append("📝 Implement code quality checks and review processes")
            elif area == 'collaboration':
                recommendations.append("🤝 Enhance inter-agent communication protocols")
            elif area == 'resource_efficiency':
                recommendations.append("⚡ Optimize resource usage and reduce waste")
        
        # Based on trends
        trend_direction = trend_analysis.get('overall_trend', {}).get('direction', 'stable')
        if trend_direction == 'declining':
            recommendations.append("⚠️ Address performance decline immediately")
        elif trend_direction == 'stable':
            recommendations.append("🚀 Consider new optimization strategies to drive improvement")
        
        return recommendations
    
    async def _generate_area_specific_recommendations(self, agent_id: str, area: str, 
                                                    snapshot: PerformanceSnapshot) -> List[Dict[str, Any]]:
        """Generate specific recommendations for an improvement area."""
        recommendations = []
        
        if area == 'task_execution':
            recommendations.extend([
                {
                    'type': 'parameter_adjustment',
                    'title': 'Increase confidence threshold',
                    'description': 'Raise decision confidence threshold to improve accuracy',
                    'estimated_impact': 0.15,
                    'implementation_effort': 'low',
                    'risk_level': 'low'
                },
                {
                    'type': 'strategy_change',
                    'title': 'Implement task validation',
                    'description': 'Add pre-execution task validation to catch issues early',
                    'estimated_impact': 0.20,
                    'implementation_effort': 'medium',
                    'risk_level': 'low'
                }
            ])
        
        elif area == 'code_quality':
            recommendations.extend([
                {
                    'type': 'process_improvement',
                    'title': 'Add automated code review',
                    'description': 'Implement automated code quality checks before execution',
                    'estimated_impact': 0.25,
                    'implementation_effort': 'medium',
                    'risk_level': 'low'
                }
            ])
        
        elif area == 'collaboration':
            recommendations.extend([
                {
                    'type': 'communication_enhancement',
                    'title': 'Increase status update frequency',
                    'description': 'Send more frequent progress updates to other agents',
                    'estimated_impact': 0.10,
                    'implementation_effort': 'low',
                    'risk_level': 'low'
                }
            ])
        
        elif area == 'resource_efficiency':
            recommendations.extend([
                {
                    'type': 'resource_optimization',
                    'title': 'Implement resource pooling',
                    'description': 'Share resources more efficiently across agents',
                    'estimated_impact': 0.30,
                    'implementation_effort': 'high',
                    'risk_level': 'medium'
                }
            ])
        
        return recommendations
    
    def _estimate_implementation_timeline(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate implementation timeline for recommendations."""
        total_effort = 0
        effort_mapping = {'low': 1, 'medium': 3, 'high': 8}
        
        for rec in recommendations:
            effort = rec.get('implementation_effort', 'medium')
            total_effort += effort_mapping.get(effort, 3)
        
        return {
            'total_effort_hours': total_effort,
            'estimated_days': max(1, total_effort // 8),
            'phases': [
                {'phase': 'Low effort optimizations', 'duration': '1-2 hours'},
                {'phase': 'Medium effort improvements', 'duration': '1-2 days'},
                {'phase': 'High effort transformations', 'duration': '1-2 weeks'}
            ]
        }
    
    def _get_monitoring_metrics(self, optimization_type: OptimizationType) -> List[str]:
        """Get metrics to monitor during experiment."""
        base_metrics = ['overall_score', 'task_success_rate', 'error_rate']
        
        type_specific_metrics = {
            OptimizationType.PERFORMANCE: ['average_task_time', 'task_success_rate'],
            OptimizationType.ACCURACY: ['decision_accuracy', 'error_rate'],
            OptimizationType.RESOURCE_USAGE: ['resource_utilization'],
            OptimizationType.COLLABORATION: ['collaboration_rating'],
            OptimizationType.LEARNING_RATE: ['learning_velocity']
        }
        
        return base_metrics + type_specific_metrics.get(optimization_type, [])
    
    def _define_success_criteria(self, experiment: OptimizationExperiment) -> Dict[str, Any]:
        """Define success criteria for an experiment."""
        return {
            'minimum_improvement': 0.05,  # 5% improvement
            'statistical_significance': 0.95,
            'no_regression_in_other_metrics': True,
            'experiment_duration': '1-2 hours',
            'safety_thresholds': {
                'max_error_rate_increase': 0.1,
                'max_resource_increase': 0.2
            }
        }
    
    def _assess_experiment_risk(self, experiment: OptimizationExperiment) -> Dict[str, Any]:
        """Assess the risk level of an experiment."""
        risk_factors = []
        risk_level = 'low'
        
        # Check parameter changes
        param_changes = len(experiment.parameters_after)
        if param_changes > 3:
            risk_factors.append("Multiple parameter changes")
            risk_level = 'medium'
        
        # Check optimization type
        if experiment.optimization_type in [OptimizationType.RESOURCE_USAGE]:
            risk_factors.append("Resource-sensitive optimization")
            risk_level = 'medium'
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'mitigation_strategies': ['Automatic rollback on failure', 'Continuous monitoring']
        }
    
    def _predict_experiment_outcomes(self, experiment: OptimizationExperiment) -> Dict[str, Any]:
        """Predict potential experiment outcomes."""
        base_score = experiment.performance_before.overall_score()
        
        # Estimate improvement based on optimization type
        improvement_estimates = {
            OptimizationType.PERFORMANCE: 0.10,
            OptimizationType.ACCURACY: 0.08,
            OptimizationType.EFFICIENCY: 0.12,
            OptimizationType.RESOURCE_USAGE: 0.15,
            OptimizationType.COLLABORATION: 0.06,
            OptimizationType.LEARNING_RATE: 0.07
        }
        
        estimated_improvement = improvement_estimates.get(experiment.optimization_type, 0.08)
        
        return {
            'expected_improvement': estimated_improvement,
            'optimistic_scenario': estimated_improvement * 1.5,
            'conservative_scenario': estimated_improvement * 0.5,
            'success_probability': 0.7,
            'expected_duration': '1-2 hours'
        }
    
    def _generate_cross_agent_recommendations(self, comparison: Dict[str, Any], 
                                            insights: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on cross-agent analysis."""
        recommendations = []
        
        # Recommendations based on rankings
        rankings = comparison.get('rankings', [])
        if len(rankings) > 1:
            top_performer = rankings[0]
            recommendations.append(f"Study {top_performer}'s strategies for improvement")
        
        # Recommendations based on best practices
        best_practices = comparison.get('best_practices', [])
        if best_practices:
            recommendations.append("Implement identified best practices across underperforming agents")
        
        # Recommendations based on common improvement areas
        improvements = comparison.get('improvement_opportunities', [])
        if improvements:
            recommendations.append("Address common improvement areas systematically")
        
        return recommendations
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train the Self-Optimization Agent with experiment results."""
        try:
            for data_point in training_data:
                if data_point.data_type == DataType.SYSTEM_METRICS:
                    # Process experiment results
                    experiment_results = data_point.metadata.get('experiment_results')
                    if experiment_results:
                        # Learn from successful experiments
                        if experiment_results.get('success'):
                            insight = LearningInsight(
                                insight_id=str(uuid.uuid4()),
                                category='optimization_success',
                                title=f"Successful {experiment_results['optimization_type']} optimization",
                                description=f"Experiment improved performance by {experiment_results.get('improvement', 0):.2%}",
                                supporting_experiments=[experiment_results['experiment_id']],
                                confidence=0.8,
                                impact_score=experiment_results.get('improvement', 0),
                                applicability=[experiment_results.get('agent_id', 'unknown')],
                                created_at=datetime.now(),
                                validated=True
                            )
                            self.learning_insights.append(insight)
            
            logger.info(f"Self-Optimization Agent trained on {len(training_data)} experiment results")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    async def evaluate(self, test_data: List[DataPoint]) -> Dict[str, float]:
        """Evaluate the Self-Optimization Agent performance."""
        successful_optimizations = 0
        total_experiments = len(test_data)
        
        for data_point in test_data:
            experiment_data = data_point.metadata.get('experiment_data')
            if experiment_data:
                # Simulate optimization recommendation
                try:
                    prediction = await self.predict({
                        'type': 'optimization_recommendation',
                        'agent_id': experiment_data.get('agent_id'),
                        'performance_data': experiment_data.get('performance_data')
                    })
                    
                    if prediction.confidence > 0.6:
                        successful_optimizations += 1
                        
                except Exception as e:
                    logger.warning(f"Evaluation error: {e}")
                    continue
        
        accuracy = successful_optimizations / total_experiments if total_experiments > 0 else 0.0
        
        return {
            'optimization_accuracy': accuracy,
            'total_experiments_analyzed': total_experiments,
            'successful_recommendations': successful_optimizations,
            'learning_insights_generated': len(self.learning_insights),
            'active_experiments': len(self.experiment_manager.active_experiments),
            'completed_experiments': len(self.experiment_manager.completed_experiments)
        }
    
    async def get_optimization_insights(self) -> Dict[str, Any]:
        """Get comprehensive optimization insights."""
        return {
            'learning_insights': [insight.to_knowledge_item() for insight in self.learning_insights],
            'experiment_summary': {
                'total_experiments': len(self.experiment_manager.completed_experiments),
                'success_rate': self._calculate_experiment_success_rate(),
                'most_successful_optimizations': self._get_most_successful_optimizations(),
                'common_failure_patterns': self._identify_failure_patterns()
            },
            'performance_trends': await self._get_global_performance_trends(),
            'optimization_recommendations': self._generate_global_recommendations()
        }
    
    def _calculate_experiment_success_rate(self) -> float:
        """Calculate overall experiment success rate."""
        if not self.experiment_manager.completed_experiments:
            return 0.0
        
        successful = len([exp for exp in self.experiment_manager.completed_experiments if exp.success])
        total = len(self.experiment_manager.completed_experiments)
        
        return successful / total
    
    def _get_most_successful_optimizations(self) -> List[Dict[str, Any]]:
        """Get the most successful optimization types."""
        optimization_results = defaultdict(list)
        
        for exp in self.experiment_manager.completed_experiments:
            if exp.success and exp.performance_after:
                improvement = exp.performance_after.overall_score() - exp.performance_before.overall_score()
                optimization_results[exp.optimization_type.value].append(improvement)
        
        # Calculate average improvement for each type
        avg_improvements = {}
        for opt_type, improvements in optimization_results.items():
            if improvements:
                avg_improvements[opt_type] = statistics.mean(improvements)
        
        # Sort by average improvement
        sorted_optimizations = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)
        
        return [{'type': opt_type, 'avg_improvement': improvement} 
                for opt_type, improvement in sorted_optimizations[:5]]
    
    def _identify_failure_patterns(self) -> List[str]:
        """Identify common patterns in failed experiments."""
        failed_experiments = [exp for exp in self.experiment_manager.completed_experiments if not exp.success]
        
        patterns = []
        
        # Analyze failure patterns
        failure_types = defaultdict(int)
        for exp in failed_experiments:
            failure_types[exp.optimization_type.value] += 1
        
        for opt_type, count in failure_types.items():
            if count > 2:  # More than 2 failures of the same type
                patterns.append(f"Frequent failures in {opt_type} optimizations")
        
        return patterns
    
    async def _get_global_performance_trends(self) -> Dict[str, Any]:
        """Get global performance trends across all agents."""
        all_snapshots = []
        for agent_history in self.performance_analyzer.performance_history.values():
            all_snapshots.extend(agent_history)
        
        if len(all_snapshots) < 5:
            return {"status": "insufficient_data"}
        
        # Sort by timestamp
        all_snapshots.sort(key=lambda x: x.timestamp)
        
        # Calculate trends
        overall_scores = [snapshot.overall_score() for snapshot in all_snapshots]
        global_trend = self.performance_analyzer._calculate_trend(overall_scores)
        
        return {
            'global_trend': global_trend,
            'total_agents_monitored': len(self.performance_analyzer.performance_history),
            'total_snapshots': len(all_snapshots),
            'average_performance': statistics.mean(overall_scores),
            'performance_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        }
    
    def _generate_global_recommendations(self) -> List[str]:
        """Generate global optimization recommendations."""
        recommendations = []
        
        # Based on successful experiments
        if self._calculate_experiment_success_rate() < 0.7:
            recommendations.append("Review experiment design methodology - success rate is low")
        
        # Based on learning insights
        high_impact_insights = [insight for insight in self.learning_insights if insight.impact_score > 0.15]
        if high_impact_insights:
            recommendations.append(f"Implement {len(high_impact_insights)} high-impact optimization insights")
        
        # Based on performance trends
        if len(self.performance_analyzer.performance_history) > 3:
            recommendations.append("Continue monitoring performance trends for optimization opportunities")
        
        return recommendations


async def create_self_optimization_agent(agent_id: str) -> SelfOptimizationAgent:
    """Factory function to create a new Self-Optimization Agent."""
    anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
    return SelfOptimizationAgent(agent_id, anthropic_client)
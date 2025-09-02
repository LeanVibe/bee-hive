"""
Utility Functions for AgentManagementAPI v2

Provides shared utilities, helpers, and common operations for the consolidated
agent management API with performance optimization and intelligent caching.
"""

import uuid
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, deque
import json
import structlog

try:
    from ....models.agent import AgentStatus, AgentType
except ImportError:
    from enum import Enum
    
    class AgentStatus(Enum):
        ACTIVE = "ACTIVE"
        INACTIVE = "INACTIVE"
    
    class AgentType(Enum):
        CLAUDE = "claude"
        OPENAI = "openai"

try:
    from ....core.coordination import CoordinationMode, ProjectStatus
except ImportError:
    from enum import Enum
    
    class CoordinationMode(Enum):
        PARALLEL = "PARALLEL"
        SEQUENTIAL = "SEQUENTIAL"
    
    class ProjectStatus(Enum):
        ACTIVE = "ACTIVE"
        COMPLETED = "COMPLETED"

logger = structlog.get_logger()


# ========================================
# Agent Management Utilities
# ========================================

class AgentCapabilityMatcher:
    """
    Intelligent agent capability matching for optimal task assignment.
    
    Provides advanced capability analysis with machine learning-inspired
    scoring algorithms for optimal agent-task matching.
    """
    
    def __init__(self):
        self.capability_weights = {
            'exact_match': 1.0,
            'related_skill': 0.8,
            'general_ability': 0.6,
            'learning_potential': 0.4
        }
        self.experience_multipliers = {
            'expert': 1.2,
            'senior': 1.1,
            'intermediate': 1.0,
            'junior': 0.9,
            'beginner': 0.7
        }
    
    def calculate_match_score(
        self,
        agent_capabilities: List[str],
        agent_specializations: List[str],
        agent_experience: str,
        agent_proficiency: float,
        task_requirements: List[str],
        task_complexity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive match score between agent and task.
        
        Returns detailed scoring breakdown with optimization recommendations.
        """
        try:
            # Base capability matching
            capability_score = self._calculate_capability_score(
                agent_capabilities, task_requirements
            )
            
            # Specialization bonus
            specialization_score = self._calculate_specialization_score(
                agent_specializations, task_requirements
            )
            
            # Experience factor
            experience_multiplier = self.experience_multipliers.get(
                agent_experience.lower(), 1.0
            )
            
            # Proficiency adjustment
            proficiency_adjustment = min(1.2, max(0.8, agent_proficiency))
            
            # Task complexity consideration
            complexity_factor = 1.0 - (task_complexity - agent_proficiency) * 0.5
            complexity_factor = max(0.3, min(1.2, complexity_factor))
            
            # Calculate final score
            base_score = (capability_score * 0.4 + specialization_score * 0.6)
            adjusted_score = base_score * experience_multiplier * proficiency_adjustment * complexity_factor
            final_score = min(1.0, max(0.0, adjusted_score))
            
            return {
                'overall_score': final_score,
                'capability_score': capability_score,
                'specialization_score': specialization_score,
                'experience_multiplier': experience_multiplier,
                'proficiency_adjustment': proficiency_adjustment,
                'complexity_factor': complexity_factor,
                'recommendation': self._generate_recommendation(final_score),
                'improvement_suggestions': self._generate_improvement_suggestions(
                    agent_capabilities, agent_specializations, task_requirements, final_score
                )
            }
            
        except Exception as e:
            logger.warning("Capability matching failed", error=str(e))
            return {
                'overall_score': 0.5,
                'capability_score': 0.5,
                'specialization_score': 0.5,
                'recommendation': 'uncertain',
                'error': str(e)
            }
    
    def _calculate_capability_score(
        self,
        agent_capabilities: List[str],
        task_requirements: List[str]
    ) -> float:
        """Calculate score based on capability overlap."""
        if not task_requirements:
            return 0.8  # Default score if no specific requirements
        
        matches = 0
        total_requirements = len(task_requirements)
        
        for requirement in task_requirements:
            requirement_lower = requirement.lower()
            
            # Check for exact matches
            if requirement_lower in [cap.lower() for cap in agent_capabilities]:
                matches += self.capability_weights['exact_match']
                continue
            
            # Check for related skills
            related_found = False
            for capability in agent_capabilities:
                if self._are_capabilities_related(requirement_lower, capability.lower()):
                    matches += self.capability_weights['related_skill']
                    related_found = True
                    break
            
            if not related_found:
                # Check for general abilities
                if self._has_general_ability(agent_capabilities, requirement_lower):
                    matches += self.capability_weights['general_ability']
        
        return min(1.0, matches / total_requirements)
    
    def _calculate_specialization_score(
        self,
        agent_specializations: List[str],
        task_requirements: List[str]
    ) -> float:
        """Calculate score based on specialization relevance."""
        if not agent_specializations or not task_requirements:
            return 0.7  # Neutral score
        
        relevance_scores = []
        
        for specialization in agent_specializations:
            spec_score = 0
            for requirement in task_requirements:
                if self._is_specialization_relevant(specialization, requirement):
                    spec_score += 1
            
            relevance_scores.append(spec_score / len(task_requirements))
        
        return max(relevance_scores) if relevance_scores else 0.5
    
    def _are_capabilities_related(self, cap1: str, cap2: str) -> bool:
        """Check if two capabilities are related."""
        related_groups = [
            ['python', 'django', 'flask', 'fastapi', 'backend'],
            ['javascript', 'typescript', 'react', 'vue', 'angular', 'frontend'],
            ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'devops'],
            ['testing', 'pytest', 'jest', 'selenium', 'qa'],
            ['database', 'sql', 'postgresql', 'mysql', 'mongodb'],
            ['api', 'rest', 'graphql', 'microservices']
        ]
        
        for group in related_groups:
            if cap1 in group and cap2 in group:
                return True
        
        return False
    
    def _has_general_ability(self, capabilities: List[str], requirement: str) -> bool:
        """Check if agent has general ability for requirement."""
        general_abilities = {
            'programming': ['python', 'javascript', 'java', 'go', 'rust'],
            'web_development': ['frontend', 'backend', 'fullstack'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud'],
            'database': ['sql', 'nosql', 'database']
        }
        
        for ability, related_caps in general_abilities.items():
            if requirement in ability and any(cap.lower() in related_caps for cap in capabilities):
                return True
        
        return False
    
    def _is_specialization_relevant(self, specialization: str, requirement: str) -> bool:
        """Check if specialization is relevant to requirement."""
        spec_lower = specialization.lower()
        req_lower = requirement.lower()
        
        # Direct match or containment
        if spec_lower in req_lower or req_lower in spec_lower:
            return True
        
        # Domain relevance
        domain_mappings = {
            'backend_developer': ['api', 'database', 'server', 'backend'],
            'frontend_developer': ['ui', 'ux', 'frontend', 'javascript'],
            'devops_engineer': ['deployment', 'infrastructure', 'docker', 'kubernetes'],
            'qa_engineer': ['testing', 'quality', 'automation', 'validation']
        }
        
        for domain, keywords in domain_mappings.items():
            if spec_lower == domain and any(keyword in req_lower for keyword in keywords):
                return True
        
        return False
    
    def _generate_recommendation(self, score: float) -> str:
        """Generate recommendation based on match score."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.8:
            return 'very_good'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.6:
            return 'fair'
        elif score >= 0.5:
            return 'marginal'
        else:
            return 'poor'
    
    def _generate_improvement_suggestions(
        self,
        capabilities: List[str],
        specializations: List[str],
        requirements: List[str],
        score: float
    ) -> List[str]:
        """Generate suggestions for improving agent-task match."""
        suggestions = []
        
        if score < 0.7:
            missing_capabilities = [
                req for req in requirements
                if not any(self._are_capabilities_related(req.lower(), cap.lower()) for cap in capabilities)
            ]
            
            if missing_capabilities:
                suggestions.append(f"Consider training in: {', '.join(missing_capabilities[:3])}")
        
        if not specializations:
            suggestions.append("Agent would benefit from defined specializations")
        
        if score < 0.5:
            suggestions.append("Consider alternative agent assignment or task modification")
        
        return suggestions


class AgentPerformanceAnalyzer:
    """
    Advanced agent performance analysis and optimization recommendations.
    
    Provides comprehensive performance metrics analysis with trend detection
    and optimization insights for continuous improvement.
    """
    
    def __init__(self):
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.baseline_metrics = {
            'response_time_ms': 150,
            'success_rate': 0.95,
            'task_completion_rate': 0.9,
            'efficiency_score': 0.85
        }
    
    def analyze_agent_performance(
        self,
        agent_id: str,
        performance_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive agent performance analysis with trends and recommendations.
        """
        try:
            current_metrics = self._extract_current_metrics(performance_data)
            
            # Add to history
            if agent_id not in self.performance_history:
                self.performance_history[agent_id] = deque(maxlen=100)
            
            self.performance_history[agent_id].append({
                'timestamp': datetime.utcnow(),
                'metrics': current_metrics
            })
            
            # Historical analysis
            if historical_data:
                trend_analysis = self._analyze_trends(historical_data)
            else:
                trend_analysis = self._analyze_trends(list(self.performance_history[agent_id]))
            
            # Performance scoring
            performance_score = self._calculate_performance_score(current_metrics)
            
            # Optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                current_metrics, trend_analysis, performance_score
            )
            
            # Anomaly detection
            anomalies = self._detect_anomalies(current_metrics, agent_id)
            
            return {
                'agent_id': agent_id,
                'current_metrics': current_metrics,
                'performance_score': performance_score,
                'trend_analysis': trend_analysis,
                'recommendations': recommendations,
                'anomalies': anomalies,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'health_status': self._determine_health_status(performance_score, anomalies)
            }
            
        except Exception as e:
            logger.error("Performance analysis failed", agent_id=agent_id, error=str(e))
            return {
                'agent_id': agent_id,
                'error': str(e),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _extract_current_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract standardized metrics from performance data."""
        return {
            'response_time_ms': performance_data.get('average_response_time', 0) * 1000,
            'success_rate': performance_data.get('success_rate', 0.0),
            'task_completion_rate': performance_data.get('task_completion_rate', 0.0),
            'efficiency_score': performance_data.get('efficiency_score', 0.0),
            'error_rate': performance_data.get('error_rate', 0.0),
            'uptime_percentage': performance_data.get('uptime_percentage', 100.0),
            'context_utilization': performance_data.get('context_window_usage', 0.0)
        }
    
    def _analyze_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        if len(historical_data) < 2:
            return {'trend': 'insufficient_data'}
        
        metrics_over_time = defaultdict(list)
        
        for data_point in historical_data:
            timestamp = data_point.get('timestamp')
            metrics = data_point.get('metrics', {})
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_over_time[metric].append(value)
        
        trends = {}
        for metric, values in metrics_over_time.items():
            if len(values) >= 2:
                # Simple linear trend calculation
                x = list(range(len(values)))
                y = values
                
                # Calculate slope
                n = len(x)
                slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
                
                # Determine trend direction
                if abs(slope) < 0.001:  # Threshold for stability
                    trend_direction = 'stable'
                elif slope > 0:
                    trend_direction = 'improving' if metric in ['success_rate', 'efficiency_score'] else 'degrading'
                else:
                    trend_direction = 'degrading' if metric in ['success_rate', 'efficiency_score'] else 'improving'
                
                trends[metric] = {
                    'direction': trend_direction,
                    'slope': slope,
                    'current_value': values[-1],
                    'change_percentage': ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                }
        
        return trends
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score (0-1)."""
        weights = {
            'success_rate': 0.3,
            'efficiency_score': 0.25,
            'response_time_ms': 0.2,  # Inverse scoring
            'task_completion_rate': 0.15,
            'uptime_percentage': 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'response_time_ms':
                    # Inverse scoring for response time (lower is better)
                    normalized = max(0, min(1, 1 - (metrics[metric] - 50) / 500))
                elif metric == 'uptime_percentage':
                    normalized = metrics[metric] / 100.0
                else:
                    normalized = metrics[metric]
                
                score += normalized * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    def _generate_optimization_recommendations(
        self,
        metrics: Dict[str, float],
        trends: Dict[str, Any],
        performance_score: float
    ) -> List[Dict[str, str]]:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # Response time optimization
        if metrics.get('response_time_ms', 0) > 200:
            recommendations.append({
                'category': 'performance',
                'priority': 'high',
                'recommendation': 'Optimize response time through caching or algorithm improvements',
                'expected_impact': 'Reduce response time by 20-40%'
            })
        
        # Success rate improvement
        if metrics.get('success_rate', 1.0) < 0.9:
            recommendations.append({
                'category': 'reliability',
                'priority': 'critical',
                'recommendation': 'Investigate and resolve failure patterns',
                'expected_impact': 'Improve success rate to >95%'
            })
        
        # Efficiency enhancement
        if metrics.get('efficiency_score', 0.0) < 0.8:
            recommendations.append({
                'category': 'efficiency',
                'priority': 'medium',
                'recommendation': 'Review task assignment and capability matching',
                'expected_impact': 'Increase efficiency by 10-20%'
            })
        
        # Trend-based recommendations
        for metric, trend_data in trends.items():
            if trend_data.get('direction') == 'degrading':
                recommendations.append({
                    'category': 'trend_alert',
                    'priority': 'medium',
                    'recommendation': f'Address degrading trend in {metric}',
                    'expected_impact': 'Prevent further performance decline'
                })
        
        return recommendations
    
    def _detect_anomalies(self, current_metrics: Dict[str, float], agent_id: str) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis."""
        anomalies = []
        
        # Compare against baseline
        for metric, value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                
                # Simple threshold-based anomaly detection
                if metric == 'response_time_ms':
                    if value > baseline * 2:  # More than 2x baseline
                        anomalies.append({
                            'metric': metric,
                            'type': 'performance_degradation',
                            'severity': 'high' if value > baseline * 3 else 'medium',
                            'current_value': value,
                            'baseline_value': baseline
                        })
                else:
                    if value < baseline * 0.7:  # Less than 70% of baseline
                        anomalies.append({
                            'metric': metric,
                            'type': 'underperformance',
                            'severity': 'high' if value < baseline * 0.5 else 'medium',
                            'current_value': value,
                            'baseline_value': baseline
                        })
        
        return anomalies
    
    def _determine_health_status(self, performance_score: float, anomalies: List[Dict[str, Any]]) -> str:
        """Determine overall agent health status."""
        critical_anomalies = len([a for a in anomalies if a.get('severity') == 'high'])
        
        if critical_anomalies > 0:
            return 'critical'
        elif performance_score < 0.6:
            return 'degraded'
        elif performance_score < 0.8:
            return 'warning'
        else:
            return 'healthy'


# ========================================
# Coordination Utilities
# ========================================

class CoordinationOptimizer:
    """
    Intelligent coordination optimization for multi-agent projects.
    
    Provides advanced algorithms for optimal agent assignment, workload
    balancing, and coordination mode selection.
    """
    
    def __init__(self):
        self.coordination_cache = {}
        self.optimization_history = defaultdict(list)
    
    def optimize_agent_assignment(
        self,
        project_requirements: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        coordination_mode: str = 'parallel'
    ) -> Dict[str, Any]:
        """
        Optimize agent assignment for project using advanced matching algorithms.
        """
        try:
            cache_key = self._generate_cache_key(project_requirements, available_agents, coordination_mode)
            
            # Check cache first
            if cache_key in self.coordination_cache:
                cached_result = self.coordination_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    return cached_result['result']
            
            # Perform optimization
            optimization_result = self._perform_assignment_optimization(
                project_requirements, available_agents, coordination_mode
            )
            
            # Cache result
            self.coordination_cache[cache_key] = {
                'result': optimization_result,
                'timestamp': datetime.utcnow(),
                'ttl_seconds': 300  # 5 minute cache
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error("Agent assignment optimization failed", error=str(e))
            return {
                'assignments': [],
                'optimization_score': 0.0,
                'error': str(e)
            }
    
    def _perform_assignment_optimization(
        self,
        requirements: Dict[str, Any],
        agents: List[Dict[str, Any]],
        coordination_mode: str
    ) -> Dict[str, Any]:
        """Perform the actual assignment optimization."""
        matcher = AgentCapabilityMatcher()
        task_requirements = requirements.get('required_capabilities', [])
        project_complexity = requirements.get('complexity', 0.5)
        
        # Score all agents for the project
        agent_scores = []
        for agent in agents:
            score_data = matcher.calculate_match_score(
                agent.get('capabilities', []),
                agent.get('specializations', []),
                agent.get('experience_level', 'intermediate'),
                agent.get('proficiency', 0.8),
                task_requirements,
                project_complexity
            )
            
            agent_scores.append({
                'agent_id': agent.get('agent_id'),
                'agent_data': agent,
                'score': score_data['overall_score'],
                'score_details': score_data
            })
        
        # Sort by score
        agent_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select optimal team based on coordination mode
        optimal_assignments = self._select_optimal_team(
            agent_scores, coordination_mode, requirements
        )
        
        # Calculate optimization metrics
        optimization_score = self._calculate_optimization_score(optimal_assignments)
        
        return {
            'assignments': optimal_assignments,
            'optimization_score': optimization_score,
            'coordination_mode': coordination_mode,
            'selection_rationale': self._generate_selection_rationale(optimal_assignments),
            'performance_predictions': self._predict_team_performance(optimal_assignments)
        }
    
    def _select_optimal_team(
        self,
        scored_agents: List[Dict[str, Any]],
        coordination_mode: str,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select optimal team based on coordination mode and requirements."""
        max_team_size = requirements.get('max_team_size', 5)
        min_team_size = requirements.get('min_team_size', 2)
        
        assignments = []
        
        if coordination_mode.lower() == 'parallel':
            # For parallel coordination, select diverse, high-scoring agents
            assignments = self._select_parallel_team(scored_agents, max_team_size, min_team_size)
        elif coordination_mode.lower() == 'sequential':
            # For sequential coordination, prioritize individual excellence
            assignments = self._select_sequential_team(scored_agents, max_team_size, min_team_size)
        else:
            # Hybrid or adaptive mode
            assignments = self._select_adaptive_team(scored_agents, max_team_size, min_team_size)
        
        return assignments
    
    def _select_parallel_team(
        self,
        scored_agents: List[Dict[str, Any]],
        max_size: int,
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Select team optimized for parallel coordination."""
        selected = []
        used_specializations = set()
        
        # First, select agents with diverse specializations
        for agent_data in scored_agents:
            if len(selected) >= max_size:
                break
            
            agent = agent_data['agent_data']
            specializations = set(agent.get('specializations', []))
            
            # Prefer agents with unique specializations
            if not specializations.intersection(used_specializations) or len(selected) < min_size:
                selected.append({
                    'agent_id': agent_data['agent_id'],
                    'assignment_role': self._determine_assignment_role(agent, selected),
                    'match_score': agent_data['score'],
                    'specializations': list(specializations),
                    'coordination_role': 'parallel_contributor'
                })
                used_specializations.update(specializations)
        
        return selected
    
    def _select_sequential_team(
        self,
        scored_agents: List[Dict[str, Any]],
        max_size: int,
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Select team optimized for sequential coordination."""
        selected = []
        
        # For sequential mode, prioritize individual capability over diversity
        for agent_data in scored_agents[:max_size]:
            selected.append({
                'agent_id': agent_data['agent_id'],
                'assignment_role': self._determine_assignment_role(agent_data['agent_data'], selected),
                'match_score': agent_data['score'],
                'specializations': agent_data['agent_data'].get('specializations', []),
                'coordination_role': 'sequential_contributor',
                'sequence_priority': len(selected) + 1
            })
        
        return selected
    
    def _select_adaptive_team(
        self,
        scored_agents: List[Dict[str, Any]],
        max_size: int,
        min_size: int
    ) -> List[Dict[str, Any]]:
        """Select team using adaptive coordination strategy."""
        # Hybrid approach: balance individual excellence and team diversity
        selected = []
        
        # Select top performers first
        top_performers = scored_agents[:max(2, min_size)]
        for agent_data in top_performers:
            selected.append({
                'agent_id': agent_data['agent_id'],
                'assignment_role': self._determine_assignment_role(agent_data['agent_data'], selected),
                'match_score': agent_data['score'],
                'specializations': agent_data['agent_data'].get('specializations', []),
                'coordination_role': 'adaptive_contributor'
            })
        
        # Fill remaining slots with diverse specialists
        if len(selected) < max_size:
            remaining_slots = max_size - len(selected)
            used_specs = set()
            for agent in selected:
                used_specs.update(agent['specializations'])
            
            for agent_data in scored_agents[len(selected):]:
                if len(selected) >= max_size:
                    break
                
                agent_specs = set(agent_data['agent_data'].get('specializations', []))
                if not agent_specs.intersection(used_specs):
                    selected.append({
                        'agent_id': agent_data['agent_id'],
                        'assignment_role': self._determine_assignment_role(agent_data['agent_data'], selected),
                        'match_score': agent_data['score'],
                        'specializations': list(agent_specs),
                        'coordination_role': 'adaptive_contributor'
                    })
                    used_specs.update(agent_specs)
        
        return selected
    
    def _determine_assignment_role(self, agent: Dict[str, Any], existing_team: List[Dict[str, Any]]) -> str:
        """Determine the specific role for an agent in the team."""
        specializations = agent.get('specializations', [])
        
        # Role mapping based on specializations
        role_mappings = {
            'backend_developer': 'Backend Development Lead' if not any('Backend' in m.get('assignment_role', '') for m in existing_team) else 'Backend Developer',
            'frontend_developer': 'Frontend Development Lead' if not any('Frontend' in m.get('assignment_role', '') for m in existing_team) else 'Frontend Developer',
            'devops_engineer': 'DevOps Lead' if not any('DevOps' in m.get('assignment_role', '') for m in existing_team) else 'DevOps Engineer',
            'qa_engineer': 'QA Lead' if not any('QA' in m.get('assignment_role', '') for m in existing_team) else 'QA Engineer'
        }
        
        for spec in specializations:
            if spec.lower() in role_mappings:
                return role_mappings[spec.lower()]
        
        return f"Specialist - {specializations[0] if specializations else 'General'}"
    
    def _calculate_optimization_score(self, assignments: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score for the team assignment."""
        if not assignments:
            return 0.0
        
        # Weighted factors for optimization score
        avg_match_score = sum(a['match_score'] for a in assignments) / len(assignments)
        
        # Diversity bonus (more specializations = better)
        all_specializations = set()
        for assignment in assignments:
            all_specializations.update(assignment.get('specializations', []))
        diversity_score = min(1.0, len(all_specializations) / 8.0)  # Normalize to max 8 specializations
        
        # Team size factor (closer to optimal size = better)
        optimal_size = 4  # Assumed optimal team size
        size_score = 1.0 - abs(len(assignments) - optimal_size) / optimal_size
        
        # Combined optimization score
        optimization_score = (
            avg_match_score * 0.5 +
            diversity_score * 0.3 +
            size_score * 0.2
        )
        
        return round(optimization_score, 3)
    
    def _generate_selection_rationale(self, assignments: List[Dict[str, Any]]) -> str:
        """Generate human-readable rationale for team selection."""
        if not assignments:
            return "No suitable agents found for assignment"
        
        avg_score = sum(a['match_score'] for a in assignments) / len(assignments)
        specializations = set()
        for assignment in assignments:
            specializations.update(assignment.get('specializations', []))
        
        return (
            f"Selected {len(assignments)} agents with average match score of {avg_score:.2f}. "
            f"Team provides {len(specializations)} distinct specializations for comprehensive coverage."
        )
    
    def _predict_team_performance(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict team performance metrics based on assignments."""
        if not assignments:
            return {}
        
        avg_score = sum(a['match_score'] for a in assignments) / len(assignments)
        
        # Performance predictions based on match scores and team composition
        predicted_success_rate = min(0.98, 0.7 + avg_score * 0.25)
        predicted_efficiency = min(0.95, avg_score * 0.9)
        predicted_completion_time_factor = max(0.8, 2.0 - avg_score * 1.2)  # Lower is faster
        
        return {
            'predicted_success_rate': predicted_success_rate,
            'predicted_efficiency': predicted_efficiency,
            'predicted_completion_time_factor': predicted_completion_time_factor,
            'confidence_level': avg_score,
            'risk_factors': self._identify_risk_factors(assignments)
        }
    
    def _identify_risk_factors(self, assignments: List[Dict[str, Any]]) -> List[str]:
        """Identify potential risk factors in team composition."""
        risks = []
        
        # Check for low match scores
        low_score_agents = [a for a in assignments if a['match_score'] < 0.6]
        if low_score_agents:
            risks.append(f"{len(low_score_agents)} agent(s) with suboptimal capability match")
        
        # Check for lack of diversity
        all_specs = set()
        for assignment in assignments:
            all_specs.update(assignment.get('specializations', []))
        if len(all_specs) < 3:
            risks.append("Limited specialization diversity may impact project coverage")
        
        # Check for team size
        if len(assignments) < 2:
            risks.append("Very small team size may limit redundancy and collaboration")
        elif len(assignments) > 6:
            risks.append("Large team size may introduce coordination overhead")
        
        return risks
    
    def _generate_cache_key(
        self,
        requirements: Dict[str, Any],
        agents: List[Dict[str, Any]],
        coordination_mode: str
    ) -> str:
        """Generate cache key for optimization results."""
        key_data = {
            'requirements': requirements,
            'agent_ids': [a.get('agent_id') for a in agents],
            'coordination_mode': coordination_mode
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached optimization result is still valid."""
        timestamp = cached_data.get('timestamp')
        ttl_seconds = cached_data.get('ttl_seconds', 300)
        
        if not timestamp:
            return False
        
        age_seconds = (datetime.utcnow() - timestamp).total_seconds()
        return age_seconds < ttl_seconds


# ========================================
# General Utility Functions
# ========================================

def validate_agent_data(agent_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate agent data structure and content.
    
    Returns tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Required fields
    required_fields = ['id', 'name', 'role', 'status']
    for field in required_fields:
        if field not in agent_data or not agent_data[field]:
            errors.append(f"Missing or empty required field: {field}")
    
    # Validate agent ID format
    agent_id = agent_data.get('id')
    if agent_id:
        try:
            uuid.UUID(str(agent_id))
        except ValueError:
            # Allow string IDs but validate length
            if len(str(agent_id)) < 3 or len(str(agent_id)) > 100:
                errors.append("Agent ID must be valid UUID or string of 3-100 characters")
    
    # Validate status
    status = agent_data.get('status')
    if status:
        valid_statuses = [s.value for s in AgentStatus]
        if str(status).upper() not in valid_statuses:
            errors.append(f"Invalid status. Valid statuses: {valid_statuses}")
    
    # Validate capabilities (if present)
    capabilities = agent_data.get('capabilities')
    if capabilities is not None:
        if not isinstance(capabilities, list):
            errors.append("Capabilities must be a list")
        elif len(capabilities) > 50:
            errors.append("Too many capabilities (maximum 50)")
    
    return len(errors) == 0, errors


def normalize_agent_response(agent_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize agent data to standard response format.
    
    Handles different input formats and ensures consistent output.
    """
    try:
        normalized = {
            'id': str(agent_data.get('id', '')),
            'name': str(agent_data.get('name', f"Agent-{str(agent_data.get('id', ''))[:8]}")),
            'role': str(agent_data.get('role', 'unknown')),
            'type': str(agent_data.get('type', 'claude_code')),
            'status': str(agent_data.get('status', 'unknown')),
            'capabilities': agent_data.get('capabilities', []),
            'config': agent_data.get('config', {}),
            'created_at': agent_data.get('created_at', datetime.utcnow().isoformat()),
            'updated_at': agent_data.get('updated_at'),
            'last_active': agent_data.get('last_active'),
            'current_task_id': agent_data.get('current_task_id'),
            'total_tasks_completed': int(agent_data.get('total_tasks_completed', 0)),
            'total_tasks_failed': int(agent_data.get('total_tasks_failed', 0)),
            'average_response_time': float(agent_data.get('average_response_time', 0.0))
        }
        
        # Ensure timestamps are in ISO format
        for timestamp_field in ['created_at', 'updated_at', 'last_active']:
            value = normalized.get(timestamp_field)
            if value and not isinstance(value, str):
                if hasattr(value, 'isoformat'):
                    normalized[timestamp_field] = value.isoformat()
                else:
                    normalized[timestamp_field] = str(value)
        
        return normalized
        
    except Exception as e:
        logger.warning("Agent response normalization failed", error=str(e))
        return agent_data


def calculate_system_health_score(
    agents: List[Dict[str, Any]],
    projects: List[Dict[str, Any]] = None,
    metrics: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculate overall system health score based on agents, projects, and metrics.
    
    Returns comprehensive health assessment with recommendations.
    """
    try:
        health_data = {
            'overall_score': 0.0,
            'component_scores': {},
            'health_status': 'unknown',
            'recommendations': [],
            'critical_issues': []
        }
        
        # Agent health component (40% of overall score)
        agent_health = _calculate_agent_health_component(agents)
        health_data['component_scores']['agents'] = agent_health
        
        # Project health component (30% of overall score)
        project_health = _calculate_project_health_component(projects or [])
        health_data['component_scores']['projects'] = project_health
        
        # System metrics component (30% of overall score)
        metrics_health = _calculate_metrics_health_component(metrics or {})
        health_data['component_scores']['metrics'] = metrics_health
        
        # Calculate overall score
        overall_score = (
            agent_health['score'] * 0.4 +
            project_health['score'] * 0.3 +
            metrics_health['score'] * 0.3
        )
        health_data['overall_score'] = round(overall_score, 3)
        
        # Determine health status
        if overall_score >= 0.9:
            health_data['health_status'] = 'excellent'
        elif overall_score >= 0.8:
            health_data['health_status'] = 'good'
        elif overall_score >= 0.7:
            health_data['health_status'] = 'fair'
        elif overall_score >= 0.5:
            health_data['health_status'] = 'degraded'
        else:
            health_data['health_status'] = 'critical'
        
        # Collect recommendations and critical issues
        for component in health_data['component_scores'].values():
            health_data['recommendations'].extend(component.get('recommendations', []))
            health_data['critical_issues'].extend(component.get('critical_issues', []))
        
        return health_data
        
    except Exception as e:
        logger.error("System health calculation failed", error=str(e))
        return {
            'overall_score': 0.0,
            'health_status': 'error',
            'error': str(e)
        }


def _calculate_agent_health_component(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate health score for agent component."""
    if not agents:
        return {
            'score': 0.0,
            'critical_issues': ['No agents available'],
            'recommendations': ['Deploy and activate agent system']
        }
    
    total_agents = len(agents)
    active_agents = len([a for a in agents if a.get('status', '').lower() in ['active', 'idle']])
    failed_agents = len([a for a in agents if a.get('status', '').lower() in ['failed', 'error']])
    
    # Calculate component scores
    availability_score = active_agents / total_agents if total_agents > 0 else 0
    reliability_score = max(0, (total_agents - failed_agents) / total_agents) if total_agents > 0 else 0
    
    # Performance analysis
    avg_response_time = sum(a.get('average_response_time', 0) for a in agents) / total_agents
    performance_score = max(0, min(1, 1 - (avg_response_time - 0.1) / 0.5))  # 100ms baseline, 600ms max
    
    overall_score = (availability_score * 0.4 + reliability_score * 0.4 + performance_score * 0.2)
    
    # Generate recommendations and critical issues
    recommendations = []
    critical_issues = []
    
    if availability_score < 0.8:
        critical_issues.append(f'Low agent availability: {active_agents}/{total_agents} agents active')
        recommendations.append('Investigate and resolve inactive agent issues')
    
    if failed_agents > 0:
        critical_issues.append(f'{failed_agents} agents in failed state')
        recommendations.append('Restart or replace failed agents')
    
    if avg_response_time > 0.3:
        recommendations.append('Optimize agent response times through performance tuning')
    
    return {
        'score': overall_score,
        'availability_score': availability_score,
        'reliability_score': reliability_score,
        'performance_score': performance_score,
        'active_agents': active_agents,
        'total_agents': total_agents,
        'failed_agents': failed_agents,
        'recommendations': recommendations,
        'critical_issues': critical_issues
    }


def _calculate_project_health_component(projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate health score for project component."""
    if not projects:
        return {
            'score': 0.8,  # Neutral score for no projects
            'recommendations': ['Consider creating projects to utilize agent capabilities']
        }
    
    total_projects = len(projects)
    completed_projects = len([p for p in projects if p.get('status', '').lower() == 'completed'])
    failed_projects = len([p for p in projects if p.get('status', '').lower() in ['failed', 'cancelled']])
    active_projects = len([p for p in projects if p.get('status', '').lower() in ['active', 'in_progress']])
    
    # Success rate
    success_rate = completed_projects / max(1, completed_projects + failed_projects)
    
    # Activity level (having active projects is good)
    activity_score = min(1.0, active_projects / max(1, total_projects))
    
    overall_score = success_rate * 0.7 + activity_score * 0.3
    
    recommendations = []
    critical_issues = []
    
    if success_rate < 0.8:
        critical_issues.append(f'Low project success rate: {success_rate:.1%}')
        recommendations.append('Review project failure patterns and improve coordination')
    
    if failed_projects > total_projects * 0.2:
        critical_issues.append(f'High project failure rate: {failed_projects}/{total_projects}')
    
    return {
        'score': overall_score,
        'success_rate': success_rate,
        'activity_score': activity_score,
        'total_projects': total_projects,
        'completed_projects': completed_projects,
        'failed_projects': failed_projects,
        'active_projects': active_projects,
        'recommendations': recommendations,
        'critical_issues': critical_issues
    }


def _calculate_metrics_health_component(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate health score for system metrics component."""
    # Default healthy metrics if none provided
    default_metrics = {
        'average_response_time': 0.15,
        'error_rate': 0.02,
        'uptime_percentage': 99.5,
        'memory_usage_percentage': 60,
        'cpu_usage_percentage': 40
    }
    
    effective_metrics = {**default_metrics, **metrics}
    
    # Score individual metrics
    response_time_score = max(0, min(1, 1 - (effective_metrics['average_response_time'] - 0.1) / 0.4))
    error_rate_score = max(0, min(1, 1 - effective_metrics['error_rate'] / 0.1))
    uptime_score = effective_metrics['uptime_percentage'] / 100
    memory_score = max(0, min(1, 1 - max(0, effective_metrics['memory_usage_percentage'] - 80) / 20))
    cpu_score = max(0, min(1, 1 - max(0, effective_metrics['cpu_usage_percentage'] - 80) / 20))
    
    overall_score = (
        response_time_score * 0.25 +
        error_rate_score * 0.25 +
        uptime_score * 0.25 +
        memory_score * 0.125 +
        cpu_score * 0.125
    )
    
    recommendations = []
    critical_issues = []
    
    if effective_metrics['average_response_time'] > 0.5:
        critical_issues.append('High system response times detected')
        recommendations.append('Investigate performance bottlenecks and optimize system resources')
    
    if effective_metrics['error_rate'] > 0.05:
        critical_issues.append('High system error rate detected')
        recommendations.append('Review error logs and address recurring issues')
    
    if effective_metrics['uptime_percentage'] < 99:
        critical_issues.append('System uptime below target')
        recommendations.append('Improve system reliability and fault tolerance')
    
    return {
        'score': overall_score,
        'response_time_score': response_time_score,
        'error_rate_score': error_rate_score,
        'uptime_score': uptime_score,
        'memory_score': memory_score,
        'cpu_score': cpu_score,
        'metrics': effective_metrics,
        'recommendations': recommendations,
        'critical_issues': critical_issues
    }
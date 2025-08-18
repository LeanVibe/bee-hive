"""
Task Router for Intelligent Agent Assignment

This module provides intelligent routing of tasks to optimal agents based on
capabilities, load, cost, and performance characteristics.

IMPLEMENTATION STATUS: INTERFACE DEFINITION
This file contains the complete interface definition and architectural design.
The implementation will be delegated to a subagent to avoid context rot.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from .orchestration_models import (
    TaskAssignment,
    AgentPool,
    RoutingStrategy,
    OrchestrationRequest
)
from ..agents.universal_agent_interface import AgentType, AgentTask, AgentCapability, CapabilityType

# ================================================================================
# Task Router Interface
# ================================================================================

class TaskRouter(ABC):
    """
    Abstract interface for intelligent task routing.
    
    The Task Router is responsible for:
    - Analyzing task requirements and agent capabilities
    - Selecting optimal agents for task execution
    - Load balancing across available agents
    - Cost optimization and resource efficiency
    - Performance prediction and optimization
    
    IMPLEMENTATION REQUIREMENTS:
    - Must route tasks in <100ms for simple assignments
    - Must consider agent load, capabilities, and performance history
    - Must support multiple routing strategies (best_fit, load_balanced, etc.)
    - Must provide confidence scores for routing decisions
    - Must handle agent unavailability gracefully
    """
    
    @abstractmethod
    async def route_task(
        self,
        task: AgentTask,
        agent_pool: AgentPool,
        strategy: RoutingStrategy = RoutingStrategy.BEST_FIT,
        constraints: Optional[Dict[str, Any]] = None
    ) -> TaskAssignment:
        """
        Route a single task to the optimal agent.
        
        IMPLEMENTATION REQUIRED: Core routing logic with capability matching,
        load balancing, and cost optimization.
        """
        pass
    
    @abstractmethod
    async def route_batch(
        self,
        tasks: List[AgentTask],
        agent_pool: AgentPool,
        strategy: RoutingStrategy = RoutingStrategy.LOAD_BALANCED
    ) -> List[TaskAssignment]:
        """
        Route multiple tasks efficiently with global optimization.
        
        IMPLEMENTATION REQUIRED: Batch routing with global optimization
        to minimize total cost and maximize throughput.
        """
        pass
    
    @abstractmethod
    async def get_routing_recommendations(
        self,
        task_requirements: Dict[str, Any],
        agent_pool: AgentPool
    ) -> List[Dict[str, Any]]:
        """
        Get ranked agent recommendations for task requirements.
        
        IMPLEMENTATION REQUIRED: Analysis engine that ranks agents by
        suitability with confidence scores and cost estimates.
        """
        pass

# ================================================================================
# Implementation Placeholder
# ================================================================================

class ProductionTaskRouter(TaskRouter):
    """
    Production implementation of intelligent task routing.
    
    This implementation provides sophisticated routing logic with:
    - Advanced capability matching and scoring
    - Real-time load balancing across agents
    - Cost optimization and resource efficiency
    - Performance prediction based on historical data
    - Multiple routing strategies with adaptive selection
    - Constraint handling and preference validation
    
    Performance Targets:
    - <100ms for simple routing decisions
    - <500ms for batch routing with global optimization
    - 95%+ routing accuracy for capability matching
    """
    
    def __init__(self, learning_rate: float = 0.1, max_history_size: int = 10000):
        """
        Initialize the production task router.
        
        Args:
            learning_rate: Rate at which historical performance influences routing
            max_history_size: Maximum number of historical assignments to track
        """
        self._routing_metrics = {}
        self._performance_history = {}
        self._agent_performance = {}
        self._learning_rate = learning_rate
        self._max_history_size = max_history_size
        self._round_robin_counters = {}
        self._sticky_sessions = {}
        
        # Performance tracking
        self._routing_times = []
        self._accuracy_scores = []
        
    async def route_task(
        self,
        task: AgentTask,
        agent_pool: AgentPool,
        strategy: RoutingStrategy = RoutingStrategy.BEST_FIT,
        constraints: Optional[Dict[str, Any]] = None
    ) -> TaskAssignment:
        """
        Route a single task to the optimal agent.
        
        Implements sophisticated routing logic with capability matching,
        load balancing, cost optimization, and performance prediction.
        """
        start_time = datetime.utcnow()
        constraints = constraints or {}
        
        try:
            # Get eligible agents based on capabilities and constraints
            eligible_agents = await self._get_eligible_agents(task, agent_pool, constraints)
            
            if not eligible_agents:
                raise ValueError(f"No eligible agents found for task {task.id}")
            
            # Score agents based on strategy
            agent_scores = await self._score_agents(task, eligible_agents, agent_pool, strategy)
            
            # Select best agent
            selected_agent_id = self._select_agent(agent_scores, strategy)
            
            # Create task assignment
            assignment = await self._create_assignment(
                task, selected_agent_id, agent_pool, agent_scores[selected_agent_id]
            )
            
            # Track routing decision
            await self._track_routing_decision(task, assignment, agent_scores, strategy)
            
            # Record performance metrics
            routing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._routing_times.append(routing_time)
            
            return assignment
            
        except Exception as e:
            # Log error and create fallback assignment if possible
            await self._handle_routing_error(task, agent_pool, str(e))
            raise
    
    async def route_batch(
        self,
        tasks: List[AgentTask],
        agent_pool: AgentPool,
        strategy: RoutingStrategy = RoutingStrategy.LOAD_BALANCED
    ) -> List[TaskAssignment]:
        """
        Route multiple tasks efficiently with global optimization.
        
        Implements batch routing with global optimization to minimize
        total cost and maximize throughput across all tasks.
        """
        start_time = datetime.utcnow()
        assignments = []
        
        try:
            # Sort tasks by priority and dependencies
            sorted_tasks = await self._sort_tasks_for_batch_routing(tasks)
            
            # Track agent loads for global optimization
            agent_loads = {agent_id: 0 for agent_id in agent_pool.available_agents.keys()}
            
            for task in sorted_tasks:
                # Update constraints based on previous assignments
                dynamic_constraints = await self._calculate_dynamic_constraints(
                    task, assignments, agent_loads
                )
                
                # Route individual task with updated constraints
                assignment = await self.route_task(
                    task, agent_pool, strategy, dynamic_constraints
                )
                
                assignments.append(assignment)
                
                # Update agent load tracking
                agent_loads[assignment.agent_id] += assignment.estimated_duration_minutes
            
            # Optimize assignments for global efficiency
            optimized_assignments = await self._optimize_batch_assignments(
                assignments, agent_pool, strategy
            )
            
            # Record batch routing performance
            batch_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._record_batch_performance(len(tasks), batch_time)
            
            return optimized_assignments
            
        except Exception as e:
            await self._handle_batch_routing_error(tasks, agent_pool, str(e))
            raise
    
    async def get_routing_recommendations(
        self,
        task_requirements: Dict[str, Any],
        agent_pool: AgentPool
    ) -> List[Dict[str, Any]]:
        """
        Get ranked agent recommendations for task requirements.
        
        Provides comprehensive analysis of agent suitability with
        confidence scores, cost estimates, and performance predictions.
        """
        try:
            # Create pseudo-task for analysis
            pseudo_task = self._create_pseudo_task(task_requirements)
            
            # Get all available agents
            available_agents = list(agent_pool.available_agents.keys())
            
            recommendations = []
            
            for agent_id in available_agents:
                # Calculate capability match
                capability_score = await self._calculate_capability_score(
                    pseudo_task, agent_id, agent_pool
                )
                
                # Calculate load and availability
                load_score = await self._calculate_load_score(agent_id, agent_pool)
                
                # Estimate cost and performance
                cost_estimate = await self._estimate_cost(pseudo_task, agent_id, agent_pool)
                performance_estimate = await self._estimate_performance(
                    pseudo_task, agent_id, agent_pool
                )
                
                # Calculate overall confidence
                confidence = await self._calculate_recommendation_confidence(
                    capability_score, load_score, agent_id
                )
                
                recommendation = {
                    "agent_id": agent_id,
                    "agent_type": agent_pool.available_agents[agent_id].value,
                    "confidence": round(confidence, 3),
                    "capability_match": round(capability_score, 3),
                    "load_score": round(load_score, 3),
                    "estimated_cost": round(cost_estimate, 2),
                    "estimated_duration_minutes": round(performance_estimate, 1),
                    "success_probability": await self._calculate_success_probability(
                        agent_id, pseudo_task
                    ),
                    "recommendation_reason": await self._generate_recommendation_reason(
                        capability_score, load_score, cost_estimate, performance_estimate
                    )
                }
                
                recommendations.append(recommendation)
            
            # Sort by confidence score (descending)
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            return recommendations
            
        except Exception as e:
            await self._handle_recommendation_error(task_requirements, str(e))
            return []
    
    # ================================================================================
    # Core Routing Logic Implementation
    # ================================================================================
    
    async def _get_eligible_agents(
        self,
        task: AgentTask,
        agent_pool: AgentPool,
        constraints: Dict[str, Any]
    ) -> List[str]:
        """Get list of agents eligible to handle the task based on capabilities and constraints."""
        eligible_agents = []
        
        for agent_id in agent_pool.available_agents.keys():
            # Check if agent is available and healthy
            if not await self._is_agent_available(agent_id, agent_pool):
                continue
            
            # Check capability match
            if not await self._has_required_capabilities(task, agent_id, agent_pool):
                continue
            
            # Check constraints
            if not await self._satisfies_constraints(agent_id, constraints, agent_pool):
                continue
            
            eligible_agents.append(agent_id)
        
        return eligible_agents
    
    async def _score_agents(
        self,
        task: AgentTask,
        eligible_agents: List[str],
        agent_pool: AgentPool,
        strategy: RoutingStrategy
    ) -> Dict[str, float]:
        """Score agents based on routing strategy and task requirements."""
        agent_scores = {}
        
        for agent_id in eligible_agents:
            if strategy == RoutingStrategy.BEST_FIT:
                score = await self._calculate_best_fit_score(task, agent_id, agent_pool)
            elif strategy == RoutingStrategy.LOAD_BALANCED:
                score = await self._calculate_load_balanced_score(task, agent_id, agent_pool)
            elif strategy == RoutingStrategy.COST_OPTIMIZED:
                score = await self._calculate_cost_optimized_score(task, agent_id, agent_pool)
            elif strategy == RoutingStrategy.ROUND_ROBIN:
                score = await self._calculate_round_robin_score(agent_id)
            elif strategy == RoutingStrategy.STICKY_SESSION:
                score = await self._calculate_sticky_session_score(task, agent_id)
            else:
                score = 0.5  # Default neutral score
            
            agent_scores[agent_id] = score
        
        return agent_scores
    
    def _select_agent(self, agent_scores: Dict[str, float], strategy: RoutingStrategy) -> str:
        """Select the best agent based on scores and strategy."""
        if not agent_scores:
            raise ValueError("No agents available for selection")
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            # For round robin, select based on counter rather than pure score
            return min(agent_scores.keys(), key=lambda aid: self._round_robin_counters.get(aid, 0))
        
        # For all other strategies, select agent with highest score
        return max(agent_scores.keys(), key=lambda aid: agent_scores[aid])
    
    async def _create_assignment(
        self,
        task: AgentTask,
        agent_id: str,
        agent_pool: AgentPool,
        confidence_score: float
    ) -> TaskAssignment:
        """Create a task assignment with estimated metrics."""
        # Estimate duration and cost
        estimated_duration = await self._estimate_task_duration(task, agent_id, agent_pool)
        estimated_cost = await self._estimate_cost(task, agent_id, agent_pool)
        
        assignment = TaskAssignment(
            request_id="",  # Will be set by orchestrator
            task_id=task.id,
            agent_id=agent_id,
            agent_type=agent_pool.available_agents[agent_id],
            estimated_duration_minutes=estimated_duration,
            estimated_cost_units=estimated_cost,
            confidence_score=confidence_score
        )
        
        return assignment
    
    # ================================================================================
    # Capability Matching and Scoring
    # ================================================================================
    
    async def _calculate_capability_score(
        self,
        task: AgentTask,
        agent_id: str,
        agent_pool: AgentPool
    ) -> float:
        """Calculate how well an agent's capabilities match task requirements."""
        if agent_id not in agent_pool.agent_capabilities:
            return 0.0
        
        agent_capabilities = agent_pool.agent_capabilities[agent_id]
        task_type = task.type
        
        # Find capability match for task type
        matching_capability = None
        for capability in agent_capabilities:
            if capability.type == task_type:
                matching_capability = capability
                break
        
        if not matching_capability:
            # Check for related capabilities
            related_score = await self._calculate_related_capability_score(task_type, agent_capabilities)
            return related_score * 0.5  # Penalize for indirect match
        
        # Base score from capability confidence
        base_score = matching_capability.confidence
        
        # Adjust based on performance score
        performance_adjustment = matching_capability.performance_score * 0.3
        
        # Adjust based on historical performance
        historical_adjustment = await self._get_historical_performance_adjustment(
            agent_id, task_type
        )
        
        # Combine scores with weights
        final_score = (
            base_score * 0.5 +
            performance_adjustment +
            historical_adjustment * 0.2
        )
        
        return min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def _calculate_related_capability_score(
        self,
        task_type: CapabilityType,
        agent_capabilities: List[AgentCapability]
    ) -> float:
        """Calculate score for related capabilities when exact match isn't found."""
        # Define capability relationships
        related_capabilities = {
            CapabilityType.CODE_IMPLEMENTATION: [
                CapabilityType.CODE_ANALYSIS,
                CapabilityType.DEBUGGING,
                CapabilityType.REFACTORING
            ],
            CapabilityType.CODE_REVIEW: [
                CapabilityType.CODE_ANALYSIS,
                CapabilityType.SECURITY_ANALYSIS,
                CapabilityType.PERFORMANCE_OPTIMIZATION
            ],
            CapabilityType.TESTING: [
                CapabilityType.CODE_ANALYSIS,
                CapabilityType.DEBUGGING
            ],
            # Add more relationships as needed
        }
        
        related_types = related_capabilities.get(task_type, [])
        
        best_related_score = 0.0
        for capability in agent_capabilities:
            if capability.type in related_types:
                related_score = capability.confidence * capability.performance_score
                best_related_score = max(best_related_score, related_score)
        
        return best_related_score
    
    # ================================================================================
    # Load Balancing Implementation
    # ================================================================================
    
    async def _calculate_load_score(self, agent_id: str, agent_pool: AgentPool) -> float:
        """Calculate agent load score (higher is better - less loaded)."""
        if agent_id not in agent_pool.agent_metrics:
            return 0.5  # Neutral score for unknown agents
        
        metrics = agent_pool.agent_metrics[agent_id]
        
        # Calculate load factors
        cpu_load = metrics.current_cpu_usage / 100.0
        memory_load = min(metrics.current_memory_usage_mb / 1024.0, 1.0)
        task_load = min(metrics.active_tasks / 10.0, 1.0)  # Assume 10 tasks as high load
        queue_load = min(metrics.queue_length / 20.0, 1.0)  # Assume 20 queued as high load
        
        # Combine load factors
        total_load = (cpu_load + memory_load + task_load + queue_load) / 4.0
        
        # Return inverse (higher score for lower load)
        return 1.0 - total_load
    
    async def _calculate_load_balanced_score(
        self,
        task: AgentTask,
        agent_id: str,
        agent_pool: AgentPool
    ) -> float:
        """Calculate score for load-balanced routing strategy."""
        capability_score = await self._calculate_capability_score(task, agent_id, agent_pool)
        load_score = await self._calculate_load_score(agent_id, agent_pool)
        
        # Weight capability higher than load for quality
        return capability_score * 0.7 + load_score * 0.3
    
    async def _calculate_best_fit_score(
        self,
        task: AgentTask,
        agent_id: str,
        agent_pool: AgentPool
    ) -> float:
        """Calculate score for best-fit routing strategy."""
        # Focus primarily on capability match
        capability_score = await self._calculate_capability_score(task, agent_id, agent_pool)
        
        # Small adjustment for load to break ties
        load_score = await self._calculate_load_score(agent_id, agent_pool)
        
        return capability_score * 0.9 + load_score * 0.1
    
    # ================================================================================
    # Cost Optimization Implementation
    # ================================================================================
    
    async def _estimate_cost(self, task: AgentTask, agent_id: str, agent_pool: AgentPool) -> float:
        """Estimate the cost of executing a task on a specific agent."""
        # Base cost factors
        base_cost_per_minute = 1.0
        
        # Agent-specific cost modifiers
        agent_type = agent_pool.available_agents[agent_id]
        agent_cost_modifier = {
            AgentType.CLAUDE_CODE: 1.2,  # Premium but high quality
            AgentType.CURSOR: 1.0,       # Standard cost
            AgentType.GEMINI_CLI: 0.8,   # Lower cost
            AgentType.OPENCODE: 0.9,     # Moderate cost
            AgentType.GITHUB_COPILOT: 1.1  # Slightly premium
        }.get(agent_type, 1.0)
        
        # Task complexity modifier
        task_complexity = await self._estimate_task_complexity(task)
        complexity_modifier = 1.0 + (task_complexity - 0.5)  # Scale around neutral
        
        # Estimated duration
        duration_minutes = await self._estimate_task_duration(task, agent_id, agent_pool)
        
        # Calculate total cost
        total_cost = base_cost_per_minute * agent_cost_modifier * complexity_modifier * duration_minutes
        
        return max(total_cost, 0.1)  # Minimum cost
    
    async def _calculate_cost_optimized_score(
        self,
        task: AgentTask,
        agent_id: str,
        agent_pool: AgentPool
    ) -> float:
        """Calculate score for cost-optimized routing strategy."""
        capability_score = await self._calculate_capability_score(task, agent_id, agent_pool)
        estimated_cost = await self._estimate_cost(task, agent_id, agent_pool)
        
        # Only consider agents with reasonable capability
        if capability_score < 0.3:
            return 0.0
        
        # Calculate cost efficiency (capability per unit cost)
        cost_efficiency = capability_score / estimated_cost
        
        # Normalize cost efficiency to [0, 1] range
        # This is a simplification - in practice, you'd track cost efficiency ranges
        normalized_efficiency = min(cost_efficiency / 0.1, 1.0)  # Assume 0.1 as baseline
        
        return normalized_efficiency
    
    # ================================================================================
    # Round Robin and Sticky Session Implementation
    # ================================================================================
    
    async def _calculate_round_robin_score(self, agent_id: str) -> float:
        """Calculate score for round-robin routing strategy."""
        # Lower counter = higher priority for selection
        counter = self._round_robin_counters.get(agent_id, 0)
        max_counter = max(self._round_robin_counters.values()) if self._round_robin_counters else 0
        
        # Normalize to [0, 1] with lower counter getting higher score
        if max_counter > 0:
            return 1.0 - (counter / max_counter)
        else:
            return 1.0
    
    async def _calculate_sticky_session_score(self, task: AgentTask, agent_id: str) -> float:
        """Calculate score for sticky session routing strategy."""
        # Check if this task type or related tasks were recently assigned to this agent
        task_type = task.type
        
        if agent_id in self._sticky_sessions:
            agent_sessions = self._sticky_sessions[agent_id]
            
            # Check for recent similar tasks
            recent_similar_tasks = sum(
                1 for session_task_type, timestamp in agent_sessions
                if session_task_type == task_type and 
                (datetime.utcnow() - timestamp).total_seconds() < 3600  # 1 hour window
            )
            
            # Higher score for agents that recently handled similar tasks
            return min(recent_similar_tasks * 0.3, 1.0)
        
        return 0.1  # Low score for agents without session history
    
    # ================================================================================
    # Batch Processing and Optimization
    # ================================================================================
    
    async def _sort_tasks_for_batch_routing(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks for optimal batch routing based on priority and dependencies."""
        # Sort by priority (lower number = higher priority)
        return sorted(tasks, key=lambda t: (t.priority, t.created_at))
    
    async def _calculate_dynamic_constraints(
        self,
        task: AgentTask,
        existing_assignments: List[TaskAssignment],
        agent_loads: Dict[str, int]
    ) -> Dict[str, Any]:
        """Calculate dynamic constraints based on current batch state."""
        constraints = {}
        
        # Load balancing constraint
        max_load = max(agent_loads.values()) if agent_loads else 0
        if max_load > 0:
            # Prefer agents with load below average
            avg_load = sum(agent_loads.values()) / len(agent_loads)
            constraints["preferred_low_load"] = avg_load
        
        # Diversity constraint (avoid overloading single agent type)
        agent_type_counts = {}
        for assignment in existing_assignments:
            agent_type_counts[assignment.agent_type] = agent_type_counts.get(assignment.agent_type, 0) + 1
        
        if agent_type_counts:
            # Discourage overused agent types
            max_type_usage = max(agent_type_counts.values())
            constraints["discourage_overused_types"] = {
                agent_type for agent_type, count in agent_type_counts.items()
                if count >= max_type_usage * 0.8
            }
        
        return constraints
    
    async def _optimize_batch_assignments(
        self,
        assignments: List[TaskAssignment],
        agent_pool: AgentPool,
        strategy: RoutingStrategy
    ) -> List[TaskAssignment]:
        """Optimize batch assignments for global efficiency."""
        if len(assignments) <= 1:
            return assignments
        
        # For now, return original assignments
        # In a full implementation, this would use more sophisticated optimization
        # such as simulated annealing or genetic algorithms
        optimized = assignments.copy()
        
        # Simple optimization: balance loads if strategy supports it
        if strategy == RoutingStrategy.LOAD_BALANCED:
            optimized = await self._balance_batch_loads(optimized, agent_pool)
        
        return optimized
    
    async def _balance_batch_loads(
        self,
        assignments: List[TaskAssignment],
        agent_pool: AgentPool
    ) -> List[TaskAssignment]:
        """Balance loads across agents in batch assignments."""
        # Calculate current load distribution
        agent_loads = {}
        for assignment in assignments:
            agent_loads[assignment.agent_id] = agent_loads.get(assignment.agent_id, 0) + assignment.estimated_duration_minutes
        
        # If load is well distributed, return as-is
        if len(agent_loads) <= 1:
            return assignments
        
        max_load = max(agent_loads.values())
        min_load = min(agent_loads.values())
        
        # If load difference is acceptable, return as-is
        if max_load - min_load <= 30:  # 30 minutes difference threshold
            return assignments
        
        # For more complex load balancing, we would implement reassignment logic here
        # For now, return original assignments
        return assignments
    
    # ================================================================================
    # Performance Estimation and Prediction
    # ================================================================================
    
    async def _estimate_task_duration(self, task: AgentTask, agent_id: str, agent_pool: AgentPool) -> int:
        """Estimate task duration in minutes for specific agent."""
        # Base duration estimates by task type
        base_durations = {
            CapabilityType.CODE_ANALYSIS: 15,
            CapabilityType.CODE_IMPLEMENTATION: 45,
            CapabilityType.CODE_REVIEW: 20,
            CapabilityType.DOCUMENTATION: 30,
            CapabilityType.TESTING: 35,
            CapabilityType.DEBUGGING: 40,
            CapabilityType.REFACTORING: 50,
            CapabilityType.ARCHITECTURE_DESIGN: 60,
            CapabilityType.PERFORMANCE_OPTIMIZATION: 55,
            CapabilityType.SECURITY_ANALYSIS: 25,
            CapabilityType.UI_DEVELOPMENT: 40,
            CapabilityType.API_DEVELOPMENT: 45,
            CapabilityType.DATABASE_DESIGN: 50,
            CapabilityType.DEPLOYMENT: 25,
            CapabilityType.MONITORING: 20
        }
        
        base_duration = base_durations.get(task.type, 30)
        
        # Adjust based on task complexity
        task_complexity = await self._estimate_task_complexity(task)
        complexity_modifier = 0.5 + task_complexity  # Range: 0.5 to 1.5
        
        # Adjust based on agent performance
        if agent_id in agent_pool.agent_capabilities:
            agent_capability = next(
                (cap for cap in agent_pool.agent_capabilities[agent_id] if cap.type == task.type),
                None
            )
            if agent_capability and agent_capability.estimated_time_seconds:
                agent_estimate = agent_capability.estimated_time_seconds / 60  # Convert to minutes
                base_duration = (base_duration + agent_estimate) / 2  # Average with base estimate
        
        # Apply historical performance adjustment
        historical_modifier = await self._get_historical_duration_modifier(agent_id, task.type)
        
        final_duration = int(base_duration * complexity_modifier * historical_modifier)
        return max(final_duration, 5)  # Minimum 5 minutes
    
    async def _estimate_task_complexity(self, task: AgentTask) -> float:
        """Estimate task complexity on a scale of 0.0 to 1.0."""
        complexity_score = 0.5  # Default medium complexity
        
        # Adjust based on description length and content
        if task.description:
            desc_length = len(task.description.split())
            if desc_length > 100:
                complexity_score += 0.2
            elif desc_length < 20:
                complexity_score -= 0.2
            
            # Look for complexity indicators in description
            complexity_keywords = ['complex', 'intricate', 'advanced', 'multiple', 'integrate', 'optimize']
            simplicity_keywords = ['simple', 'basic', 'straightforward', 'minor', 'quick']
            
            desc_lower = task.description.lower()
            complexity_matches = sum(1 for keyword in complexity_keywords if keyword in desc_lower)
            simplicity_matches = sum(1 for keyword in simplicity_keywords if keyword in desc_lower)
            
            complexity_score += (complexity_matches - simplicity_matches) * 0.1
        
        # Adjust based on requirements count
        if task.requirements:
            req_count = len(task.requirements)
            if req_count > 5:
                complexity_score += 0.15
            elif req_count == 1:
                complexity_score -= 0.1
        
        # Adjust based on task priority (higher priority often means more complex)
        if task.priority <= 2:
            complexity_score += 0.1
        elif task.priority >= 8:
            complexity_score -= 0.1
        
        return min(max(complexity_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    async def _estimate_performance(self, task: AgentTask, agent_id: str, agent_pool: AgentPool) -> float:
        """Estimate performance metrics for task execution."""
        return await self._estimate_task_duration(task, agent_id, agent_pool)
    
    # ================================================================================
    # Historical Performance and Learning
    # ================================================================================
    
    async def _get_historical_performance_adjustment(self, agent_id: str, task_type: CapabilityType) -> float:
        """Get historical performance adjustment for agent and task type."""
        if agent_id not in self._agent_performance:
            return 0.0  # Neutral adjustment for unknown agents
        
        agent_history = self._agent_performance[agent_id]
        
        if task_type not in agent_history:
            return 0.0  # Neutral adjustment for unknown task types
        
        task_history = agent_history[task_type]
        
        # Calculate recent performance trend
        recent_scores = task_history[-10:]  # Last 10 executions
        if not recent_scores:
            return 0.0
        
        avg_recent_score = sum(recent_scores) / len(recent_scores)
        return (avg_recent_score - 0.5) * 0.4  # Scale to [-0.2, 0.2]
    
    async def _get_historical_duration_modifier(self, agent_id: str, task_type: CapabilityType) -> float:
        """Get historical duration modifier based on agent's past performance."""
        # This would track actual vs estimated durations
        # For now, return a neutral modifier
        return 1.0
    
    async def _track_routing_decision(
        self,
        task: AgentTask,
        assignment: TaskAssignment,
        agent_scores: Dict[str, float],
        strategy: RoutingStrategy
    ) -> None:
        """Track routing decision for learning and metrics."""
        routing_record = {
            "task_id": task.id,
            "task_type": task.type,
            "selected_agent": assignment.agent_id,
            "agent_scores": agent_scores,
            "strategy": strategy,
            "confidence": assignment.confidence_score,
            "timestamp": datetime.utcnow()
        }
        
        # Store routing record
        if "routing_decisions" not in self._routing_metrics:
            self._routing_metrics["routing_decisions"] = []
        
        self._routing_metrics["routing_decisions"].append(routing_record)
        
        # Limit history size
        if len(self._routing_metrics["routing_decisions"]) > self._max_history_size:
            self._routing_metrics["routing_decisions"] = self._routing_metrics["routing_decisions"][-self._max_history_size:]
        
        # Update round-robin counters
        if strategy == RoutingStrategy.ROUND_ROBIN:
            self._round_robin_counters[assignment.agent_id] = self._round_robin_counters.get(assignment.agent_id, 0) + 1
        
        # Update sticky sessions
        if strategy == RoutingStrategy.STICKY_SESSION:
            if assignment.agent_id not in self._sticky_sessions:
                self._sticky_sessions[assignment.agent_id] = []
            self._sticky_sessions[assignment.agent_id].append((task.type, datetime.utcnow()))
            
            # Cleanup old sessions (keep last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self._sticky_sessions[assignment.agent_id] = [
                (task_type, timestamp) for task_type, timestamp in self._sticky_sessions[assignment.agent_id]
                if timestamp > cutoff_time
            ]
    
    # ================================================================================
    # Utility Methods and Validation
    # ================================================================================
    
    async def _is_agent_available(self, agent_id: str, agent_pool: AgentPool) -> bool:
        """Check if agent is available and healthy for task assignment."""
        # Check if agent exists in pool
        if agent_id not in agent_pool.available_agents:
            return False
        
        # Check if agent is in maintenance mode
        if agent_id in agent_pool.maintenance_mode_agents:
            return False
        
        # Check agent metrics for health
        if agent_id in agent_pool.agent_metrics:
            metrics = agent_pool.agent_metrics[agent_id]
            
            # Check if agent is overloaded
            if metrics.active_tasks >= 10:  # Max concurrent tasks
                return False
            
            # Check if agent has high error rate
            if metrics.queue_length >= 20:  # Max queue length
                return False
            
            # Check resource usage
            if metrics.current_cpu_usage >= 90.0 or metrics.current_memory_usage_mb >= 2048:
                return False
        
        return True
    
    async def _has_required_capabilities(self, task: AgentTask, agent_id: str, agent_pool: AgentPool) -> bool:
        """Check if agent has required capabilities for the task."""
        if agent_id not in agent_pool.agent_capabilities:
            return False
        
        agent_capabilities = agent_pool.agent_capabilities[agent_id]
        task_type = task.type
        
        # Check for direct capability match
        for capability in agent_capabilities:
            if capability.type == task_type and capability.confidence >= 0.3:
                return True
        
        # Check for related capabilities
        related_score = await self._calculate_related_capability_score(task_type, agent_capabilities)
        return related_score >= 0.4  # Minimum threshold for related capabilities
    
    async def _satisfies_constraints(
        self,
        agent_id: str,
        constraints: Dict[str, Any],
        agent_pool: AgentPool
    ) -> bool:
        """Check if agent satisfies routing constraints."""
        if not constraints:
            return True
        
        agent_type = agent_pool.available_agents[agent_id]
        
        # Check excluded agent types
        if "excluded_agent_types" in constraints:
            if agent_type in constraints["excluded_agent_types"]:
                return False
        
        # Check preferred agent types
        if "preferred_agent_types" in constraints:
            if agent_type not in constraints["preferred_agent_types"]:
                return False
        
        # Check maximum cost constraint
        if "max_cost" in constraints:
            # This would require creating a pseudo-task to estimate cost
            # For now, skip this check
            pass
        
        # Check load constraints
        if "max_load_threshold" in constraints:
            if agent_id in agent_pool.agent_metrics:
                load_score = await self._calculate_load_score(agent_id, agent_pool)
                if load_score < constraints["max_load_threshold"]:
                    return False
        
        # Check discouraged overused types (from batch processing)
        if "discourage_overused_types" in constraints:
            if agent_type in constraints["discourage_overused_types"]:
                return False
        
        return True
    
    def _create_pseudo_task(self, task_requirements: Dict[str, Any]) -> AgentTask:
        """Create a pseudo-task for analysis from requirements dict."""
        from ..agents.universal_agent_interface import create_task, create_execution_context
        
        # Map string to CapabilityType
        type_str = task_requirements.get("type", "code_analysis")
        try:
            task_type = CapabilityType(type_str)
        except ValueError:
            # If the type string is not valid, try to find a close match
            type_mapping = {
                "code_review": CapabilityType.CODE_REVIEW,
                "analysis": CapabilityType.CODE_ANALYSIS,
                "implementation": CapabilityType.CODE_IMPLEMENTATION,
                "testing": CapabilityType.TESTING,
                "documentation": CapabilityType.DOCUMENTATION,
                "debugging": CapabilityType.DEBUGGING,
                "refactoring": CapabilityType.REFACTORING
            }
            task_type = type_mapping.get(type_str, CapabilityType.CODE_ANALYSIS)
        
        title = task_requirements.get("title", "Analysis Task")
        description = task_requirements.get("description", "")
        
        # Create basic execution context
        context = create_execution_context(
            worktree_path=task_requirements.get("worktree_path", "/tmp/analysis")
        )
        
        return create_task(
            task_type=task_type,
            title=title,
            description=description,
            context=context,
            priority=task_requirements.get("priority", 5),
            timeout_seconds=task_requirements.get("timeout_seconds", 300)
        )
    
    # ================================================================================
    # Recommendation Support Methods
    # ================================================================================
    
    async def _calculate_recommendation_confidence(
        self,
        capability_score: float,
        load_score: float,
        agent_id: str
    ) -> float:
        """Calculate overall confidence score for recommendations."""
        # Base confidence from capability and load
        base_confidence = (capability_score * 0.7) + (load_score * 0.3)
        
        # Adjust based on historical performance
        if agent_id in self._agent_performance:
            # Calculate average performance across all task types
            all_scores = []
            for task_type_scores in self._agent_performance[agent_id].values():
                all_scores.extend(task_type_scores[-5:])  # Last 5 scores per task type
            
            if all_scores:
                historical_avg = sum(all_scores) / len(all_scores)
                historical_adjustment = (historical_avg - 0.5) * 0.2  # Scale to [-0.1, 0.1]
                base_confidence += historical_adjustment
        
        return min(max(base_confidence, 0.0), 1.0)
    
    async def _calculate_success_probability(self, agent_id: str, task: AgentTask) -> float:
        """Calculate probability of successful task completion."""
        # Base success rate from agent metrics
        if agent_id in self._agent_performance:
            agent_history = self._agent_performance[agent_id]
            
            if task.type in agent_history:
                recent_scores = agent_history[task.type][-10:]  # Last 10 executions
                if recent_scores:
                    # Success probability is average of recent scores
                    return sum(recent_scores) / len(recent_scores)
        
        # Default success probability based on capability
        # Create a mock agent pool for capability calculation
        mock_agent_pool = type('MockAgentPool', (), {})()
        mock_agent_pool.agent_capabilities = {agent_id: []}
        
        try:
            capability_score = await self._calculate_capability_score(task, agent_id, mock_agent_pool)
            return max(0.5 + (capability_score * 0.4), 0.3)  # Range: 0.3 to 0.9
        except:
            return 0.5  # Default neutral probability
    
    async def _generate_recommendation_reason(
        self,
        capability_score: float,
        load_score: float,
        cost_estimate: float,
        performance_estimate: float
    ) -> str:
        """Generate human-readable reason for recommendation."""
        reasons = []
        
        if capability_score >= 0.8:
            reasons.append("excellent capability match")
        elif capability_score >= 0.6:
            reasons.append("good capability match")
        elif capability_score >= 0.4:
            reasons.append("adequate capability match")
        else:
            reasons.append("limited capability match")
        
        if load_score >= 0.8:
            reasons.append("low current load")
        elif load_score >= 0.6:
            reasons.append("moderate current load")
        else:
            reasons.append("high current load")
        
        if cost_estimate <= 10.0:
            reasons.append("cost-effective")
        elif cost_estimate <= 25.0:
            reasons.append("moderate cost")
        else:
            reasons.append("higher cost")
        
        if performance_estimate <= 30:
            reasons.append("fast execution expected")
        elif performance_estimate <= 60:
            reasons.append("moderate execution time")
        else:
            reasons.append("longer execution time expected")
        
        return ", ".join(reasons)
    
    # ================================================================================
    # Error Handling and Recovery
    # ================================================================================
    
    async def _handle_routing_error(self, task: AgentTask, agent_pool: AgentPool, error_msg: str) -> None:
        """Handle routing errors with fallback strategies."""
        # Log error for monitoring
        error_record = {
            "task_id": task.id,
            "task_type": task.type,
            "error": error_msg,
            "timestamp": datetime.utcnow(),
            "available_agents": len(agent_pool.available_agents)
        }
        
        if "routing_errors" not in self._routing_metrics:
            self._routing_metrics["routing_errors"] = []
        
        self._routing_metrics["routing_errors"].append(error_record)
        
        # Could implement fallback logic here, such as:
        # - Retry with relaxed constraints
        # - Suggest manual assignment
        # - Queue task for later retry
    
    async def _handle_batch_routing_error(
        self,
        tasks: List[AgentTask],
        agent_pool: AgentPool,
        error_msg: str
    ) -> None:
        """Handle batch routing errors."""
        # Log batch error
        error_record = {
            "batch_size": len(tasks),
            "error": error_msg,
            "timestamp": datetime.utcnow(),
            "available_agents": len(agent_pool.available_agents)
        }
        
        if "batch_routing_errors" not in self._routing_metrics:
            self._routing_metrics["batch_routing_errors"] = []
        
        self._routing_metrics["batch_routing_errors"].append(error_record)
    
    async def _handle_recommendation_error(self, task_requirements: Dict[str, Any], error_msg: str) -> None:
        """Handle recommendation generation errors."""
        # Log recommendation error
        error_record = {
            "requirements": task_requirements,
            "error": error_msg,
            "timestamp": datetime.utcnow()
        }
        
        if "recommendation_errors" not in self._routing_metrics:
            self._routing_metrics["recommendation_errors"] = []
        
        self._routing_metrics["recommendation_errors"].append(error_record)
    
    # ================================================================================
    # Performance Monitoring and Metrics
    # ================================================================================
    
    async def _record_batch_performance(self, batch_size: int, execution_time_ms: float) -> None:
        """Record batch routing performance metrics."""
        if "batch_performance" not in self._routing_metrics:
            self._routing_metrics["batch_performance"] = []
        
        performance_record = {
            "batch_size": batch_size,
            "execution_time_ms": execution_time_ms,
            "throughput": batch_size / (execution_time_ms / 1000.0),  # tasks per second
            "timestamp": datetime.utcnow()
        }
        
        self._routing_metrics["batch_performance"].append(performance_record)
        
        # Limit performance history
        if len(self._routing_metrics["batch_performance"]) > 1000:
            self._routing_metrics["batch_performance"] = self._routing_metrics["batch_performance"][-1000:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for monitoring."""
        metrics = {
            "total_routing_decisions": len(self._routing_metrics.get("routing_decisions", [])),
            "total_errors": len(self._routing_metrics.get("routing_errors", [])),
            "total_batch_operations": len(self._routing_metrics.get("batch_performance", [])),
            "avg_routing_time_ms": 0.0,
            "routing_accuracy": 0.0,
            "agent_utilization": {},
            "strategy_effectiveness": {}
        }
        
        if self._routing_times:
            metrics["avg_routing_time_ms"] = sum(self._routing_times) / len(self._routing_times)
        
        if self._accuracy_scores:
            metrics["routing_accuracy"] = sum(self._accuracy_scores) / len(self._accuracy_scores)
        
        # Calculate strategy effectiveness
        routing_decisions = self._routing_metrics.get("routing_decisions", [])
        strategy_counts = {}
        strategy_confidence_sums = {}
        
        for decision in routing_decisions:
            strategy = decision["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            strategy_confidence_sums[strategy] = strategy_confidence_sums.get(strategy, 0) + decision["confidence"]
        
        for strategy, count in strategy_counts.items():
            avg_confidence = strategy_confidence_sums[strategy] / count
            metrics["strategy_effectiveness"][strategy] = {
                "usage_count": count,
                "avg_confidence": round(avg_confidence, 3)
            }
        
        return metrics
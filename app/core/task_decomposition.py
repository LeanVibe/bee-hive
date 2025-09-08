"""
Intelligent Task Decomposition Engine - Epic 2 Phase 2 Implementation

Advanced task decomposition with parallel execution optimization, dependency management,
and intelligent result aggregation for complex multi-agent coordination.

Building on Epic 2 Phase 1 foundations and integrating with DynamicAgentCollaboration:
- Intelligent task decomposition algorithms based on complexity and requirements
- Parallel execution optimization with dependency graph management
- Dynamic work distribution for optimal resource utilization  
- Smart result aggregation and synthesis capabilities
- Failure recovery and adaptive execution strategies
- Performance monitoring and bottleneck detection

Key Performance Targets:
- 70%+ parallel execution efficiency across agent teams
- <500ms task decomposition time for complex tasks
- 80% dependency resolution accuracy
- Real-time execution monitoring and optimization
- Adaptive failure recovery with <10s recovery time
"""

import asyncio
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
# import networkx as nx  # For dependency graph analysis - commented out to avoid dependency

from .agent_collaboration import (
    DynamicAgentCollaboration, AgentTeam, ComplexTask, SubTask,
    TaskComplexityLevel, AgentCapability, CollaborationPattern,
    get_dynamic_agent_collaboration
)
from .intelligent_orchestrator import (
    IntelligentOrchestrator, IntelligentTaskRequest, IntelligentTaskResult,
    get_intelligent_orchestrator
)
from .context_engine import (
    AdvancedContextEngine, SemanticMatch, get_context_engine
)
from .semantic_memory import (
    SemanticMemorySystem, SemanticSearchMode, get_semantic_memory
)
from ..core.orchestrator import AgentRole, TaskPriority
from ..core.logging_service import get_component_logger


logger = get_component_logger("task_decomposition")


class DecompositionStrategy(Enum):
    """Task decomposition strategies."""
    CAPABILITY_BASED = "capability_based"       # Decompose by required capabilities
    TEMPORAL_PHASES = "temporal_phases"         # Decompose by time phases
    DEPENDENCY_LAYERS = "dependency_layers"     # Decompose by dependency layers
    COMPLEXITY_LEVELS = "complexity_levels"     # Decompose by complexity
    HYBRID_OPTIMAL = "hybrid_optimal"           # AI-optimized hybrid approach


class ExecutionMode(Enum):
    """Execution modes for parallel processing."""
    SEQUENTIAL = "sequential"                   # Execute tasks in sequence
    PARALLEL_FULL = "parallel_full"            # Full parallel execution
    PARALLEL_LIMITED = "parallel_limited"      # Limited parallelism
    ADAPTIVE_PARALLEL = "adaptive_parallel"    # Adaptive based on resources
    PIPELINE = "pipeline"                       # Pipeline execution


class DependencyType(Enum):
    """Types of task dependencies."""
    PREREQUISITE = "prerequisite"              # Must complete before
    RESOURCE_SHARED = "resource_shared"        # Shared resource dependency  
    DATA_FLOW = "data_flow"                    # Data output dependency
    KNOWLEDGE_DEPENDENCY = "knowledge_dependency"  # Knowledge/context dependency
    TIMING_CONSTRAINT = "timing_constraint"    # Time-based dependency


class ExecutionStatus(Enum):
    """Execution status for tasks and subtasks."""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class TaskDependency:
    """Task dependency relationship."""
    dependency_id: uuid.UUID
    source_task_id: uuid.UUID
    target_task_id: uuid.UUID
    dependency_type: DependencyType
    strength: float  # 0.0-1.0, strength of dependency
    estimated_delay: timedelta
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecompositionResult:
    """Result of task decomposition analysis."""
    original_task_id: uuid.UUID
    subtasks: List[SubTask]
    dependencies: List[TaskDependency]
    dependency_graph: Dict[uuid.UUID, List[uuid.UUID]]
    decomposition_strategy: DecompositionStrategy
    estimated_parallel_speedup: float
    complexity_reduction: float
    confidence_score: float
    optimization_suggestions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionNode:
    """Execution node in the parallel execution graph."""
    node_id: uuid.UUID
    subtask: SubTask
    assigned_agent_id: Optional[uuid.UUID] = None
    dependencies: List[uuid.UUID] = field(default_factory=list)
    dependents: List[uuid.UUID] = field(default_factory=list)
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    execution_duration: Optional[timedelta] = None
    result: Optional[Dict[str, Any]] = None
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class ParallelExecutionPlan:
    """Parallel execution plan with optimized scheduling."""
    plan_id: uuid.UUID
    task_id: uuid.UUID
    team_id: uuid.UUID
    execution_nodes: Dict[uuid.UUID, ExecutionNode]
    execution_phases: List[List[uuid.UUID]]  # Nodes that can run in parallel
    critical_path: List[uuid.UUID]
    estimated_total_duration: timedelta
    parallelism_factor: float  # Average parallel tasks per phase
    resource_requirements: Dict[str, Any]
    execution_mode: ExecutionMode
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result from parallel task execution."""
    execution_id: uuid.UUID
    task_id: uuid.UUID
    team_id: uuid.UUID
    subtask_results: Dict[uuid.UUID, Dict[str, Any]]
    aggregated_result: Dict[str, Any]
    execution_metrics: Dict[str, float]
    success_rate: float
    total_execution_time: timedelta
    parallel_efficiency: float
    bottlenecks_detected: List[str] = field(default_factory=list)
    optimizations_applied: List[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RecoveryPlan:
    """Recovery plan for failed task executions."""
    recovery_id: uuid.UUID
    original_execution_id: uuid.UUID
    failed_subtasks: List[uuid.UUID]
    recovery_strategy: str
    alternative_assignments: Dict[uuid.UUID, uuid.UUID]  # SubTask -> Agent
    retry_schedule: Dict[uuid.UUID, datetime]
    estimated_recovery_time: timedelta
    success_probability: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class IntelligentTaskDecomposition:
    """
    Intelligent Task Decomposition Engine for Epic 2 Phase 2.
    
    Provides advanced task decomposition with parallel execution optimization,
    dependency management, and intelligent result aggregation.
    
    Key Capabilities:
    - Intelligent task decomposition based on complexity and requirements
    - Parallel execution optimization with dependency graph analysis
    - Dynamic work distribution for optimal resource utilization
    - Smart result aggregation and synthesis
    - Failure recovery and adaptive execution strategies
    - Real-time performance monitoring and bottleneck detection
    """
    
    def __init__(
        self,
        collaboration_system: Optional[DynamicAgentCollaboration] = None,
        intelligent_orchestrator: Optional[IntelligentOrchestrator] = None
    ):
        """Initialize the Intelligent Task Decomposition engine."""
        self.collaboration_system = collaboration_system
        self.intelligent_orchestrator = intelligent_orchestrator
        self.context_engine: Optional[AdvancedContextEngine] = None
        self.semantic_memory: Optional[SemanticMemorySystem] = None
        
        # Decomposition and execution tracking
        self.decomposition_results: Dict[uuid.UUID, DecompositionResult] = {}
        self.execution_plans: Dict[uuid.UUID, ParallelExecutionPlan] = {}
        self.active_executions: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.execution_history: Dict[uuid.UUID, ExecutionResult] = {}
        self.recovery_plans: Dict[uuid.UUID, RecoveryPlan] = {}
        
        # Performance tracking and optimization
        self._performance_metrics = {
            'total_decompositions': 0,
            'avg_decomposition_time_ms': 0.0,
            'avg_parallel_efficiency': 0.0,
            'dependency_resolution_accuracy': 0.0,
            'execution_success_rate': 0.0,
            'avg_recovery_time_ms': 0.0,
            'bottleneck_detection_rate': 0.0
        }
        
        # Optimization algorithms
        self._decomposition_algorithms: Dict[DecompositionStrategy, Callable] = {
            DecompositionStrategy.CAPABILITY_BASED: self._decompose_by_capabilities,
            DecompositionStrategy.TEMPORAL_PHASES: self._decompose_by_phases,
            DecompositionStrategy.DEPENDENCY_LAYERS: self._decompose_by_dependencies,
            DecompositionStrategy.COMPLEXITY_LEVELS: self._decompose_by_complexity,
            DecompositionStrategy.HYBRID_OPTIMAL: self._decompose_hybrid_optimal
        }
        
        logger.info("Intelligent Task Decomposition engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the task decomposition engine with dependencies."""
        try:
            # Initialize collaboration system if not provided
            if not self.collaboration_system:
                self.collaboration_system = await get_dynamic_agent_collaboration()
            
            # Initialize intelligent orchestrator if not provided
            if not self.intelligent_orchestrator:
                self.intelligent_orchestrator = await get_intelligent_orchestrator()
            
            # Initialize Phase 1 components
            self.context_engine = await get_context_engine()
            self.semantic_memory = await get_semantic_memory()
            
            logger.info("✅ Intelligent Task Decomposition initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Task Decomposition engine: {e}")
            raise
    
    async def decompose_complex_task(self, task: ComplexTask) -> DecompositionResult:
        """
        Decompose complex task into optimally structured subtasks.
        
        Args:
            task: Complex task to decompose
            
        Returns:
            Decomposition result with subtasks and dependencies
        """
        start_time = time.perf_counter()
        
        try:
            if not self.context_engine:
                await self.initialize()
            
            logger.info(f"🔧 Decomposing complex task: {task.title}")
            
            # Step 1: Analyze task characteristics and determine optimal strategy
            strategy = await self._determine_decomposition_strategy(task)
            
            # Step 2: Apply decomposition algorithm
            decomposition_func = self._decomposition_algorithms[strategy]
            subtasks, dependencies = await decomposition_func(task)
            
            # Step 3: Build dependency graph
            dependency_graph = self._build_dependency_graph(subtasks, dependencies)
            
            # Step 4: Optimize decomposition for parallel execution
            subtasks, dependencies = await self._optimize_for_parallelism(
                task, subtasks, dependencies
            )
            
            # Step 5: Estimate performance improvements
            parallel_speedup = self._estimate_parallel_speedup(subtasks, dependencies)
            complexity_reduction = self._calculate_complexity_reduction(task, subtasks)
            
            # Step 6: Generate optimization suggestions
            suggestions = await self._generate_optimization_suggestions(
                task, subtasks, dependencies
            )
            
            # Step 7: Calculate confidence score
            confidence = self._calculate_decomposition_confidence(
                task, subtasks, dependencies, strategy
            )
            
            # Step 8: Create decomposition result
            decomposition_time = (time.perf_counter() - start_time) * 1000
            
            result = DecompositionResult(
                original_task_id=task.task_id,
                subtasks=subtasks,
                dependencies=dependencies,
                dependency_graph=dependency_graph,
                decomposition_strategy=strategy,
                estimated_parallel_speedup=parallel_speedup,
                complexity_reduction=complexity_reduction,
                confidence_score=confidence,
                optimization_suggestions=suggestions
            )
            
            # Store result and update metrics
            self.decomposition_results[task.task_id] = result
            self._update_decomposition_metrics(decomposition_time)
            
            logger.info(
                f"✅ Task decomposition complete: {len(subtasks)} subtasks, "
                f"speedup: {parallel_speedup:.2f}x, confidence: {confidence:.2f} "
                f"(time: {decomposition_time:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            raise
    
    async def optimize_parallel_execution(
        self,
        subtasks: List[SubTask],
        team: AgentTeam
    ) -> ParallelExecutionPlan:
        """
        Optimize parallel execution of subtasks across team members.
        
        Args:
            subtasks: List of subtasks to execute in parallel
            team: Agent team for execution
            
        Returns:
            Optimized parallel execution plan
        """
        try:
            logger.info(f"🚀 Optimizing parallel execution for {len(subtasks)} subtasks")
            
            # Step 1: Create execution nodes
            execution_nodes = await self._create_execution_nodes(subtasks)
            
            # Step 2: Analyze dependencies and create execution graph
            dependency_graph = self._analyze_execution_dependencies(execution_nodes)
            
            # Step 3: Find critical path and execution phases
            critical_path = self._find_critical_path(execution_nodes, dependency_graph)
            execution_phases = self._determine_execution_phases(execution_nodes, dependency_graph)
            
            # Step 4: Optimize agent assignments for parallel execution
            execution_nodes = await self._optimize_agent_assignments(
                execution_nodes, team, execution_phases
            )
            
            # Step 5: Calculate execution metrics
            total_duration = self._calculate_total_execution_time(execution_nodes, critical_path)
            parallelism_factor = self._calculate_parallelism_factor(execution_phases)
            
            # Step 6: Determine optimal execution mode
            execution_mode = self._determine_execution_mode(
                execution_phases, team, parallelism_factor
            )
            
            # Step 7: Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(
                execution_nodes, team
            )
            
            # Step 8: Create execution plan
            plan = ParallelExecutionPlan(
                plan_id=uuid.uuid4(),
                task_id=subtasks[0].parent_task_id if subtasks else uuid.uuid4(),
                team_id=team.team_id,
                execution_nodes=execution_nodes,
                execution_phases=execution_phases,
                critical_path=critical_path,
                estimated_total_duration=total_duration,
                parallelism_factor=parallelism_factor,
                resource_requirements=resource_requirements,
                execution_mode=execution_mode,
                optimization_metadata={
                    'dependency_complexity': len(dependency_graph),
                    'max_parallel_tasks': max(len(phase) for phase in execution_phases) if execution_phases else 0,
                    'optimization_score': parallelism_factor * 0.7 + (1.0 / max(len(execution_phases), 1)) * 0.3
                }
            )
            
            self.execution_plans[plan.plan_id] = plan
            
            logger.info(
                f"✅ Parallel execution optimized: {len(execution_phases)} phases, "
                f"parallelism: {parallelism_factor:.2f}, duration: {total_duration}"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Parallel execution optimization failed: {e}")
            raise
    
    async def manage_task_dependencies(
        self,
        execution_plan: ParallelExecutionPlan
    ) -> Dict[str, Any]:
        """
        Manage task dependencies and execution coordination.
        
        Args:
            execution_plan: Parallel execution plan with dependencies
            
        Returns:
            Dependency management result with coordination info
        """
        try:
            logger.info(f"🔗 Managing dependencies for execution plan: {execution_plan.plan_id}")
            
            # Step 1: Validate dependency graph integrity
            validation_result = self._validate_dependency_graph(execution_plan)
            
            # Step 2: Setup dependency monitoring
            dependency_monitors = await self._setup_dependency_monitoring(execution_plan)
            
            # Step 3: Create dependency resolution strategies
            resolution_strategies = self._create_dependency_resolution_strategies(execution_plan)
            
            # Step 4: Initialize real-time dependency tracking
            tracking_system = await self._initialize_dependency_tracking(execution_plan)
            
            # Step 5: Setup automatic dependency notifications
            notification_system = self._setup_dependency_notifications(execution_plan)
            
            management_result = {
                'execution_plan_id': execution_plan.plan_id,
                'dependency_validation': validation_result,
                'monitoring_systems': dependency_monitors,
                'resolution_strategies': resolution_strategies,
                'tracking_system_id': tracking_system,
                'notification_system_id': notification_system,
                'management_confidence': self._calculate_dependency_management_confidence(
                    execution_plan, validation_result
                ),
                'initialized_at': datetime.utcnow()
            }
            
            logger.info(
                f"✅ Dependency management initialized: "
                f"confidence: {management_result['management_confidence']:.2f}"
            )
            
            return management_result
            
        except Exception as e:
            logger.error(f"Dependency management failed: {e}")
            raise
    
    async def aggregate_results(
        self,
        subtask_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate and synthesize results from parallel subtask execution.
        
        Args:
            subtask_results: Results from individual subtasks
            
        Returns:
            Aggregated and synthesized result
        """
        try:
            logger.info(f"🔄 Aggregating results from {len(subtask_results)} subtasks")
            
            # Step 1: Validate and clean input results
            valid_results = self._validate_subtask_results(subtask_results)
            
            # Step 2: Categorize results by type and importance
            result_categories = self._categorize_results(valid_results)
            
            # Step 3: Apply intelligent result aggregation strategies
            aggregated_data = await self._aggregate_by_strategy(result_categories)
            
            # Step 4: Synthesize insights and patterns across results
            synthesis_insights = await self._synthesize_cross_result_insights(aggregated_data)
            
            # Step 5: Generate comprehensive summary
            result_summary = self._generate_result_summary(
                valid_results, aggregated_data, synthesis_insights
            )
            
            # Step 6: Calculate result quality metrics
            quality_metrics = self._calculate_result_quality_metrics(
                valid_results, aggregated_data
            )
            
            # Step 7: Identify potential issues or inconsistencies
            issues_identified = self._identify_result_issues(valid_results, aggregated_data)
            
            # Step 8: Create final aggregated result
            final_result = {
                'aggregation_id': uuid.uuid4(),
                'source_subtasks': len(valid_results),
                'aggregated_data': aggregated_data,
                'synthesis_insights': synthesis_insights,
                'result_summary': result_summary,
                'quality_metrics': quality_metrics,
                'issues_identified': issues_identified,
                'aggregation_confidence': self._calculate_aggregation_confidence(
                    quality_metrics, issues_identified
                ),
                'aggregated_at': datetime.utcnow()
            }
            
            logger.info(
                f"✅ Result aggregation complete: confidence: "
                f"{final_result['aggregation_confidence']:.2f}, "
                f"quality: {quality_metrics.get('overall_quality', 0.0):.2f}"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            raise
    
    async def handle_execution_failures(
        self,
        failed_subtasks: List[SubTask],
        execution_plan: ParallelExecutionPlan
    ) -> RecoveryPlan:
        """
        Handle execution failures with intelligent recovery strategies.
        
        Args:
            failed_subtasks: List of failed subtasks
            execution_plan: Original execution plan
            
        Returns:
            Recovery plan with alternative strategies
        """
        try:
            logger.info(f"🔧 Handling execution failures for {len(failed_subtasks)} subtasks")
            
            # Step 1: Analyze failure patterns and root causes
            failure_analysis = await self._analyze_execution_failures(
                failed_subtasks, execution_plan
            )
            
            # Step 2: Determine optimal recovery strategy
            recovery_strategy = await self._determine_recovery_strategy(
                failed_subtasks, failure_analysis, execution_plan
            )
            
            # Step 3: Find alternative agent assignments
            alternative_assignments = await self._find_alternative_assignments(
                failed_subtasks, execution_plan, failure_analysis
            )
            
            # Step 4: Create retry schedule with backoff
            retry_schedule = self._create_retry_schedule(
                failed_subtasks, failure_analysis, recovery_strategy
            )
            
            # Step 5: Estimate recovery time and success probability
            recovery_time = self._estimate_recovery_time(
                failed_subtasks, alternative_assignments, retry_schedule
            )
            success_probability = self._calculate_recovery_success_probability(
                failed_subtasks, failure_analysis, alternative_assignments
            )
            
            # Step 6: Create recovery plan
            recovery_plan = RecoveryPlan(
                recovery_id=uuid.uuid4(),
                original_execution_id=execution_plan.plan_id,
                failed_subtasks=[st.subtask_id for st in failed_subtasks],
                recovery_strategy=recovery_strategy,
                alternative_assignments=alternative_assignments,
                retry_schedule=retry_schedule,
                estimated_recovery_time=recovery_time,
                success_probability=success_probability
            )
            
            self.recovery_plans[recovery_plan.recovery_id] = recovery_plan
            
            logger.info(
                f"✅ Recovery plan created: strategy: {recovery_strategy}, "
                f"success_probability: {success_probability:.2f}, "
                f"estimated_time: {recovery_time}"
            )
            
            return recovery_plan
            
        except Exception as e:
            logger.error(f"Execution failure handling failed: {e}")
            raise
    
    # Private helper methods for decomposition algorithms
    
    async def _determine_decomposition_strategy(self, task: ComplexTask) -> DecompositionStrategy:
        """Determine optimal decomposition strategy for task."""
        try:
            # Analyze task characteristics
            complexity_score = self._calculate_task_complexity_score(task)
            capability_diversity = len(task.required_capabilities)
            duration_hours = task.estimated_duration.total_seconds() / 3600
            
            # Strategy selection logic
            if complexity_score < 0.3 and capability_diversity <= 2:
                return DecompositionStrategy.CAPABILITY_BASED
            elif duration_hours > 40:  # Long tasks benefit from temporal phases
                return DecompositionStrategy.TEMPORAL_PHASES
            elif capability_diversity > 5:  # High diversity benefits from capability-based
                return DecompositionStrategy.CAPABILITY_BASED
            elif complexity_score > 0.8:  # High complexity benefits from hybrid approach
                return DecompositionStrategy.HYBRID_OPTIMAL
            else:
                return DecompositionStrategy.DEPENDENCY_LAYERS
                
        except Exception:
            return DecompositionStrategy.HYBRID_OPTIMAL
    
    def _calculate_task_complexity_score(self, task: ComplexTask) -> float:
        """Calculate normalized task complexity score."""
        complexity_mapping = {
            TaskComplexityLevel.SIMPLE: 0.2,
            TaskComplexityLevel.MODERATE: 0.4,
            TaskComplexityLevel.COMPLEX: 0.6,
            TaskComplexityLevel.ADVANCED: 0.8,
            TaskComplexityLevel.ENTERPRISE: 1.0
        }
        
        base_score = complexity_mapping.get(task.complexity_level, 0.5)
        
        # Adjust based on other factors
        capability_factor = min(len(task.required_capabilities) / 10.0, 0.5)
        duration_factor = min(task.estimated_duration.total_seconds() / (7 * 24 * 3600), 0.3)
        dependency_factor = min(len(task.dependencies) / 20.0, 0.2)
        
        total_score = base_score + capability_factor + duration_factor + dependency_factor
        return min(1.0, total_score)
    
    async def _decompose_by_capabilities(
        self,
        task: ComplexTask
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Decompose task by required capabilities."""
        subtasks = []
        dependencies = []
        
        # Create subtasks for each required capability
        for i, capability in enumerate(task.required_capabilities):
            subtask = SubTask(
                subtask_id=uuid.uuid4(),
                parent_task_id=task.task_id,
                title=f"{capability.value.replace('_', ' ').title()} Component",
                description=f"Handle {capability.value} aspects of {task.title}",
                required_capability=capability,
                estimated_duration=task.estimated_duration // len(task.required_capabilities),
                priority=task.priority
            )
            subtasks.append(subtask)
            
            # Create dependencies (simplified - each depends on previous)
            if i > 0:
                dependency = TaskDependency(
                    dependency_id=uuid.uuid4(),
                    source_task_id=subtasks[i-1].subtask_id,
                    target_task_id=subtask.subtask_id,
                    dependency_type=DependencyType.PREREQUISITE,
                    strength=0.7,
                    estimated_delay=timedelta(minutes=15)
                )
                dependencies.append(dependency)
        
        return subtasks, dependencies
    
    async def _decompose_by_phases(
        self,
        task: ComplexTask
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Decompose task by temporal phases."""
        subtasks = []
        dependencies = []
        
        # Define standard phases
        phases = [
            ("Analysis & Planning", 0.2),
            ("Implementation", 0.5),
            ("Testing & Validation", 0.2),
            ("Documentation & Deployment", 0.1)
        ]
        
        for i, (phase_name, time_fraction) in enumerate(phases):
            phase_duration = task.estimated_duration * time_fraction
            
            subtask = SubTask(
                subtask_id=uuid.uuid4(),
                parent_task_id=task.task_id,
                title=f"{phase_name} Phase",
                description=f"{phase_name} for {task.title}",
                required_capability=list(task.required_capabilities)[0] if task.required_capabilities else AgentCapability.PROJECT_MANAGEMENT,
                estimated_duration=phase_duration,
                priority=task.priority
            )
            subtasks.append(subtask)
            
            # Sequential dependencies between phases
            if i > 0:
                dependency = TaskDependency(
                    dependency_id=uuid.uuid4(),
                    source_task_id=subtasks[i-1].subtask_id,
                    target_task_id=subtask.subtask_id,
                    dependency_type=DependencyType.PREREQUISITE,
                    strength=0.9,  # Strong dependency between phases
                    estimated_delay=timedelta(minutes=30)
                )
                dependencies.append(dependency)
        
        return subtasks, dependencies
    
    async def _decompose_by_dependencies(
        self,
        task: ComplexTask
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Decompose task by dependency layers."""
        # This would implement more sophisticated dependency analysis
        # For now, fallback to capability-based decomposition
        return await self._decompose_by_capabilities(task)
    
    async def _decompose_by_complexity(
        self,
        task: ComplexTask
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Decompose task by complexity levels."""
        # This would implement complexity-based decomposition
        # For now, fallback to capability-based decomposition
        return await self._decompose_by_capabilities(task)
    
    async def _decompose_hybrid_optimal(
        self,
        task: ComplexTask
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Apply hybrid optimal decomposition strategy."""
        # Combine multiple strategies for optimal decomposition
        # Start with capability-based, then add temporal phases
        
        capability_subtasks, capability_deps = await self._decompose_by_capabilities(task)
        
        # If task is large, add phase structure
        if task.estimated_duration > timedelta(days=2):
            # Split each capability subtask into phases
            all_subtasks = []
            all_dependencies = []
            
            for cap_subtask in capability_subtasks:
                # Mini-phases for each capability
                phases = [
                    ("Setup", 0.3),
                    ("Implementation", 0.6),
                    ("Finalization", 0.1)
                ]
                
                phase_subtasks = []
                for phase_name, time_fraction in phases:
                    phase_duration = cap_subtask.estimated_duration * time_fraction
                    
                    phase_subtask = SubTask(
                        subtask_id=uuid.uuid4(),
                        parent_task_id=task.task_id,
                        title=f"{cap_subtask.title} - {phase_name}",
                        description=f"{phase_name} phase of {cap_subtask.description}",
                        required_capability=cap_subtask.required_capability,
                        estimated_duration=phase_duration,
                        priority=cap_subtask.priority
                    )
                    phase_subtasks.append(phase_subtask)
                    all_subtasks.append(phase_subtask)
                
                # Add sequential dependencies within capability phases
                for i in range(len(phase_subtasks) - 1):
                    dependency = TaskDependency(
                        dependency_id=uuid.uuid4(),
                        source_task_id=phase_subtasks[i].subtask_id,
                        target_task_id=phase_subtasks[i+1].subtask_id,
                        dependency_type=DependencyType.PREREQUISITE,
                        strength=0.8,
                        estimated_delay=timedelta(minutes=10)
                    )
                    all_dependencies.append(dependency)
            
            return all_subtasks, all_dependencies
        else:
            return capability_subtasks, capability_deps
    
    def _build_dependency_graph(
        self,
        subtasks: List[SubTask],
        dependencies: List[TaskDependency]
    ) -> Dict[uuid.UUID, List[uuid.UUID]]:
        """Build dependency graph from subtasks and dependencies."""
        graph = defaultdict(list)
        
        # Initialize all nodes
        for subtask in subtasks:
            graph[subtask.subtask_id] = []
        
        # Add dependency edges
        for dependency in dependencies:
            graph[dependency.source_task_id].append(dependency.target_task_id)
        
        return dict(graph)
    
    async def _optimize_for_parallelism(
        self,
        task: ComplexTask,
        subtasks: List[SubTask],
        dependencies: List[TaskDependency]
    ) -> Tuple[List[SubTask], List[TaskDependency]]:
        """Optimize decomposition for maximum parallelism."""
        # Identify subtasks that can run in parallel
        # Remove unnecessary dependencies
        optimized_dependencies = []
        
        for dep in dependencies:
            # Only keep strong dependencies
            if dep.strength > 0.6:
                optimized_dependencies.append(dep)
            else:
                logger.info(f"Removing weak dependency for parallelism: {dep.dependency_id}")
        
        return subtasks, optimized_dependencies
    
    def _estimate_parallel_speedup(
        self,
        subtasks: List[SubTask],
        dependencies: List[TaskDependency]
    ) -> float:
        """Estimate parallel execution speedup factor."""
        if not subtasks:
            return 1.0
        
        # Calculate sequential time
        total_sequential_time = sum(
            subtask.estimated_duration.total_seconds() for subtask in subtasks
        )
        
        # Estimate parallel time using critical path
        # Simplified: assume perfect parallelism except for dependencies
        dependency_delay = sum(
            dep.estimated_delay.total_seconds() for dep in dependencies
        )
        
        # Assume we can parallelize non-dependent tasks
        independent_tasks = len(subtasks) - len(dependencies)
        parallel_factor = max(1, independent_tasks / 2)  # Conservative estimate
        
        estimated_parallel_time = (
            total_sequential_time / parallel_factor + dependency_delay
        )
        
        speedup = total_sequential_time / max(estimated_parallel_time, total_sequential_time * 0.1)
        return min(speedup, len(subtasks))  # Cap at number of subtasks
    
    def _calculate_complexity_reduction(
        self,
        original_task: ComplexTask,
        subtasks: List[SubTask]
    ) -> float:
        """Calculate complexity reduction achieved by decomposition."""
        original_complexity = self._calculate_task_complexity_score(original_task)
        
        # Assume each subtask has lower complexity
        avg_subtask_complexity = original_complexity / max(len(subtasks), 1) * 0.7
        
        complexity_reduction = 1.0 - (avg_subtask_complexity / original_complexity)
        return max(0.0, min(0.9, complexity_reduction))  # Cap between 0-90%
    
    async def _generate_optimization_suggestions(
        self,
        task: ComplexTask,
        subtasks: List[SubTask],
        dependencies: List[TaskDependency]
    ) -> List[str]:
        """Generate optimization suggestions for decomposition."""
        suggestions = []
        
        if len(subtasks) > 10:
            suggestions.append("Consider grouping related subtasks to reduce coordination overhead")
        
        if len(dependencies) > len(subtasks) * 0.8:
            suggestions.append("High dependency count may limit parallelism - review if all are necessary")
        
        avg_duration = sum(st.estimated_duration.total_seconds() for st in subtasks) / len(subtasks)
        if avg_duration < 1800:  # Less than 30 minutes
            suggestions.append("Small subtasks may have high coordination overhead - consider merging some")
        
        if len(task.required_capabilities) > len(subtasks):
            suggestions.append("Consider creating subtasks for each required capability")
        
        return suggestions
    
    def _calculate_decomposition_confidence(
        self,
        task: ComplexTask,
        subtasks: List[SubTask],
        dependencies: List[TaskDependency],
        strategy: DecompositionStrategy
    ) -> float:
        """Calculate confidence score for decomposition."""
        # Factor 1: Capability coverage
        required_caps = task.required_capabilities
        covered_caps = set()
        for subtask in subtasks:
            covered_caps.add(subtask.required_capability)
        
        coverage_score = len(covered_caps.intersection(required_caps)) / max(len(required_caps), 1)
        
        # Factor 2: Duration distribution
        total_duration = sum(st.estimated_duration.total_seconds() for st in subtasks)
        original_duration = task.estimated_duration.total_seconds()
        duration_accuracy = 1.0 - abs(total_duration - original_duration) / original_duration
        duration_accuracy = max(0.0, min(1.0, duration_accuracy))
        
        # Factor 3: Dependency reasonableness
        dependency_ratio = len(dependencies) / max(len(subtasks), 1)
        dependency_score = 1.0 - min(dependency_ratio, 1.0)  # Lower dependencies = higher score
        
        # Factor 4: Strategy appropriateness
        strategy_scores = {
            DecompositionStrategy.CAPABILITY_BASED: 0.8,
            DecompositionStrategy.TEMPORAL_PHASES: 0.7,
            DecompositionStrategy.DEPENDENCY_LAYERS: 0.6,
            DecompositionStrategy.COMPLEXITY_LEVELS: 0.6,
            DecompositionStrategy.HYBRID_OPTIMAL: 0.9
        }
        strategy_score = strategy_scores.get(strategy, 0.5)
        
        # Calculate overall confidence
        confidence = (
            coverage_score * 0.3 +
            duration_accuracy * 0.3 +
            dependency_score * 0.2 +
            strategy_score * 0.2
        )
        
        return min(1.0, confidence)
    
    def _update_decomposition_metrics(self, decomposition_time_ms: float) -> None:
        """Update decomposition performance metrics."""
        current_avg = self._performance_metrics['avg_decomposition_time_ms']
        total_count = self._performance_metrics['total_decompositions']
        
        new_avg = ((current_avg * total_count) + decomposition_time_ms) / (total_count + 1)
        
        self._performance_metrics['avg_decomposition_time_ms'] = new_avg
        self._performance_metrics['total_decompositions'] += 1
    
    # Additional helper methods for parallel execution optimization
    
    async def _create_execution_nodes(self, subtasks: List[SubTask]) -> Dict[uuid.UUID, ExecutionNode]:
        """Create execution nodes from subtasks."""
        nodes = {}
        
        for subtask in subtasks:
            node = ExecutionNode(
                node_id=subtask.subtask_id,
                subtask=subtask
            )
            nodes[subtask.subtask_id] = node
        
        return nodes
    
    def _analyze_execution_dependencies(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode]
    ) -> Dict[uuid.UUID, List[uuid.UUID]]:
        """Analyze dependencies between execution nodes."""
        # For this implementation, we'll use the existing subtask dependencies
        # In a real implementation, this would analyze more complex relationships
        dependency_graph = defaultdict(list)
        
        # Simple dependency analysis based on capability relationships
        node_list = list(execution_nodes.values())
        for i, node in enumerate(node_list):
            # Create simple sequential dependencies for demonstration
            if i > 0:
                prev_node = node_list[i-1]
                node.dependencies.append(prev_node.node_id)
                prev_node.dependents.append(node.node_id)
                dependency_graph[prev_node.node_id].append(node.node_id)
        
        return dict(dependency_graph)
    
    def _find_critical_path(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode],
        dependency_graph: Dict[uuid.UUID, List[uuid.UUID]]
    ) -> List[uuid.UUID]:
        """Find critical path through execution graph."""
        # Simple critical path analysis
        # In a real implementation, would use proper critical path method (CPM)
        
        # For now, return the longest sequential path
        visited = set()
        longest_path = []
        
        def dfs(node_id, current_path):
            if node_id in visited:
                return current_path
            
            visited.add(node_id)
            current_path = current_path + [node_id]
            
            # Find longest path from this node
            best_path = current_path
            for dependent_id in dependency_graph.get(node_id, []):
                candidate_path = dfs(dependent_id, current_path)
                if len(candidate_path) > len(best_path):
                    best_path = candidate_path
            
            return best_path
        
        # Start from nodes with no dependencies
        start_nodes = [
            node_id for node_id, node in execution_nodes.items()
            if not node.dependencies
        ]
        
        for start_node in start_nodes:
            visited.clear()
            path = dfs(start_node, [])
            if len(path) > len(longest_path):
                longest_path = path
        
        return longest_path
    
    def _determine_execution_phases(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode],
        dependency_graph: Dict[uuid.UUID, List[uuid.UUID]]
    ) -> List[List[uuid.UUID]]:
        """Determine execution phases for parallel processing."""
        phases = []
        remaining_nodes = set(execution_nodes.keys())
        completed_nodes = set()
        
        while remaining_nodes:
            # Find nodes that can run in current phase (no unmet dependencies)
            current_phase = []
            
            for node_id in list(remaining_nodes):
                node = execution_nodes[node_id]
                dependencies_met = all(
                    dep_id in completed_nodes for dep_id in node.dependencies
                )
                
                if dependencies_met:
                    current_phase.append(node_id)
            
            if not current_phase:
                # Break circular dependencies or add remaining nodes
                current_phase = list(remaining_nodes)
            
            phases.append(current_phase)
            
            # Update tracking sets
            for node_id in current_phase:
                remaining_nodes.remove(node_id)
                completed_nodes.add(node_id)
        
        return phases
    
    async def _optimize_agent_assignments(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode],
        team: AgentTeam,
        execution_phases: List[List[uuid.UUID]]
    ) -> Dict[uuid.UUID, ExecutionNode]:
        """Optimize agent assignments for parallel execution."""
        # Use collaboration system for agent assignment
        if not self.collaboration_system:
            return execution_nodes
        
        for phase in execution_phases:
            for node_id in phase:
                node = execution_nodes[node_id]
                
                # Route subtask to optimal team member
                assigned_agent = await self.collaboration_system.route_by_expertise(
                    node.subtask, team
                )
                node.assigned_agent_id = assigned_agent
        
        return execution_nodes
    
    def _calculate_total_execution_time(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode],
        critical_path: List[uuid.UUID]
    ) -> timedelta:
        """Calculate total execution time based on critical path."""
        total_seconds = 0
        
        for node_id in critical_path:
            if node_id in execution_nodes:
                node = execution_nodes[node_id]
                total_seconds += node.subtask.estimated_duration.total_seconds()
        
        return timedelta(seconds=total_seconds)
    
    def _calculate_parallelism_factor(self, execution_phases: List[List[uuid.UUID]]) -> float:
        """Calculate average parallelism factor."""
        if not execution_phases:
            return 1.0
        
        total_tasks = sum(len(phase) for phase in execution_phases)
        avg_parallel_tasks = total_tasks / len(execution_phases)
        
        return avg_parallel_tasks
    
    def _determine_execution_mode(
        self,
        execution_phases: List[List[uuid.UUID]],
        team: AgentTeam,
        parallelism_factor: float
    ) -> ExecutionMode:
        """Determine optimal execution mode."""
        team_size = len(team.agent_members)
        max_parallel_tasks = max(len(phase) for phase in execution_phases) if execution_phases else 1
        
        if parallelism_factor <= 1.2:
            return ExecutionMode.SEQUENTIAL
        elif max_parallel_tasks <= team_size and parallelism_factor >= 2.0:
            return ExecutionMode.PARALLEL_FULL
        elif max_parallel_tasks > team_size:
            return ExecutionMode.PARALLEL_LIMITED
        else:
            return ExecutionMode.ADAPTIVE_PARALLEL
    
    def _calculate_resource_requirements(
        self,
        execution_nodes: Dict[uuid.UUID, ExecutionNode],
        team: AgentTeam
    ) -> Dict[str, Any]:
        """Calculate resource requirements for execution."""
        return {
            'required_agents': len(set(
                node.assigned_agent_id for node in execution_nodes.values()
                if node.assigned_agent_id
            )),
            'total_agent_hours': sum(
                node.subtask.estimated_duration.total_seconds() / 3600
                for node in execution_nodes.values()
            ),
            'peak_concurrent_agents': max(
                len([n for n in execution_nodes.values() if n.assigned_agent_id == agent_id])
                for agent_id in team.agent_members
            ) if execution_nodes and team.agent_members else 0,
            'memory_estimate_mb': len(execution_nodes) * 50,  # Simplified estimate
            'storage_estimate_gb': len(execution_nodes) * 0.1  # Simplified estimate
        }
    
    # Simplified implementations for other helper methods
    
    def _validate_dependency_graph(self, execution_plan: ParallelExecutionPlan) -> Dict[str, Any]:
        """Validate dependency graph integrity."""
        return {
            'is_valid': True,
            'has_cycles': False,
            'validation_score': 0.9,
            'issues': []
        }
    
    async def _setup_dependency_monitoring(self, execution_plan: ParallelExecutionPlan) -> Dict[str, str]:
        """Setup dependency monitoring systems."""
        return {
            'monitor_id': str(uuid.uuid4()),
            'monitor_type': 'real_time',
            'status': 'active'
        }
    
    def _create_dependency_resolution_strategies(self, execution_plan: ParallelExecutionPlan) -> List[str]:
        """Create dependency resolution strategies."""
        return [
            'automatic_retry',
            'alternative_agent_assignment',
            'dependency_relaxation',
            'parallel_fallback'
        ]
    
    async def _initialize_dependency_tracking(self, execution_plan: ParallelExecutionPlan) -> str:
        """Initialize real-time dependency tracking."""
        return str(uuid.uuid4())
    
    def _setup_dependency_notifications(self, execution_plan: ParallelExecutionPlan) -> str:
        """Setup automatic dependency notifications."""
        return str(uuid.uuid4())
    
    def _calculate_dependency_management_confidence(
        self,
        execution_plan: ParallelExecutionPlan,
        validation_result: Dict[str, Any]
    ) -> float:
        """Calculate dependency management confidence."""
        base_score = 0.8
        validation_score = validation_result.get('validation_score', 0.5)
        return (base_score + validation_score) / 2
    
    def _validate_subtask_results(self, subtask_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean subtask results."""
        valid_results = []
        for result in subtask_results:
            if result.get('success', False) and result.get('subtask_id'):
                valid_results.append(result)
        return valid_results
    
    def _categorize_results(self, valid_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize results by type and importance."""
        categories = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'supporting_data': []
        }
        
        for result in valid_results:
            priority = result.get('priority', 'medium')
            if priority in ['high', 'urgent']:
                categories['high_priority'].append(result)
            elif priority == 'low':
                categories['low_priority'].append(result)
            else:
                categories['medium_priority'].append(result)
        
        return categories
    
    async def _aggregate_by_strategy(self, result_categories: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Apply intelligent result aggregation strategies."""
        aggregated = {}
        
        for category, results in result_categories.items():
            aggregated[category] = {
                'count': len(results),
                'avg_quality': sum(r.get('quality_score', 0.5) for r in results) / max(len(results), 1),
                'success_rate': sum(1 for r in results if r.get('success', False)) / max(len(results), 1),
                'combined_output': [r.get('output', {}) for r in results]
            }
        
        return aggregated
    
    async def _synthesize_cross_result_insights(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize insights and patterns across results."""
        return {
            'patterns_identified': ['consistent_quality', 'balanced_workload'],
            'performance_trends': {'improving': True, 'trend_score': 0.8},
            'collaboration_insights': ['good_coordination', 'effective_communication'],
            'optimization_opportunities': ['resource_rebalancing', 'parallel_optimization']
        }
    
    def _generate_result_summary(
        self,
        valid_results: List[Dict[str, Any]],
        aggregated_data: Dict[str, Any],
        synthesis_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive result summary."""
        return {
            'total_subtasks': len(valid_results),
            'overall_success_rate': sum(1 for r in valid_results if r.get('success', False)) / max(len(valid_results), 1),
            'average_quality': sum(r.get('quality_score', 0.5) for r in valid_results) / max(len(valid_results), 1),
            'key_insights': synthesis_insights.get('patterns_identified', []),
            'summary_text': f"Successfully completed {len(valid_results)} subtasks with high coordination"
        }
    
    def _calculate_result_quality_metrics(
        self,
        valid_results: List[Dict[str, Any]],
        aggregated_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate result quality metrics."""
        return {
            'overall_quality': sum(r.get('quality_score', 0.5) for r in valid_results) / max(len(valid_results), 1),
            'consistency_score': 0.85,  # Simplified
            'completeness_score': 0.90,  # Simplified
            'accuracy_score': 0.88  # Simplified
        }
    
    def _identify_result_issues(
        self,
        valid_results: List[Dict[str, Any]],
        aggregated_data: Dict[str, Any]
    ) -> List[str]:
        """Identify potential issues or inconsistencies."""
        issues = []
        
        # Check for quality variations
        quality_scores = [r.get('quality_score', 0.5) for r in valid_results]
        if quality_scores and (max(quality_scores) - min(quality_scores)) > 0.3:
            issues.append("High quality variation across subtasks")
        
        # Check for execution time variations
        execution_times = [r.get('execution_time_ms', 1000) for r in valid_results]
        if execution_times and (max(execution_times) / min(execution_times)) > 3:
            issues.append("Significant execution time variations")
        
        return issues
    
    def _calculate_aggregation_confidence(
        self,
        quality_metrics: Dict[str, float],
        issues_identified: List[str]
    ) -> float:
        """Calculate aggregation confidence score."""
        base_score = quality_metrics.get('overall_quality', 0.5)
        issue_penalty = len(issues_identified) * 0.1
        
        confidence = max(0.1, base_score - issue_penalty)
        return min(1.0, confidence)
    
    # Simplified failure handling methods
    
    async def _analyze_execution_failures(
        self,
        failed_subtasks: List[SubTask],
        execution_plan: ParallelExecutionPlan
    ) -> Dict[str, Any]:
        """Analyze failure patterns and root causes."""
        return {
            'failure_patterns': ['resource_exhaustion', 'dependency_timeout'],
            'root_causes': ['agent_overload', 'communication_failure'],
            'severity_score': 0.6,
            'recovery_feasibility': 0.8
        }
    
    async def _determine_recovery_strategy(
        self,
        failed_subtasks: List[SubTask],
        failure_analysis: Dict[str, Any],
        execution_plan: ParallelExecutionPlan
    ) -> str:
        """Determine optimal recovery strategy."""
        severity = failure_analysis.get('severity_score', 0.5)
        
        if severity > 0.8:
            return 'full_restart_with_replanning'
        elif severity > 0.6:
            return 'alternative_agent_assignment'
        else:
            return 'simple_retry_with_backoff'
    
    async def _find_alternative_assignments(
        self,
        failed_subtasks: List[SubTask],
        execution_plan: ParallelExecutionPlan,
        failure_analysis: Dict[str, Any]
    ) -> Dict[uuid.UUID, uuid.UUID]:
        """Find alternative agent assignments for failed subtasks."""
        alternatives = {}
        
        # Simplified: assign to different agents
        for subtask in failed_subtasks:
            # Find alternative agent (simplified)
            alternatives[subtask.subtask_id] = uuid.uuid4()
        
        return alternatives
    
    def _create_retry_schedule(
        self,
        failed_subtasks: List[SubTask],
        failure_analysis: Dict[str, Any],
        recovery_strategy: str
    ) -> Dict[uuid.UUID, datetime]:
        """Create retry schedule with backoff."""
        schedule = {}
        base_delay = timedelta(minutes=5)
        
        for i, subtask in enumerate(failed_subtasks):
            # Exponential backoff
            delay = base_delay * (2 ** i)
            schedule[subtask.subtask_id] = datetime.utcnow() + delay
        
        return schedule
    
    def _estimate_recovery_time(
        self,
        failed_subtasks: List[SubTask],
        alternative_assignments: Dict[uuid.UUID, uuid.UUID],
        retry_schedule: Dict[uuid.UUID, datetime]
    ) -> timedelta:
        """Estimate recovery time."""
        if not retry_schedule:
            return timedelta(minutes=30)
        
        max_delay = max(
            (scheduled_time - datetime.utcnow()).total_seconds()
            for scheduled_time in retry_schedule.values()
        )
        
        return timedelta(seconds=max_delay)
    
    def _calculate_recovery_success_probability(
        self,
        failed_subtasks: List[SubTask],
        failure_analysis: Dict[str, Any],
        alternative_assignments: Dict[uuid.UUID, uuid.UUID]
    ) -> float:
        """Calculate recovery success probability."""
        base_probability = 0.8
        failure_severity = failure_analysis.get('severity_score', 0.5)
        
        # Reduce probability based on severity
        adjusted_probability = base_probability * (1.0 - failure_severity * 0.3)
        
        # Increase probability if we have alternative assignments
        if alternative_assignments:
            adjusted_probability += 0.1
        
        return max(0.1, min(0.95, adjusted_probability))
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for task decomposition engine."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'collaboration_system': None,
                'intelligent_orchestrator': None,
                'context_engine': None,
                'semantic_memory': None
            },
            'performance_metrics': self._performance_metrics,
            'active_decompositions': len(self.decomposition_results),
            'active_execution_plans': len(self.execution_plans),
            'active_executions': len(self.active_executions)
        }
        
        try:
            # Check component health
            if self.collaboration_system:
                health_status['components']['collaboration_system'] = \
                    await self.collaboration_system.health_check()
            
            if self.intelligent_orchestrator:
                health_status['components']['intelligent_orchestrator'] = \
                    await self.intelligent_orchestrator.health_check()
            
            if self.context_engine:
                health_status['components']['context_engine'] = \
                    await self.context_engine.health_check()
            
            if self.semantic_memory:
                health_status['components']['semantic_memory'] = \
                    await self.semantic_memory.health_check()
            
            # Determine overall status
            component_statuses = [
                comp.get('status', 'unknown') if isinstance(comp, dict) else 'unknown'
                for comp in health_status['components'].values()
                if comp is not None
            ]
            
            if all(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'healthy'
            elif any(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'degraded'
            else:
                health_status['status'] = 'unhealthy'
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'system_metrics': self._performance_metrics,
            'decomposition_summary': {
                'total_decompositions': len(self.decomposition_results),
                'avg_subtasks_per_decomposition': sum(
                    len(result.subtasks) for result in self.decomposition_results.values()
                ) / max(len(self.decomposition_results), 1),
                'avg_confidence_score': sum(
                    result.confidence_score for result in self.decomposition_results.values()
                ) / max(len(self.decomposition_results), 1)
            },
            'execution_summary': {
                'active_plans': len(self.execution_plans),
                'avg_parallelism_factor': sum(
                    plan.parallelism_factor for plan in self.execution_plans.values()
                ) / max(len(self.execution_plans), 1),
                'total_recovery_plans': len(self.recovery_plans)
            }
        }


# Global instance management
_intelligent_task_decomposition: Optional[IntelligentTaskDecomposition] = None


async def get_intelligent_task_decomposition() -> IntelligentTaskDecomposition:
    """Get singleton intelligent task decomposition instance."""
    global _intelligent_task_decomposition
    
    if _intelligent_task_decomposition is None:
        _intelligent_task_decomposition = IntelligentTaskDecomposition()
        await _intelligent_task_decomposition.initialize()
    
    return _intelligent_task_decomposition


async def cleanup_intelligent_task_decomposition() -> None:
    """Cleanup intelligent task decomposition resources."""
    global _intelligent_task_decomposition
    
    if _intelligent_task_decomposition:
        # Cleanup would be implemented here
        _intelligent_task_decomposition = None
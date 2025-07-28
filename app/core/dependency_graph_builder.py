"""
Advanced Dependency Graph Builder for LeanVibe Agent Hive 2.0 Workflow Engine

Production-ready DAG (Directed Acyclic Graph) construction, validation, and analysis
with critical path calculation, bottleneck detection, and parallel execution optimization.
"""

import uuid
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import structlog

from ..models.workflow import Workflow
from ..models.task import Task

logger = structlog.get_logger()


@dataclass
class DependencyNode:
    """Represents a task node in the dependency graph."""
    task_id: str
    task_type: str
    estimated_duration: int
    dependencies: Set[str]
    dependents: Set[str]
    depth: int = 0
    critical_path: bool = False
    earliest_start: int = 0
    latest_start: int = 0


@dataclass
class ExecutionBatch:
    """Represents a batch of tasks that can execute in parallel."""
    batch_number: int
    task_ids: List[str]
    estimated_duration: int
    parallel_capacity: int
    dependencies_satisfied: Set[str]


@dataclass
class CriticalPath:
    """Represents the critical path through the workflow."""
    task_sequence: List[str]
    total_duration: int
    bottleneck_tasks: List[str]
    optimization_opportunities: List[Dict[str, Any]]


@dataclass
class DependencyAnalysis:
    """Complete analysis of workflow dependencies."""
    execution_batches: List[ExecutionBatch]
    critical_path: CriticalPath
    total_estimated_duration: int
    max_parallel_tasks: int
    dependency_violations: List[str]
    optimization_suggestions: List[Dict[str, Any]]


class DependencyGraphBuilder:
    """
    Advanced dependency graph builder with DAG validation and analysis.
    
    Features:
    - Topological sorting with batching for parallel execution
    - Critical path analysis for performance optimization
    - Bottleneck detection and resolution suggestions
    - Circular dependency detection with detailed error reporting
    - Dynamic graph modification with validation
    - Performance optimization for large workflows (1000+ tasks)
    """
    
    def __init__(self):
        """Initialize the dependency graph builder."""
        self.nodes: Dict[str, DependencyNode] = {}
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.analysis_cache: Dict[str, DependencyAnalysis] = {}
        
        # Performance metrics
        self.build_time_ms = 0
        self.validation_time_ms = 0
        self.analysis_time_ms = 0
        
        logger.info("DependencyGraphBuilder initialized")
    
    def build_graph(self, workflow: Workflow, tasks: List[Task]) -> DependencyAnalysis:
        """
        Build and analyze dependency graph for a workflow.
        
        Args:
            workflow: Workflow object with dependencies
            tasks: List of Task objects in the workflow
            
        Returns:
            DependencyAnalysis with execution plan and optimization suggestions
        """
        start_time = datetime.utcnow()
        
        try:
            # Clear previous state
            self._reset_graph()
            
            # Build task lookup
            task_lookup = {str(task.id): task for task in tasks}
            
            # Create nodes for each task
            self._create_nodes(workflow, task_lookup)
            
            # Build adjacency lists
            self._build_adjacency_lists(workflow)
            
            # Validate graph for cycles and consistency
            validation_errors = self._validate_graph()
            if validation_errors:
                raise ValueError(f"Graph validation failed: {'; '.join(validation_errors)}")
            
            # Calculate critical path and timing
            self._calculate_critical_path()
            
            # Generate execution batches
            execution_batches = self._generate_execution_batches()
            
            # Perform optimization analysis
            critical_path = self._analyze_critical_path()
            optimization_suggestions = self._generate_optimization_suggestions()
            
            # Create analysis result
            analysis = DependencyAnalysis(
                execution_batches=execution_batches,
                critical_path=critical_path,
                total_estimated_duration=critical_path.total_duration,
                max_parallel_tasks=max(len(batch.task_ids) for batch in execution_batches) if execution_batches else 0,
                dependency_violations=[],
                optimization_suggestions=optimization_suggestions
            )
            
            # Cache the analysis
            cache_key = self._generate_cache_key(workflow)
            self.analysis_cache[cache_key] = analysis
            
            # Update metrics
            self.analysis_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            logger.info(
                "✅ Dependency graph built and analyzed",
                workflow_id=str(workflow.id),
                total_tasks=len(tasks),
                execution_batches=len(execution_batches),
                critical_path_duration=critical_path.total_duration,
                analysis_time_ms=self.analysis_time_ms
            )
            
            return analysis
            
        except Exception as e:
            logger.error(
                "❌ Failed to build dependency graph",
                workflow_id=str(workflow.id),
                error=str(e)
            )
            raise
    
    def validate_dependencies(self, workflow: Workflow, tasks: List[Task]) -> List[str]:
        """
        Validate workflow dependencies without full analysis.
        
        Args:
            workflow: Workflow object with dependencies
            tasks: List of Task objects in the workflow
            
        Returns:
            List of validation error messages
        """
        start_time = datetime.utcnow()
        
        try:
            # Clear previous state
            self._reset_graph()
            
            # Build task lookup
            task_lookup = {str(task.id): task for task in tasks}
            
            # Create nodes for each task
            self._create_nodes(workflow, task_lookup)
            
            # Build adjacency lists
            self._build_adjacency_lists(workflow)
            
            # Validate graph
            validation_errors = self._validate_graph()
            
            # Update metrics
            self.validation_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            logger.info(
                "✅ Dependency validation completed",
                workflow_id=str(workflow.id),
                validation_errors=len(validation_errors),
                validation_time_ms=self.validation_time_ms
            )
            
            return validation_errors
            
        except Exception as e:
            logger.error(
                "❌ Failed to validate dependencies",
                workflow_id=str(workflow.id),
                error=str(e)
            )
            return [f"Validation error: {str(e)}"]
    
    def add_task_dependency(
        self, 
        task_id: str, 
        depends_on_task_id: str,
        estimated_duration: int = 30
    ) -> bool:
        """
        Dynamically add a task dependency to the graph.
        
        Args:
            task_id: ID of the task that depends on another
            depends_on_task_id: ID of the task that this task depends on
            estimated_duration: Estimated duration for timing calculations
            
        Returns:
            True if dependency was added successfully, False if it would create a cycle
        """
        try:
            # Create nodes if they don't exist
            if task_id not in self.nodes:
                self.nodes[task_id] = DependencyNode(
                    task_id=task_id,
                    task_type="unknown",
                    estimated_duration=estimated_duration,
                    dependencies=set(),
                    dependents=set()
                )
            
            if depends_on_task_id not in self.nodes:
                self.nodes[depends_on_task_id] = DependencyNode(
                    task_id=depends_on_task_id,
                    task_type="unknown",
                    estimated_duration=estimated_duration,
                    dependencies=set(),
                    dependents=set()
                )
            
            # Check if adding this dependency would create a cycle
            if self._would_create_cycle(task_id, depends_on_task_id):
                logger.warning(
                    "Cannot add dependency - would create cycle",
                    task_id=task_id,
                    depends_on_task_id=depends_on_task_id
                )
                return False
            
            # Add the dependency
            self.nodes[task_id].dependencies.add(depends_on_task_id)
            self.nodes[depends_on_task_id].dependents.add(task_id)
            self.adjacency_list[depends_on_task_id].add(task_id)
            self.reverse_adjacency_list[task_id].add(depends_on_task_id)
            
            # Clear analysis cache as graph has changed
            self.analysis_cache.clear()
            
            logger.info(
                "✅ Task dependency added",
                task_id=task_id,
                depends_on_task_id=depends_on_task_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "❌ Failed to add task dependency",
                task_id=task_id,
                depends_on_task_id=depends_on_task_id,
                error=str(e)
            )
            return False
    
    def remove_task_dependency(self, task_id: str, depends_on_task_id: str) -> bool:
        """
        Dynamically remove a task dependency from the graph.
        
        Args:
            task_id: ID of the task to remove dependency from
            depends_on_task_id: ID of the dependency to remove
            
        Returns:
            True if dependency was removed successfully
        """
        try:
            if task_id not in self.nodes or depends_on_task_id not in self.nodes:
                logger.warning(
                    "Cannot remove dependency - nodes not found",
                    task_id=task_id,
                    depends_on_task_id=depends_on_task_id
                )
                return False
            
            # Remove the dependency
            self.nodes[task_id].dependencies.discard(depends_on_task_id)
            self.nodes[depends_on_task_id].dependents.discard(task_id)
            self.adjacency_list[depends_on_task_id].discard(task_id)
            self.reverse_adjacency_list[task_id].discard(depends_on_task_id)
            
            # Clear analysis cache as graph has changed
            self.analysis_cache.clear()
            
            logger.info(
                "✅ Task dependency removed",
                task_id=task_id,
                depends_on_task_id=depends_on_task_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "❌ Failed to remove task dependency",
                task_id=task_id,
                depends_on_task_id=depends_on_task_id,
                error=str(e)
            )
            return False
    
    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """
        Get list of tasks that are ready to execute based on completed dependencies.
        
        Args:
            completed_tasks: Set of task IDs that have been completed
            
        Returns:
            List of task IDs ready for execution
        """
        ready_tasks = []
        
        for task_id, node in self.nodes.items():
            if task_id in completed_tasks:
                continue
            
            # Check if all dependencies are completed
            if node.dependencies.issubset(completed_tasks):
                ready_tasks.append(task_id)
        
        logger.debug(
            "Ready tasks calculated",
            ready_tasks_count=len(ready_tasks),
            completed_tasks_count=len(completed_tasks)
        )
        
        return ready_tasks
    
    def get_blocking_tasks(self, task_id: str) -> List[str]:
        """
        Get list of tasks that are blocking the execution of a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            List of task IDs that must be completed before this task can run
        """
        if task_id not in self.nodes:
            return []
        
        return list(self.nodes[task_id].dependencies)
    
    def get_dependent_tasks(self, task_id: str) -> List[str]:
        """
        Get list of tasks that depend on a specific task.
        
        Args:
            task_id: ID of the task to check
            
        Returns:
            List of task IDs that depend on this task
        """
        if task_id not in self.nodes:
            return []
        
        return list(self.nodes[task_id].dependents)
    
    def calculate_impact_analysis(self, task_id: str) -> Dict[str, Any]:
        """
        Calculate the impact of a task delay or failure on the workflow.
        
        Args:
            task_id: ID of the task to analyze
            
        Returns:
            Impact analysis including affected tasks and timeline impact
        """
        if task_id not in self.nodes:
            return {"error": "Task not found in graph"}
        
        node = self.nodes[task_id]
        
        # Find all downstream tasks using BFS
        affected_tasks = set()
        queue = deque([task_id])
        visited = set()
        
        while queue:
            current_task = queue.popleft()
            if current_task in visited:
                continue
            
            visited.add(current_task)
            
            # Add dependents to queue
            for dependent in self.nodes[current_task].dependents:
                if dependent not in visited:
                    queue.append(dependent)
                    affected_tasks.add(dependent)
        
        # Calculate timeline impact for critical path tasks
        timeline_impact = 0
        if node.critical_path:
            timeline_impact = node.estimated_duration
        
        return {
            "task_id": task_id,
            "is_critical_path": node.critical_path,
            "affected_tasks": list(affected_tasks),
            "affected_tasks_count": len(affected_tasks),
            "timeline_impact_minutes": timeline_impact,
            "dependency_depth": node.depth,
            "immediate_dependents": list(node.dependents)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and analysis metrics."""
        return {
            "build_time_ms": self.build_time_ms,
            "validation_time_ms": self.validation_time_ms,
            "analysis_time_ms": self.analysis_time_ms,
            "nodes_count": len(self.nodes),
            "edges_count": sum(len(deps) for deps in self.adjacency_list.values()),
            "cache_entries": len(self.analysis_cache),
            "max_dependency_depth": max((node.depth for node in self.nodes.values()), default=0)
        }
    
    # Private methods
    
    def _reset_graph(self) -> None:
        """Clear the graph state."""
        self.nodes.clear()
        self.adjacency_list.clear()
        self.reverse_adjacency_list.clear()
    
    def _create_nodes(self, workflow: Workflow, task_lookup: Dict[str, Task]) -> None:
        """Create dependency nodes for all tasks in the workflow."""
        if not workflow.task_ids:
            return
        
        for task_id in workflow.task_ids:
            task_id_str = str(task_id)
            task = task_lookup.get(task_id_str)
            
            if not task:
                logger.warning(f"Task {task_id_str} not found in task lookup")
                continue
            
            self.nodes[task_id_str] = DependencyNode(
                task_id=task_id_str,
                task_type=task.task_type.value if task.task_type else "unknown",
                estimated_duration=task.estimated_effort or 30,
                dependencies=set(),
                dependents=set()
            )
    
    def _build_adjacency_lists(self, workflow: Workflow) -> None:
        """Build adjacency lists from workflow dependencies."""
        if not workflow.dependencies:
            return
        
        for task_id, deps in workflow.dependencies.items():
            if task_id not in self.nodes:
                continue
            
            for dep_id in deps:
                if dep_id not in self.nodes:
                    continue
                
                # Add to adjacency lists
                self.adjacency_list[dep_id].add(task_id)
                self.reverse_adjacency_list[task_id].add(dep_id)
                
                # Update node relationships
                self.nodes[task_id].dependencies.add(dep_id)
                self.nodes[dep_id].dependents.add(task_id)
    
    def _validate_graph(self) -> List[str]:
        """Validate the dependency graph for cycles and consistency."""
        errors = []
        
        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dependent in self.adjacency_list[task_id]:
                if has_cycle(dependent):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check each connected component
        for task_id in self.nodes.keys():
            if task_id not in visited:
                if has_cycle(task_id):
                    errors.append(f"Circular dependency detected involving task {task_id}")
        
        # Check for orphaned dependencies
        for task_id, node in self.nodes.items():
            for dep_id in node.dependencies:
                if dep_id not in self.nodes:
                    errors.append(f"Task {task_id} depends on non-existent task {dep_id}")
        
        return errors
    
    def _would_create_cycle(self, task_id: str, depends_on_task_id: str) -> bool:
        """Check if adding a dependency would create a cycle."""
        # Use DFS to check if there's already a path from depends_on_task_id to task_id
        visited = set()
        
        def dfs(current: str, target: str) -> bool:
            if current == target:
                return True
            if current in visited:
                return False
            
            visited.add(current)
            
            for dependent in self.adjacency_list[current]:
                if dfs(dependent, target):
                    return True
            
            return False
        
        return dfs(depends_on_task_id, task_id)
    
    def _calculate_critical_path(self) -> None:
        """Calculate critical path and timing information for all nodes."""
        # Calculate dependency depths and earliest start times using topological sort
        in_degree = {task_id: len(node.dependencies) for task_id, node in self.nodes.items()}
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        
        while queue:
            current_task = queue.popleft()
            current_node = self.nodes[current_task]
            
            # Calculate earliest start time
            if current_node.dependencies:
                current_node.earliest_start = max(
                    self.nodes[dep].earliest_start + self.nodes[dep].estimated_duration
                    for dep in current_node.dependencies
                )
            
            # Update depth
            if current_node.dependencies:
                current_node.depth = max(self.nodes[dep].depth for dep in current_node.dependencies) + 1
            
            # Process dependents
            for dependent in current_node.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Calculate latest start times (backward pass)
        self._calculate_latest_start_times()
        
        # Mark critical path tasks
        for node in self.nodes.values():
            node.critical_path = (node.earliest_start == node.latest_start)
    
    def _calculate_latest_start_times(self) -> None:
        """Calculate latest start times using backward pass."""
        # Find leaf nodes (tasks with no dependents)
        leaf_nodes = [task_id for task_id, node in self.nodes.items() if not node.dependents]
        
        # Initialize latest start times for leaf nodes
        for task_id in leaf_nodes:
            node = self.nodes[task_id]
            node.latest_start = node.earliest_start
        
        # Process nodes in reverse topological order
        processed = set(leaf_nodes)
        queue = deque()
        
        # Add nodes whose dependents have all been processed
        for task_id, node in self.nodes.items():
            if task_id not in processed and node.dependents.issubset(processed):
                queue.append(task_id)
        
        while queue:
            current_task = queue.popleft()
            current_node = self.nodes[current_task]
            
            if current_node.dependents:
                # Latest start is the minimum of (dependent's latest start - this task's duration)
                current_node.latest_start = min(
                    self.nodes[dep].latest_start - current_node.estimated_duration
                    for dep in current_node.dependents
                )
            
            processed.add(current_task)
            
            # Add new nodes whose dependents have all been processed
            for task_id, node in self.nodes.items():
                if task_id not in processed and node.dependents.issubset(processed):
                    queue.append(task_id)
    
    def _generate_execution_batches(self) -> List[ExecutionBatch]:
        """Generate execution batches using topological sorting."""
        in_degree = {task_id: len(node.dependencies) for task_id, node in self.nodes.items()}
        batches = []
        batch_number = 0
        processed = set()
        
        while len(processed) < len(self.nodes):
            # Find all tasks with no remaining dependencies
            ready_tasks = [
                task_id for task_id, degree in in_degree.items()
                if degree == 0 and task_id not in processed
            ]
            
            if not ready_tasks:
                break  # No more tasks can be processed (shouldn't happen with valid DAG)
            
            # Calculate batch duration and capacity
            batch_duration = max(self.nodes[task_id].estimated_duration for task_id in ready_tasks)
            batch_capacity = len(ready_tasks)
            
            batch = ExecutionBatch(
                batch_number=batch_number,
                task_ids=ready_tasks,
                estimated_duration=batch_duration,
                parallel_capacity=batch_capacity,
                dependencies_satisfied=processed.copy()
            )
            batches.append(batch)
            
            # Mark tasks as processed and update in-degrees
            for task_id in ready_tasks:
                processed.add(task_id)
                for dependent in self.nodes[task_id].dependents:
                    in_degree[dependent] -= 1
            
            batch_number += 1
        
        return batches
    
    def _analyze_critical_path(self) -> CriticalPath:
        """Analyze the critical path through the workflow."""
        critical_tasks = [task_id for task_id, node in self.nodes.items() if node.critical_path]
        
        if not critical_tasks:
            return CriticalPath(
                task_sequence=[],
                total_duration=0,
                bottleneck_tasks=[],
                optimization_opportunities=[]
            )
        
        # Build critical path sequence
        critical_path_sequence = self._build_critical_path_sequence(critical_tasks)
        
        # Calculate total duration
        total_duration = sum(self.nodes[task_id].estimated_duration for task_id in critical_path_sequence)
        
        # Identify bottleneck tasks (tasks with high impact on total duration)
        bottleneck_tasks = self._identify_bottleneck_tasks(critical_path_sequence)
        
        # Generate optimization opportunities
        optimization_opportunities = self._generate_critical_path_optimizations(critical_path_sequence)
        
        return CriticalPath(
            task_sequence=critical_path_sequence,
            total_duration=total_duration,
            bottleneck_tasks=bottleneck_tasks,
            optimization_opportunities=optimization_opportunities
        )
    
    def _build_critical_path_sequence(self, critical_tasks: List[str]) -> List[str]:
        """Build the critical path sequence from critical tasks."""
        if not critical_tasks:
            return []
        
        # Find the starting task (no critical dependencies)
        start_tasks = []
        for task_id in critical_tasks:
            critical_deps = [dep for dep in self.nodes[task_id].dependencies if dep in critical_tasks]
            if not critical_deps:
                start_tasks.append(task_id)
        
        if not start_tasks:
            return critical_tasks  # Fallback to any order
        
        # Build sequence using topological sort on critical tasks only
        sequence = []
        visited = set()
        
        def dfs(task_id: str):
            if task_id in visited or task_id not in critical_tasks:
                return
            
            visited.add(task_id)
            
            # Visit dependencies first
            for dep in self.nodes[task_id].dependencies:
                if dep in critical_tasks:
                    dfs(dep)
            
            sequence.append(task_id)
        
        for start_task in start_tasks:
            dfs(start_task)
        
        return sequence
    
    def _identify_bottleneck_tasks(self, critical_path_sequence: List[str]) -> List[str]:
        """Identify bottleneck tasks in the critical path."""
        if not critical_path_sequence:
            return []
        
        # Tasks are bottlenecks if they have high duration or many dependents
        bottlenecks = []
        avg_duration = sum(self.nodes[task_id].estimated_duration for task_id in critical_path_sequence) / len(critical_path_sequence)
        
        for task_id in critical_path_sequence:
            node = self.nodes[task_id]
            
            # High duration tasks
            if node.estimated_duration > avg_duration * 1.5:
                bottlenecks.append(task_id)
            
            # Tasks with many dependents
            elif len(node.dependents) > 3:
                bottlenecks.append(task_id)
        
        return bottlenecks
    
    def _generate_critical_path_optimizations(self, critical_path_sequence: List[str]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for the critical path."""
        optimizations = []
        
        for task_id in critical_path_sequence:
            node = self.nodes[task_id]
            
            # Suggest parallelization opportunities
            if len(node.dependents) > 1:
                parallel_dependents = [dep for dep in node.dependents if dep not in critical_path_sequence]
                if parallel_dependents:
                    optimizations.append({
                        "type": "parallelization",
                        "task_id": task_id,
                        "suggestion": f"Consider parallelizing {len(parallel_dependents)} dependent tasks",
                        "impact": "medium",
                        "parallel_tasks": parallel_dependents
                    })
            
            # Suggest duration optimization for high-duration tasks
            if node.estimated_duration > 60:  # > 1 hour
                optimizations.append({
                    "type": "duration_optimization",
                    "task_id": task_id,
                    "suggestion": f"Task has high duration ({node.estimated_duration} minutes) - consider breaking down",
                    "impact": "high",
                    "current_duration": node.estimated_duration
                })
        
        return optimizations
    
    def _generate_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate general optimization suggestions for the workflow."""
        suggestions = []
        
        # Analyze parallel execution opportunities
        max_parallel = max(len(batch.task_ids) for batch in self._generate_execution_batches()) if self.nodes else 0
        if max_parallel < len(self.nodes) / 3:  # Less than 1/3 parallelization
            suggestions.append({
                "type": "parallelization",
                "suggestion": "Low parallelization detected - consider reducing task dependencies",
                "impact": "high",
                "max_parallel_tasks": max_parallel,
                "total_tasks": len(self.nodes)
            })
        
        # Analyze dependency depth
        max_depth = max((node.depth for node in self.nodes.values()), default=0)
        if max_depth > 5:
            suggestions.append({
                "type": "dependency_depth",
                "suggestion": f"Deep dependency chain detected (depth: {max_depth}) - consider flattening",
                "impact": "medium",
                "max_depth": max_depth
            })
        
        return suggestions
    
    def _generate_cache_key(self, workflow: Workflow) -> str:
        """Generate cache key for workflow analysis."""
        return f"{workflow.id}_{hash(str(sorted(workflow.dependencies.items())) if workflow.dependencies else '')}"
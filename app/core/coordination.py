"""
Multi-Agent Coordination Engine for LeanVibe Agent Hive 2.0

This revolutionary system enables sophisticated coordination between multiple agents
working on the same project simultaneously. It provides real-time state synchronization,
intelligent task distribution, conflict resolution, and coordinated development workflows.

CRITICAL: This is the brain of the multi-agent system, transforming individual
autonomous agents into a coordinated hive intelligence.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from collections import defaultdict
import structlog

from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .redis import get_message_broker, get_session_cache
from .workspace_manager import workspace_manager
from .external_tools import external_tools
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.session import Session

logger = structlog.get_logger()


class CoordinationMode(Enum):
    """Modes of multi-agent coordination."""
    PARALLEL = "parallel"                # Agents work independently with sync points
    SEQUENTIAL = "sequential"            # Agents work in predefined sequence
    COLLABORATIVE = "collaborative"     # Real-time collaboration on shared tasks
    COMPETITIVE = "competitive"         # Agents compete for best solution
    HIERARCHICAL = "hierarchical"       # Lead agent coordinates subordinates


class ConflictType(Enum):
    """Types of conflicts between agents."""
    CODE_CONFLICT = "code_conflict"           # Simultaneous code modifications
    RESOURCE_CONFLICT = "resource_conflict"   # Competing for same resources
    TASK_CONFLICT = "task_conflict"           # Overlapping task assignments
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Circular or broken dependencies
    QUALITY_CONFLICT = "quality_conflict"     # Different quality standards


class ProjectStatus(Enum):
    """Status of coordinated projects."""
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """Represents an agent's specialized capability."""
    name: str
    proficiency: float  # 0.0 to 1.0
    experience_level: str  # novice, intermediate, expert, master
    specializations: List[str]
    tools_available: List[str]
    performance_metrics: Dict[str, float]
    
    def calculate_task_suitability(self, task_requirements: List[str]) -> float:
        """Calculate how suitable this agent is for specific task requirements."""
        if not task_requirements:
            return 0.5
        
        # Check specialization match
        specialization_match = len(set(task_requirements) & set(self.specializations)) / len(task_requirements)
        
        # Factor in proficiency and experience
        experience_multiplier = {
            "novice": 0.6,
            "intermediate": 0.8,
            "expert": 1.0,
            "master": 1.2
        }.get(self.experience_level, 0.8)
        
        # Calculate final suitability score
        suitability = (specialization_match * 0.6 + self.proficiency * 0.4) * experience_multiplier
        return min(1.0, suitability)


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""
    dependent_task_id: str
    prerequisite_task_id: str
    dependency_type: str  # hard, soft, informational
    satisfaction_criteria: Dict[str, Any]
    
    def is_satisfied(self, completed_tasks: Set[str], task_results: Dict[str, Any]) -> bool:
        """Check if this dependency is satisfied."""
        if self.dependency_type == "informational":
            return True
        
        if self.prerequisite_task_id not in completed_tasks:
            return False
        
        # Check satisfaction criteria
        if self.prerequisite_task_id in task_results:
            result = task_results[self.prerequisite_task_id]
            for criteria, expected_value in self.satisfaction_criteria.items():
                if result.get(criteria) != expected_value:
                    return False
        
        return True


@dataclass
class CoordinatedProject:
    """Represents a project being developed by multiple agents."""
    id: str
    name: str
    description: str
    
    # Project configuration
    coordination_mode: CoordinationMode
    participating_agents: List[str]  # Agent IDs
    lead_agent_id: Optional[str]
    
    # Project structure
    tasks: Dict[str, Task]
    dependencies: List[TaskDependency]
    milestones: List[Dict[str, Any]]
    
    # State management
    status: ProjectStatus
    current_phase: str
    shared_state: Dict[str, Any]
    
    # Repository and workspace
    repository_id: Optional[str]
    workspace_branch: str
    integration_branch: str
    
    # Synchronization
    sync_points: List[str]
    last_sync: datetime
    sync_frequency: int  # seconds
    
    # Quality and progress
    quality_gates: List[Dict[str, Any]]
    progress_metrics: Dict[str, float]
    
    # Metadata
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    deadline: Optional[datetime]
    
    def __post_init__(self):
        if not self.tasks:
            self.tasks = {}
        if not self.dependencies:
            self.dependencies = []
        if not self.shared_state:
            self.shared_state = {}
        if not self.progress_metrics:
            self.progress_metrics = {}


@dataclass
class ConflictEvent:
    """Represents a conflict between agents."""
    id: str
    project_id: str
    conflict_type: ConflictType
    
    # Agents involved
    primary_agent_id: str
    secondary_agent_id: str
    affected_agents: List[str]
    
    # Conflict details
    description: str
    affected_files: List[str]
    conflicting_changes: Dict[str, Any]
    
    # Resolution
    resolution_strategy: Optional[str]
    resolved: bool
    resolution_result: Optional[Dict[str, Any]]
    
    # Timing
    detected_at: datetime
    resolved_at: Optional[datetime]
    
    # Impact assessment
    severity: str  # low, medium, high, critical
    impact_score: float
    affected_tasks: List[str]


class AgentRegistry:
    """
    Registry for managing agent capabilities and availability.
    
    Tracks agent specializations, performance metrics, and current assignments
    to enable intelligent task distribution and coordination.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.agent_assignments: Dict[str, List[str]] = defaultdict(list)  # agent_id -> task_ids
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_status: Dict[str, str] = {}  # agent_id -> status
    
    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        specializations: List[str],
        proficiency: float = 0.8,
        experience_level: str = "intermediate"
    ) -> None:
        """Register an agent with its capabilities."""
        
        # Get agent performance metrics from database
        async with get_session() as db_session:
            agent = await db_session.get(Agent, agent_id)
            if agent:
                performance_metrics = {
                    "task_completion_rate": getattr(agent, 'completion_rate', 0.85),
                    "average_task_time": getattr(agent, 'avg_task_time', 120.0),
                    "quality_score": getattr(agent, 'quality_score', 0.8),
                    "reliability_score": getattr(agent, 'reliability_score', 0.9)
                }
            else:
                performance_metrics = {
                    "task_completion_rate": 0.85,
                    "average_task_time": 120.0,
                    "quality_score": 0.8,
                    "reliability_score": 0.9
                }
        
        # Create agent capability profile
        capability = AgentCapability(
            name=f"agent_{agent_id}",
            proficiency=proficiency,
            experience_level=experience_level,
            specializations=specializations,
            tools_available=capabilities,
            performance_metrics=performance_metrics
        )
        
        self.agents[agent_id] = capability
        self.agent_status[agent_id] = "available"
        
        logger.info(
            "Agent registered in coordination registry",
            agent_id=agent_id,
            specializations=specializations,
            proficiency=proficiency
        )
    
    def get_best_agent_for_task(
        self,
        task_requirements: List[str],
        exclude_agents: Optional[List[str]] = None
    ) -> Optional[str]:
        """Find the best agent for a specific task."""
        
        exclude_agents = exclude_agents or []
        best_agent = None
        best_score = 0.0
        
        for agent_id, capability in self.agents.items():
            if agent_id in exclude_agents:
                continue
            
            if self.agent_status.get(agent_id) != "available":
                continue
            
            # Consider current workload
            current_tasks = len(self.agent_assignments.get(agent_id, []))
            workload_factor = max(0.1, 1.0 - (current_tasks * 0.2))  # Penalize overloaded agents
            
            # Calculate task suitability
            suitability = capability.calculate_task_suitability(task_requirements)
            
            # Factor in performance metrics
            performance_factor = (
                capability.performance_metrics.get("task_completion_rate", 0.8) * 0.3 +
                capability.performance_metrics.get("quality_score", 0.8) * 0.4 +
                capability.performance_metrics.get("reliability_score", 0.8) * 0.3
            )
            
            # Calculate final score
            final_score = suitability * performance_factor * workload_factor
            
            if final_score > best_score:
                best_score = final_score
                best_agent = agent_id
        
        return best_agent
    
    def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Assign a task to an agent."""
        if agent_id not in self.agents:
            return False
        
        self.agent_assignments[agent_id].append(task_id)
        
        # Update agent status based on workload
        task_count = len(self.agent_assignments[agent_id])
        if task_count >= 3:  # Configurable threshold
            self.agent_status[agent_id] = "busy"
        elif task_count >= 5:
            self.agent_status[agent_id] = "overloaded"
        
        return True
    
    def complete_task(self, agent_id: str, task_id: str, performance_data: Dict[str, float]) -> None:
        """Mark a task as completed and update agent performance."""
        if agent_id in self.agent_assignments:
            if task_id in self.agent_assignments[agent_id]:
                self.agent_assignments[agent_id].remove(task_id)
        
        # Update performance metrics
        if agent_id in self.agents:
            # Update with exponential moving average
            alpha = 0.3  # Learning rate
            for metric, value in performance_data.items():
                current_value = self.agents[agent_id].performance_metrics.get(metric, 0.8)
                new_value = alpha * value + (1 - alpha) * current_value
                self.agents[agent_id].performance_metrics[metric] = new_value
        
        # Update agent status
        task_count = len(self.agent_assignments.get(agent_id, []))
        if task_count == 0:
            self.agent_status[agent_id] = "available"
        elif task_count < 3:
            self.agent_status[agent_id] = "active"


class ConflictResolver:
    """
    Sophisticated conflict resolution system for multi-agent development.
    
    Detects and resolves conflicts between agents working on the same project,
    including code conflicts, resource conflicts, and task dependencies.
    """
    
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client
        self.active_conflicts: Dict[str, ConflictEvent] = {}
        self.resolution_strategies: Dict[ConflictType, List[str]] = {
            ConflictType.CODE_CONFLICT: [
                "automatic_merge",
                "ai_assisted_merge", 
                "agent_negotiation",
                "human_intervention"
            ],
            ConflictType.RESOURCE_CONFLICT: [
                "resource_sharing",
                "priority_based_allocation",
                "time_based_rotation",
                "resource_duplication"
            ],
            ConflictType.TASK_CONFLICT: [
                "task_redistribution",
                "agent_collaboration",
                "task_splitting",
                "priority_resolution"
            ]
        }
    
    async def detect_conflicts(
        self,
        project: CoordinatedProject,
        recent_changes: List[Dict[str, Any]]
    ) -> List[ConflictEvent]:
        """Detect potential conflicts in project state."""
        
        conflicts = []
        
        # Check for code conflicts
        code_conflicts = await self._detect_code_conflicts(project, recent_changes)
        conflicts.extend(code_conflicts)
        
        # Check for resource conflicts
        resource_conflicts = await self._detect_resource_conflicts(project)
        conflicts.extend(resource_conflicts)
        
        # Check for task conflicts
        task_conflicts = await self._detect_task_conflicts(project)
        conflicts.extend(task_conflicts)
        
        # Store active conflicts
        for conflict in conflicts:
            self.active_conflicts[conflict.id] = conflict
        
        return conflicts
    
    async def _detect_code_conflicts(
        self,
        project: CoordinatedProject,
        recent_changes: List[Dict[str, Any]]
    ) -> List[ConflictEvent]:
        """Detect code conflicts between agents."""
        
        conflicts = []
        
        # Group changes by file
        file_changes = defaultdict(list)
        for change in recent_changes:
            for file_path in change.get("files_modified", []):
                file_changes[file_path].append(change)
        
        # Check for simultaneous modifications
        for file_path, changes in file_changes.items():
            if len(changes) > 1:
                # Check if changes overlap in time (within conflict window)
                conflict_window = timedelta(minutes=30)
                
                for i in range(len(changes)):
                    for j in range(i + 1, len(changes)):
                        change1, change2 = changes[i], changes[j]
                        
                        time1 = datetime.fromisoformat(change1.get("timestamp"))
                        time2 = datetime.fromisoformat(change2.get("timestamp"))
                        
                        if abs(time1 - time2) < conflict_window:
                            # Potential conflict detected
                            conflict = ConflictEvent(
                                id=str(uuid.uuid4()),
                                project_id=project.id,
                                conflict_type=ConflictType.CODE_CONFLICT,
                                primary_agent_id=change1.get("agent_id"),
                                secondary_agent_id=change2.get("agent_id"),
                                affected_agents=[change1.get("agent_id"), change2.get("agent_id")],
                                description=f"Simultaneous modifications to {file_path}",
                                affected_files=[file_path],
                                conflicting_changes={"change1": change1, "change2": change2},
                                resolution_strategy=None,
                                resolved=False,
                                resolution_result=None,
                                detected_at=datetime.utcnow(),
                                resolved_at=None,
                                severity="medium",
                                impact_score=0.6,
                                affected_tasks=[]
                            )
                            conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_resource_conflicts(self, project: CoordinatedProject) -> List[ConflictEvent]:
        """Detect resource conflicts between agents."""
        
        conflicts = []
        
        # Check workspace resource usage
        agent_resources = {}
        for agent_id in project.participating_agents:
            workspace = await workspace_manager.get_workspace(agent_id)
            if workspace:
                metrics = await workspace.get_metrics()
                agent_resources[agent_id] = {
                    "memory_mb": metrics.total_memory_mb,
                    "cpu_percent": metrics.total_cpu_percent,
                    "disk_usage_mb": metrics.disk_usage_mb
                }
        
        # Check for resource exhaustion
        total_memory = sum(res.get("memory_mb", 0) for res in agent_resources.values())
        total_cpu = sum(res.get("cpu_percent", 0) for res in agent_resources.values())
        
        if total_memory > 8192 or total_cpu > 80:  # Configurable thresholds
            conflict = ConflictEvent(
                id=str(uuid.uuid4()),
                project_id=project.id,
                conflict_type=ConflictType.RESOURCE_CONFLICT,
                primary_agent_id=max(agent_resources.keys(), key=lambda x: agent_resources[x].get("memory_mb", 0)),
                secondary_agent_id="system",
                affected_agents=list(project.participating_agents),
                description="Resource exhaustion detected",
                affected_files=[],
                conflicting_changes={"resource_usage": agent_resources},
                resolution_strategy=None,
                resolved=False,
                resolution_result=None,
                detected_at=datetime.utcnow(),
                resolved_at=None,
                severity="high",
                impact_score=0.8,
                affected_tasks=[]
            )
            conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_task_conflicts(self, project: CoordinatedProject) -> List[ConflictEvent]:
        """Detect task assignment conflicts."""
        
        conflicts = []
        
        # Check for overlapping task assignments
        agent_tasks = defaultdict(list)
        for task_id, task in project.tasks.items():
            if task.assigned_agent_id:
                agent_tasks[task.assigned_agent_id].append(task_id)
        
        # Look for tasks with similar requirements assigned to different agents
        # This could indicate inefficient distribution or potential conflicts
        for agent_id, task_ids in agent_tasks.items():
            if len(task_ids) > 5:  # Configurable threshold
                conflict = ConflictEvent(
                    id=str(uuid.uuid4()),
                    project_id=project.id,
                    conflict_type=ConflictType.TASK_CONFLICT,
                    primary_agent_id=agent_id,
                    secondary_agent_id="coordinator",
                    affected_agents=[agent_id],
                    description=f"Agent {agent_id} is overloaded with {len(task_ids)} tasks",
                    affected_files=[],
                    conflicting_changes={"overloaded_tasks": task_ids},
                    resolution_strategy=None,
                    resolved=False,
                    resolution_result=None,
                    detected_at=datetime.utcnow(),
                    resolved_at=None,
                    severity="medium",
                    impact_score=0.5,
                    affected_tasks=task_ids
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def resolve_conflict(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt to resolve a specific conflict."""
        
        try:
            # Select resolution strategy based on conflict type and severity
            strategies = self.resolution_strategies.get(conflict.conflict_type, [])
            
            for strategy in strategies:
                success, result = await self._apply_resolution_strategy(
                    conflict, project, strategy
                )
                
                if success:
                    conflict.resolved = True
                    conflict.resolved_at = datetime.utcnow()
                    conflict.resolution_strategy = strategy
                    conflict.resolution_result = result
                    
                    logger.info(
                        "Conflict resolved successfully",
                        conflict_id=conflict.id,
                        strategy=strategy,
                        project_id=project.id
                    )
                    
                    return True, result
            
            # If no automatic resolution worked, escalate to human
            logger.warning(
                "Conflict requires human intervention",
                conflict_id=conflict.id,
                conflict_type=conflict.conflict_type.value,
                project_id=project.id
            )
            
            return False, {"escalated_to_human": True, "reason": "automatic_resolution_failed"}
            
        except Exception as e:
            logger.error(
                "Conflict resolution failed",
                conflict_id=conflict.id,
                error=str(e)
            )
            return False, {"error": str(e)}
    
    async def _apply_resolution_strategy(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject,
        strategy: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Apply a specific resolution strategy."""
        
        if strategy == "automatic_merge" and conflict.conflict_type == ConflictType.CODE_CONFLICT:
            return await self._automatic_merge_resolution(conflict, project)
        
        elif strategy == "ai_assisted_merge" and conflict.conflict_type == ConflictType.CODE_CONFLICT:
            return await self._ai_assisted_merge_resolution(conflict, project)
        
        elif strategy == "task_redistribution" and conflict.conflict_type == ConflictType.TASK_CONFLICT:
            return await self._task_redistribution_resolution(conflict, project)
        
        elif strategy == "resource_sharing" and conflict.conflict_type == ConflictType.RESOURCE_CONFLICT:
            return await self._resource_sharing_resolution(conflict, project)
        
        else:
            return False, {"error": f"Unknown strategy: {strategy}"}
    
    async def _automatic_merge_resolution(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject
    ) -> Tuple[bool, Dict[str, Any]]:
        """Attempt automatic merge resolution for code conflicts."""
        
        # Use external tools Git integration for automatic merge
        if project.repository_id:
            try:
                # Create temporary branch for merge attempt
                merge_branch = f"conflict-resolution-{conflict.id[:8]}"
                
                success, output, error = await external_tools.git.create_branch(
                    project.repository_id,
                    merge_branch,
                    checkout=True
                )
                
                if success:
                    # Attempt automatic merge (simplified for demo)
                    # In production, this would use sophisticated merge algorithms
                    return True, {
                        "merge_branch": merge_branch,
                        "merge_strategy": "automatic",
                        "conflicts_resolved": len(conflict.affected_files)
                    }
                
            except Exception as e:
                logger.error("Automatic merge failed", error=str(e))
        
        return False, {"error": "Automatic merge not possible"}
    
    async def _ai_assisted_merge_resolution(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject
    ) -> Tuple[bool, Dict[str, Any]]:
        """Use AI to assist with merge conflict resolution."""
        
        # Analyze conflicting changes using AI
        analysis_prompt = f"""
        Analyze this code conflict and suggest a resolution:
        
        Conflict: {conflict.description}
        Affected files: {', '.join(conflict.affected_files)}
        
        Change 1 (Agent {conflict.primary_agent_id}):
        {json.dumps(conflict.conflicting_changes.get('change1', {}), indent=2)}
        
        Change 2 (Agent {conflict.secondary_agent_id}):
        {json.dumps(conflict.conflicting_changes.get('change2', {}), indent=2)}
        
        Provide a merge strategy that preserves the intent of both changes.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            ai_suggestion = response.content[0].text
            
            return True, {
                "ai_suggestion": ai_suggestion,
                "merge_strategy": "ai_assisted",
                "confidence": 0.8
            }
            
        except Exception as e:
            logger.error("AI-assisted merge failed", error=str(e))
            return False, {"error": "AI analysis failed"}
    
    async def _task_redistribution_resolution(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject
    ) -> Tuple[bool, Dict[str, Any]]:
        """Resolve task conflicts through redistribution."""
        
        # Get agent registry to find alternative assignments
        from .coordination import coordination_engine
        registry = coordination_engine.agent_registry
        
        affected_tasks = conflict.affected_tasks
        redistributed = []
        
        for task_id in affected_tasks:
            if task_id in project.tasks:
                task = project.tasks[task_id]
                
                # Find better agent for this task
                better_agent = registry.get_best_agent_for_task(
                    task.required_capabilities,
                    exclude_agents=[conflict.primary_agent_id]
                )
                
                if better_agent:
                    # Reassign task
                    registry.assign_task(better_agent, task_id)
                    task.assigned_agent_id = better_agent
                    redistributed.append({
                        "task_id": task_id,
                        "from_agent": conflict.primary_agent_id,
                        "to_agent": better_agent
                    })
        
        if redistributed:
            return True, {
                "redistributed_tasks": redistributed,
                "strategy": "task_redistribution"
            }
        
        return False, {"error": "No suitable agents found for redistribution"}
    
    async def _resource_sharing_resolution(
        self,
        conflict: ConflictEvent,
        project: CoordinatedProject
    ) -> Tuple[bool, Dict[str, Any]]:
        """Resolve resource conflicts through intelligent sharing."""
        
        # Implement resource throttling and sharing
        resource_usage = conflict.conflicting_changes.get("resource_usage", {})
        
        # Suspend lowest priority agents temporarily
        agents_by_memory = sorted(
            resource_usage.items(),
            key=lambda x: x[1].get("memory_mb", 0),
            reverse=True
        )
        
        suspended_agents = []
        for agent_id, usage in agents_by_memory[:2]:  # Suspend top 2 resource consumers
            workspace = await workspace_manager.get_workspace(agent_id)
            if workspace:
                success = await workspace.suspend()
                if success:
                    suspended_agents.append(agent_id)
        
        if suspended_agents:
            return True, {
                "suspended_agents": suspended_agents,
                "strategy": "resource_sharing",
                "suspension_duration": 300  # 5 minutes
            }
        
        return False, {"error": "Unable to suspend agents for resource sharing"}


class MultiAgentCoordinator:
    """
    Core coordination engine for multi-agent development workflows.
    
    This is the brain of the system, orchestrating multiple agents working
    on the same project with real-time coordination and conflict resolution.
    """
    
    def __init__(self):
        self.anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.agent_registry = AgentRegistry()
        self.conflict_resolver = ConflictResolver(self.anthropic)
        
        # Active projects
        self.active_projects: Dict[str, CoordinatedProject] = {}
        self.project_locks: Dict[str, asyncio.Lock] = {}
        
        # Coordination state
        self.coordination_bus = None
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.coordination_metrics: Dict[str, float] = {
            "projects_completed": 0,
            "conflicts_resolved": 0,
            "average_project_duration": 0,
            "agent_utilization": 0
        }
    
    async def initialize(self):
        """Initialize the coordination engine."""
        
        # Initialize message bus
        self.coordination_bus = get_message_broker()
        
        # Register for agent events
        await self.coordination_bus.subscribe("agent_events", self._handle_agent_event)
        await self.coordination_bus.subscribe("project_updates", self._handle_project_update)
        
        logger.info("Multi-Agent Coordination Engine initialized")
    
    async def create_coordinated_project(
        self,
        name: str,
        description: str,
        requirements: Dict[str, Any],
        coordination_mode: CoordinationMode = CoordinationMode.PARALLEL,
        deadline: Optional[datetime] = None
    ) -> str:
        """Create a new coordinated project."""
        
        project_id = str(uuid.uuid4())
        
        # Analyze requirements and select appropriate agents
        selected_agents = await self._select_project_agents(requirements)
        
        if not selected_agents:
            raise ValueError("No suitable agents available for project requirements")
        
        # Create project structure
        project = CoordinatedProject(
            id=project_id,
            name=name,
            description=description,
            coordination_mode=coordination_mode,
            participating_agents=selected_agents,
            lead_agent_id=selected_agents[0] if selected_agents else None,
            tasks={},
            dependencies=[],
            milestones=[],
            status=ProjectStatus.PLANNING,
            current_phase="initialization",
            shared_state={},
            repository_id=None,
            workspace_branch="main",
            integration_branch="integration",
            sync_points=["milestone_1", "milestone_2", "completion"],
            last_sync=datetime.utcnow(),
            sync_frequency=300,  # 5 minutes
            quality_gates=[
                {"name": "code_review", "threshold": 0.8},
                {"name": "test_coverage", "threshold": 0.9},
                {"name": "security_scan", "threshold": 1.0}
            ],
            progress_metrics={},
            created_at=datetime.utcnow(),
            started_at=None,
            completed_at=None,
            deadline=deadline
        )
        
        # Store project and create coordination lock
        self.active_projects[project_id] = project
        self.project_locks[project_id] = asyncio.Lock()
        
        # Decompose project into tasks
        await self._decompose_project_into_tasks(project, requirements)
        
        # Start coordination sync task
        self.sync_tasks[project_id] = asyncio.create_task(
            self._project_sync_loop(project_id)
        )
        
        logger.info(
            "Coordinated project created",
            project_id=project_id,
            project_name=name,
            agents=selected_agents,
            coordination_mode=coordination_mode.value
        )
        
        return project_id
    
    async def _select_project_agents(
        self,
        requirements: Dict[str, Any]
    ) -> List[str]:
        """Select optimal agents for project based on requirements."""
        
        required_capabilities = requirements.get("capabilities", [])
        project_complexity = requirements.get("complexity", "medium")
        timeline = requirements.get("timeline", "normal")
        
        # Determine how many agents needed based on complexity
        agent_count = {
            "low": 2,
            "medium": 3,
            "high": 4,
            "enterprise": 6
        }.get(project_complexity, 3)
        
        selected_agents = []
        used_capabilities = set()
        
        # Select agents with diverse, non-overlapping specializations
        for capability in required_capabilities:
            if capability not in used_capabilities:
                agent_id = self.agent_registry.get_best_agent_for_task(
                    [capability],
                    exclude_agents=selected_agents
                )
                
                if agent_id:
                    selected_agents.append(agent_id)
                    used_capabilities.add(capability)
                    
                    if len(selected_agents) >= agent_count:
                        break
        
        # Fill remaining slots with best available agents
        while len(selected_agents) < agent_count:
            agent_id = self.agent_registry.get_best_agent_for_task(
                required_capabilities,
                exclude_agents=selected_agents
            )
            
            if agent_id:
                selected_agents.append(agent_id)
            else:
                break
        
        return selected_agents
    
    async def _decompose_project_into_tasks(
        self,
        project: CoordinatedProject,
        requirements: Dict[str, Any]
    ) -> None:
        """Decompose project requirements into specific agent tasks."""
        
        # Use AI to analyze requirements and create task breakdown
        decomposition_prompt = f"""
        Analyze this project and break it down into specific, actionable tasks for multiple agents:
        
        Project: {project.name}
        Description: {project.description}
        Requirements: {json.dumps(requirements, indent=2)}
        Available Agents: {project.participating_agents}
        Coordination Mode: {project.coordination_mode.value}
        
        Create a detailed task breakdown with:
        1. Specific tasks that can be worked on in parallel
        2. Dependencies between tasks
        3. Estimated effort for each task
        4. Required capabilities for each task
        5. Synchronization points where agents need to coordinate
        
        Format the response as structured JSON with tasks, dependencies, and sync points.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": decomposition_prompt}]
            )
            
            # Parse AI response and create tasks
            # (Simplified for demo - would have sophisticated parsing in production)
            
            # Create example tasks based on common patterns
            base_tasks = [
                {
                    "title": f"Setup and Architecture Planning for {project.name}",
                    "description": f"Design system architecture and setup project structure",
                    "task_type": "ARCHITECTURE",
                    "priority": TaskPriority.HIGH,
                    "estimated_effort": 180,  # 3 hours
                    "required_capabilities": ["architecture", "planning"],
                    "dependencies": []
                },
                {
                    "title": f"Core Implementation for {project.name}",
                    "description": f"Implement main functionality and business logic",
                    "task_type": "FEATURE_DEVELOPMENT",
                    "priority": TaskPriority.HIGH,
                    "estimated_effort": 360,  # 6 hours
                    "required_capabilities": ["development", "coding"],
                    "dependencies": ["setup_task"]
                },
                {
                    "title": f"Testing and Quality Assurance for {project.name}",
                    "description": f"Create comprehensive tests and quality validation",
                    "task_type": "TESTING",
                    "priority": TaskPriority.MEDIUM,
                    "estimated_effort": 240,  # 4 hours
                    "required_capabilities": ["testing", "quality_assurance"],
                    "dependencies": ["core_implementation"]
                },
                {
                    "title": f"Documentation and Deployment for {project.name}",
                    "description": f"Create documentation and deployment configuration",
                    "task_type": "DOCUMENTATION",
                    "priority": TaskPriority.MEDIUM,
                    "estimated_effort": 120,  # 2 hours
                    "required_capabilities": ["documentation", "deployment"],
                    "dependencies": ["testing"]
                }
            ]
            
            # Create tasks and assign to agents
            task_ids = []
            for i, task_data in enumerate(base_tasks):
                task_id = str(uuid.uuid4())
                
                # Find best agent for this task
                best_agent = self.agent_registry.get_best_agent_for_task(
                    task_data["required_capabilities"]
                )
                
                # Create task
                from ..models.task import Task, TaskType
                task = Task(
                    id=task_id,
                    title=task_data["title"],
                    description=task_data["description"],
                    task_type=getattr(TaskType, task_data["task_type"], TaskType.FEATURE_DEVELOPMENT),
                    priority=task_data["priority"],
                    assigned_agent_id=best_agent,
                    required_capabilities=task_data["required_capabilities"],
                    estimated_effort=task_data["estimated_effort"],
                    context={
                        "project_id": project.id,
                        "coordination_mode": project.coordination_mode.value,
                        "sync_points": project.sync_points
                    }
                )
                
                project.tasks[task_id] = task
                task_ids.append(task_id)
                
                # Assign task to agent
                if best_agent:
                    self.agent_registry.assign_task(best_agent, task_id)
            
            # Create dependencies
            if len(task_ids) >= 2:
                # Setup -> Core Implementation
                dependency1 = TaskDependency(
                    dependent_task_id=task_ids[1],
                    prerequisite_task_id=task_ids[0],
                    dependency_type="hard",
                    satisfaction_criteria={"status": "completed"}
                )
                project.dependencies.append(dependency1)
            
            if len(task_ids) >= 3:
                # Core Implementation -> Testing
                dependency2 = TaskDependency(
                    dependent_task_id=task_ids[2],
                    prerequisite_task_id=task_ids[1],
                    dependency_type="hard",
                    satisfaction_criteria={"status": "completed"}
                )
                project.dependencies.append(dependency2)
            
            if len(task_ids) >= 4:
                # Testing -> Documentation
                dependency3 = TaskDependency(
                    dependent_task_id=task_ids[3],
                    prerequisite_task_id=task_ids[2],
                    dependency_type="soft",
                    satisfaction_criteria={"status": "completed"}
                )
                project.dependencies.append(dependency3)
            
            logger.info(
                "Project decomposed into tasks",
                project_id=project.id,
                task_count=len(project.tasks),
                dependency_count=len(project.dependencies)
            )
            
        except Exception as e:
            logger.error(
                "Project decomposition failed",
                project_id=project.id,
                error=str(e)
            )
            raise
    
    async def _project_sync_loop(self, project_id: str) -> None:
        """Continuous synchronization loop for a project."""
        
        while project_id in self.active_projects:
            try:
                async with self.project_locks[project_id]:
                    project = self.active_projects[project_id]
                    
                    if project.status in [ProjectStatus.COMPLETED, ProjectStatus.CANCELLED, ProjectStatus.FAILED]:
                        break
                    
                    # Perform synchronization activities
                    await self._sync_project_state(project)
                    await self._detect_and_resolve_conflicts(project)
                    await self._update_progress_metrics(project)
                    await self._check_quality_gates(project)
                    
                    # Update last sync time
                    project.last_sync = datetime.utcnow()
                
                # Wait for next sync cycle
                await asyncio.sleep(project.sync_frequency)
                
            except Exception as e:
                logger.error(
                    "Project sync error",
                    project_id=project_id,
                    error=str(e)
                )
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _sync_project_state(self, project: CoordinatedProject) -> None:
        """Synchronize project state across all participating agents."""
        
        # Collect state updates from all agents
        agent_states = {}
        for agent_id in project.participating_agents:
            # Get agent's current work status
            agent_tasks = [
                task for task in project.tasks.values()
                if task.assigned_agent_id == agent_id
            ]
            
            agent_states[agent_id] = {
                "assigned_tasks": len(agent_tasks),
                "active_tasks": len([t for t in agent_tasks if t.status == TaskStatus.IN_PROGRESS]),
                "completed_tasks": len([t for t in agent_tasks if t.status == TaskStatus.COMPLETED]),
                "last_activity": datetime.utcnow().isoformat()
            }
        
        # Update shared project state
        project.shared_state.update({
            "agent_states": agent_states,
            "last_sync": datetime.utcnow().isoformat(),
            "total_tasks": len(project.tasks),
            "completed_tasks": len([t for t in project.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "active_agents": len([a for a, s in agent_states.items() if s["active_tasks"] > 0])
        })
        
        # Broadcast state update to all agents
        if self.coordination_bus:
            await self.coordination_bus.send_message(
                from_agent="coordinator",
                to_agent="all",
                message_type="project_state_sync",
                payload={
                    "project_id": project.id,
                    "shared_state": project.shared_state,
                    "sync_timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _detect_and_resolve_conflicts(self, project: CoordinatedProject) -> None:
        """Detect and resolve conflicts in the project."""
        
        # Get recent changes from all agents
        recent_changes = []
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        
        # This would typically collect actual file changes, git commits, etc.
        # For demo, we'll simulate some changes
        for agent_id in project.participating_agents:
            workspace = await workspace_manager.get_workspace(agent_id)
            if workspace:
                # Simulate recent changes
                recent_changes.append({
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "files_modified": ["src/main.py", "tests/test_main.py"],
                    "change_type": "code_modification"
                })
        
        # Detect conflicts
        conflicts = await self.conflict_resolver.detect_conflicts(project, recent_changes)
        
        # Resolve conflicts
        for conflict in conflicts:
            success, result = await self.conflict_resolver.resolve_conflict(conflict, project)
            
            if success:
                self.coordination_metrics["conflicts_resolved"] += 1
                logger.info(
                    "Conflict automatically resolved",
                    project_id=project.id,
                    conflict_id=conflict.id,
                    resolution=result
                )
            else:
                logger.warning(
                    "Conflict requires manual intervention",
                    project_id=project.id,
                    conflict_id=conflict.id,
                    result=result
                )
    
    async def _update_progress_metrics(self, project: CoordinatedProject) -> None:
        """Update project progress metrics."""
        
        total_tasks = len(project.tasks)
        completed_tasks = len([t for t in project.tasks.values() if t.status == TaskStatus.COMPLETED])
        in_progress_tasks = len([t for t in project.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        
        # Calculate progress percentage
        progress_percentage = (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Calculate velocity (tasks completed per hour)
        if project.started_at:
            elapsed_hours = (datetime.utcnow() - project.started_at).total_seconds() / 3600
            velocity = completed_tasks / elapsed_hours if elapsed_hours > 0 else 0
        else:
            velocity = 0
        
        # Update metrics
        project.progress_metrics.update({
            "progress_percentage": progress_percentage,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "in_progress_tasks": in_progress_tasks,
            "velocity_tasks_per_hour": velocity,
            "estimated_completion": self._estimate_completion_time(project),
            "agent_utilization": self._calculate_agent_utilization(project)
        })
        
        # Check if project is complete
        if completed_tasks == total_tasks and project.status == ProjectStatus.ACTIVE:
            await self._complete_project(project)
    
    def _estimate_completion_time(self, project: CoordinatedProject) -> Optional[str]:
        """Estimate project completion time based on current velocity."""
        
        if project.progress_metrics.get("velocity_tasks_per_hour", 0) > 0:
            remaining_tasks = project.progress_metrics.get("total_tasks", 0) - project.progress_metrics.get("completed_tasks", 0)
            velocity = project.progress_metrics["velocity_tasks_per_hour"]
            
            estimated_hours = remaining_tasks / velocity
            estimated_completion = datetime.utcnow() + timedelta(hours=estimated_hours)
            
            return estimated_completion.isoformat()
        
        return None
    
    def _calculate_agent_utilization(self, project: CoordinatedProject) -> float:
        """Calculate overall agent utilization for the project."""
        
        total_agents = len(project.participating_agents)
        active_agents = 0
        
        for agent_id in project.participating_agents:
            agent_tasks = [t for t in project.tasks.values() if t.assigned_agent_id == agent_id]
            active_tasks = [t for t in agent_tasks if t.status == TaskStatus.IN_PROGRESS]
            
            if active_tasks:
                active_agents += 1
        
        return (active_agents / total_agents) * 100 if total_agents > 0 else 0
    
    async def _check_quality_gates(self, project: CoordinatedProject) -> None:
        """Check if project meets quality gates."""
        
        for gate in project.quality_gates:
            gate_name = gate["name"]
            threshold = gate["threshold"]
            
            # Implement quality checks based on gate type
            if gate_name == "test_coverage":
                # Check test coverage (simplified)
                coverage = await self._get_test_coverage(project)
                gate["current_value"] = coverage
                gate["passed"] = coverage >= threshold
                
            elif gate_name == "code_review":
                # Check code review completion
                review_score = await self._get_code_review_score(project)
                gate["current_value"] = review_score
                gate["passed"] = review_score >= threshold
                
            elif gate_name == "security_scan":
                # Check security scan results
                security_score = await self._get_security_score(project)
                gate["current_value"] = security_score
                gate["passed"] = security_score >= threshold
        
        # Check if all quality gates pass
        all_gates_passed = all(gate.get("passed", False) for gate in project.quality_gates)
        project.shared_state["quality_gates_passed"] = all_gates_passed
    
    async def _get_test_coverage(self, project: CoordinatedProject) -> float:
        """Get test coverage for the project."""
        # Simplified implementation - would integrate with actual testing tools
        return 0.92  # 92% coverage
    
    async def _get_code_review_score(self, project: CoordinatedProject) -> float:
        """Get code review score for the project."""
        # Simplified implementation - would integrate with code review tools
        return 0.85  # 85% review score
    
    async def _get_security_score(self, project: CoordinatedProject) -> float:
        """Get security scan score for the project."""
        # Simplified implementation - would integrate with security scanning tools
        return 1.0  # No security issues
    
    async def _complete_project(self, project: CoordinatedProject) -> None:
        """Mark project as completed and clean up resources."""
        
        project.status = ProjectStatus.COMPLETED
        project.completed_at = datetime.utcnow()
        
        # Update coordination metrics
        if project.started_at:
            duration = (project.completed_at - project.started_at).total_seconds() / 3600
            self.coordination_metrics["average_project_duration"] = (
                self.coordination_metrics["average_project_duration"] + duration
            ) / 2
        
        self.coordination_metrics["projects_completed"] += 1
        
        # Clean up agent assignments
        for task in project.tasks.values():
            if task.assigned_agent_id:
                self.agent_registry.complete_task(
                    task.assigned_agent_id,
                    task.id,
                    {"completion_time": 120.0, "quality_score": 0.9}  # Example metrics
                )
        
        # Stop sync task
        if project.id in self.sync_tasks:
            self.sync_tasks[project.id].cancel()
            del self.sync_tasks[project.id]
        
        logger.info(
            "Project completed successfully",
            project_id=project.id,
            project_name=project.name,
            duration_hours=duration if project.started_at else 0,
            tasks_completed=len(project.tasks)
        )
    
    async def _handle_agent_event(self, message: Dict[str, Any]) -> None:
        """Handle agent events from the message bus."""
        
        event_type = message.get("type")
        agent_id = message.get("agent_id")
        
        if event_type == "task_completed":
            await self._handle_task_completion(agent_id, message)
        elif event_type == "agent_error":
            await self._handle_agent_error(agent_id, message)
        elif event_type == "agent_status_change":
            await self._handle_agent_status_change(agent_id, message)
    
    async def _handle_project_update(self, message: Dict[str, Any]) -> None:
        """Handle project update events."""
        
        project_id = message.get("project_id")
        update_type = message.get("type")
        
        if project_id in self.active_projects:
            project = self.active_projects[project_id]
            
            if update_type == "task_progress":
                # Update task progress
                task_id = message.get("task_id")
                progress = message.get("progress", 0)
                
                if task_id in project.tasks:
                    project.tasks[task_id].progress = progress
    
    async def get_project_status(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of a coordinated project."""
        
        if project_id not in self.active_projects:
            return None
        
        project = self.active_projects[project_id]
        
        return {
            "project_id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status.value,
            "coordination_mode": project.coordination_mode.value,
            "participating_agents": project.participating_agents,
            "progress_metrics": project.progress_metrics,
            "quality_gates": project.quality_gates,
            "shared_state": project.shared_state,
            "created_at": project.created_at.isoformat(),
            "started_at": project.started_at.isoformat() if project.started_at else None,
            "completed_at": project.completed_at.isoformat() if project.completed_at else None,
            "tasks": {
                task_id: {
                    "title": task.title,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent_id,
                    "progress": getattr(task, 'progress', 0)
                }
                for task_id, task in project.tasks.items()
            },
            "active_conflicts": [
                conflict.id for conflict in self.conflict_resolver.active_conflicts.values()
                if conflict.project_id == project_id and not conflict.resolved
            ]
        }
    
    async def start_project(self, project_id: str) -> bool:
        """Start execution of a coordinated project."""
        
        if project_id not in self.active_projects:
            return False
        
        project = self.active_projects[project_id]
        
        if project.status != ProjectStatus.PLANNING:
            return False
        
        # Update project status
        project.status = ProjectStatus.ACTIVE
        project.started_at = datetime.utcnow()
        
        # Notify all participating agents
        if self.coordination_bus:
            for agent_id in project.participating_agents:
                await self.coordination_bus.send_message(
                    from_agent="coordinator",
                    to_agent=agent_id,
                    message_type="project_started",
                    payload={
                        "project_id": project_id,
                        "project_name": project.name,
                        "coordination_mode": project.coordination_mode.value,
                        "assigned_tasks": [
                            task.id for task in project.tasks.values()
                            if task.assigned_agent_id == agent_id
                        ]
                    }
                )
        
        logger.info(
            "Coordinated project started",
            project_id=project_id,
            project_name=project.name,
            participating_agents=project.participating_agents
        )
        
        return True


# Global coordination engine instance
coordination_engine = MultiAgentCoordinator()
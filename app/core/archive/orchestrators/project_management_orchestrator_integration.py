"""
Project Management Integration for SimpleOrchestrator

Integrates the comprehensive project management system with the existing
SimpleOrchestrator, providing seamless task routing, agent assignment,
and workflow coordination.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc

from .simple_orchestrator import SimpleOrchestrator
from .kanban_state_machine import KanbanStateMachine, StateTransitionResult
from .intelligent_task_router import IntelligentTaskRouter, TaskRoutingContext, RoutingStrategy
from .capability_matcher import CapabilityMatcher
from ..models.project_management import (
    Project, Epic, PRD, Task as ProjectTask, KanbanState, 
    TaskType, TaskPriority
)
from ..models.task import Task as LegacyTask, TaskStatus
from ..models.agent import Agent
from ..services.project_management_service import ProjectManagementService
from .logging_service import get_component_logger

logger = get_component_logger("project_management_orchestrator")


@dataclass
class TaskMigrationMapping:
    """Mapping between legacy and new task systems."""
    
    legacy_task_id: uuid.UUID
    project_task_id: uuid.UUID
    migration_status: str  # 'pending', 'migrated', 'failed'
    migration_date: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class WorkflowTransitionEvent:
    """Event triggered by workflow transitions."""
    
    entity_type: str
    entity_id: uuid.UUID
    old_state: KanbanState
    new_state: KanbanState
    triggered_by: Optional[uuid.UUID] = None
    automation_actions: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.automation_actions is None:
            self.automation_actions = []


class ProjectManagementOrchestratorIntegration:
    """
    Integration layer between project management system and SimpleOrchestrator.
    
    Provides bidirectional synchronization, workflow automation, and 
    intelligent task routing across both systems.
    """
    
    def __init__(self, db_session: Session, simple_orchestrator: SimpleOrchestrator):
        """
        Initialize the integration.
        
        Args:
            db_session: Database session
            simple_orchestrator: Instance of SimpleOrchestrator
        """
        self.db_session = db_session
        self.orchestrator = simple_orchestrator
        self.kanban_machine = KanbanStateMachine(db_session)
        self.project_service = ProjectManagementService(db_session)
        self.task_router = IntelligentTaskRouter(db_session)
        self.capability_matcher = CapabilityMatcher(db_session)
        
        # Migration tracking
        self.migration_mappings: Dict[uuid.UUID, TaskMigrationMapping] = {}
        
        # Event handlers
        self.workflow_event_handlers: List[callable] = []
        
        # Integration status
        self.is_initialized = False
    
    def initialize_integration(self) -> None:
        """Initialize the integration and set up event handlers."""
        try:
            # Register workflow event handlers
            self._register_workflow_event_handlers()
            
            # Set up task synchronization
            self._setup_task_synchronization()
            
            # Initialize migration tracking
            self._initialize_migration_tracking()
            
            self.is_initialized = True
            logger.info("Project management orchestrator integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize project management integration: {e}")
            raise
    
    # Task Management Integration
    
    def create_project_task_from_legacy(
        self,
        legacy_task: LegacyTask,
        prd_id: uuid.UUID,
        auto_assign: bool = True
    ) -> ProjectTask:
        """
        Create a project management task from a legacy task.
        
        Args:
            legacy_task: Legacy task to convert
            prd_id: PRD ID to associate with
            auto_assign: Whether to auto-assign based on legacy assignment
            
        Returns:
            Created project task
        """
        try:
            # Map legacy task properties to new system
            project_task = ProjectTask(
                title=legacy_task.title,
                description=legacy_task.description,
                prd_id=prd_id,
                task_type=self._map_legacy_task_type(legacy_task.task_type),
                priority=self._map_legacy_priority(legacy_task.priority),
                estimated_effort_minutes=legacy_task.estimated_effort or None,
                context=legacy_task.context or {},
                required_capabilities=legacy_task.required_capabilities or [],
                kanban_state=self._map_legacy_status_to_kanban(legacy_task.status)
            )
            
            # Auto-assign if requested and legacy task was assigned
            if auto_assign and legacy_task.assigned_agent_id:
                project_task.assigned_agent_id = legacy_task.assigned_agent_id
            
            # Generate short ID and save
            project_task.ensure_short_id(self.db_session)
            self.db_session.add(project_task)
            self.db_session.flush()
            
            # Track migration
            mapping = TaskMigrationMapping(
                legacy_task_id=legacy_task.id,
                project_task_id=project_task.id,
                migration_status='migrated',
                migration_date=datetime.utcnow()
            )
            self.migration_mappings[legacy_task.id] = mapping
            
            logger.info(f"Created project task {project_task.get_display_id()} from legacy task {legacy_task.id}")
            return project_task
            
        except Exception as e:
            logger.error(f"Failed to create project task from legacy: {e}")
            raise
    
    def sync_task_assignment(
        self,
        project_task: ProjectTask,
        agent_id: Optional[uuid.UUID] = None,
        sync_to_legacy: bool = True
    ) -> None:
        """
        Synchronize task assignment between systems.
        
        Args:
            project_task: Project task to sync
            agent_id: Agent to assign to (if None, unassign)
            sync_to_legacy: Whether to sync back to legacy system
        """
        try:
            old_agent_id = project_task.assigned_agent_id
            
            # Update project task assignment
            if agent_id:
                project_task.assign_to_agent(agent_id)
            else:
                project_task.assigned_agent_id = None
                project_task.assigned_at = None
            
            self.db_session.flush()
            
            # Sync to legacy system if mapping exists
            if sync_to_legacy:
                legacy_mapping = self._find_legacy_mapping(project_task.id)
                if legacy_mapping:
                    legacy_task = self.db_session.query(LegacyTask).filter(
                        LegacyTask.id == legacy_mapping.legacy_task_id
                    ).first()
                    
                    if legacy_task:
                        legacy_task.assigned_agent_id = agent_id
                        if agent_id:
                            legacy_task.assigned_at = datetime.utcnow()
                            if legacy_task.status == TaskStatus.PENDING:
                                legacy_task.status = TaskStatus.ASSIGNED
            
            # Notify orchestrator of assignment change
            if self.orchestrator:
                self.orchestrator._handle_task_assignment_change(
                    project_task.id, old_agent_id, agent_id
                )
            
            logger.info(f"Synced task assignment: {project_task.get_display_id()} -> {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to sync task assignment: {e}")
            raise
    
    def intelligent_task_routing(
        self,
        project_task: ProjectTask,
        routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
        consider_workload: bool = True
    ) -> Optional[Agent]:
        """
        Use intelligent routing to assign project tasks.
        
        Args:
            project_task: Task to route
            routing_strategy: Routing strategy to use
            consider_workload: Whether to consider agent workload
            
        Returns:
            Best agent for the task, or None if none suitable
        """
        try:
            # Create routing context
            context = TaskRoutingContext(
                task_id=project_task.id,
                task_type=project_task.task_type.value,
                priority=project_task.priority.value,
                required_capabilities=project_task.required_capabilities or [],
                estimated_effort=project_task.estimated_effort_minutes or 60,
                due_date=project_task.due_date,
                dependencies=project_task.dependencies or [],
                context_data=project_task.context or {}
            )
            
            # Get routing recommendation
            result = self.task_router.route_task(context, routing_strategy)
            
            if result and result.recommended_agent:
                # Assign the task
                self.sync_task_assignment(project_task, result.recommended_agent.id)
                
                logger.info(f"Intelligently routed task {project_task.get_display_id()} to agent {result.recommended_agent.id}")
                return result.recommended_agent
            
            logger.warning(f"No suitable agent found for task {project_task.get_display_id()}")
            return None
            
        except Exception as e:
            logger.error(f"Intelligent task routing failed: {e}")
            return None
    
    # Workflow Integration
    
    def handle_workflow_transition(
        self,
        entity,
        old_state: KanbanState,
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID] = None
    ) -> List[str]:
        """
        Handle workflow state transitions and trigger orchestrator actions.
        
        Args:
            entity: Entity that transitioned
            old_state: Previous kanban state
            new_state: New kanban state
            agent_id: Agent who triggered transition
            
        Returns:
            List of automation actions performed
        """
        try:
            automation_actions = []
            
            # Create workflow event
            event = WorkflowTransitionEvent(
                entity_type=entity.__class__.__name__,
                entity_id=entity.id,
                old_state=old_state,
                new_state=new_state,
                triggered_by=agent_id
            )
            
            # Handle task-specific transitions
            if isinstance(entity, ProjectTask):
                actions = self._handle_task_workflow_transition(entity, old_state, new_state, agent_id)
                automation_actions.extend(actions)
                
                # Update orchestrator workload tracking
                if self.orchestrator and entity.assigned_agent_id:
                    self.orchestrator._update_agent_workload(entity.assigned_agent_id)
            
            # Handle PRD transitions
            elif isinstance(entity, PRD):
                actions = self._handle_prd_workflow_transition(entity, old_state, new_state, agent_id)
                automation_actions.extend(actions)
            
            # Handle Epic transitions
            elif isinstance(entity, Epic):
                actions = self._handle_epic_workflow_transition(entity, old_state, new_state, agent_id)
                automation_actions.extend(actions)
            
            # Handle Project transitions
            elif isinstance(entity, Project):
                actions = self._handle_project_workflow_transition(entity, old_state, new_state, agent_id)
                automation_actions.extend(actions)
            
            event.automation_actions = automation_actions
            
            # Notify event handlers
            for handler in self.workflow_event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.warning(f"Workflow event handler failed: {e}")
            
            return automation_actions
            
        except Exception as e:
            logger.error(f"Failed to handle workflow transition: {e}")
            return []
    
    def auto_advance_workflow(self, limit: int = 50) -> Dict[str, int]:
        """
        Automatically advance workflow items that are ready to progress.
        
        Args:
            limit: Maximum items to process per type
            
        Returns:
            Dictionary with counts of advanced items by type
        """
        try:
            results = {"tasks": 0, "prds": 0, "epics": 0, "projects": 0}
            
            # Auto-advance tasks
            task_results = self.project_service.auto_advance_ready_work(limit)
            results["tasks"] = len(task_results.get("tasks", []))
            results["prds"] = len(task_results.get("prds", []))
            results["epics"] = len(task_results.get("epics", []))
            results["projects"] = len(task_results.get("projects", []))
            
            # Sync with orchestrator
            if self.orchestrator:
                self.orchestrator._trigger_workflow_evaluation()
            
            logger.info(f"Auto-advanced workflow: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Auto-advance workflow failed: {e}")
            return {"tasks": 0, "prds": 0, "epics": 0, "projects": 0}
    
    # Agent Workload Management
    
    def get_agent_project_workload(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get comprehensive workload analysis for an agent across project hierarchy.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Workload analysis dictionary
        """
        try:
            # Get project tasks assigned to agent
            project_tasks = self.db_session.query(ProjectTask).filter(
                ProjectTask.assigned_agent_id == agent_id
            ).all()
            
            # Get legacy tasks for comparison
            legacy_tasks = self.db_session.query(LegacyTask).filter(
                LegacyTask.assigned_agent_id == agent_id
            ).all()
            
            # Analyze workload
            workload = {
                "agent_id": str(agent_id),
                "project_tasks": {
                    "total": len(project_tasks),
                    "by_state": {},
                    "by_priority": {},
                    "estimated_hours": 0
                },
                "legacy_tasks": {
                    "total": len(legacy_tasks),
                    "by_status": {},
                    "estimated_hours": 0
                },
                "workload_score": 0.0,
                "recommendations": []
            }
            
            # Analyze project tasks
            for task in project_tasks:
                state = task.kanban_state.value
                priority = task.priority.name
                
                workload["project_tasks"]["by_state"][state] = workload["project_tasks"]["by_state"].get(state, 0) + 1
                workload["project_tasks"]["by_priority"][priority] = workload["project_tasks"]["by_priority"].get(priority, 0) + 1
                
                if task.estimated_effort_minutes:
                    workload["project_tasks"]["estimated_hours"] += task.estimated_effort_minutes / 60.0
            
            # Analyze legacy tasks
            for task in legacy_tasks:
                status = task.status.value
                workload["legacy_tasks"]["by_status"][status] = workload["legacy_tasks"]["by_status"].get(status, 0) + 1
                
                if task.estimated_effort:
                    workload["legacy_tasks"]["estimated_hours"] += task.estimated_effort / 60.0
            
            # Calculate workload score
            total_hours = workload["project_tasks"]["estimated_hours"] + workload["legacy_tasks"]["estimated_hours"]
            workload["workload_score"] = min(1.0, total_hours / 40.0)  # Assume 40 hour capacity
            
            # Generate recommendations
            if workload["workload_score"] > 0.8:
                workload["recommendations"].append("Agent is near capacity - consider redistributing tasks")
            
            in_progress_count = workload["project_tasks"]["by_state"].get("in_progress", 0)
            if in_progress_count > 3:
                workload["recommendations"].append("Too many tasks in progress - focus on completion")
            
            return workload
            
        except Exception as e:
            logger.error(f"Failed to get agent workload: {e}")
            return {"agent_id": str(agent_id), "error": str(e)}
    
    def rebalance_workloads(
        self,
        max_workload_score: float = 0.8,
        min_workload_score: float = 0.2
    ) -> Dict[str, Any]:
        """
        Rebalance workloads across agents by reassigning tasks.
        
        Args:
            max_workload_score: Maximum allowed workload score
            min_workload_score: Minimum workload score to consider for receiving tasks
            
        Returns:
            Rebalancing results
        """
        try:
            # Get all active agents
            agents = self.db_session.query(Agent).filter(
                Agent.status == "active"
            ).all()
            
            if not agents:
                return {"error": "No active agents available"}
            
            # Analyze workloads
            agent_workloads = {}
            for agent in agents:
                workload = self.get_agent_project_workload(agent.id)
                agent_workloads[agent.id] = workload
            
            # Identify overloaded and underutilized agents
            overloaded_agents = [
                agent_id for agent_id, workload in agent_workloads.items()
                if workload.get("workload_score", 0) > max_workload_score
            ]
            
            available_agents = [
                agent_id for agent_id, workload in agent_workloads.items()
                if workload.get("workload_score", 0) < min_workload_score
            ]
            
            if not overloaded_agents or not available_agents:
                return {"message": "No rebalancing needed", "overloaded": len(overloaded_agents), "available": len(available_agents)}
            
            # Reassign tasks
            reassignments = []
            for overloaded_agent in overloaded_agents:
                # Get reassignable tasks (READY or BACKLOG state)
                tasks_to_reassign = self.db_session.query(ProjectTask).filter(
                    and_(
                        ProjectTask.assigned_agent_id == overloaded_agent,
                        ProjectTask.kanban_state.in_([KanbanState.READY, KanbanState.BACKLOG])
                    )
                ).limit(3).all()  # Limit reassignments per agent
                
                for task in tasks_to_reassign:
                    if available_agents:
                        new_agent = available_agents.pop(0)
                        old_agent = task.assigned_agent_id
                        
                        self.sync_task_assignment(task, new_agent, sync_to_legacy=True)
                        
                        reassignments.append({
                            "task_id": task.get_display_id(),
                            "from_agent": str(old_agent),
                            "to_agent": str(new_agent)
                        })
                        
                        # Update available capacity
                        estimated_hours = (task.estimated_effort_minutes or 60) / 60.0
                        agent_workloads[new_agent]["workload_score"] += estimated_hours / 40.0
                        
                        if agent_workloads[new_agent]["workload_score"] >= min_workload_score:
                            break
            
            self.db_session.commit()
            
            return {
                "reassignments": reassignments,
                "count": len(reassignments),
                "overloaded_agents": len(overloaded_agents),
                "available_agents": len([aid for aid in agent_workloads if agent_workloads[aid]["workload_score"] < min_workload_score])
            }
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to rebalance workloads: {e}")
            return {"error": str(e)}
    
    # Integration Status and Health
    
    def get_integration_health(self) -> Dict[str, Any]:
        """Get health status of the integration."""
        try:
            health = {
                "initialized": self.is_initialized,
                "timestamp": datetime.utcnow().isoformat(),
                "components": {
                    "kanban_machine": self.kanban_machine is not None,
                    "project_service": self.project_service is not None,
                    "task_router": self.task_router is not None,
                    "orchestrator": self.orchestrator is not None
                },
                "statistics": {
                    "migration_mappings": len(self.migration_mappings),
                    "event_handlers": len(self.workflow_event_handlers)
                },
                "database": {
                    "session_active": self.db_session.is_active if self.db_session else False
                }
            }
            
            # Get entity counts
            if self.db_session:
                health["entity_counts"] = {
                    "projects": self.db_session.query(Project).count(),
                    "epics": self.db_session.query(Epic).count(),
                    "prds": self.db_session.query(PRD).count(),
                    "project_tasks": self.db_session.query(ProjectTask).count(),
                    "legacy_tasks": self.db_session.query(LegacyTask).count()
                }
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to get integration health: {e}")
            return {"error": str(e), "initialized": False}
    
    # Private helper methods
    
    def _register_workflow_event_handlers(self) -> None:
        """Register workflow event handlers."""
        self.workflow_event_handlers.extend([
            self._log_workflow_events,
            self._update_orchestrator_metrics,
            self._trigger_automation_rules
        ])
    
    def _setup_task_synchronization(self) -> None:
        """Set up bidirectional task synchronization."""
        # This would set up database triggers or event listeners
        # to keep legacy and project tasks synchronized
        pass
    
    def _initialize_migration_tracking(self) -> None:
        """Initialize migration tracking from existing data."""
        # Load existing migration mappings from database or create tracking table
        pass
    
    def _map_legacy_task_type(self, legacy_type) -> TaskType:
        """Map legacy task type to new TaskType enum."""
        type_mapping = {
            "FEATURE_DEVELOPMENT": TaskType.FEATURE_DEVELOPMENT,
            "BUG_FIX": TaskType.BUG_FIX,
            "REFACTORING": TaskType.REFACTORING,
            "TESTING": TaskType.TESTING,
            "DOCUMENTATION": TaskType.DOCUMENTATION,
            "ARCHITECTURE": TaskType.ARCHITECTURE,
            "DEPLOYMENT": TaskType.DEPLOYMENT,
            "CODE_REVIEW": TaskType.CODE_REVIEW,
            "RESEARCH": TaskType.RESEARCH,
            "OPTIMIZATION": TaskType.OPTIMIZATION
        }
        
        if hasattr(legacy_type, 'value'):
            return type_mapping.get(legacy_type.value, TaskType.FEATURE_DEVELOPMENT)
        return type_mapping.get(str(legacy_type), TaskType.FEATURE_DEVELOPMENT)
    
    def _map_legacy_priority(self, legacy_priority) -> TaskPriority:
        """Map legacy priority to new TaskPriority enum."""
        if hasattr(legacy_priority, 'value'):
            value = legacy_priority.value
        else:
            value = str(legacy_priority)
        
        priority_mapping = {
            "LOW": TaskPriority.LOW,
            "MEDIUM": TaskPriority.MEDIUM,
            "HIGH": TaskPriority.HIGH,
            "CRITICAL": TaskPriority.CRITICAL,
            "1": TaskPriority.LOW,
            "5": TaskPriority.MEDIUM,
            "8": TaskPriority.HIGH,
            "10": TaskPriority.CRITICAL
        }
        
        return priority_mapping.get(str(value), TaskPriority.MEDIUM)
    
    def _map_legacy_status_to_kanban(self, legacy_status: TaskStatus) -> KanbanState:
        """Map legacy task status to kanban state."""
        status_mapping = {
            TaskStatus.PENDING: KanbanState.BACKLOG,
            TaskStatus.ASSIGNED: KanbanState.READY,
            TaskStatus.IN_PROGRESS: KanbanState.IN_PROGRESS,
            TaskStatus.BLOCKED: KanbanState.BLOCKED,
            TaskStatus.COMPLETED: KanbanState.DONE,
            TaskStatus.FAILED: KanbanState.BLOCKED,
            TaskStatus.CANCELLED: KanbanState.CANCELLED
        }
        
        return status_mapping.get(legacy_status, KanbanState.BACKLOG)
    
    def _find_legacy_mapping(self, project_task_id: uuid.UUID) -> Optional[TaskMigrationMapping]:
        """Find legacy task mapping for a project task."""
        for mapping in self.migration_mappings.values():
            if mapping.project_task_id == project_task_id:
                return mapping
        return None
    
    def _handle_task_workflow_transition(
        self,
        task: ProjectTask,
        old_state: KanbanState,
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID]
    ) -> List[str]:
        """Handle task workflow transitions."""
        actions = []
        
        # Auto-assign if moving to IN_PROGRESS and no assignee
        if new_state == KanbanState.IN_PROGRESS and not task.assigned_agent_id:
            if agent_id:
                self.sync_task_assignment(task, agent_id)
                actions.append(f"Auto-assigned task to agent {agent_id}")
        
        # Suggest next tasks when task completes
        if new_state == KanbanState.DONE:
            # Find dependent tasks and auto-start if possible
            dependent_tasks = self.db_session.query(ProjectTask).filter(
                ProjectTask.dependencies.contains([task.id])
            ).all()
            
            for dep_task in dependent_tasks:
                if dep_task.kanban_state == KanbanState.READY:
                    # Check if all dependencies are done
                    all_deps_done = True
                    for dep_id in dep_task.dependencies or []:
                        dep = self.db_session.query(ProjectTask).filter(ProjectTask.id == dep_id).first()
                        if not dep or dep.kanban_state != KanbanState.DONE:
                            all_deps_done = False
                            break
                    
                    if all_deps_done:
                        result = self.kanban_machine.transition_entity_state(
                            dep_task, KanbanState.IN_PROGRESS, agent_id, "Auto-started: dependencies complete"
                        )
                        if result.success:
                            actions.append(f"Auto-started dependent task {dep_task.get_display_id()}")
        
        return actions
    
    def _handle_prd_workflow_transition(
        self,
        prd: PRD,
        old_state: KanbanState,
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID]
    ) -> List[str]:
        """Handle PRD workflow transitions."""
        actions = []
        
        # Auto-generate implementation tasks when PRD approved
        if new_state == KanbanState.DONE and old_state in [KanbanState.REVIEW, KanbanState.IN_PROGRESS]:
            if not prd.tasks:  # Only if no tasks exist
                try:
                    generated_tasks = self.project_service.auto_generate_implementation_tasks(
                        prd.id, "standard"
                    )
                    actions.append(f"Auto-generated {len(generated_tasks)} implementation tasks")
                except Exception as e:
                    logger.warning(f"Failed to auto-generate tasks for PRD {prd.get_display_id()}: {e}")
        
        return actions
    
    def _handle_epic_workflow_transition(
        self,
        epic: Epic,
        old_state: KanbanState,
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID]
    ) -> List[str]:
        """Handle Epic workflow transitions."""
        actions = []
        
        # Notify when epic completes
        if new_state == KanbanState.DONE:
            actions.append(f"Epic {epic.get_display_id()} completed - notifying stakeholders")
        
        return actions
    
    def _handle_project_workflow_transition(
        self,
        project: Project,
        old_state: KanbanState,
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID]
    ) -> List[str]:
        """Handle Project workflow transitions."""
        actions = []
        
        # Generate project completion metrics
        if new_state == KanbanState.DONE:
            stats = self.project_service.get_project_hierarchy_stats(project.id)
            actions.append(f"Project completed - {stats.task_count} tasks, {stats.average_task_completion_time_hours:.1f}h avg completion")
        
        return actions
    
    # Event handlers
    
    def _log_workflow_events(self, event: WorkflowTransitionEvent) -> None:
        """Log workflow events for audit trail."""
        logger.info(f"Workflow transition: {event.entity_type} {event.entity_id} {event.old_state.value} -> {event.new_state.value}")
    
    def _update_orchestrator_metrics(self, event: WorkflowTransitionEvent) -> None:
        """Update orchestrator metrics based on workflow events."""
        if self.orchestrator:
            # Update workflow metrics in orchestrator
            pass
    
    def _trigger_automation_rules(self, event: WorkflowTransitionEvent) -> None:
        """Trigger automation rules based on workflow events."""
        # Implement custom automation rules here
        pass


def create_project_management_integration(
    db_session: Session,
    orchestrator: SimpleOrchestrator
) -> ProjectManagementOrchestratorIntegration:
    """
    Factory function to create and initialize project management integration.
    
    Args:
        db_session: Database session
        orchestrator: SimpleOrchestrator instance
        
    Returns:
        Initialized integration instance
    """
    integration = ProjectManagementOrchestratorIntegration(db_session, orchestrator)
    integration.initialize_integration()
    return integration
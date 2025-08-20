"""
Kanban State Machine for LeanVibe Agent Hive 2.0

Manages state transitions, validation rules, and workflow automation
for all project management entities (Projects, Epics, PRDs, Tasks).
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass, field

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ..models.project_management import KanbanState, Project, Epic, PRD, ProjectTask
from .logging_service import get_component_logger

logger = get_component_logger("kanban_state_machine")


@dataclass
class TransitionRule:
    """Defines a state transition rule with validation conditions."""
    
    from_state: KanbanState
    to_state: KanbanState
    conditions: List[Callable] = field(default_factory=list)
    auto_actions: List[Callable] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class StateTransitionResult:
    """Result of a state transition attempt."""
    
    success: bool
    old_state: KanbanState
    new_state: KanbanState
    entity_id: uuid.UUID
    entity_type: str
    timestamp: datetime
    agent_id: Optional[uuid.UUID] = None
    reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    auto_actions_performed: List[str] = field(default_factory=list)


@dataclass
class WorkflowMetrics:
    """Metrics for kanban workflow performance."""
    
    entity_type: str
    total_entities: int
    state_counts: Dict[KanbanState, int]
    average_cycle_time_days: Optional[float] = None
    bottleneck_states: List[KanbanState] = field(default_factory=list)
    throughput_per_day: float = 0.0
    wip_limits_violated: List[str] = field(default_factory=list)


class KanbanStateMachine:
    """
    Central state machine for managing Kanban workflows across all project entities.
    
    Handles state transitions, validation, automation, and metrics collection
    for Projects, Epics, PRDs, and Tasks.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize the Kanban state machine.
        
        Args:
            db_session: Database session for entity operations
        """
        self.db_session = db_session
        self.transition_rules = self._initialize_transition_rules()
        self.wip_limits = self._initialize_wip_limits()
        self.automation_rules = self._initialize_automation_rules()
    
    def _initialize_transition_rules(self) -> Dict[str, List[TransitionRule]]:
        """Initialize state transition rules for each entity type."""
        
        # Common transitions for all entities
        common_rules = [
            TransitionRule(
                from_state=KanbanState.BACKLOG,
                to_state=KanbanState.READY,
                description="Move from backlog to ready"
            ),
            TransitionRule(
                from_state=KanbanState.READY,
                to_state=KanbanState.IN_PROGRESS,
                description="Start work on ready item"
            ),
            TransitionRule(
                from_state=KanbanState.IN_PROGRESS,
                to_state=KanbanState.REVIEW,
                description="Submit for review"
            ),
            TransitionRule(
                from_state=KanbanState.REVIEW,
                to_state=KanbanState.DONE,
                description="Complete reviewed item"
            ),
            TransitionRule(
                from_state=KanbanState.REVIEW,
                to_state=KanbanState.IN_PROGRESS,
                description="Return to development for fixes"
            ),
            TransitionRule(
                from_state=KanbanState.IN_PROGRESS,
                to_state=KanbanState.BLOCKED,
                description="Block due to impediment"
            ),
            TransitionRule(
                from_state=KanbanState.BLOCKED,
                to_state=KanbanState.READY,
                description="Unblock and return to ready"
            ),
            TransitionRule(
                from_state=KanbanState.DONE,
                to_state=KanbanState.REVIEW,
                description="Reopen for additional changes"
            ),
            # Cancellation paths
            TransitionRule(
                from_state=KanbanState.BACKLOG,
                to_state=KanbanState.CANCELLED,
                description="Cancel backlog item"
            ),
            TransitionRule(
                from_state=KanbanState.READY,
                to_state=KanbanState.CANCELLED,
                description="Cancel ready item"
            ),
            TransitionRule(
                from_state=KanbanState.IN_PROGRESS,
                to_state=KanbanState.CANCELLED,
                description="Cancel in-progress item"
            ),
            TransitionRule(
                from_state=KanbanState.CANCELLED,
                to_state=KanbanState.BACKLOG,
                description="Reactivate cancelled item"
            )
        ]
        
        return {
            "Project": common_rules.copy(),
            "Epic": common_rules.copy(),
            "PRD": self._get_prd_specific_rules() + common_rules,
            "Task": self._get_task_specific_rules() + common_rules
        }
    
    def _get_prd_specific_rules(self) -> List[TransitionRule]:
        """Get PRD-specific transition rules."""
        return [
            TransitionRule(
                from_state=KanbanState.REVIEW,
                to_state=KanbanState.DONE,
                conditions=[self._prd_has_approvals],
                description="Complete PRD after approvals"
            )
        ]
    
    def _get_task_specific_rules(self) -> List[TransitionRule]:
        """Get Task-specific transition rules."""
        return [
            TransitionRule(
                from_state=KanbanState.READY,
                to_state=KanbanState.IN_PROGRESS,
                conditions=[self._task_dependencies_met],
                description="Start task when dependencies are met"
            ),
            TransitionRule(
                from_state=KanbanState.REVIEW,
                to_state=KanbanState.DONE,
                conditions=[self._task_quality_gates_passed],
                description="Complete task after quality gates"
            )
        ]
    
    def _initialize_wip_limits(self) -> Dict[str, Dict[KanbanState, int]]:
        """Initialize Work-In-Progress limits for each entity type and state."""
        return {
            "Project": {
                KanbanState.IN_PROGRESS: 5,  # Max 5 active projects
                KanbanState.REVIEW: 3
            },
            "Epic": {
                KanbanState.IN_PROGRESS: 10,
                KanbanState.REVIEW: 5
            },
            "PRD": {
                KanbanState.IN_PROGRESS: 15,
                KanbanState.REVIEW: 8
            },
            "Task": {
                KanbanState.IN_PROGRESS: 50,
                KanbanState.REVIEW: 20
            }
        }
    
    def _initialize_automation_rules(self) -> Dict[str, List[Callable]]:
        """Initialize automation rules that trigger on state changes."""
        return {
            "Task": [
                self._auto_start_dependent_tasks,
                self._auto_update_parent_prd_progress,
                self._auto_notify_assignee
            ],
            "PRD": [
                self._auto_update_parent_epic_progress,
                self._auto_create_implementation_tasks
            ],
            "Epic": [
                self._auto_update_parent_project_progress
            ],
            "Project": [
                self._auto_update_project_metrics
            ]
        }
    
    def transition_entity_state(
        self,
        entity: Union[Project, Epic, PRD, ProjectTask],
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None,
        force: bool = False
    ) -> StateTransitionResult:
        """
        Attempt to transition an entity to a new kanban state.
        
        Args:
            entity: The entity to transition
            new_state: Target kanban state
            agent_id: ID of agent performing transition
            reason: Reason for the transition
            force: Skip validation rules if True
            
        Returns:
            StateTransitionResult with transition details
        """
        entity_type = entity.__class__.__name__
        old_state = entity.kanban_state
        
        result = StateTransitionResult(
            success=False,
            old_state=old_state,
            new_state=new_state,
            entity_id=entity.id,
            entity_type=entity_type,
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            reason=reason
        )
        
        # Check if transition is valid
        if not force and not self._is_valid_transition(entity_type, old_state, new_state):
            result.errors.append(f"Invalid transition from {old_state.value} to {new_state.value}")
            return result
        
        # Check WIP limits
        if not force and not self._check_wip_limits(entity_type, new_state):
            result.errors.append(f"WIP limit exceeded for {new_state.value} state")
            return result
        
        # Validate conditions
        if not force:
            rule = self._find_transition_rule(entity_type, old_state, new_state)
            if rule and not self._validate_conditions(entity, rule):
                result.errors.append("Transition conditions not met")
                return result
        
        try:
            # Perform the transition
            entity.kanban_state = new_state
            
            # Update entity-specific timestamps
            self._update_entity_timestamps(entity, old_state, new_state)
            
            # Record transition in entity's history
            if hasattr(entity, 'state_history'):
                transition_record = {
                    'from_state': old_state.value,
                    'to_state': new_state.value,
                    'timestamp': result.timestamp.isoformat(),
                    'agent_id': str(agent_id) if agent_id else None,
                    'reason': reason
                }
                
                if entity.state_history is None:
                    entity.state_history = []
                entity.state_history.append(transition_record)
            
            # Perform automation actions
            auto_actions = self._perform_automation_actions(entity, old_state, new_state)
            result.auto_actions_performed = auto_actions
            
            # Commit to database
            self.db_session.add(entity)
            self.db_session.commit()
            
            result.success = True
            
            logger.info(f"State transition successful", extra={
                'entity_type': entity_type,
                'entity_id': str(entity.id),
                'from_state': old_state.value,
                'to_state': new_state.value,
                'agent_id': str(agent_id) if agent_id else None
            })
            
        except Exception as e:
            self.db_session.rollback()
            result.errors.append(f"Database error: {str(e)}")
            logger.error(f"State transition failed", extra={
                'entity_type': entity_type,
                'entity_id': str(entity.id),
                'error': str(e)
            })
        
        return result
    
    def bulk_transition_entities(
        self,
        entities: List[Union[Project, Epic, PRD, ProjectTask]],
        new_state: KanbanState,
        agent_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None
    ) -> List[StateTransitionResult]:
        """
        Bulk transition multiple entities to a new state.
        
        Args:
            entities: List of entities to transition
            new_state: Target kanban state
            agent_id: ID of agent performing transitions
            reason: Reason for the transitions
            
        Returns:
            List of StateTransitionResult for each entity
        """
        results = []
        
        for entity in entities:
            result = self.transition_entity_state(
                entity, new_state, agent_id, reason
            )
            results.append(result)
        
        return results
    
    def get_workflow_metrics(
        self,
        entity_type: str,
        date_range_days: int = 30
    ) -> WorkflowMetrics:
        """
        Calculate workflow metrics for an entity type.
        
        Args:
            entity_type: Type of entity (Project, Epic, PRD, Task)
            date_range_days: Number of days to include in metrics
            
        Returns:
            WorkflowMetrics with performance data
        """
        model_class = self._get_model_class(entity_type)
        
        if not model_class:
            return WorkflowMetrics(
                entity_type=entity_type,
                total_entities=0,
                state_counts={}
            )
        
        # Get entities within date range
        start_date = datetime.utcnow() - timedelta(days=date_range_days)
        
        entities = self.db_session.query(model_class).filter(
            model_class.created_at >= start_date
        ).all()
        
        # Calculate state distribution
        state_counts = {state: 0 for state in KanbanState}
        for entity in entities:
            state_counts[entity.kanban_state] += 1
        
        # Calculate cycle times for completed entities
        cycle_times = []
        throughput_count = 0
        
        for entity in entities:
            if (entity.kanban_state == KanbanState.DONE and 
                hasattr(entity, 'actual_start') and hasattr(entity, 'actual_completion') and
                entity.actual_start and entity.actual_completion):
                
                cycle_time = (entity.actual_completion - entity.actual_start).days
                cycle_times.append(cycle_time)
                throughput_count += 1
        
        # Calculate metrics
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else None
        throughput_per_day = throughput_count / date_range_days if date_range_days > 0 else 0
        
        # Identify bottlenecks (states with high WIP)
        bottleneck_states = []
        wip_limits = self.wip_limits.get(entity_type, {})
        
        for state, count in state_counts.items():
            limit = wip_limits.get(state, float('inf'))
            if count > limit * 0.8:  # 80% of limit
                bottleneck_states.append(state)
        
        # Check WIP violations
        wip_violations = []
        for state, limit in wip_limits.items():
            if state_counts[state] > limit:
                wip_violations.append(f"{state.value}: {state_counts[state]}/{limit}")
        
        return WorkflowMetrics(
            entity_type=entity_type,
            total_entities=len(entities),
            state_counts=state_counts,
            average_cycle_time_days=avg_cycle_time,
            bottleneck_states=bottleneck_states,
            throughput_per_day=throughput_per_day,
            wip_limits_violated=wip_violations
        )
    
    def get_entities_by_state(
        self,
        entity_type: str,
        state: KanbanState,
        limit: int = 100
    ) -> List[Union[Project, Epic, PRD, ProjectTask]]:
        """Get entities of a specific type in a specific state."""
        model_class = self._get_model_class(entity_type)
        
        if not model_class:
            return []
        
        return self.db_session.query(model_class).filter(
            model_class.kanban_state == state
        ).limit(limit).all()
    
    def get_blocked_entities(
        self,
        entity_type: Optional[str] = None
    ) -> List[Union[Project, Epic, PRD, ProjectTask]]:
        """Get all blocked entities, optionally filtered by type."""
        entities = []
        
        model_classes = []
        if entity_type:
            model_class = self._get_model_class(entity_type)
            if model_class:
                model_classes = [model_class]
        else:
            model_classes = [Project, Epic, PRD, ProjectTask]
        
        for model_class in model_classes:
            blocked = self.db_session.query(model_class).filter(
                model_class.kanban_state == KanbanState.BLOCKED
            ).all()
            entities.extend(blocked)
        
        return entities
    
    def auto_transition_ready_tasks(self, limit: int = 50) -> List[StateTransitionResult]:
        """
        Automatically transition tasks from READY to IN_PROGRESS when conditions are met.
        
        Args:
            limit: Maximum number of tasks to process
            
        Returns:
            List of transition results
        """
        ready_tasks = self.get_entities_by_state("Task", KanbanState.READY, limit)
        results = []
        
        for task in ready_tasks:
            # Check if task can be auto-started
            if (self._task_dependencies_met(task) and 
                not self._check_wip_limits("Task", KanbanState.IN_PROGRESS)):
                
                result = self.transition_entity_state(
                    task,
                    KanbanState.IN_PROGRESS,
                    reason="Auto-transition: dependencies met"
                )
                results.append(result)
        
        return results
    
    # Validation methods
    
    def _is_valid_transition(self, entity_type: str, from_state: KanbanState, to_state: KanbanState) -> bool:
        """Check if a state transition is valid for the entity type."""
        rules = self.transition_rules.get(entity_type, [])
        
        for rule in rules:
            if rule.from_state == from_state and rule.to_state == to_state:
                return True
        
        return False
    
    def _check_wip_limits(self, entity_type: str, state: KanbanState) -> bool:
        """Check if transitioning to a state would violate WIP limits."""
        limits = self.wip_limits.get(entity_type, {})
        limit = limits.get(state)
        
        if limit is None:
            return True  # No limit set
        
        # Count current entities in this state
        model_class = self._get_model_class(entity_type)
        if not model_class:
            return True
        
        current_count = self.db_session.query(model_class).filter(
            model_class.kanban_state == state
        ).count()
        
        return current_count < limit
    
    def _find_transition_rule(self, entity_type: str, from_state: KanbanState, to_state: KanbanState) -> Optional[TransitionRule]:
        """Find the transition rule for a specific state change."""
        rules = self.transition_rules.get(entity_type, [])
        
        for rule in rules:
            if rule.from_state == from_state and rule.to_state == to_state:
                return rule
        
        return None
    
    def _validate_conditions(self, entity: Union[Project, Epic, PRD, ProjectTask], rule: TransitionRule) -> bool:
        """Validate all conditions for a transition rule."""
        for condition in rule.conditions:
            if not condition(entity):
                return False
        
        return True
    
    # Condition methods
    
    def _prd_has_approvals(self, prd: PRD) -> bool:
        """Check if PRD has required approvals."""
        return len(prd.approved_by or []) > 0
    
    def _task_dependencies_met(self, task: ProjectTask) -> bool:
        """Check if all task dependencies are completed."""
        if not task.dependencies:
            return True
        
        # Query dependent tasks
        dependent_tasks = self.db_session.query(ProjectTask).filter(
            ProjectTask.id.in_(task.dependencies)
        ).all()
        
        # All dependencies must be DONE
        return all(dep.kanban_state == KanbanState.DONE for dep in dependent_tasks)
    
    def _task_quality_gates_passed(self, task: ProjectTask) -> bool:
        """Check if task quality gates have been passed."""
        # Simplified - in real implementation, this would check actual quality gates
        return True
    
    # Helper methods
    
    def _get_model_class(self, entity_type: str) -> Optional[Type]:
        """Get the SQLAlchemy model class for an entity type."""
        type_map = {
            "Project": Project,
            "Epic": Epic,
            "PRD": PRD,
            "Task": ProjectTask
        }
        return type_map.get(entity_type)
    
    def _update_entity_timestamps(self, entity: Union[Project, Epic, PRD, ProjectTask], old_state: KanbanState, new_state: KanbanState) -> None:
        """Update entity timestamps based on state transitions."""
        now = datetime.utcnow()
        
        # Start tracking
        if new_state == KanbanState.IN_PROGRESS:
            if hasattr(entity, 'actual_start') and not entity.actual_start:
                entity.actual_start = now
        
        # Completion tracking
        elif new_state == KanbanState.DONE:
            if hasattr(entity, 'actual_completion') and not entity.actual_completion:
                entity.actual_completion = now
    
    def _perform_automation_actions(self, entity: Union[Project, Epic, PRD, ProjectTask], old_state: KanbanState, new_state: KanbanState) -> List[str]:
        """Perform automated actions triggered by state transitions."""
        entity_type = entity.__class__.__name__
        automation_rules = self.automation_rules.get(entity_type, [])
        actions_performed = []
        
        for action in automation_rules:
            try:
                action_result = action(entity, old_state, new_state)
                if action_result:
                    actions_performed.append(action_result)
            except Exception as e:
                logger.warning(f"Automation action failed", extra={
                    'entity_type': entity_type,
                    'entity_id': str(entity.id),
                    'action': action.__name__,
                    'error': str(e)
                })
        
        return actions_performed
    
    # Automation action methods
    
    def _auto_start_dependent_tasks(self, task: ProjectTask, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Auto-start tasks that were waiting for this task completion."""
        if new_state != KanbanState.DONE:
            return None
        
        # Find tasks that depend on this task
        dependent_tasks = self.db_session.query(ProjectTask).filter(
            ProjectTask.dependencies.contains([task.id])
        ).all()
        
        started_count = 0
        for dep_task in dependent_tasks:
            if (dep_task.kanban_state == KanbanState.READY and 
                self._task_dependencies_met(dep_task)):
                
                dep_task.kanban_state = KanbanState.IN_PROGRESS
                dep_task.actual_start = datetime.utcnow()
                started_count += 1
        
        if started_count > 0:
            return f"Auto-started {started_count} dependent tasks"
        
        return None
    
    def _auto_update_parent_prd_progress(self, task: ProjectTask, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Update parent PRD progress when task state changes."""
        if not task.prd:
            return None
        
        prd = task.prd
        task_counts = prd.get_task_count_by_status()
        
        # Auto-transition PRD based on task progress
        total_tasks = sum(task_counts.values())
        done_tasks = task_counts.get('done', 0)
        
        if total_tasks > 0 and done_tasks == total_tasks:
            if prd.kanban_state != KanbanState.DONE:
                prd.kanban_state = KanbanState.DONE
                return f"Auto-completed PRD {prd.get_display_id()}"
        
        return None
    
    def _auto_update_parent_epic_progress(self, prd: PRD, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Update parent Epic progress when PRD state changes."""
        if not prd.epic:
            return None
        
        epic = prd.epic
        prd_counts = epic.get_prd_count_by_status()
        
        # Auto-transition Epic based on PRD progress
        total_prds = sum(prd_counts.values())
        done_prds = prd_counts.get(PRDStatus.COMPLETED, 0)
        
        if total_prds > 0 and done_prds == total_prds:
            if epic.kanban_state != KanbanState.DONE:
                epic.kanban_state = KanbanState.DONE
                return f"Auto-completed Epic {epic.get_display_id()}"
        
        return None
    
    def _auto_update_parent_project_progress(self, epic: Epic, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Update parent Project progress when Epic state changes."""
        if not epic.project:
            return None
        
        project = epic.project
        completion_pct = project.get_completion_percentage()
        
        # Auto-transition Project to REVIEW when 80% complete
        if completion_pct >= 80.0 and project.kanban_state == KanbanState.IN_PROGRESS:
            project.kanban_state = KanbanState.REVIEW
            return f"Auto-moved Project {project.get_display_id()} to review (80% complete)"
        
        # Auto-complete Project when 100% complete
        elif completion_pct >= 100.0 and project.kanban_state != KanbanState.DONE:
            project.kanban_state = KanbanState.DONE
            return f"Auto-completed Project {project.get_display_id()}"
        
        return None
    
    def _auto_update_project_metrics(self, project: Project, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Update project metrics when project state changes."""
        # This could trigger analytics updates, reporting, etc.
        return f"Updated metrics for project {project.get_display_id()}"
    
    def _auto_create_implementation_tasks(self, prd: PRD, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Auto-create implementation tasks when PRD is approved."""
        if new_state != KanbanState.DONE or old_state == KanbanState.DONE:
            return None
        
        # This would create standard implementation tasks based on PRD requirements
        # For now, just return a placeholder
        return f"Would create implementation tasks for PRD {prd.get_display_id()}"
    
    def _auto_notify_assignee(self, task: ProjectTask, old_state: KanbanState, new_state: KanbanState) -> Optional[str]:
        """Send notification to assignee about state change."""
        if not task.assigned_agent_id:
            return None
        
        # This would send actual notifications
        return f"Notified agent {task.assigned_agent_id} of task state change"
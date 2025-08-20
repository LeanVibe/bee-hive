"""
Project Management Models for LeanVibe Agent Hive 2.0

Comprehensive hierarchy: Projects → Epics → PRDs → Tasks
with Kanban workflow management and Short ID integration.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum

from sqlalchemy import (
    Column, String, Text, DateTime, JSON, Enum as SQLEnum, 
    Integer, ForeignKey, Boolean, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, validates

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, UUIDArray, StringArray
from .short_id_mixin import ShortIdMixin
from ..core.short_id_generator import EntityType


# Kanban State Definitions
class KanbanState(Enum):
    """Universal Kanban states for all project entities."""
    BACKLOG = "backlog"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ProjectStatus(Enum):
    """Project lifecycle status."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class EpicStatus(Enum):
    """Epic lifecycle status."""
    DRAFT = "draft"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PRDStatus(Enum):
    """PRD lifecycle status."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    IN_DEVELOPMENT = "in_development"
    COMPLETED = "completed"
    DEPRECATED = "deprecated"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class TaskType(Enum):
    """Enhanced task types for comprehensive project management."""
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    OPTIMIZATION = "optimization"
    PLANNING = "planning"
    COORDINATION = "coordination"
    ANALYSIS = "analysis"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"


class Project(Base, ShortIdMixin):
    """
    Top-level project container for organizing development work.
    
    Projects contain multiple epics and provide strategic context
    for all development activities.
    """
    
    __tablename__ = "projects"
    ENTITY_TYPE = EntityType.PROJECT
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Status and lifecycle
    status = Column(ENUM(ProjectStatus, name='projectstatus'), 
                   nullable=False, default=ProjectStatus.PLANNING, index=True)
    kanban_state = Column(ENUM(KanbanState, name='kanban_state_project'), 
                         nullable=False, default=KanbanState.BACKLOG, index=True)
    
    # Project metadata
    objectives = Column(JSON, nullable=True, default=list)
    success_criteria = Column(JSON, nullable=True, default=list)
    stakeholders = Column(StringArray(), nullable=True, default=list)
    tags = Column(StringArray(), nullable=True, default=list)
    
    # Timeline and planning
    start_date = Column(DateTime(timezone=True), nullable=True)
    target_end_date = Column(DateTime(timezone=True), nullable=True)
    actual_end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Ownership and assignment
    owner_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True)
    assigned_agents = Column(UUIDArray(), nullable=True, default=list)
    
    # Configuration and context
    configuration = Column(JSON, nullable=True, default=dict)
    external_links = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    epics = relationship("Epic", back_populates="project", cascade="all, delete-orphan")
    owner_agent = relationship("Agent", foreign_keys=[owner_agent_id])
    
    # Constraints
    __table_args__ = (
        Index('ix_projects_status_kanban', 'status', 'kanban_state'),
        Index('ix_projects_timeline', 'start_date', 'target_end_date'),
        CheckConstraint('start_date IS NULL OR target_end_date IS NULL OR start_date <= target_end_date',
                       name='check_project_dates'),
    )
    
    def __init__(self, **kwargs):
        """Initialize project with proper defaults."""
        if 'status' not in kwargs:
            kwargs['status'] = ProjectStatus.PLANNING
        if 'kanban_state' not in kwargs:
            kwargs['kanban_state'] = KanbanState.BACKLOG
        if 'objectives' not in kwargs:
            kwargs['objectives'] = []
        if 'success_criteria' not in kwargs:
            kwargs['success_criteria'] = []
        if 'stakeholders' not in kwargs:
            kwargs['stakeholders'] = []
        if 'tags' not in kwargs:
            kwargs['tags'] = []
        if 'assigned_agents' not in kwargs:
            kwargs['assigned_agents'] = []
        if 'configuration' not in kwargs:
            kwargs['configuration'] = {}
        if 'external_links' not in kwargs:
            kwargs['external_links'] = {}
        
        super().__init__(**kwargs)
    
    def can_transition_to_state(self, new_state: KanbanState) -> bool:
        """Check if project can transition to new kanban state."""
        valid_transitions = {
            KanbanState.BACKLOG: {KanbanState.READY, KanbanState.CANCELLED},
            KanbanState.READY: {KanbanState.IN_PROGRESS, KanbanState.BACKLOG, KanbanState.CANCELLED},
            KanbanState.IN_PROGRESS: {KanbanState.REVIEW, KanbanState.BLOCKED, KanbanState.CANCELLED},
            KanbanState.REVIEW: {KanbanState.DONE, KanbanState.IN_PROGRESS, KanbanState.BLOCKED},
            KanbanState.DONE: {KanbanState.REVIEW},  # Allow reopening
            KanbanState.BLOCKED: {KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.CANCELLED},
            KanbanState.CANCELLED: {KanbanState.BACKLOG}  # Allow reactivation
        }
        
        return new_state in valid_transitions.get(self.kanban_state, set())
    
    def get_completion_percentage(self) -> float:
        """Calculate completion percentage based on epics."""
        if not self.epics:
            return 0.0
        
        completed_epics = sum(1 for epic in self.epics if epic.status == EpicStatus.COMPLETED)
        return (completed_epics / len(self.epics)) * 100.0
    
    def get_epic_count_by_status(self) -> Dict[EpicStatus, int]:
        """Get count of epics by status."""
        counts = {status: 0 for status in EpicStatus}
        for epic in self.epics:
            counts[epic.status] += 1
        return counts


class Epic(Base, ShortIdMixin):
    """
    Epic-level work containers within projects.
    
    Epics represent major features or initiatives that contain
    multiple PRDs and provide mid-level planning scope.
    """
    
    __tablename__ = "epics"
    ENTITY_TYPE = EntityType.EPIC
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    project_id = Column(DatabaseAgnosticUUID(), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Status and lifecycle
    status = Column(ENUM(EpicStatus, name='epicstatus'), 
                   nullable=False, default=EpicStatus.DRAFT, index=True)
    kanban_state = Column(ENUM(KanbanState, name='kanban_state_epic'), 
                         nullable=False, default=KanbanState.BACKLOG, index=True)
    
    # Epic metadata
    user_stories = Column(JSON, nullable=True, default=list)
    acceptance_criteria = Column(JSON, nullable=True, default=list)
    business_value = Column(Text, nullable=True)
    technical_notes = Column(Text, nullable=True)
    
    # Priority and planning
    priority = Column(ENUM(TaskPriority, name='epic_priority'), 
                     nullable=False, default=TaskPriority.MEDIUM, index=True)
    estimated_story_points = Column(Integer, nullable=True)
    actual_story_points = Column(Integer, nullable=True)
    
    # Timeline
    start_date = Column(DateTime(timezone=True), nullable=True)
    target_end_date = Column(DateTime(timezone=True), nullable=True)
    actual_end_date = Column(DateTime(timezone=True), nullable=True)
    
    # Assignment
    owner_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True)
    assigned_agents = Column(UUIDArray(), nullable=True, default=list)
    
    # Dependencies
    dependencies = Column(UUIDArray(), nullable=True, default=list, 
                         comment="Epic IDs this epic depends on")
    blocking_epics = Column(UUIDArray(), nullable=True, default=list,
                           comment="Epic IDs this epic blocks")
    
    # Configuration
    labels = Column(StringArray(), nullable=True, default=list)
    custom_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="epics")
    prds = relationship("PRD", back_populates="epic", cascade="all, delete-orphan")
    owner_agent = relationship("Agent", foreign_keys=[owner_agent_id])
    
    # Constraints
    __table_args__ = (
        Index('ix_epics_project_status', 'project_id', 'status'),
        Index('ix_epics_priority_kanban', 'priority', 'kanban_state'),
        Index('ix_epics_timeline', 'start_date', 'target_end_date'),
        CheckConstraint('start_date IS NULL OR target_end_date IS NULL OR start_date <= target_end_date',
                       name='check_epic_dates'),
    )
    
    def __init__(self, **kwargs):
        """Initialize epic with proper defaults."""
        if 'status' not in kwargs:
            kwargs['status'] = EpicStatus.DRAFT
        if 'kanban_state' not in kwargs:
            kwargs['kanban_state'] = KanbanState.BACKLOG
        if 'priority' not in kwargs:
            kwargs['priority'] = TaskPriority.MEDIUM
        if 'user_stories' not in kwargs:
            kwargs['user_stories'] = []
        if 'acceptance_criteria' not in kwargs:
            kwargs['acceptance_criteria'] = []
        if 'assigned_agents' not in kwargs:
            kwargs['assigned_agents'] = []
        if 'dependencies' not in kwargs:
            kwargs['dependencies'] = []
        if 'blocking_epics' not in kwargs:
            kwargs['blocking_epics'] = []
        if 'labels' not in kwargs:
            kwargs['labels'] = []
        if 'custom_metadata' not in kwargs:
            kwargs['custom_metadata'] = {}
        
        super().__init__(**kwargs)
    
    def can_transition_to_state(self, new_state: KanbanState) -> bool:
        """Check if epic can transition to new kanban state."""
        # Same transitions as Project
        valid_transitions = {
            KanbanState.BACKLOG: {KanbanState.READY, KanbanState.CANCELLED},
            KanbanState.READY: {KanbanState.IN_PROGRESS, KanbanState.BACKLOG, KanbanState.CANCELLED},
            KanbanState.IN_PROGRESS: {KanbanState.REVIEW, KanbanState.BLOCKED, KanbanState.CANCELLED},
            KanbanState.REVIEW: {KanbanState.DONE, KanbanState.IN_PROGRESS, KanbanState.BLOCKED},
            KanbanState.DONE: {KanbanState.REVIEW},
            KanbanState.BLOCKED: {KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.CANCELLED},
            KanbanState.CANCELLED: {KanbanState.BACKLOG}
        }
        
        return new_state in valid_transitions.get(self.kanban_state, set())
    
    def get_prd_count_by_status(self) -> Dict[PRDStatus, int]:
        """Get count of PRDs by status."""
        counts = {status: 0 for status in PRDStatus}
        for prd in self.prds:
            counts[prd.status] += 1
        return counts


class PRD(Base, ShortIdMixin):
    """
    Product Requirements Document for detailed feature specification.
    
    PRDs contain the detailed requirements and specifications for
    features and contain multiple implementation tasks.
    """
    
    __tablename__ = "prds"
    ENTITY_TYPE = EntityType.PRD
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    epic_id = Column(DatabaseAgnosticUUID(), ForeignKey("epics.id"), nullable=False, index=True)
    
    # Status and lifecycle
    status = Column(ENUM(PRDStatus, name='prdstatus'), 
                   nullable=False, default=PRDStatus.DRAFT, index=True)
    kanban_state = Column(ENUM(KanbanState, name='kanban_state_prd'), 
                         nullable=False, default=KanbanState.BACKLOG, index=True)
    
    # PRD content
    requirements = Column(JSON, nullable=True, default=list, 
                         comment="Functional requirements")
    technical_requirements = Column(JSON, nullable=True, default=list,
                                   comment="Technical requirements")
    acceptance_criteria = Column(JSON, nullable=True, default=list)
    user_flows = Column(JSON, nullable=True, default=list)
    mockups_wireframes = Column(JSON, nullable=True, default=dict,
                               comment="Links to design assets")
    
    # Planning and estimation
    priority = Column(ENUM(TaskPriority, name='prd_priority'), 
                     nullable=False, default=TaskPriority.MEDIUM, index=True)
    estimated_effort_days = Column(Integer, nullable=True)
    complexity_score = Column(Integer, nullable=True, 
                             comment="1-10 complexity rating")
    
    # Approval workflow
    reviewers = Column(UUIDArray(), nullable=True, default=list,
                      comment="Agent IDs assigned to review")
    approved_by = Column(UUIDArray(), nullable=True, default=list,
                        comment="Agent IDs who approved")
    approval_date = Column(DateTime(timezone=True), nullable=True)
    
    # Assignment
    owner_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True)
    assigned_agents = Column(UUIDArray(), nullable=True, default=list)
    
    # Dependencies
    dependencies = Column(UUIDArray(), nullable=True, default=list,
                         comment="PRD IDs this PRD depends on")
    blocking_prds = Column(UUIDArray(), nullable=True, default=list,
                          comment="PRD IDs this PRD blocks")
    
    # Version control
    version = Column(String(20), nullable=False, default="1.0")
    previous_version_id = Column(DatabaseAgnosticUUID(), 
                                ForeignKey("prds.id"), nullable=True)
    
    # Configuration
    tags = Column(StringArray(), nullable=True, default=list)
    external_references = Column(JSON, nullable=True, default=dict)
    custom_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    epic = relationship("Epic", back_populates="prds")
    tasks = relationship("ProjectTask", back_populates="prd", cascade="all, delete-orphan")
    owner_agent = relationship("Agent", foreign_keys=[owner_agent_id])
    previous_version = relationship("PRD", remote_side=[id])
    
    # Constraints
    __table_args__ = (
        Index('ix_prds_epic_status', 'epic_id', 'status'),
        Index('ix_prds_priority_kanban', 'priority', 'kanban_state'),
        Index('ix_prds_version', 'version'),
        CheckConstraint("complexity_score IS NULL OR (complexity_score >= 1 AND complexity_score <= 10)",
                       name='check_complexity_range'),
    )
    
    def __init__(self, **kwargs):
        """Initialize PRD with proper defaults."""
        if 'status' not in kwargs:
            kwargs['status'] = PRDStatus.DRAFT
        if 'kanban_state' not in kwargs:
            kwargs['kanban_state'] = KanbanState.BACKLOG
        if 'priority' not in kwargs:
            kwargs['priority'] = TaskPriority.MEDIUM
        if 'version' not in kwargs:
            kwargs['version'] = "1.0"
        if 'requirements' not in kwargs:
            kwargs['requirements'] = []
        if 'technical_requirements' not in kwargs:
            kwargs['technical_requirements'] = []
        if 'acceptance_criteria' not in kwargs:
            kwargs['acceptance_criteria'] = []
        if 'user_flows' not in kwargs:
            kwargs['user_flows'] = []
        if 'mockups_wireframes' not in kwargs:
            kwargs['mockups_wireframes'] = {}
        if 'reviewers' not in kwargs:
            kwargs['reviewers'] = []
        if 'approved_by' not in kwargs:
            kwargs['approved_by'] = []
        if 'assigned_agents' not in kwargs:
            kwargs['assigned_agents'] = []
        if 'dependencies' not in kwargs:
            kwargs['dependencies'] = []
        if 'blocking_prds' not in kwargs:
            kwargs['blocking_prds'] = []
        if 'tags' not in kwargs:
            kwargs['tags'] = []
        if 'external_references' not in kwargs:
            kwargs['external_references'] = {}
        if 'custom_metadata' not in kwargs:
            kwargs['custom_metadata'] = {}
        
        super().__init__(**kwargs)
    
    def can_transition_to_state(self, new_state: KanbanState) -> bool:
        """Check if PRD can transition to new kanban state."""
        # Similar to other entities but with approval considerations
        valid_transitions = {
            KanbanState.BACKLOG: {KanbanState.READY, KanbanState.CANCELLED},
            KanbanState.READY: {KanbanState.IN_PROGRESS, KanbanState.BACKLOG, KanbanState.CANCELLED},
            KanbanState.IN_PROGRESS: {KanbanState.REVIEW, KanbanState.BLOCKED, KanbanState.CANCELLED},
            KanbanState.REVIEW: {KanbanState.DONE, KanbanState.IN_PROGRESS, KanbanState.BLOCKED},
            KanbanState.DONE: {KanbanState.REVIEW},
            KanbanState.BLOCKED: {KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.CANCELLED},
            KanbanState.CANCELLED: {KanbanState.BACKLOG}
        }
        
        return new_state in valid_transitions.get(self.kanban_state, set())
    
    def is_approved(self) -> bool:
        """Check if PRD has been approved."""
        return self.status == PRDStatus.APPROVED and len(self.approved_by or []) > 0
    
    def get_task_count_by_status(self) -> Dict[str, int]:
        """Get count of tasks by their kanban state."""
        counts = {state.value: 0 for state in KanbanState}
        for task in self.tasks:
            counts[task.kanban_state.value] += 1
        return counts


class ProjectTask(Base, ShortIdMixin):
    """
    Enhanced Task model with project management hierarchy and Kanban workflow.
    
    Tasks are the atomic units of work that implement PRD requirements
    and can be assigned to agents for execution.
    """
    
    __tablename__ = "project_tasks"
    ENTITY_TYPE = EntityType.TASK
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    prd_id = Column(DatabaseAgnosticUUID(), ForeignKey("prds.id"), nullable=False, index=True)
    
    # Task classification
    task_type = Column(ENUM(TaskType, name='tasktype'), 
                      nullable=False, default=TaskType.FEATURE_DEVELOPMENT, index=True)
    kanban_state = Column(ENUM(KanbanState, name='kanban_state_task'), 
                         nullable=False, default=KanbanState.BACKLOG, index=True)
    priority = Column(ENUM(TaskPriority, name='taskpriority'), 
                     nullable=False, default=TaskPriority.MEDIUM, index=True)
    
    # Assignment and ownership
    assigned_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True, index=True)
    created_by_agent_id = Column(DatabaseAgnosticUUID(), ForeignKey("agents.id"), nullable=True)
    
    # Task relationships
    dependencies = Column(UUIDArray(), nullable=True, default=list,
                         comment="Task IDs this task depends on")
    blocking_tasks = Column(UUIDArray(), nullable=True, default=list,
                           comment="Task IDs this task blocks")
    
    # Task context and configuration
    context = Column(JSON, nullable=True, default=dict,
                    comment="Execution context and parameters")
    required_capabilities = Column(StringArray(), nullable=True, default=list,
                                  comment="Agent capabilities required")
    
    # Estimation and effort tracking
    estimated_effort_minutes = Column(Integer, nullable=True)
    actual_effort_minutes = Column(Integer, nullable=True)
    complexity_points = Column(Integer, nullable=True,
                              comment="Story points or complexity score")
    
    # Execution results
    result = Column(JSON, nullable=True, default=dict,
                   comment="Task execution results")
    error_message = Column(Text, nullable=True)
    
    # Retry and resilience
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    
    # Quality and validation
    acceptance_criteria = Column(JSON, nullable=True, default=list)
    test_requirements = Column(JSON, nullable=True, default=list)
    quality_gates = Column(JSON, nullable=True, default=list)
    
    # Timeline and scheduling
    due_date = Column(DateTime(timezone=True), nullable=True)
    scheduled_start = Column(DateTime(timezone=True), nullable=True)
    actual_start = Column(DateTime(timezone=True), nullable=True)
    actual_completion = Column(DateTime(timezone=True), nullable=True)
    
    # State transition tracking
    state_history = Column(JSON, nullable=True, default=list,
                          comment="History of state transitions")
    
    # Configuration
    tags = Column(StringArray(), nullable=True, default=list)
    external_references = Column(JSON, nullable=True, default=dict)
    custom_metadata = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    prd = relationship("PRD", back_populates="tasks")
    assigned_agent = relationship("Agent", foreign_keys=[assigned_agent_id])
    created_by = relationship("Agent", foreign_keys=[created_by_agent_id])
    
    # Constraints
    __table_args__ = (
        Index('ix_tasks_prd_kanban', 'prd_id', 'kanban_state'),
        Index('ix_tasks_assigned_kanban', 'assigned_agent_id', 'kanban_state'),
        Index('ix_tasks_priority_due', 'priority', 'due_date'),
        Index('ix_tasks_type_kanban', 'task_type', 'kanban_state'),
        CheckConstraint('retry_count >= 0', name='check_retry_count_positive'),
        CheckConstraint('max_retries >= 0', name='check_max_retries_positive'),
        CheckConstraint('estimated_effort_minutes IS NULL OR estimated_effort_minutes > 0',
                       name='check_estimated_effort_positive'),
        CheckConstraint('complexity_points IS NULL OR (complexity_points >= 1 AND complexity_points <= 21)',
                       name='check_complexity_points_range'),
    )
    
    def __init__(self, **kwargs):
        """Initialize task with proper defaults."""
        if 'task_type' not in kwargs:
            kwargs['task_type'] = TaskType.FEATURE_DEVELOPMENT
        if 'kanban_state' not in kwargs:
            kwargs['kanban_state'] = KanbanState.BACKLOG
        if 'priority' not in kwargs:
            kwargs['priority'] = TaskPriority.MEDIUM
        if 'retry_count' not in kwargs:
            kwargs['retry_count'] = 0
        if 'max_retries' not in kwargs:
            kwargs['max_retries'] = 3
        if 'dependencies' not in kwargs:
            kwargs['dependencies'] = []
        if 'blocking_tasks' not in kwargs:
            kwargs['blocking_tasks'] = []
        if 'context' not in kwargs:
            kwargs['context'] = {}
        if 'required_capabilities' not in kwargs:
            kwargs['required_capabilities'] = []
        if 'result' not in kwargs:
            kwargs['result'] = {}
        if 'acceptance_criteria' not in kwargs:
            kwargs['acceptance_criteria'] = []
        if 'test_requirements' not in kwargs:
            kwargs['test_requirements'] = []
        if 'quality_gates' not in kwargs:
            kwargs['quality_gates'] = []
        if 'state_history' not in kwargs:
            kwargs['state_history'] = []
        if 'tags' not in kwargs:
            kwargs['tags'] = []
        if 'external_references' not in kwargs:
            kwargs['external_references'] = {}
        if 'custom_metadata' not in kwargs:
            kwargs['custom_metadata'] = {}
        
        super().__init__(**kwargs)
    
    def can_transition_to_state(self, new_state: KanbanState) -> bool:
        """Check if task can transition to new kanban state."""
        valid_transitions = {
            KanbanState.BACKLOG: {KanbanState.READY, KanbanState.CANCELLED},
            KanbanState.READY: {KanbanState.IN_PROGRESS, KanbanState.BACKLOG, KanbanState.CANCELLED},
            KanbanState.IN_PROGRESS: {KanbanState.REVIEW, KanbanState.BLOCKED, KanbanState.CANCELLED},
            KanbanState.REVIEW: {KanbanState.DONE, KanbanState.IN_PROGRESS, KanbanState.BLOCKED},
            KanbanState.DONE: {KanbanState.REVIEW},  # Allow reopening for fixes
            KanbanState.BLOCKED: {KanbanState.READY, KanbanState.IN_PROGRESS, KanbanState.CANCELLED},
            KanbanState.CANCELLED: {KanbanState.BACKLOG}  # Allow reactivation
        }
        
        return new_state in valid_transitions.get(self.kanban_state, set())
    
    def transition_to_state(self, new_state: KanbanState, agent_id: Optional[uuid.UUID] = None, 
                           reason: Optional[str] = None) -> bool:
        """
        Transition task to new kanban state with validation and history tracking.
        
        Args:
            new_state: Target kanban state
            agent_id: ID of agent performing the transition
            reason: Optional reason for the transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        if not self.can_transition_to_state(new_state):
            return False
        
        # Record state transition
        transition_record = {
            'from_state': self.kanban_state.value,
            'to_state': new_state.value,
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': str(agent_id) if agent_id else None,
            'reason': reason
        }
        
        # Update state history
        if self.state_history is None:
            self.state_history = []
        self.state_history.append(transition_record)
        
        # Update actual timestamps based on state
        if new_state == KanbanState.IN_PROGRESS and not self.actual_start:
            self.actual_start = datetime.utcnow()
        elif new_state == KanbanState.DONE and not self.actual_completion:
            self.actual_completion = datetime.utcnow()
        
        # Update the state
        old_state = self.kanban_state
        self.kanban_state = new_state
        
        return True
    
    def can_be_started(self) -> bool:
        """Check if task can be started based on dependencies."""
        if not self.dependencies:
            return True
        
        # This would need to query the database to check dependency states
        # For now, return True - actual implementation would verify all dependencies are DONE
        return True
    
    def assign_to_agent(self, agent_id: uuid.UUID) -> None:
        """Assign task to a specific agent."""
        self.assigned_agent_id = agent_id
        
        # Auto-transition to READY if currently in BACKLOG
        if self.kanban_state == KanbanState.BACKLOG:
            self.transition_to_state(KanbanState.READY, agent_id, "Auto-transition on assignment")
    
    def calculate_urgency_score(self) -> float:
        """Calculate urgency score based on priority, due date, and dependencies."""
        base_score = self.priority.value / 10.0
        
        # Due date urgency
        if self.due_date:
            days_until_due = (self.due_date - datetime.utcnow()).days
            if days_until_due <= 0:
                return 1.0  # Overdue = maximum urgency
            elif days_until_due <= 1:
                return min(1.0, base_score + 0.3)
            elif days_until_due <= 7:
                return min(1.0, base_score + 0.1)
        
        # Blocking other tasks increases urgency
        if self.blocking_tasks:
            base_score = min(1.0, base_score + 0.1 * len(self.blocking_tasks))
        
        return base_score
    
    def get_project(self) -> Optional["Project"]:
        """Get the project this task belongs to."""
        if self.prd and self.prd.epic:
            return self.prd.epic.project
        return None
    
    def get_epic(self) -> Optional["Epic"]:
        """Get the epic this task belongs to."""
        if self.prd:
            return self.prd.epic
        return None
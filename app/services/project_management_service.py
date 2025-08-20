"""
Project Management Service for LeanVibe Agent Hive 2.0

High-level service layer for project management operations,
providing business logic and coordination between models,
state machine, and external systems.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, func, desc

from ..core.kanban_state_machine import KanbanStateMachine, StateTransitionResult
from ..models.project_management import (
    Project, Epic, PRD, ProjectTask, KanbanState, ProjectStatus, 
    EpicStatus, PRDStatus, TaskType, TaskPriority
)
from ..models.agent import Agent
from ..core.logging_service import get_component_logger

logger = get_component_logger("project_management_service")


@dataclass
class ProjectHierarchyStats:
    """Statistics for a project hierarchy."""
    
    project_count: int = 0
    epic_count: int = 0
    prd_count: int = 0
    task_count: int = 0
    
    # Progress tracking
    completed_projects: int = 0
    completed_epics: int = 0
    completed_prds: int = 0
    completed_tasks: int = 0
    
    # State distribution
    state_distribution: Dict[KanbanState, int] = None
    
    # Time metrics
    average_project_duration_days: Optional[float] = None
    average_task_completion_time_hours: Optional[float] = None
    
    def __post_init__(self):
        if self.state_distribution is None:
            self.state_distribution = {state: 0 for state in KanbanState}


@dataclass
class WorkloadAnalysis:
    """Agent workload analysis."""
    
    agent_id: uuid.UUID
    assigned_tasks: int
    in_progress_tasks: int
    total_estimated_hours: float
    overdue_tasks: int
    workload_score: float  # 0.0 to 1.0+, >1.0 indicates overload


class ProjectManagementService:
    """
    High-level service for project management operations.
    
    Provides business logic layer on top of the data models
    and integrates with the Kanban state machine for workflow management.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize the project management service.
        
        Args:
            db_session: Database session for operations
        """
        self.db_session = db_session
        self.kanban_machine = KanbanStateMachine(db_session)
    
    # Project Operations
    
    def create_project_with_initial_structure(
        self,
        name: str,
        description: Optional[str] = None,
        template_type: Optional[str] = None,
        owner_agent_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> Tuple[Project, List[Epic]]:
        """
        Create a project with initial structure based on template.
        
        Args:
            name: Project name
            description: Project description
            template_type: Template to use for initial structure
            owner_agent_id: Owner agent ID
            **kwargs: Additional project attributes
            
        Returns:
            Tuple of (created project, initial epics)
        """
        try:
            # Create project
            project = Project(
                name=name,
                description=description,
                owner_agent_id=owner_agent_id,
                **kwargs
            )
            project.ensure_short_id(self.db_session)
            
            self.db_session.add(project)
            self.db_session.flush()  # Get project ID
            
            # Create initial structure based on template
            initial_epics = []
            if template_type:
                initial_epics = self._create_template_epics(project, template_type)
            
            self.db_session.commit()
            
            logger.info(f"Created project {project.get_display_id()} with {len(initial_epics)} initial epics")
            return project, initial_epics
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to create project: {e}")
            raise
    
    def get_project_hierarchy_stats(self, project_id: uuid.UUID) -> ProjectHierarchyStats:
        """Get comprehensive statistics for a project hierarchy."""
        project = self.db_session.query(Project).options(
            joinedload(Project.epics).joinedload(Epic.prds).joinedload(PRD.tasks)
        ).filter(Project.id == project_id).first()
        
        if not project:
            return ProjectHierarchyStats()
        
        stats = ProjectHierarchyStats()
        stats.project_count = 1
        
        # Count epics and their states
        stats.epic_count = len(project.epics)
        stats.completed_epics = sum(1 for epic in project.epics 
                                   if epic.kanban_state == KanbanState.DONE)
        
        # Count PRDs and tasks
        all_tasks = []
        for epic in project.epics:
            stats.prd_count += len(epic.prds)
            stats.completed_prds += sum(1 for prd in epic.prds 
                                       if prd.kanban_state == KanbanState.DONE)
            
            for prd in epic.prds:
                all_tasks.extend(prd.tasks)
        
        stats.task_count = len(all_tasks)
        stats.completed_tasks = sum(1 for task in all_tasks 
                                   if task.kanban_state == KanbanState.DONE)
        
        # State distribution across all tasks
        for task in all_tasks:
            stats.state_distribution[task.kanban_state] += 1
        
        # Calculate time metrics
        if project.created_at and project.actual_end_date:
            duration = (project.actual_end_date - project.created_at).days
            stats.average_project_duration_days = float(duration)
        
        # Calculate average task completion time
        completed_tasks_with_times = [
            task for task in all_tasks 
            if (task.kanban_state == KanbanState.DONE and 
                task.actual_start and task.actual_completion)
        ]
        
        if completed_tasks_with_times:
            total_hours = sum(
                (task.actual_completion - task.actual_start).total_seconds() / 3600
                for task in completed_tasks_with_times
            )
            stats.average_task_completion_time_hours = total_hours / len(completed_tasks_with_times)
        
        return stats
    
    # Epic Operations
    
    def create_epic_with_standard_prds(
        self,
        project_id: uuid.UUID,
        name: str,
        description: Optional[str] = None,
        standard_prd_types: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[Epic, List[PRD]]:
        """
        Create an epic with standard PRD structure.
        
        Args:
            project_id: Parent project ID
            name: Epic name
            description: Epic description
            standard_prd_types: Types of standard PRDs to create
            **kwargs: Additional epic attributes
            
        Returns:
            Tuple of (created epic, initial PRDs)
        """
        try:
            epic = Epic(
                name=name,
                description=description,
                project_id=project_id,
                **kwargs
            )
            epic.ensure_short_id(self.db_session)
            
            self.db_session.add(epic)
            self.db_session.flush()
            
            # Create standard PRDs
            initial_prds = []
            if standard_prd_types:
                for prd_type in standard_prd_types:
                    prd = self._create_standard_prd(epic, prd_type)
                    if prd:
                        initial_prds.append(prd)
            
            self.db_session.commit()
            
            logger.info(f"Created epic {epic.get_display_id()} with {len(initial_prds)} standard PRDs")
            return epic, initial_prds
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to create epic: {e}")
            raise
    
    def analyze_epic_complexity(self, epic_id: uuid.UUID) -> Dict[str, Any]:
        """Analyze epic complexity and provide recommendations."""
        epic = self.db_session.query(Epic).options(
            joinedload(Epic.prds).joinedload(PRD.tasks)
        ).filter(Epic.id == epic_id).first()
        
        if not epic:
            return {"error": "Epic not found"}
        
        analysis = {
            "epic_id": str(epic_id),
            "epic_name": epic.name,
            "total_prds": len(epic.prds),
            "total_tasks": sum(len(prd.tasks) for prd in epic.prds),
            "estimated_story_points": epic.estimated_story_points,
            "complexity_factors": [],
            "risk_level": "LOW",
            "recommendations": []
        }
        
        # Analyze complexity factors
        if analysis["total_prds"] > 10:
            analysis["complexity_factors"].append("High number of PRDs")
            analysis["risk_level"] = "MEDIUM"
        
        if analysis["total_tasks"] > 50:
            analysis["complexity_factors"].append("High number of tasks")
            analysis["risk_level"] = "HIGH"
        
        # Check for interdependencies
        dependency_count = sum(len(epic.dependencies or []) for epic in [epic])
        if dependency_count > 3:
            analysis["complexity_factors"].append("Multiple dependencies")
            analysis["risk_level"] = "HIGH"
        
        # Generate recommendations
        if analysis["risk_level"] == "HIGH":
            analysis["recommendations"].extend([
                "Consider breaking epic into smaller epics",
                "Assign dedicated product owner",
                "Plan for more frequent reviews"
            ])
        
        return analysis
    
    # PRD Operations
    
    def create_prd_from_template(
        self,
        epic_id: uuid.UUID,
        title: str,
        template_name: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PRD:
        """
        Create a PRD from a predefined template.
        
        Args:
            epic_id: Parent epic ID
            title: PRD title
            template_name: Template to use
            context: Context variables for template
            **kwargs: Additional PRD attributes
            
        Returns:
            Created PRD
        """
        try:
            # Load template
            template = self._load_prd_template(template_name)
            if not template:
                raise ValueError(f"Template not found: {template_name}")
            
            # Apply context to template
            filled_template = self._fill_prd_template(template, context or {})
            
            # Create PRD
            prd = PRD(
                title=title,
                epic_id=epic_id,
                requirements=filled_template.get("requirements", []),
                technical_requirements=filled_template.get("technical_requirements", []),
                acceptance_criteria=filled_template.get("acceptance_criteria", []),
                user_flows=filled_template.get("user_flows", []),
                **kwargs
            )
            prd.ensure_short_id(self.db_session)
            
            self.db_session.add(prd)
            self.db_session.commit()
            
            logger.info(f"Created PRD {prd.get_display_id()} from template {template_name}")
            return prd
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to create PRD from template: {e}")
            raise
    
    def auto_generate_implementation_tasks(
        self,
        prd_id: uuid.UUID,
        task_generation_strategy: str = "standard"
    ) -> List[ProjectTask]:
        """
        Auto-generate implementation tasks for a PRD.
        
        Args:
            prd_id: PRD ID
            task_generation_strategy: Strategy for task generation
            
        Returns:
            List of generated tasks
        """
        prd = self.db_session.query(PRD).filter(PRD.id == prd_id).first()
        if not prd:
            raise ValueError("PRD not found")
        
        try:
            generated_tasks = []
            
            # Generate tasks based on strategy
            if task_generation_strategy == "standard":
                generated_tasks = self._generate_standard_implementation_tasks(prd)
            elif task_generation_strategy == "comprehensive":
                generated_tasks = self._generate_comprehensive_implementation_tasks(prd)
            elif task_generation_strategy == "minimal":
                generated_tasks = self._generate_minimal_implementation_tasks(prd)
            
            # Save generated tasks
            for task in generated_tasks:
                task.ensure_short_id(self.db_session)
                self.db_session.add(task)
            
            self.db_session.commit()
            
            logger.info(f"Generated {len(generated_tasks)} tasks for PRD {prd.get_display_id()}")
            return generated_tasks
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to generate tasks: {e}")
            raise
    
    # Task Operations
    
    def smart_task_assignment(
        self,
        task_ids: List[uuid.UUID],
        assignment_strategy: str = "balanced"
    ) -> Dict[uuid.UUID, uuid.UUID]:
        """
        Smart assignment of tasks to agents based on workload and capabilities.
        
        Args:
            task_ids: List of task IDs to assign
            assignment_strategy: Strategy for assignment
            
        Returns:
            Dictionary mapping task_id -> agent_id
        """
        tasks = self.db_session.query(ProjectTask).filter(
            ProjectTask.id.in_(task_ids)
        ).all()
        
        if not tasks:
            return {}
        
        # Get available agents
        available_agents = self.db_session.query(Agent).filter(
            Agent.status == "active"  # Assuming agent has status field
        ).all()
        
        if not available_agents:
            logger.warning("No available agents for task assignment")
            return {}
        
        # Analyze agent workloads
        workload_analysis = self._analyze_agent_workloads(available_agents)
        
        assignments = {}
        
        try:
            for task in tasks:
                # Find best agent based on strategy
                best_agent = self._find_best_agent_for_task(
                    task, available_agents, workload_analysis, assignment_strategy
                )
                
                if best_agent:
                    # Assign task
                    task.assign_to_agent(best_agent.id)
                    assignments[task.id] = best_agent.id
                    
                    # Update workload analysis
                    estimated_hours = (task.estimated_effort_minutes or 60) / 60.0
                    workload_analysis[best_agent.id].assigned_tasks += 1
                    workload_analysis[best_agent.id].total_estimated_hours += estimated_hours
            
            self.db_session.commit()
            
            logger.info(f"Assigned {len(assignments)} tasks to agents")
            return assignments
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to assign tasks: {e}")
            raise
    
    def get_task_recommendations(
        self,
        agent_id: uuid.UUID,
        limit: int = 10
    ) -> List[Tuple[ProjectTask, float]]:
        """
        Get task recommendations for an agent based on capabilities and workload.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of recommendations
            
        Returns:
            List of (task, suitability_score) tuples
        """
        agent = self.db_session.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            return []
        
        # Get unassigned tasks in READY state
        available_tasks = self.db_session.query(ProjectTask).filter(
            and_(
                ProjectTask.assigned_agent_id.is_(None),
                ProjectTask.kanban_state == KanbanState.READY
            )
        ).all()
        
        # Score tasks based on suitability
        scored_tasks = []
        for task in available_tasks:
            score = self._calculate_task_suitability_score(agent, task)
            scored_tasks.append((task, score))
        
        # Sort by score and return top recommendations
        scored_tasks.sort(key=lambda x: x[1], reverse=True)
        return scored_tasks[:limit]
    
    # Workflow and State Management
    
    def bulk_workflow_operation(
        self,
        entity_ids: List[uuid.UUID],
        operation: str,
        parameters: Dict[str, Any]
    ) -> List[StateTransitionResult]:
        """
        Perform bulk workflow operations on entities.
        
        Args:
            entity_ids: List of entity IDs
            operation: Operation to perform
            parameters: Operation parameters
            
        Returns:
            List of operation results
        """
        results = []
        
        try:
            for entity_id in entity_ids:
                # Find entity across all types
                entity = self._find_entity_by_id(entity_id)
                if not entity:
                    continue
                
                # Perform operation
                if operation == "transition_state":
                    new_state = KanbanState(parameters["new_state"])
                    result = self.kanban_machine.transition_entity_state(
                        entity, new_state,
                        agent_id=parameters.get("agent_id"),
                        reason=parameters.get("reason"),
                        force=parameters.get("force", False)
                    )
                    results.append(result)
                
                elif operation == "assign_agent":
                    if hasattr(entity, 'assign_to_agent'):
                        agent_id = uuid.UUID(parameters["agent_id"])
                        entity.assign_to_agent(agent_id)
                        # Create a mock result for consistency
                        result = StateTransitionResult(
                            success=True,
                            old_state=entity.kanban_state,
                            new_state=entity.kanban_state,
                            entity_id=entity.id,
                            entity_type=entity.__class__.__name__,
                            timestamp=datetime.utcnow()
                        )
                        results.append(result)
            
            self.db_session.commit()
            return results
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Bulk operation failed: {e}")
            raise
    
    def auto_advance_ready_work(self, limit: int = 50) -> Dict[str, List[StateTransitionResult]]:
        """
        Automatically advance work items that are ready to move forward.
        
        Args:
            limit: Maximum number of items to process per entity type
            
        Returns:
            Dictionary of results by entity type
        """
        results = {
            "tasks": [],
            "prds": [],
            "epics": [],
            "projects": []
        }
        
        try:
            # Auto-advance tasks
            results["tasks"] = self.kanban_machine.auto_transition_ready_tasks(limit)
            
            # Auto-advance PRDs that have all tasks complete
            ready_prds = self.kanban_machine.get_entities_by_state("PRD", KanbanState.REVIEW, limit)
            for prd in ready_prds:
                if all(task.kanban_state == KanbanState.DONE for task in prd.tasks):
                    result = self.kanban_machine.transition_entity_state(
                        prd, KanbanState.DONE, reason="All tasks completed"
                    )
                    results["prds"].append(result)
            
            # Similar logic for epics and projects
            # ... (implementation similar to PRDs)
            
            return results
            
        except Exception as e:
            logger.error(f"Auto-advance failed: {e}")
            raise
    
    # Analytics and Reporting
    
    def get_productivity_metrics(
        self,
        date_range_days: int = 30,
        agent_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Get productivity metrics for the specified time period.
        
        Args:
            date_range_days: Number of days to analyze
            agent_id: Optional agent ID to filter by
            
        Returns:
            Productivity metrics dictionary
        """
        start_date = datetime.utcnow() - timedelta(days=date_range_days)
        
        query = self.db_session.query(Task).filter(Task.created_at >= start_date)
        
        if agent_id:
            query = query.filter(Task.assigned_agent_id == agent_id)
        
        tasks = query.all()
        
        # Calculate metrics
        total_tasks = len(tasks)
        completed_tasks = len([t for t in tasks if t.kanban_state == KanbanState.DONE])
        in_progress_tasks = len([t for t in tasks if t.kanban_state == KanbanState.IN_PROGRESS])
        
        # Velocity (completed tasks per day)
        velocity = completed_tasks / date_range_days if date_range_days > 0 else 0
        
        # Average completion time
        completed_with_times = [
            t for t in tasks 
            if (t.kanban_state == KanbanState.DONE and 
                t.actual_start and t.actual_completion)
        ]
        
        avg_completion_time = None
        if completed_with_times:
            total_time = sum(
                (t.actual_completion - t.actual_start).total_seconds() / 3600
                for t in completed_with_times
            )
            avg_completion_time = total_time / len(completed_with_times)
        
        return {
            "period_days": date_range_days,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "completion_rate": (completed_tasks / total_tasks) if total_tasks > 0 else 0,
            "velocity_per_day": velocity,
            "average_completion_time_hours": avg_completion_time,
            "agent_id": str(agent_id) if agent_id else None
        }
    
    # Private helper methods
    
    def _create_template_epics(self, project: Project, template_type: str) -> List[Epic]:
        """Create initial epics based on project template."""
        templates = {
            "web_application": [
                {"name": "Authentication & Authorization", "priority": TaskPriority.HIGH},
                {"name": "Core Features", "priority": TaskPriority.HIGH},
                {"name": "User Interface", "priority": TaskPriority.MEDIUM},
                {"name": "API Development", "priority": TaskPriority.MEDIUM},
                {"name": "Testing & QA", "priority": TaskPriority.MEDIUM},
                {"name": "Deployment & DevOps", "priority": TaskPriority.LOW}
            ],
            "data_analysis": [
                {"name": "Data Collection", "priority": TaskPriority.HIGH},
                {"name": "Data Processing", "priority": TaskPriority.HIGH},
                {"name": "Analysis & Modeling", "priority": TaskPriority.MEDIUM},
                {"name": "Visualization", "priority": TaskPriority.MEDIUM},
                {"name": "Reporting", "priority": TaskPriority.LOW}
            ]
        }
        
        epic_configs = templates.get(template_type, [])
        epics = []
        
        for config in epic_configs:
            epic = Epic(
                name=config["name"],
                project_id=project.id,
                priority=config["priority"]
            )
            epic.ensure_short_id(self.db_session)
            epics.append(epic)
            self.db_session.add(epic)
        
        return epics
    
    def _create_standard_prd(self, epic: Epic, prd_type: str) -> Optional[PRD]:
        """Create a standard PRD of the specified type."""
        prd_templates = {
            "requirements": {
                "title": f"{epic.name} - Requirements Analysis",
                "description": "Detailed requirements analysis and specification"
            },
            "design": {
                "title": f"{epic.name} - Design Document",
                "description": "Technical design and architecture specification"
            },
            "implementation": {
                "title": f"{epic.name} - Implementation Plan",
                "description": "Implementation roadmap and task breakdown"
            }
        }
        
        template = prd_templates.get(prd_type)
        if not template:
            return None
        
        prd = PRD(
            title=template["title"],
            description=template["description"],
            epic_id=epic.id
        )
        prd.ensure_short_id(self.db_session)
        self.db_session.add(prd)
        
        return prd
    
    def _load_prd_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Load PRD template by name."""
        # In a real implementation, this would load from a template system
        # For now, return a simple template structure
        templates = {
            "feature_prd": {
                "requirements": [
                    "Define user stories",
                    "Specify functional requirements",
                    "Document non-functional requirements"
                ],
                "technical_requirements": [
                    "Architecture overview",
                    "Technology stack",
                    "Performance requirements"
                ],
                "acceptance_criteria": [
                    "User acceptance criteria",
                    "Technical acceptance criteria",
                    "Quality criteria"
                ]
            }
        }
        
        return templates.get(template_name)
    
    def _fill_prd_template(self, template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Fill PRD template with context variables."""
        # Simple template filling - in production, use a proper template engine
        filled = {}
        
        for key, value in template.items():
            if isinstance(value, list):
                filled[key] = [item.format(**context) for item in value]
            elif isinstance(value, str):
                filled[key] = value.format(**context)
            else:
                filled[key] = value
        
        return filled
    
    def _generate_standard_implementation_tasks(self, prd: PRD) -> List[ProjectTask]:
        """Generate standard implementation tasks for a PRD."""
        tasks = []
        
        # Standard task types based on PRD content
        standard_tasks = [
            {"title": "Design review and planning", "type": TaskType.PLANNING},
            {"title": "Implementation", "type": TaskType.FEATURE_DEVELOPMENT},
            {"title": "Unit testing", "type": TaskType.TESTING},
            {"title": "Integration testing", "type": TaskType.TESTING},
            {"title": "Documentation", "type": TaskType.DOCUMENTATION},
            {"title": "Code review", "type": TaskType.CODE_REVIEW}
        ]
        
        for task_config in standard_tasks:
            task = ProjectTask(
                title=f"{prd.title} - {task_config['title']}",
                description=f"Auto-generated task for {prd.title}",
                prd_id=prd.id,
                task_type=task_config["type"],
                priority=prd.priority
            )
            tasks.append(task)
        
        return tasks
    
    def _generate_comprehensive_implementation_tasks(self, prd: PRD) -> List[ProjectTask]:
        """Generate comprehensive implementation tasks for a PRD."""
        # More detailed task generation based on PRD complexity
        tasks = self._generate_standard_implementation_tasks(prd)
        
        # Add additional tasks for complex PRDs
        if prd.complexity_score and prd.complexity_score >= 7:
            additional_tasks = [
                {"title": "Architecture review", "type": TaskType.ARCHITECTURE},
                {"title": "Performance testing", "type": TaskType.TESTING},
                {"title": "Security review", "type": TaskType.SECURITY},
                {"title": "Deployment planning", "type": TaskType.DEPLOYMENT}
            ]
            
            for task_config in additional_tasks:
                task = ProjectTask(
                    title=f"{prd.title} - {task_config['title']}",
                    prd_id=prd.id,
                    task_type=task_config["type"],
                    priority=prd.priority
                )
                tasks.append(task)
        
        return tasks
    
    def _generate_minimal_implementation_tasks(self, prd: PRD) -> List[ProjectTask]:
        """Generate minimal implementation tasks for a PRD."""
        return [
            ProjectTask(
                title=f"{prd.title} - Implementation",
                description="Implementation task",
                prd_id=prd.id,
                task_type=TaskType.FEATURE_DEVELOPMENT,
                priority=prd.priority
            ),
            ProjectTask(
                title=f"{prd.title} - Testing",
                description="Testing task",
                prd_id=prd.id,
                task_type=TaskType.TESTING,
                priority=prd.priority
            )
        ]
    
    def _analyze_agent_workloads(self, agents: List[Agent]) -> Dict[uuid.UUID, WorkloadAnalysis]:
        """Analyze current workloads for a list of agents."""
        workloads = {}
        
        for agent in agents:
            # Get agent's assigned tasks
            assigned_tasks = self.db_session.query(ProjectTask).filter(
                ProjectTask.assigned_agent_id == agent.id
            ).all()
            
            in_progress_count = len([t for t in assigned_tasks 
                                   if t.kanban_state == KanbanState.IN_PROGRESS])
            
            total_hours = sum(
                (t.estimated_effort_minutes or 60) / 60.0 for t in assigned_tasks
                if t.kanban_state in [KanbanState.READY, KanbanState.IN_PROGRESS]
            )
            
            overdue_count = len([
                t for t in assigned_tasks 
                if t.due_date and t.due_date < datetime.utcnow() and 
                   t.kanban_state != KanbanState.DONE
            ])
            
            # Calculate workload score (simplified)
            workload_score = min(1.0, total_hours / 40.0)  # Assume 40 hour capacity
            
            workloads[agent.id] = WorkloadAnalysis(
                agent_id=agent.id,
                assigned_tasks=len(assigned_tasks),
                in_progress_tasks=in_progress_count,
                total_estimated_hours=total_hours,
                overdue_tasks=overdue_count,
                workload_score=workload_score
            )
        
        return workloads
    
    def _find_best_agent_for_task(
        self,
        task: ProjectTask,
        available_agents: List[Agent],
        workload_analysis: Dict[uuid.UUID, WorkloadAnalysis],
        strategy: str
    ) -> Optional[Agent]:
        """Find the best agent for a task based on strategy."""
        if not available_agents:
            return None
        
        if strategy == "balanced":
            # Choose agent with lowest workload
            return min(available_agents, 
                      key=lambda a: workload_analysis[a.id].workload_score)
        
        elif strategy == "capability_match":
            # Choose based on capability matching (simplified)
            # In real implementation, match required_capabilities with agent skills
            return available_agents[0]  # Placeholder
        
        elif strategy == "round_robin":
            # Simple round-robin assignment
            return available_agents[len(available_agents) % task.id.int % len(available_agents)]
        
        return available_agents[0]  # Default fallback
    
    def _calculate_task_suitability_score(self, agent: Agent, task: ProjectTask) -> float:
        """Calculate how suitable an agent is for a task."""
        score = 0.5  # Base score
        
        # Factor in task type matching (simplified)
        # In real implementation, match with agent capabilities
        
        # Factor in workload
        workload = self._analyze_agent_workloads([agent]).get(agent.id)
        if workload:
            # Lower workload = higher suitability
            score += (1.0 - workload.workload_score) * 0.3
        
        # Factor in priority matching
        if task.priority == TaskPriority.HIGH:
            score += 0.2
        
        return min(1.0, score)
    
    def _find_entity_by_id(self, entity_id: uuid.UUID):
        """Find an entity by ID across all entity types."""
        for model_class in [Project, Epic, PRD, ProjectTask]:
            entity = self.db_session.query(model_class).filter(
                model_class.id == entity_id
            ).first()
            if entity:
                return entity
        return None
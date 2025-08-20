"""
Autonomous Project Execution Engine
Real-time project execution and delivery automation with comprehensive tracking.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
import redis.asyncio as redis
import aiohttp

from app.core.database import get_async_session
from app.core.redis import get_redis_client


class ProjectPhase(Enum):
    """Project execution phases."""
    INITIATION = "initiation"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    CLOSURE = "closure"


class ProjectStatus(Enum):
    """Project status levels."""
    ACTIVE = "active"
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    BEHIND_SCHEDULE = "behind_schedule"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """AI agent roles in project execution."""
    REQUIREMENTS_ANALYST = "requirements_analyst"
    SOLUTION_ARCHITECT = "solution_architect"
    FULL_STACK_DEVELOPER = "full_stack_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    QA_ENGINEER = "qa_engineer"
    DEVOPS_ENGINEER = "devops_engineer"
    UI_UX_SPECIALIST = "ui_ux_specialist"
    SECURITY_SPECIALIST = "security_specialist"
    DATABASE_SPECIALIST = "database_specialist"
    PROJECT_COORDINATOR = "project_coordinator"


class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentProfile:
    """AI agent profile and capabilities."""
    agent_id: str
    role: AgentRole
    specialization: str
    capacity: float  # 0.0 to 1.0
    skills: List[str]
    experience_level: int  # 1-10 scale
    current_workload: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    availability_schedule: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectTask:
    """Individual project task definition."""
    task_id: str
    title: str
    description: str
    assigned_agent_id: Optional[str]
    agent_role: AgentRole
    status: TaskStatus
    priority: TaskPriority
    estimated_hours: float
    actual_hours: float = 0.0
    progress_percentage: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    quality_score: Optional[float] = None
    stakeholder_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectExecution:
    """Main project execution tracking."""
    project_id: str
    customer_id: str
    service_type: str
    project_name: str
    current_status: ProjectStatus
    current_phase: ProjectPhase
    start_date: datetime
    planned_end_date: datetime
    estimated_end_date: datetime
    overall_progress: float = 0.0
    budget_allocated: float = 0.0
    budget_consumed: float = 0.0
    assigned_agents: List[AgentProfile] = field(default_factory=list)
    project_tasks: List[ProjectTask] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    stakeholder_communications: List[Dict[str, Any]] = field(default_factory=list)
    project_artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProgressUpdate:
    """Real-time progress update."""
    update_id: str
    project_id: str
    timestamp: datetime
    agent_id: str
    task_id: str
    previous_progress: float
    current_progress: float
    work_completed: str
    time_spent_hours: float
    blockers_encountered: List[str] = field(default_factory=list)
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)


class AutonomousProjectExecutor:
    """Main autonomous project execution engine."""
    
    AGENT_TEAM_TEMPLATES = {
        "mvp_development": {
            "required_agents": [
                {
                    "role": AgentRole.REQUIREMENTS_ANALYST,
                    "specialization": "business_analysis",
                    "capacity": 1.0,
                    "skills": ["requirements_gathering", "stakeholder_communication", "documentation"]
                },
                {
                    "role": AgentRole.SOLUTION_ARCHITECT,
                    "specialization": "system_architecture",
                    "capacity": 1.0,
                    "skills": ["system_design", "technology_selection", "scalability_planning"]
                },
                {
                    "role": AgentRole.FULL_STACK_DEVELOPER,
                    "specialization": "full_stack_development",
                    "capacity": 3.0,  # Pool of 3 agents
                    "skills": ["frontend_development", "backend_development", "api_development", "database_design"]
                },
                {
                    "role": AgentRole.QA_ENGINEER,
                    "specialization": "quality_assurance",
                    "capacity": 1.0,
                    "skills": ["test_automation", "manual_testing", "quality_gates"]
                },
                {
                    "role": AgentRole.DEVOPS_ENGINEER,
                    "specialization": "devops_automation",
                    "capacity": 1.0,
                    "skills": ["ci_cd_pipeline", "deployment_automation", "monitoring_setup"]
                }
            ],
            "optional_agents": [
                {
                    "role": AgentRole.UI_UX_SPECIALIST,
                    "condition": "has_ui_requirements",
                    "capacity": 0.5,
                    "skills": ["ui_design", "ux_optimization", "user_research"]
                },
                {
                    "role": AgentRole.SECURITY_SPECIALIST,
                    "condition": "has_security_requirements",
                    "capacity": 0.5,
                    "skills": ["security_audit", "vulnerability_assessment", "compliance_validation"]
                }
            ]
        }
    }
    
    TASK_TEMPLATES = {
        "mvp_development": {
            "week_1": [
                {
                    "title": "Requirements Analysis and Documentation",
                    "agent_role": AgentRole.REQUIREMENTS_ANALYST,
                    "estimated_hours": 32,
                    "priority": TaskPriority.CRITICAL,
                    "deliverables": ["requirements_specification", "user_stories", "acceptance_criteria"],
                    "acceptance_criteria": ["All requirements documented", "Stakeholder sign-off obtained"]
                },
                {
                    "title": "System Architecture Design",
                    "agent_role": AgentRole.SOLUTION_ARCHITECT,
                    "estimated_hours": 40,
                    "priority": TaskPriority.CRITICAL,
                    "deliverables": ["architecture_diagram", "technology_stack_document", "data_flow_diagram"],
                    "acceptance_criteria": ["Architecture review passed", "Technology decisions approved"]
                },
                {
                    "title": "Development Environment Setup",
                    "agent_role": AgentRole.DEVOPS_ENGINEER,
                    "estimated_hours": 16,
                    "priority": TaskPriority.HIGH,
                    "deliverables": ["development_environment", "ci_cd_pipeline", "testing_environment"],
                    "acceptance_criteria": ["All environments operational", "Automated deployment working"]
                }
            ],
            "week_2": [
                {
                    "title": "Core Backend API Development",
                    "agent_role": AgentRole.FULL_STACK_DEVELOPER,
                    "estimated_hours": 60,
                    "priority": TaskPriority.CRITICAL,
                    "deliverables": ["rest_api_endpoints", "database_schema", "api_documentation"],
                    "acceptance_criteria": ["All core APIs functional", "90% test coverage achieved"]
                },
                {
                    "title": "Frontend Core Components",
                    "agent_role": AgentRole.FULL_STACK_DEVELOPER,
                    "estimated_hours": 50,
                    "priority": TaskPriority.HIGH,
                    "deliverables": ["core_ui_components", "routing_setup", "state_management"],
                    "acceptance_criteria": ["Core user flows working", "Responsive design implemented"]
                },
                {
                    "title": "Security Implementation",
                    "agent_role": AgentRole.SECURITY_SPECIALIST,
                    "estimated_hours": 24,
                    "priority": TaskPriority.HIGH,
                    "deliverables": ["authentication_system", "authorization_middleware", "security_audit_report"],
                    "acceptance_criteria": ["Security audit passed", "No high-risk vulnerabilities"]
                }
            ]
        }
    }
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.active_projects: Dict[str, ProjectExecution] = {}
    
    async def initiate_project_execution(
        self,
        project_config: Dict[str, Any],
        guarantee_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initiate autonomous project execution with full team deployment."""
        
        project_id = f"proj_{project_config['customer_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Initiating project execution: {project_id}")
        
        try:
            # Phase 1: Agent Team Assembly (0-30 minutes)
            team_result = await self._assemble_agent_team(project_id, project_config)
            
            # Phase 2: Task Planning and Scheduling (30-60 minutes)
            task_result = await self._generate_task_plan(project_id, project_config)
            
            # Phase 3: Environment Provisioning (60-90 minutes)
            environment_result = await self._provision_project_environments(project_id, project_config)
            
            # Phase 4: Stakeholder Integration (90-120 minutes)
            stakeholder_result = await self._setup_stakeholder_integration(project_id, project_config)
            
            # Phase 5: Real-time Monitoring Setup (120-150 minutes)
            monitoring_result = await self._setup_realtime_monitoring(project_id, project_config)
            
            # Phase 6: Project Launch (150-180 minutes)
            launch_result = await self._launch_project_execution(project_id)
            
            # Create project execution record
            project_execution = ProjectExecution(
                project_id=project_id,
                customer_id=project_config["customer_id"],
                service_type=project_config["service_type"],
                project_name=project_config.get("project_name", f"Project {project_id}"),
                current_status=ProjectStatus.ACTIVE,
                current_phase=ProjectPhase.EXECUTION,
                start_date=datetime.now(),
                planned_end_date=datetime.now() + timedelta(weeks=project_config.get("timeline_weeks", 4)),
                estimated_end_date=datetime.now() + timedelta(weeks=project_config.get("timeline_weeks", 4)),
                budget_allocated=project_config.get("budget_usd", 0),
                assigned_agents=team_result["agents"],
                project_tasks=task_result["tasks"],
                milestones=task_result["milestones"]
            )
            
            # Store project execution
            self.active_projects[project_id] = project_execution
            await self._store_project_execution(project_execution)
            
            return {
                "status": "initiated",
                "project_id": project_id,
                "execution_phases": [
                    {"phase": "agent_team_assembly", "result": team_result, "duration_minutes": 30},
                    {"phase": "task_planning", "result": task_result, "duration_minutes": 30},
                    {"phase": "environment_provisioning", "result": environment_result, "duration_minutes": 30},
                    {"phase": "stakeholder_integration", "result": stakeholder_result, "duration_minutes": 30},
                    {"phase": "monitoring_setup", "result": monitoring_result, "duration_minutes": 30},
                    {"phase": "project_launch", "result": launch_result, "duration_minutes": 30}
                ],
                "total_setup_time_minutes": 180,
                "team_composition": {
                    "total_agents": len(team_result["agents"]),
                    "agent_roles": [agent.role.value for agent in team_result["agents"]],
                    "total_capacity": sum(agent.capacity for agent in team_result["agents"])
                },
                "task_breakdown": {
                    "total_tasks": len(task_result["tasks"]),
                    "estimated_total_hours": sum(task.estimated_hours for task in task_result["tasks"]),
                    "critical_path_tasks": task_result.get("critical_path", [])
                },
                "project_dashboard_url": f"https://dashboard.leanvibe.ai/projects/{project_id}",
                "first_milestone_date": project_execution.planned_end_date - timedelta(weeks=3),
                "estimated_completion": project_execution.estimated_end_date,
                "monitoring_endpoints": monitoring_result.get("endpoints", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initiate project execution: {e}")
            return {
                "status": "failed",
                "project_id": project_id,
                "error_message": str(e),
                "recovery_actions": await self._generate_recovery_actions(e, project_config)
            }
    
    async def _assemble_agent_team(
        self,
        project_id: str,
        project_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assemble optimal AI agent team for the project."""
        
        service_type = project_config["service_type"]
        
        if service_type not in self.AGENT_TEAM_TEMPLATES:
            raise ValueError(f"Unsupported service type: {service_type}")
        
        team_template = self.AGENT_TEAM_TEMPLATES[service_type]
        deployed_agents = []
        
        # Deploy required agents
        for agent_spec in team_template["required_agents"]:
            if agent_spec["capacity"] > 1.0:
                # Deploy multiple agents for roles with capacity > 1.0
                for i in range(int(agent_spec["capacity"])):
                    agent = await self._deploy_agent(project_id, agent_spec, i+1)
                    deployed_agents.append(agent)
                
                # Handle fractional capacity
                fractional_capacity = agent_spec["capacity"] - int(agent_spec["capacity"])
                if fractional_capacity > 0:
                    agent = await self._deploy_agent(project_id, agent_spec, "partial", fractional_capacity)
                    deployed_agents.append(agent)
            else:
                agent = await self._deploy_agent(project_id, agent_spec)
                deployed_agents.append(agent)
        
        # Deploy optional agents based on conditions
        for agent_spec in team_template.get("optional_agents", []):
            if await self._evaluate_agent_condition(project_config, agent_spec["condition"]):
                agent = await self._deploy_agent(project_id, agent_spec)
                deployed_agents.append(agent)
        
        # Configure inter-agent communication
        communication_result = await self._configure_agent_communication(project_id, deployed_agents)
        
        return {
            "agents": deployed_agents,
            "total_agents": len(deployed_agents),
            "total_capacity": sum(agent.capacity for agent in deployed_agents),
            "communication_configured": communication_result["success"],
            "estimated_velocity": await self._calculate_team_velocity(deployed_agents)
        }
    
    async def _deploy_agent(
        self,
        project_id: str,
        agent_spec: Dict[str, Any],
        instance_number: Union[int, str] = 1,
        capacity_override: Optional[float] = None
    ) -> AgentProfile:
        """Deploy individual AI agent."""
        
        agent_id = f"agent_{project_id}_{agent_spec['role'].value}_{instance_number}"
        
        agent = AgentProfile(
            agent_id=agent_id,
            role=agent_spec["role"],
            specialization=agent_spec["specialization"],
            capacity=capacity_override or min(agent_spec["capacity"], 1.0),
            skills=agent_spec["skills"],
            experience_level=8,  # Default high experience level
            performance_metrics={
                "task_completion_rate": 95.0,
                "quality_score": 8.7,
                "velocity_multiplier": 1.8
            }
        )
        
        # Initialize agent in Redis
        await self._initialize_agent(agent)
        
        return agent
    
    async def _generate_task_plan(
        self,
        project_id: str,
        project_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive task plan based on project requirements."""
        
        service_type = project_config["service_type"]
        timeline_weeks = project_config.get("timeline_weeks", 4)
        
        if service_type not in self.TASK_TEMPLATES:
            raise ValueError(f"No task template for service type: {service_type}")
        
        template = self.TASK_TEMPLATES[service_type]
        all_tasks = []
        milestones = []
        
        # Generate tasks for each week
        for week in range(1, timeline_weeks + 1):
            week_key = f"week_{week}"
            
            if week_key in template:
                week_tasks = template[week_key]
            else:
                # Use the last available week template for additional weeks
                last_week = max([int(k.split('_')[1]) for k in template.keys()])
                week_tasks = template[f"week_{last_week}"]
            
            for i, task_spec in enumerate(week_tasks):
                task_id = f"task_{project_id}_w{week}_{i+1}"
                
                task = ProjectTask(
                    task_id=task_id,
                    title=task_spec["title"],
                    description=task_spec.get("description", ""),
                    assigned_agent_id=None,  # Will be assigned during execution
                    agent_role=task_spec["agent_role"],
                    status=TaskStatus.PENDING,
                    priority=task_spec["priority"],
                    estimated_hours=task_spec["estimated_hours"],
                    deliverables=task_spec["deliverables"],
                    acceptance_criteria=task_spec["acceptance_criteria"],
                    due_date=datetime.now() + timedelta(weeks=week)
                )
                
                all_tasks.append(task)
            
            # Add weekly milestone
            milestone_id = f"milestone_{project_id}_week_{week}"
            milestones.append(milestone_id)
        
        # Calculate dependencies
        await self._calculate_task_dependencies(all_tasks)
        
        # Identify critical path
        critical_path = await self._identify_critical_path(all_tasks)
        
        return {
            "tasks": all_tasks,
            "milestones": milestones,
            "total_estimated_hours": sum(task.estimated_hours for task in all_tasks),
            "critical_path": critical_path,
            "task_breakdown_by_role": await self._group_tasks_by_role(all_tasks)
        }
    
    async def process_progress_update(
        self,
        project_id: str,
        progress_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process real-time progress update from agents."""
        
        update_id = f"update_{project_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Processing progress update: {update_id}")
        
        # Create progress update record
        progress_update = ProgressUpdate(
            update_id=update_id,
            project_id=project_id,
            timestamp=datetime.now(),
            agent_id=progress_data["agent_id"],
            task_id=progress_data["task_id"],
            previous_progress=progress_data.get("previous_progress", 0.0),
            current_progress=progress_data["current_progress"],
            work_completed=progress_data["work_completed"],
            time_spent_hours=progress_data["time_spent_hours"],
            blockers_encountered=progress_data.get("blockers", []),
            quality_indicators=progress_data.get("quality_indicators", {}),
            next_steps=progress_data.get("next_steps", [])
        )
        
        # Update project execution
        project = self.active_projects.get(project_id)
        if not project:
            project = await self._load_project_execution(project_id)
        
        if project:
            # Update task progress
            updated_task = await self._update_task_progress(project, progress_update)
            
            # Recalculate overall project progress
            project.overall_progress = await self._calculate_overall_progress(project)
            
            # Update project status
            project.current_status = await self._assess_project_status(project)
            
            # Check for risks and blockers
            risks = await self._assess_project_risks(project, progress_update)
            project.risk_factors.extend(risks)
            
            # Update quality metrics
            await self._update_quality_metrics(project, progress_update)
            
            # Store updates
            await self._store_progress_update(progress_update)
            await self._store_project_execution(project)
            
            # Generate stakeholder communications
            communications = await self._generate_stakeholder_communications(project, progress_update)
            
            # Trigger automated actions if needed
            automated_actions = await self._trigger_automated_actions(project, progress_update)
            
            return {
                "status": "processed",
                "update_id": update_id,
                "project_progress": project.overall_progress,
                "project_status": project.current_status.value,
                "task_updated": updated_task.task_id if updated_task else None,
                "risks_identified": len(risks),
                "communications_sent": len(communications),
                "automated_actions": automated_actions,
                "next_milestone": await self._get_next_milestone(project),
                "estimated_completion": project.estimated_end_date
            }
        else:
            raise ValueError(f"Project not found: {project_id}")
    
    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive real-time project status."""
        
        try:
            project = self.active_projects.get(project_id)
            if not project:
                project = await self._load_project_execution(project_id)
            
            if not project:
                return {"status": "error", "message": "Project not found"}
            
            # Get recent progress updates
            recent_updates = await self._get_recent_progress_updates(project_id, limit=10)
            
            # Calculate velocity metrics
            velocity_metrics = await self._calculate_velocity_metrics(project)
            
            # Get active tasks
            active_tasks = [task for task in project.project_tasks if task.status == TaskStatus.IN_PROGRESS]
            
            # Get upcoming milestones
            upcoming_milestones = await self._get_upcoming_milestones(project)
            
            return {
                "status": "success",
                "project_id": project_id,
                "project_name": project.project_name,
                "current_status": project.current_status.value,
                "current_phase": project.current_phase.value,
                "overall_progress": project.overall_progress,
                "timeline": {
                    "start_date": project.start_date.isoformat(),
                    "planned_end_date": project.planned_end_date.isoformat(),
                    "estimated_end_date": project.estimated_end_date.isoformat(),
                    "days_elapsed": (datetime.now() - project.start_date).days,
                    "days_remaining": (project.estimated_end_date - datetime.now()).days
                },
                "team_status": {
                    "total_agents": len(project.assigned_agents),
                    "active_agents": len([a for a in project.assigned_agents if a.current_workload > 0]),
                    "team_utilization": sum(a.current_workload for a in project.assigned_agents) / len(project.assigned_agents) if project.assigned_agents else 0
                },
                "task_status": {
                    "total_tasks": len(project.project_tasks),
                    "completed_tasks": len([t for t in project.project_tasks if t.status == TaskStatus.COMPLETED]),
                    "in_progress_tasks": len(active_tasks),
                    "blocked_tasks": len([t for t in project.project_tasks if t.status == TaskStatus.BLOCKED])
                },
                "quality_metrics": project.quality_metrics,
                "velocity_metrics": velocity_metrics,
                "risk_factors": project.risk_factors[-5:],  # Last 5 risks
                "active_tasks": [
                    {
                        "task_id": task.task_id,
                        "title": task.title,
                        "assigned_agent": task.assigned_agent_id,
                        "progress": task.progress_percentage,
                        "due_date": task.due_date.isoformat() if task.due_date else None
                    }
                    for task in active_tasks
                ],
                "upcoming_milestones": upcoming_milestones,
                "recent_updates": recent_updates,
                "dashboard_url": f"https://dashboard.leanvibe.ai/projects/{project_id}",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _calculate_overall_progress(self, project: ProjectExecution) -> float:
        """Calculate overall project progress based on task completion."""
        
        if not project.project_tasks:
            return 0.0
        
        total_weight = sum(task.estimated_hours for task in project.project_tasks)
        completed_weight = sum(
            task.estimated_hours * (task.progress_percentage / 100.0)
            for task in project.project_tasks
        )
        
        return (completed_weight / total_weight) * 100.0 if total_weight > 0 else 0.0
    
    async def _assess_project_status(self, project: ProjectExecution) -> ProjectStatus:
        """Assess current project status based on progress and timeline."""
        
        # Calculate expected progress based on timeline
        total_days = (project.planned_end_date - project.start_date).days
        elapsed_days = (datetime.now() - project.start_date).days
        expected_progress = (elapsed_days / total_days) * 100.0 if total_days > 0 else 0.0
        
        progress_variance = project.overall_progress - expected_progress
        
        # Determine status based on progress variance
        if progress_variance >= 10.0:
            return ProjectStatus.ON_TRACK
        elif progress_variance >= -5.0:
            return ProjectStatus.ACTIVE
        elif progress_variance >= -15.0:
            return ProjectStatus.AT_RISK
        else:
            return ProjectStatus.BEHIND_SCHEDULE
    
    async def _store_project_execution(self, project: ProjectExecution):
        """Store project execution data in Redis."""
        
        project_data = {
            "project_id": project.project_id,
            "customer_id": project.customer_id,
            "service_type": project.service_type,
            "project_name": project.project_name,
            "current_status": project.current_status.value,
            "current_phase": project.current_phase.value,
            "start_date": project.start_date.isoformat(),
            "planned_end_date": project.planned_end_date.isoformat(),
            "estimated_end_date": project.estimated_end_date.isoformat(),
            "overall_progress": project.overall_progress,
            "budget_allocated": project.budget_allocated,
            "budget_consumed": project.budget_consumed,
            "assigned_agents": [
                {
                    "agent_id": agent.agent_id,
                    "role": agent.role.value,
                    "specialization": agent.specialization,
                    "capacity": agent.capacity,
                    "current_workload": agent.current_workload
                }
                for agent in project.assigned_agents
            ],
            "project_tasks": [
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "progress_percentage": task.progress_percentage,
                    "estimated_hours": task.estimated_hours,
                    "actual_hours": task.actual_hours,
                    "assigned_agent_id": task.assigned_agent_id,
                    "due_date": task.due_date.isoformat() if task.due_date else None
                }
                for task in project.project_tasks
            ],
            "quality_metrics": project.quality_metrics,
            "risk_factors": project.risk_factors,
            "last_updated": datetime.now().isoformat()
        }
        
        await self.redis.setex(
            f"project_execution:{project.project_id}",
            86400 * 60,  # 60 days TTL
            json.dumps(project_data, default=str)
        )


# Global service instance
_project_executor: Optional[AutonomousProjectExecutor] = None


async def get_project_executor() -> AutonomousProjectExecutor:
    """Get the global autonomous project executor instance."""
    global _project_executor
    
    if _project_executor is None:
        redis_client = await get_redis_client()
        logger = logging.getLogger(__name__)
        _project_executor = AutonomousProjectExecutor(redis_client, logger)
    
    return _project_executor


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class AutonomousProjectExecutionScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            async def test_project_execution():
            """Test the autonomous project execution engine."""

            executor = await get_project_executor()

            # Sample project configuration
            project_config = {
            "customer_id": "customer_techcorp",
            "service_type": "mvp_development",
            "project_name": "E-commerce MVP",
            "timeline_weeks": 4,
            "budget_usd": 150000,
            "requirements": [
            "User authentication system",
            "Product catalog with search",
            "Shopping cart functionality",
            "Payment processing integration",
            "Admin dashboard"
            ],
            "technology_preferences": ["React", "Node.js", "PostgreSQL"],
            "compliance_requirements": ["PCI DSS"]
            }

            guarantee_config = {
            "guarantee_amount": 150000,
            "minimum_success_threshold": 80.0
            }

            # Initiate project execution
            execution_result = await executor.initiate_project_execution(
            project_config, guarantee_config
            )

            self.logger.info("Project Execution Initiation Result:")
            self.logger.info(f"Status: {execution_result['status']}")
            self.logger.info(f"Project ID: {execution_result['project_id']}")
            self.logger.info(f"Setup Time: {execution_result['total_setup_time_minutes']} minutes")
            self.logger.info(f"Team Size: {execution_result['team_composition']['total_agents']} agents")
            self.logger.info(f"Total Tasks: {execution_result['task_breakdown']['total_tasks']}")
            self.logger.info(f"Estimated Hours: {execution_result['task_breakdown']['estimated_total_hours']}")
            self.logger.info()

            if execution_result["status"] == "initiated":
            project_id = execution_result["project_id"]

            # Simulate progress update
            progress_data = {
            "agent_id": "agent_" + project_id + "_requirements_analyst_1",
            "task_id": "task_" + project_id + "_w1_1",
            "current_progress": 75.0,
            "work_completed": "Requirements analysis 75% complete. Documented 15 of 20 requirements.",
            "time_spent_hours": 24.0,
            "quality_indicators": {
            "documentation_completeness": 75.0,
            "stakeholder_approval": 90.0
            },
            "next_steps": [
            "Complete remaining 5 requirements",
            "Schedule stakeholder review session"
            ]
            }

            # Process progress update
            update_result = await executor.process_progress_update(project_id, progress_data)
            self.logger.info("Progress Update Result:")
            self.logger.info(f"Status: {update_result['status']}")
            self.logger.info(f"Project Progress: {update_result['project_progress']:.1f}%")
            self.logger.info(f"Project Status: {update_result['project_status']}")
            self.logger.info()

            # Get project status
            status = await executor.get_project_status(project_id)
            self.logger.info("Project Status:")
            self.logger.info(f"Overall Progress: {status['overall_progress']:.1f}%")
            self.logger.info(f"Current Status: {status['current_status']}")
            self.logger.info(f"Days Elapsed: {status['timeline']['days_elapsed']}")
            self.logger.info(f"Days Remaining: {status['timeline']['days_remaining']}")
            self.logger.info(f"Active Tasks: {status['task_status']['in_progress_tasks']}")

            # Run test
            await test_project_execution()
            
            return {"status": "completed"}
    
    script_main(AutonomousProjectExecutionScript)
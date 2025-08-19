"""
Development Team Augmentation Service
Production-grade service for seamless integration with existing development teams.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from urllib.parse import urlparse

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
import redis.asyncio as redis
import aiohttp
from github import Github
from jira import JIRA
import slack_sdk

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.models.workflow import WorkflowExecution
from app.schemas.workflow import WorkflowStatus, WorkflowType


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """Task status in team workflow."""
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"
    BLOCKED = "blocked"


class AgentSpecialization(Enum):
    """Agent specialization types."""
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    FULLSTACK_DEVELOPER = "fullstack_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    SECURITY_SPECIALIST = "security_specialist"
    DATA_ENGINEER = "data_engineer"
    MOBILE_DEVELOPER = "mobile_developer"


@dataclass
class TeamProfile:
    """Profile of the existing development team."""
    team_id: str
    team_name: str
    team_size: int
    technology_stack: List[str]
    development_methodology: str  # agile, scrum, kanban, etc.
    tools_used: Dict[str, str]  # tool_type -> tool_name
    communication_channels: Dict[str, str]  # channel_type -> channel_info
    current_velocity: float  # story points or tasks per sprint
    quality_metrics: Dict[str, float]
    work_patterns: Dict[str, Any]  # timezone, working hours, etc.
    integration_requirements: List[str]


@dataclass
class AugmentationAgent:
    """AI agent configured for team augmentation."""
    agent_id: str
    specialization: AgentSpecialization
    skills: List[str]
    experience_level: str  # junior, mid, senior, lead
    availability_hours: Dict[str, List[int]]  # day -> [hour_ranges]
    current_tasks: List[str]
    performance_metrics: Dict[str, float]
    team_integration_score: float
    preferred_task_types: List[str]
    collaboration_style: str


@dataclass
class TaskAssignment:
    """Task assignment to an augmentation agent."""
    assignment_id: str
    task_id: str
    agent_id: str
    team_id: str
    assigned_at: datetime
    estimated_hours: float
    actual_hours: Optional[float] = None
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    quality_score: Optional[float] = None
    human_reviewer: Optional[str] = None
    completion_criteria: List[str] = field(default_factory=list)


@dataclass
class TeamIntegrationMetrics:
    """Metrics tracking team integration success."""
    team_id: str
    integration_date: datetime
    velocity_improvement: float  # percentage
    quality_maintenance: float  # percentage
    team_satisfaction: float  # 1-10 scale
    communication_effectiveness: float  # 1-10 scale
    task_completion_rate: float  # percentage
    average_review_time: float  # hours
    conflict_resolution_time: float  # hours
    knowledge_transfer_score: float  # 1-10 scale


class TeamAnalysisAgent:
    """Agent for analyzing existing team processes and tools."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
    
    async def analyze_team_workflow(self, team_config: Dict[str, Any]) -> TeamProfile:
        """Analyze existing team workflow and create team profile."""
        
        self.logger.info(f"Analyzing team workflow for: {team_config['team_name']}")
        
        # Analyze project management tools
        pm_analysis = await self._analyze_project_management_tools(team_config)
        
        # Analyze code repositories
        repo_analysis = await self._analyze_code_repositories(team_config)
        
        # Analyze communication patterns
        comm_analysis = await self._analyze_communication_patterns(team_config)
        
        # Analyze development practices
        dev_practices = await self._analyze_development_practices(team_config)
        
        # Calculate current velocity
        velocity = await self._calculate_team_velocity(pm_analysis, repo_analysis)
        
        team_profile = TeamProfile(
            team_id=team_config["team_id"],
            team_name=team_config["team_name"],
            team_size=team_config["team_size"],
            technology_stack=repo_analysis["technologies"],
            development_methodology=pm_analysis["methodology"],
            tools_used=team_config["tools_used"],
            communication_channels=team_config["communication_channels"],
            current_velocity=velocity,
            quality_metrics=repo_analysis["quality_metrics"],
            work_patterns=comm_analysis["work_patterns"],
            integration_requirements=await self._identify_integration_requirements(
                pm_analysis, repo_analysis, comm_analysis
            )
        )
        
        # Cache team profile
        await self.redis.setex(
            f"team_profile:{team_profile.team_id}",
            86400,  # 24 hours TTL
            json.dumps(team_profile.__dict__, default=str)
        )
        
        return team_profile
    
    async def _analyze_project_management_tools(self, team_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project management tools and extract workflow patterns."""
        
        pm_tool = team_config["tools_used"].get("project_management")
        if not pm_tool:
            return {"methodology": "unknown", "workflow_stages": [], "velocity_data": []}
        
        analysis = {
            "tool": pm_tool,
            "methodology": "unknown",
            "workflow_stages": [],
            "velocity_data": [],
            "ticket_patterns": {}
        }
        
        if "jira" in pm_tool.lower():
            analysis.update(await self._analyze_jira_workflow(team_config))
        elif "azure" in pm_tool.lower():
            analysis.update(await self._analyze_azure_devops_workflow(team_config))
        elif "github" in pm_tool.lower():
            analysis.update(await self._analyze_github_projects_workflow(team_config))
        
        return analysis
    
    async def _analyze_jira_workflow(self, team_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JIRA workflow and extract patterns."""
        
        try:
            jira_config = team_config.get("jira_config", {})
            jira = JIRA(
                server=jira_config["server"],
                basic_auth=(jira_config["username"], jira_config["api_token"])
            )
            
            # Get project information
            project_key = jira_config["project_key"]
            project = jira.project(project_key)
            
            # Analyze workflow
            workflow_schemes = jira.workflow_schemes()
            workflow_stages = []
            
            for scheme in workflow_schemes:
                if scheme.name == project.name:
                    for workflow in scheme.workflows:
                        workflow_stages.extend([status.name for status in workflow.statuses])
            
            # Get recent issues for velocity calculation
            recent_issues = jira.search_issues(
                f"project = {project_key} AND updated >= -30d",
                maxResults=1000
            )
            
            # Analyze methodology from board configuration
            boards = jira.boards(projectKeyOrId=project_key)
            methodology = "scrum" if any("scrum" in board.name.lower() for board in boards) else "kanban"
            
            return {
                "methodology": methodology,
                "workflow_stages": workflow_stages,
                "recent_issues": [self._extract_jira_issue_data(issue) for issue in recent_issues],
                "velocity_data": self._calculate_jira_velocity(recent_issues)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze JIRA workflow: {e}")
            return {"methodology": "unknown", "workflow_stages": [], "velocity_data": []}
    
    async def _analyze_code_repositories(self, team_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code repositories to understand technology stack and practices."""
        
        repo_analysis = {
            "technologies": [],
            "code_quality_tools": [],
            "ci_cd_setup": {},
            "testing_practices": {},
            "quality_metrics": {}
        }
        
        github_config = team_config.get("github_config", {})
        if github_config:
            repo_analysis.update(await self._analyze_github_repositories(github_config))
        
        gitlab_config = team_config.get("gitlab_config", {})
        if gitlab_config:
            repo_analysis.update(await self._analyze_gitlab_repositories(gitlab_config))
        
        return repo_analysis
    
    async def _analyze_github_repositories(self, github_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GitHub repositories."""
        
        try:
            github = Github(github_config["access_token"])
            
            analysis = {
                "technologies": set(),
                "code_quality_tools": [],
                "ci_cd_setup": {},
                "testing_practices": {},
                "quality_metrics": {}
            }
            
            for repo_name in github_config["repositories"]:
                repo = github.get_repo(repo_name)
                
                # Detect technologies
                languages = repo.get_languages()
                analysis["technologies"].update(languages.keys())
                
                # Analyze CI/CD setup
                try:
                    workflows = repo.get_workflows()
                    for workflow in workflows:
                        analysis["ci_cd_setup"][workflow.name] = {
                            "status": workflow.state,
                            "runs": workflow.runs_count
                        }
                except Exception:
                    pass
                
                # Analyze testing practices
                try:
                    contents = repo.get_contents("")
                    test_indicators = [
                        f for f in contents 
                        if f.name.lower() in ["test", "tests", "spec", "__tests__"]
                    ]
                    analysis["testing_practices"]["test_directories"] = len(test_indicators)
                    
                    # Check for test configuration files
                    test_configs = [
                        f for f in contents 
                        if f.name.lower() in ["pytest.ini", "jest.config.js", "karma.conf.js", "phpunit.xml"]
                    ]
                    analysis["testing_practices"]["test_configs"] = [f.name for f in test_configs]
                    
                except Exception:
                    pass
                
                # Get quality metrics
                try:
                    # Get recent commits for analysis
                    commits = repo.get_commits()[:50]  # Last 50 commits
                    analysis["quality_metrics"]["recent_commits"] = len(commits)
                    analysis["quality_metrics"]["contributors"] = len(set(c.author.login for c in commits if c.author))
                    
                    # Analyze pull requests
                    pulls = repo.get_pulls(state="closed")[:20]  # Last 20 PRs
                    if pulls.totalCount > 0:
                        avg_review_time = sum(
                            (pr.merged_at - pr.created_at).total_seconds() / 3600 
                            for pr in pulls if pr.merged_at
                        ) / len([pr for pr in pulls if pr.merged_at])
                        analysis["quality_metrics"]["avg_review_time_hours"] = avg_review_time
                    
                except Exception:
                    pass
            
            analysis["technologies"] = list(analysis["technologies"])
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze GitHub repositories: {e}")
            return {"technologies": [], "quality_metrics": {}}


class AgentMatchingEngine:
    """Engine for matching agents to team requirements and tasks."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.available_agents: Dict[str, AugmentationAgent] = {}
    
    async def initialize_agent_pool(self, team_profile: TeamProfile) -> List[AugmentationAgent]:
        """Initialize a pool of agents optimized for the team."""
        
        self.logger.info(f"Initializing agent pool for team: {team_profile.team_name}")
        
        # Determine required agent specializations
        required_specializations = await self._determine_required_specializations(team_profile)
        
        # Create specialized agents
        agents = []
        for specialization, count in required_specializations.items():
            for i in range(count):
                agent = await self._create_specialized_agent(
                    specialization, 
                    team_profile,
                    f"{team_profile.team_id}_{specialization.value}_{i+1}"
                )
                agents.append(agent)
                self.available_agents[agent.agent_id] = agent
        
        # Cache agent pool
        await self.redis.setex(
            f"agent_pool:{team_profile.team_id}",
            86400,
            json.dumps([agent.__dict__ for agent in agents], default=str)
        )
        
        return agents
    
    async def match_agent_to_task(
        self, 
        task: Dict[str, Any], 
        team_profile: TeamProfile
    ) -> Optional[AugmentationAgent]:
        """Match the best available agent to a specific task."""
        
        task_requirements = await self._analyze_task_requirements(task)
        
        # Score all available agents for this task
        agent_scores = {}
        for agent_id, agent in self.available_agents.items():
            if self._is_agent_available(agent):
                score = await self._calculate_agent_task_fit_score(
                    agent, task_requirements, team_profile
                )
                agent_scores[agent_id] = score
        
        if not agent_scores:
            return None
        
        # Select the best matching agent
        best_agent_id = max(agent_scores, key=agent_scores.get)
        best_agent = self.available_agents[best_agent_id]
        
        self.logger.info(
            f"Matched agent {best_agent_id} to task {task.get('id', 'unknown')} "
            f"with score {agent_scores[best_agent_id]:.2f}"
        )
        
        return best_agent
    
    async def _determine_required_specializations(
        self, 
        team_profile: TeamProfile
    ) -> Dict[AgentSpecialization, int]:
        """Determine which agent specializations are needed and in what quantities."""
        
        specializations = {}
        
        # Analyze technology stack to determine needs
        tech_stack = set(tech.lower() for tech in team_profile.technology_stack)
        
        # Frontend specializations
        frontend_techs = {"react", "vue", "angular", "javascript", "typescript", "css", "html"}
        if tech_stack.intersection(frontend_techs):
            specializations[AgentSpecialization.FRONTEND_DEVELOPER] = 2
        
        # Backend specializations  
        backend_techs = {"python", "java", "nodejs", "go", "php", "ruby", "c#", "scala"}
        if tech_stack.intersection(backend_techs):
            specializations[AgentSpecialization.BACKEND_DEVELOPER] = 2
        
        # DevOps specializations
        devops_techs = {"docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible"}
        if tech_stack.intersection(devops_techs) or "ci/cd" in team_profile.tools_used.values():
            specializations[AgentSpecialization.DEVOPS_ENGINEER] = 1
        
        # QA specializations
        if any("test" in tool.lower() for tool in team_profile.tools_used.values()):
            specializations[AgentSpecialization.QA_ENGINEER] = 1
        
        # Security specializations
        if any("security" in req.lower() for req in team_profile.integration_requirements):
            specializations[AgentSpecialization.SECURITY_SPECIALIST] = 1
        
        # Data engineering
        data_techs = {"sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"}
        if tech_stack.intersection(data_techs):
            specializations[AgentSpecialization.DATA_ENGINEER] = 1
        
        # Mobile development
        mobile_techs = {"ios", "android", "react-native", "flutter", "swift", "kotlin"}
        if tech_stack.intersection(mobile_techs):
            specializations[AgentSpecialization.MOBILE_DEVELOPER] = 1
        
        # Default to full-stack if no specific specializations identified
        if not specializations:
            specializations[AgentSpecialization.FULLSTACK_DEVELOPER] = 2
        
        return specializations
    
    async def _create_specialized_agent(
        self,
        specialization: AgentSpecialization,
        team_profile: TeamProfile,
        agent_id: str
    ) -> AugmentationAgent:
        """Create an agent specialized for the team's needs."""
        
        # Define skills based on specialization and team tech stack
        skills = await self._get_specialization_skills(specialization, team_profile)
        
        # Determine experience level based on team complexity
        experience_level = await self._determine_experience_level(team_profile)
        
        # Set availability based on team work patterns
        availability = await self._calculate_availability_hours(team_profile.work_patterns)
        
        agent = AugmentationAgent(
            agent_id=agent_id,
            specialization=specialization,
            skills=skills,
            experience_level=experience_level,
            availability_hours=availability,
            current_tasks=[],
            performance_metrics={
                "task_completion_rate": 0.95,
                "quality_score": 0.90,
                "velocity_score": 0.85,
                "collaboration_score": 0.88
            },
            team_integration_score=0.0,  # Will be updated as agent works with team
            preferred_task_types=await self._get_preferred_task_types(specialization),
            collaboration_style="collaborative"
        )
        
        return agent


class TaskExecutionAgent:
    """Agent responsible for executing development tasks."""
    
    def __init__(self, redis_client: redis.Redis, logger: logging.Logger):
        self.redis = redis_client
        self.logger = logger
        self.active_assignments: Dict[str, TaskAssignment] = {}
    
    async def execute_task_assignment(
        self,
        assignment: TaskAssignment,
        team_profile: TeamProfile
    ) -> Dict[str, Any]:
        """Execute a task assignment with team integration."""
        
        self.logger.info(f"Starting task execution: {assignment.task_id} by agent {assignment.agent_id}")
        
        self.active_assignments[assignment.assignment_id] = assignment
        
        try:
            # Update task status to in progress
            assignment.status = TaskStatus.IN_PROGRESS
            await self._update_external_task_status(assignment, team_profile, TaskStatus.IN_PROGRESS)
            
            # Execute task phases
            result = await self._execute_task_phases(assignment, team_profile)
            
            # Update progress and metrics
            assignment.progress_percentage = 100.0
            assignment.status = TaskStatus.IN_REVIEW
            assignment.actual_hours = result.get("execution_time_hours", assignment.estimated_hours)
            
            # Submit for team review
            review_result = await self._submit_for_team_review(assignment, team_profile, result)
            
            return {
                "status": "completed",
                "assignment_id": assignment.assignment_id,
                "execution_result": result,
                "review_status": review_result,
                "metrics": {
                    "estimated_hours": assignment.estimated_hours,
                    "actual_hours": assignment.actual_hours,
                    "quality_score": result.get("quality_score", 0.0),
                    "team_integration_score": result.get("team_integration_score", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            assignment.status = TaskStatus.BLOCKED
            await self._handle_task_failure(assignment, team_profile, str(e))
            
            return {
                "status": "failed",
                "assignment_id": assignment.assignment_id,
                "error": str(e),
                "recovery_actions": await self._suggest_recovery_actions(assignment, str(e))
            }
    
    async def _execute_task_phases(
        self,
        assignment: TaskAssignment,
        team_profile: TeamProfile
    ) -> Dict[str, Any]:
        """Execute task in phases with progress tracking."""
        
        phases = [
            ("analysis", self._analyze_task_requirements),
            ("planning", self._create_implementation_plan),
            ("implementation", self._implement_solution),
            ("testing", self._run_tests_and_validation),
            ("documentation", self._update_documentation),
            ("integration", self._integrate_with_team_workflow)
        ]
        
        phase_results = {}
        total_phases = len(phases)
        
        for i, (phase_name, phase_func) in enumerate(phases):
            try:
                self.logger.info(f"Executing phase: {phase_name}")
                
                phase_result = await phase_func(assignment, team_profile)
                phase_results[phase_name] = phase_result
                
                # Update progress
                assignment.progress_percentage = ((i + 1) / total_phases) * 90  # Reserve 10% for review
                
                # Update external systems
                await self._update_task_progress(assignment, team_profile)
                
                self.logger.info(f"Phase {phase_name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Phase {phase_name} failed: {e}")
                phase_results[phase_name] = {"status": "failed", "error": str(e)}
                break
        
        return {
            "phase_results": phase_results,
            "execution_time_hours": sum(
                result.get("execution_time_hours", 0) 
                for result in phase_results.values() 
                if isinstance(result, dict)
            ),
            "quality_score": self._calculate_overall_quality_score(phase_results),
            "team_integration_score": phase_results.get("integration", {}).get("integration_score", 0.0)
        }


class TeamAugmentationService:
    """Main service for development team augmentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client: Optional[redis.Redis] = None
        self.team_analysis_agent: Optional[TeamAnalysisAgent] = None
        self.agent_matching_engine: Optional[AgentMatchingEngine] = None
        self.task_execution_agent: Optional[TaskExecutionAgent] = None
        self.active_integrations: Dict[str, TeamIntegrationMetrics] = {}
    
    async def initialize(self):
        """Initialize the service and its components."""
        self.redis_client = await get_redis_client()
        self.team_analysis_agent = TeamAnalysisAgent(self.redis_client, self.logger)
        self.agent_matching_engine = AgentMatchingEngine(self.redis_client, self.logger)
        self.task_execution_agent = TaskExecutionAgent(self.redis_client, self.logger)
        
        self.logger.info("Team Augmentation Service initialized successfully")
    
    async def start_team_integration(
        self,
        team_config: Dict[str, Any],
        augmentation_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start integration with an existing development team."""
        
        integration_id = f"team_aug_{team_config['team_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Phase 1: Team Analysis
            self.logger.info(f"Starting team analysis for integration {integration_id}")
            
            team_profile = await self.team_analysis_agent.analyze_team_workflow(team_config)
            
            # Phase 2: Agent Pool Creation
            self.logger.info(f"Creating specialized agent pool for team {team_profile.team_name}")
            
            agent_pool = await self.agent_matching_engine.initialize_agent_pool(team_profile)
            
            # Phase 3: Integration Setup
            integration_setup = await self._setup_team_integration(
                team_profile, 
                agent_pool, 
                augmentation_requirements
            )
            
            # Phase 4: Initial Task Assignment
            initial_assignments = await self._create_initial_task_assignments(
                team_profile,
                agent_pool,
                augmentation_requirements.get("initial_tasks", [])
            )
            
            # Initialize metrics tracking
            integration_metrics = TeamIntegrationMetrics(
                team_id=team_profile.team_id,
                integration_date=datetime.now(),
                velocity_improvement=0.0,
                quality_maintenance=1.0,
                team_satisfaction=0.0,
                communication_effectiveness=0.0,
                task_completion_rate=0.0,
                average_review_time=0.0,
                conflict_resolution_time=0.0,
                knowledge_transfer_score=0.0
            )
            
            self.active_integrations[integration_id] = integration_metrics
            
            # Store integration data
            integration_data = {
                "integration_id": integration_id,
                "team_profile": team_profile.__dict__,
                "agent_pool": [agent.__dict__ for agent in agent_pool],
                "integration_setup": integration_setup,
                "initial_assignments": [assignment.__dict__ for assignment in initial_assignments],
                "integration_metrics": integration_metrics.__dict__,
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
            
            await self.redis_client.setex(
                f"team_integration:{integration_id}",
                86400 * 30,  # 30 days TTL
                json.dumps(integration_data, default=str)
            )
            
            return {
                "status": "success",
                "integration_id": integration_id,
                "team_profile": team_profile.__dict__,
                "agent_pool_size": len(agent_pool),
                "integration_setup": integration_setup,
                "initial_assignments": len(initial_assignments),
                "next_steps": [
                    "Agent onboarding with team tools and processes",
                    "Initial task execution and monitoring",
                    "Team feedback collection and adjustment",
                    "Performance optimization based on results"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start team integration: {e}")
            return {
                "status": "error",
                "integration_id": integration_id,
                "error_message": str(e),
                "recommended_action": "Review team configuration and retry integration"
            }
    
    async def assign_task_to_agent(
        self,
        integration_id: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assign a task to the most suitable augmentation agent."""
        
        try:
            # Retrieve integration data
            integration_data = await self.redis_client.get(f"team_integration:{integration_id}")
            if not integration_data:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = json.loads(integration_data)
            team_profile = TeamProfile(**integration["team_profile"])
            
            # Match agent to task
            best_agent = await self.agent_matching_engine.match_agent_to_task(
                task_data, team_profile
            )
            
            if not best_agent:
                return {
                    "status": "error",
                    "message": "No suitable agent available for this task",
                    "recommended_action": "Check agent availability or adjust task requirements"
                }
            
            # Create task assignment
            assignment = TaskAssignment(
                assignment_id=f"{integration_id}_task_{task_data.get('id', datetime.now().strftime('%Y%m%d_%H%M%S'))}",
                task_id=task_data["id"],
                agent_id=best_agent.agent_id,
                team_id=team_profile.team_id,
                assigned_at=datetime.now(),
                estimated_hours=task_data.get("estimated_hours", 8.0),
                priority=TaskPriority(task_data.get("priority", "medium")),
                completion_criteria=task_data.get("completion_criteria", [])
            )
            
            # Execute task assignment
            execution_result = await self.task_execution_agent.execute_task_assignment(
                assignment, team_profile
            )
            
            return {
                "status": "success",
                "integration_id": integration_id,
                "assignment_id": assignment.assignment_id,
                "assigned_agent": best_agent.agent_id,
                "execution_result": execution_result,
                "estimated_completion": (
                    datetime.now() + timedelta(hours=assignment.estimated_hours)
                ).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to assign task to agent: {e}")
            return {
                "status": "error",
                "integration_id": integration_id,
                "error_message": str(e)
            }
    
    async def get_integration_status(self, integration_id: str) -> Dict[str, Any]:
        """Get current status of a team integration."""
        
        try:
            integration_data = await self.redis_client.get(f"team_integration:{integration_id}")
            if not integration_data:
                return {"status": "error", "message": "Integration not found"}
            
            integration = json.loads(integration_data)
            
            # Get current metrics
            current_metrics = self.active_integrations.get(integration_id)
            if current_metrics:
                integration["integration_metrics"] = current_metrics.__dict__
            
            # Get active assignments
            active_assignments = await self._get_active_assignments(integration_id)
            
            # Calculate performance metrics
            performance_summary = await self._calculate_performance_summary(
                integration_id, active_assignments
            )
            
            return {
                "status": "success",
                "integration_id": integration_id,
                "team_name": integration["team_profile"]["team_name"],
                "integration_status": integration["status"],
                "agent_pool_size": len(integration["agent_pool"]),
                "active_assignments": len(active_assignments),
                "integration_metrics": integration["integration_metrics"],
                "performance_summary": performance_summary,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get integration status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def optimize_agent_assignments(self, integration_id: str) -> Dict[str, Any]:
        """Optimize agent assignments based on performance data."""
        
        try:
            # Analyze current performance
            performance_data = await self._analyze_agent_performance(integration_id)
            
            # Identify optimization opportunities
            optimizations = await self._identify_optimization_opportunities(
                integration_id, performance_data
            )
            
            # Apply optimizations
            optimization_results = []
            for optimization in optimizations:
                result = await self._apply_optimization(integration_id, optimization)
                optimization_results.append(result)
            
            return {
                "status": "success",
                "integration_id": integration_id,
                "optimizations_applied": len(optimization_results),
                "performance_improvements": await self._calculate_performance_improvements(
                    integration_id, optimization_results
                ),
                "optimization_results": optimization_results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize agent assignments: {e}")
            return {"status": "error", "message": str(e)}


# Global service instance
_augmentation_service: Optional[TeamAugmentationService] = None


async def get_augmentation_service() -> TeamAugmentationService:
    """Get the global team augmentation service instance."""
    global _augmentation_service
    
    if _augmentation_service is None:
        _augmentation_service = TeamAugmentationService()
        await _augmentation_service.initialize()
    
    return _augmentation_service


# Usage example and testing
if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class TeamAugmentationTest(BaseScript):
        """Test the team augmentation service."""
        
        async def execute(self):
            """Execute team augmentation service test."""
            service = await get_augmentation_service()
            
            # Sample team configuration
            team_config = {
                "team_id": "dev_team_alpha",
                "team_name": "Alpha Development Team",
                "team_size": 8,
                "tools_used": {
                    "project_management": "Jira",
                    "version_control": "GitHub",
                    "communication": "Slack",
                    "ci_cd": "GitHub Actions"
                },
                "communication_channels": {
                    "slack": "#dev-team-alpha",
                    "email": "dev-alpha@company.com"
                },
                "github_config": {
                    "access_token": "github_token_here",
                    "repositories": ["company/frontend-app", "company/backend-api"]
                },
                "jira_config": {
                    "server": "https://company.atlassian.net",
                    "username": "api@company.com",
                    "api_token": "jira_token_here",
                    "project_key": "DEV"
                }
            }
            
            # Augmentation requirements
            augmentation_requirements = {
                "additional_capacity": "40%",
                "specializations_needed": ["frontend_developer", "backend_developer"],
                "timeline": "3 months",
                "initial_tasks": [
                    {
                        "id": "DEV-123",
                        "title": "Implement user authentication",
                        "priority": "high",
                        "estimated_hours": 16,
                        "completion_criteria": [
                            "JWT authentication implemented",
                            "User registration and login flows",
                            "Password reset functionality",
                            "Comprehensive test coverage"
                        ]
                    }
                ]
            }
            
            # Start team integration
            integration_result = await service.start_team_integration(
                team_config, augmentation_requirements
            )
            self.logger.info("Team integration started", result=integration_result)
            
            results = {"integration_status": integration_result.get("status")}
            
            if integration_result["status"] == "success":
                integration_id = integration_result["integration_id"]
                
                # Assign a task
                task_data = augmentation_requirements["initial_tasks"][0]
                assignment_result = await service.assign_task_to_agent(integration_id, task_data)
                self.logger.info("Task assigned to agent", result=assignment_result)
                
                # Get integration status
                status = await service.get_integration_status(integration_id)
                self.logger.info("Integration status checked", status=status)
                
                results.update({
                    "integration_id": integration_id,
                    "task_assigned": assignment_result.get("status") == "success",
                    "agents_active": len(status.get("agents", []))
                })
            
            return results
    
    script_main(TeamAugmentationTest)
"""
Demo Orchestrator for LeanVibe Agent Hive 2.0

Specialized orchestrator with demo-specific capabilities for realistic
task generation, agent personas, and scenario management. Built on top
of SimpleOrchestrator with enhanced demonstration features.

This orchestrator focuses on creating compelling, realistic demonstrations
of multi-agent coordination that showcase the platform's capabilities.
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

from .simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance, create_simple_orchestrator
from ..models.task import TaskStatus, TaskPriority

logger = structlog.get_logger(__name__)


class DemoPhase(Enum):
    """Demo phases for structured progression."""
    SETUP = "setup"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class DemoTask:
    """Enhanced task for demo scenarios."""
    id: str
    title: str
    description: str
    agent_role: str
    phase: DemoPhase
    priority: TaskPriority
    estimated_duration: int  # minutes
    dependencies: List[str]  # task IDs this depends on
    skills_required: List[str]
    deliverables: List[str]
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class AgentPersona:
    """Agent persona with realistic characteristics."""
    role: AgentRole
    name: str
    personality: str
    strengths: List[str]
    specializations: List[str]
    work_style: str
    communication_style: str
    productivity_pattern: str  # e.g., "steady", "burst", "slow-start"
    error_rate: float  # 0.0 to 1.0
    collaboration_score: float  # 0.0 to 1.0


@dataclass
class DemoScenario:
    """Complete demo scenario configuration."""
    id: str
    name: str
    description: str
    industry: str
    complexity: str  # "simple", "moderate", "complex"
    duration_minutes: int
    agents: List[AgentPersona]
    phases: List[DemoPhase]
    tasks: List[DemoTask]
    success_criteria: List[str]
    narrative: str


class DemoOrchestrator:
    """Enhanced orchestrator with demo-specific capabilities."""
    
    def __init__(self, base_orchestrator: Optional[SimpleOrchestrator] = None):
        self.base_orchestrator = base_orchestrator or create_simple_orchestrator()
        self.scenarios: Dict[str, DemoScenario] = {}
        self.active_demo: Optional[DemoScenario] = None
        self.demo_state: Dict[str, Any] = {}
        self.task_queue: List[DemoTask] = []
        self.completed_tasks: List[DemoTask] = []
        self.demo_metrics: Dict[str, Any] = {}
        self._initialize_scenarios()
    
    async def initialize(self):
        """Initialize the demo orchestrator."""
        await self.base_orchestrator.initialize()
        logger.info("Demo orchestrator initialized with realistic scenarios")
    
    def _initialize_scenarios(self):
        """Initialize built-in demo scenarios."""
        
        # E-commerce scenario
        ecommerce_scenario = self._create_ecommerce_scenario()
        self.scenarios["ecommerce"] = ecommerce_scenario
        
        # Blog platform scenario  
        blog_scenario = self._create_blog_scenario()
        self.scenarios["blog"] = blog_scenario
        
        # API service scenario
        api_scenario = self._create_api_scenario()
        self.scenarios["api"] = api_scenario
        
        logger.info(f"Initialized {len(self.scenarios)} demo scenarios")
    
    def _create_ecommerce_scenario(self) -> DemoScenario:
        """Create comprehensive e-commerce demo scenario."""
        
        # Define agent personas
        agents = [
            AgentPersona(
                role=AgentRole.BACKEND_DEVELOPER,
                name="backend-dev-01",
                personality="methodical and detail-oriented",
                strengths=["API design", "database optimization", "security"],
                specializations=["FastAPI", "PostgreSQL", "Redis", "JWT authentication"],
                work_style="systematic with thorough testing",
                communication_style="technical and precise",
                productivity_pattern="steady",
                error_rate=0.1,
                collaboration_score=0.9
            ),
            AgentPersona(
                role=AgentRole.FRONTEND_DEVELOPER,
                name="frontend-dev-02", 
                personality="creative and user-focused",
                strengths=["UI/UX design", "responsive layouts", "performance"],
                specializations=["React", "TypeScript", "Tailwind CSS", "Vite"],
                work_style="iterative with rapid prototyping",
                communication_style="visual and collaborative",
                productivity_pattern="burst",
                error_rate=0.15,
                collaboration_score=0.85
            ),
            AgentPersona(
                role=AgentRole.QA_SPECIALIST,
                name="qa-engineer-03",
                personality="thorough and quality-focused", 
                strengths=["test automation", "edge case discovery", "performance testing"],
                specializations=["Playwright", "Jest", "load testing", "security testing"],
                work_style="comprehensive and systematic",
                communication_style="detailed and analytical",
                productivity_pattern="steady",
                error_rate=0.05,
                collaboration_score=0.8
            ),
            AgentPersona(
                role=AgentRole.DEVOPS_ENGINEER,
                name="devops-specialist-04",
                personality="reliability-focused and proactive",
                strengths=["containerization", "CI/CD", "monitoring"],
                specializations=["Docker", "GitHub Actions", "Prometheus", "nginx"],
                work_style="proactive with infrastructure focus",
                communication_style="systems-thinking oriented",
                productivity_pattern="steady",
                error_rate=0.08,
                collaboration_score=0.75
            ),
            AgentPersona(
                role=AgentRole.STRATEGIC_PARTNER,
                name="project-manager-05",
                personality="organized and communicative",
                strengths=["coordination", "planning", "stakeholder management"],
                specializations=["Agile", "risk management", "technical communication"],
                work_style="collaborative with regular check-ins",
                communication_style="clear and encouraging",
                productivity_pattern="steady",
                error_rate=0.12,
                collaboration_score=0.95
            )
        ]
        
        # Define realistic tasks with dependencies
        tasks = [
            # Setup Phase
            DemoTask(
                id="setup-db-schema",
                title="Database Schema Design",
                description="Design and implement PostgreSQL schema for products, users, orders",
                agent_role="backend_developer",
                phase=DemoPhase.SETUP,
                priority=TaskPriority.HIGH,
                estimated_duration=45,
                dependencies=[],
                skills_required=["PostgreSQL", "database design", "normalization"],
                deliverables=["schema.sql", "migration files", "ER diagram"]
            ),
            DemoTask(
                id="setup-api-auth",
                title="Authentication System",
                description="Implement JWT-based authentication with user registration and login",
                agent_role="backend_developer", 
                phase=DemoPhase.SETUP,
                priority=TaskPriority.HIGH,
                estimated_duration=60,
                dependencies=["setup-db-schema"],
                skills_required=["JWT", "password hashing", "FastAPI security"],
                deliverables=["auth endpoints", "middleware", "token validation"]
            ),
            DemoTask(
                id="ui-component-library",
                title="Component Library Setup",
                description="Create reusable React components with Tailwind CSS styling",
                agent_role="frontend_developer",
                phase=DemoPhase.SETUP,
                priority=TaskPriority.HIGH,
                estimated_duration=50,
                dependencies=[],
                skills_required=["React", "TypeScript", "Tailwind CSS", "component design"],
                deliverables=["component library", "Storybook setup", "design tokens"]
            ),
            
            # Planning Phase
            DemoTask(
                id="api-product-catalog",
                title="Product Catalog API",
                description="Build REST API for product management with search and filtering",
                agent_role="backend_developer",
                phase=DemoPhase.PLANNING,
                priority=TaskPriority.HIGH,
                estimated_duration=90,
                dependencies=["setup-api-auth"],
                skills_required=["REST API", "search optimization", "caching"],
                deliverables=["product endpoints", "search API", "filtering logic"]
            ),
            DemoTask(
                id="shopping-cart-logic",
                title="Shopping Cart System",
                description="Implement session-based shopping cart with persistence",
                agent_role="backend_developer",
                phase=DemoPhase.PLANNING,
                priority=TaskPriority.MEDIUM,
                estimated_duration=70,
                dependencies=["api-product-catalog"],
                skills_required=["session management", "state persistence", "Redis"],
                deliverables=["cart API", "session handling", "state management"]
            ),
            DemoTask(
                id="ui-product-listing",
                title="Product Listing Page",
                description="Create responsive product grid with search and filtering UI",
                agent_role="frontend_developer",
                phase=DemoPhase.PLANNING,
                priority=TaskPriority.HIGH,
                estimated_duration=80,
                dependencies=["ui-component-library", "api-product-catalog"],
                skills_required=["React hooks", "responsive design", "API integration"],
                deliverables=["product grid", "search interface", "filter controls"]
            ),
            
            # Implementation Phase
            DemoTask(
                id="checkout-process",
                title="Checkout Flow",
                description="Complete checkout process with payment integration simulation",
                agent_role="backend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=120,
                dependencies=["shopping-cart-logic"],
                skills_required=["payment processing", "order management", "email notifications"],
                deliverables=["checkout API", "order processing", "payment simulation"]
            ),
            DemoTask(
                id="ui-shopping-cart",
                title="Shopping Cart Interface",
                description="Build interactive shopping cart with quantity controls",
                agent_role="frontend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=65,
                dependencies=["ui-product-listing", "shopping-cart-logic"],
                skills_required=["state management", "form handling", "animations"],
                deliverables=["cart component", "quantity controls", "cart persistence"]
            ),
            DemoTask(
                id="user-dashboard",
                title="User Account Dashboard",
                description="User profile and order history interface",
                agent_role="frontend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.MEDIUM,
                estimated_duration=85,
                dependencies=["setup-api-auth", "ui-component-library"],
                skills_required=["authentication flow", "protected routes", "data visualization"],
                deliverables=["user profile", "order history", "settings page"]
            ),
            
            # Testing Phase
            DemoTask(
                id="api-testing-suite",
                title="API Test Automation",
                description="Comprehensive API testing with edge cases and load testing",
                agent_role="qa_specialist",
                phase=DemoPhase.TESTING,
                priority=TaskPriority.HIGH,
                estimated_duration=100,
                dependencies=["checkout-process"],
                skills_required=["API testing", "load testing", "security testing"],
                deliverables=["test suite", "performance benchmarks", "security report"]
            ),
            DemoTask(
                id="e2e-testing",
                title="End-to-End Testing",
                description="Complete user journey testing with Playwright automation",
                agent_role="qa_specialist",
                phase=DemoPhase.TESTING,
                priority=TaskPriority.HIGH,
                estimated_duration=90,
                dependencies=["ui-shopping-cart", "user-dashboard"],
                skills_required=["E2E testing", "test automation", "user journey mapping"],
                deliverables=["E2E test suite", "test reports", "bug reports"]
            ),
            
            # Deployment Phase
            DemoTask(
                id="containerization",
                title="Docker Configuration",
                description="Create Docker setup for development and production deployment",
                agent_role="devops_engineer",
                phase=DemoPhase.DEPLOYMENT,
                priority=TaskPriority.HIGH,
                estimated_duration=75,
                dependencies=["api-testing-suite"],
                skills_required=["Docker", "container orchestration", "environment config"],
                deliverables=["Dockerfile", "docker-compose.yml", "deployment guide"]
            ),
            DemoTask(
                id="ci-cd-pipeline",
                title="CI/CD Pipeline",
                description="Automated build, test, and deployment pipeline",
                agent_role="devops_engineer",
                phase=DemoPhase.DEPLOYMENT,
                priority=TaskPriority.MEDIUM,
                estimated_duration=95,
                dependencies=["containerization", "e2e-testing"],
                skills_required=["GitHub Actions", "automated testing", "deployment automation"],
                deliverables=["CI pipeline", "deployment automation", "rollback procedures"]
            ),
            
            # Monitoring Phase
            DemoTask(
                id="monitoring-setup",
                title="Application Monitoring",
                description="Set up monitoring, logging, and alerting for production",
                agent_role="devops_engineer",
                phase=DemoPhase.MONITORING,
                priority=TaskPriority.MEDIUM,
                estimated_duration=60,
                dependencies=["ci-cd-pipeline"],
                skills_required=["monitoring tools", "log aggregation", "alerting"],
                deliverables=["monitoring dashboard", "log aggregation", "alert rules"]
            ),
            DemoTask(
                id="performance-optimization",
                title="Performance Optimization",
                description="Analyze and optimize application performance",
                agent_role="frontend_developer",
                phase=DemoPhase.MONITORING,
                priority=TaskPriority.MEDIUM,
                estimated_duration=55,
                dependencies=["monitoring-setup"],
                skills_required=["performance analysis", "code splitting", "caching strategies"],
                deliverables=["performance report", "optimization implementation", "benchmarks"]
            ),
            DemoTask(
                id="demo-completion",
                title="Project Documentation & Demo",
                description="Create comprehensive documentation and demo presentation",
                agent_role="project_manager",
                phase=DemoPhase.MONITORING,
                priority=TaskPriority.LOW,
                estimated_duration=40,
                dependencies=["performance-optimization"],
                skills_required=["technical writing", "presentation", "demo creation"],
                deliverables=["project documentation", "demo video", "deployment guide"]
            )
        ]
        
        return DemoScenario(
            id="ecommerce",
            name="ShopSmart E-commerce Platform",
            description="Complete e-commerce platform with user authentication, product catalog, shopping cart, and payment processing",
            industry="E-commerce",
            complexity="complex",
            duration_minutes=15,
            agents=agents,
            phases=list(DemoPhase),
            tasks=tasks,
            success_criteria=[
                "All 5 agents successfully spawned and coordinated",
                "Complete user journey from registration to checkout",
                "Responsive design working on multiple screen sizes",
                "API performance under 200ms response time",
                "90%+ test coverage achieved",
                "Successful deployment pipeline execution"
            ],
            narrative="Watch as five specialized AI agents collaborate to build a complete e-commerce platform from scratch, demonstrating realistic software development workflows with proper testing, deployment, and monitoring."
        )
    
    def _create_blog_scenario(self) -> DemoScenario:
        """Create blog platform demo scenario."""
        
        agents = [
            AgentPersona(
                role=AgentRole.BACKEND_DEVELOPER,
                name="backend-dev-01",
                personality="pragmatic and efficient",
                strengths=["API development", "content management", "caching"],
                specializations=["Node.js", "MongoDB", "Express", "JWT"],
                work_style="rapid development with clean code",
                communication_style="concise and practical",
                productivity_pattern="burst",
                error_rate=0.12,
                collaboration_score=0.8
            ),
            AgentPersona(
                role=AgentRole.FRONTEND_DEVELOPER,
                name="frontend-dev-02",
                personality="design-conscious and detail-oriented",
                strengths=["content presentation", "responsive design", "SEO"],
                specializations=["Next.js", "TypeScript", "CSS", "accessibility"],
                work_style="iterative with focus on user experience",
                communication_style="visual and user-centered",
                productivity_pattern="steady",
                error_rate=0.08,
                collaboration_score=0.85
            ),
            AgentPersona(
                role=AgentRole.QA_SPECIALIST,
                name="qa-engineer-03",
                personality="meticulous and user-focused",
                strengths=["content validation", "cross-browser testing", "accessibility"],
                specializations=["content testing", "accessibility testing", "SEO validation"],
                work_style="thorough with user perspective",
                communication_style="clear and user-focused",
                productivity_pattern="steady",
                error_rate=0.06,
                collaboration_score=0.9
            )
        ]
        
        tasks = [
            DemoTask(
                id="content-api",
                title="Content Management API",
                description="Build API for blog posts, categories, and comments",
                agent_role="backend_developer",
                phase=DemoPhase.SETUP,
                priority=TaskPriority.HIGH,
                estimated_duration=60,
                dependencies=[],
                skills_required=["REST API", "MongoDB", "content modeling"],
                deliverables=["content API", "database schema", "admin endpoints"]
            ),
            DemoTask(
                id="blog-ui",
                title="Blog Interface",
                description="Create responsive blog layout with post listing and detail views",
                agent_role="frontend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=80,
                dependencies=["content-api"],
                skills_required=["Next.js", "responsive design", "SEO optimization"],
                deliverables=["blog components", "post rendering", "navigation"]
            ),
            DemoTask(
                id="content-testing",
                title="Content & Accessibility Testing",
                description="Validate content quality and accessibility compliance",
                agent_role="qa_specialist",
                phase=DemoPhase.TESTING,
                priority=TaskPriority.MEDIUM,
                estimated_duration=45,
                dependencies=["blog-ui"],
                skills_required=["accessibility testing", "content validation", "SEO testing"],
                deliverables=["test reports", "accessibility audit", "SEO recommendations"]
            )
        ]
        
        return DemoScenario(
            id="blog",
            name="TechBlog Content Platform",
            description="Modern blog platform with content management and reader engagement features",
            industry="Content/Media",
            complexity="moderate",
            duration_minutes=10,
            agents=agents,
            phases=[DemoPhase.SETUP, DemoPhase.IMPLEMENTATION, DemoPhase.TESTING],
            tasks=tasks,
            success_criteria=[
                "Content API with full CRUD operations",
                "Responsive blog interface",
                "SEO-optimized content presentation",
                "Accessibility compliance achieved"
            ],
            narrative="Three specialized agents build a modern blog platform focusing on content quality, user experience, and technical excellence."
        )
    
    def _create_api_scenario(self) -> DemoScenario:
        """Create API service demo scenario."""
        
        agents = [
            AgentPersona(
                role=AgentRole.BACKEND_DEVELOPER,
                name="api-dev-01",
                personality="architecture-focused and performance-oriented",
                strengths=["API design", "performance optimization", "documentation"],
                specializations=["FastAPI", "async programming", "API documentation"],
                work_style="architecture-first with performance focus",
                communication_style="technical and thorough",
                productivity_pattern="steady",
                error_rate=0.05,
                collaboration_score=0.8
            ),
            AgentPersona(
                role=AgentRole.DEVOPS_ENGINEER,
                name="devops-01",
                personality="reliability-focused and automation-oriented",
                strengths=["containerization", "deployment", "monitoring"],
                specializations=["Docker", "container optimization", "deployment automation"],
                work_style="automation-first with reliability focus",
                communication_style="systems-oriented and clear",
                productivity_pattern="steady",
                error_rate=0.07,
                collaboration_score=0.75
            )
        ]
        
        tasks = [
            DemoTask(
                id="api-design",
                title="API Architecture & Design",
                description="Design high-performance REST API with comprehensive documentation",
                agent_role="backend_developer",
                phase=DemoPhase.PLANNING,
                priority=TaskPriority.HIGH,
                estimated_duration=45,
                dependencies=[],
                skills_required=["API design", "OpenAPI", "performance architecture"],
                deliverables=["API specification", "architecture docs", "performance plan"]
            ),
            DemoTask(
                id="api-implementation",
                title="API Implementation",
                description="Build FastAPI service with async performance optimization",
                agent_role="backend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=90,
                dependencies=["api-design"],
                skills_required=["FastAPI", "async/await", "database optimization"],
                deliverables=["API implementation", "performance tests", "documentation"]
            ),
            DemoTask(
                id="containerization",
                title="Production Containerization",
                description="Create optimized Docker setup for production deployment",
                agent_role="devops_engineer",
                phase=DemoPhase.DEPLOYMENT,
                priority=TaskPriority.HIGH,
                estimated_duration=50,
                dependencies=["api-implementation"],
                skills_required=["Docker optimization", "multi-stage builds", "security"],
                deliverables=["optimized Dockerfile", "deployment config", "security review"]
            )
        ]
        
        return DemoScenario(
            id="api",
            name="RESTful API Service",
            description="High-performance REST API service with containerized deployment",
            industry="Technology/Infrastructure",
            complexity="simple",
            duration_minutes=8,
            agents=agents,
            phases=[DemoPhase.PLANNING, DemoPhase.IMPLEMENTATION, DemoPhase.DEPLOYMENT],
            tasks=tasks,
            success_criteria=[
                "API performance under 50ms response time",
                "Comprehensive API documentation",
                "Production-ready containerization",
                "Deployment automation"
            ],
            narrative="Two focused agents build a high-performance API service with emphasis on architecture, performance, and production readiness."
        )
    
    async def start_demo(self, scenario_id: str) -> Dict[str, Any]:
        """Start a demo scenario."""
        if scenario_id not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        scenario = self.scenarios[scenario_id]
        self.active_demo = scenario
        self.demo_state = {
            "scenario_id": scenario_id,
            "start_time": datetime.utcnow(),
            "current_phase": DemoPhase.SETUP,
            "spawned_agents": [],
            "active_tasks": [],
            "completed_tasks": [],
            "metrics": {
                "agents_spawned": 0,
                "tasks_completed": 0,
                "total_tasks": len(scenario.tasks),
                "success_rate": 0.0
            }
        }
        
        # Initialize task queue
        self.task_queue = scenario.tasks.copy()
        self.completed_tasks = []
        
        logger.info(f"Started demo scenario: {scenario.name}", 
                   scenario_id=scenario_id, 
                   total_tasks=len(scenario.tasks))
        
        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario.name,
            "description": scenario.description,
            "agents_to_spawn": len(scenario.agents),
            "total_tasks": len(scenario.tasks),
            "estimated_duration": scenario.duration_minutes
        }
    
    async def spawn_demo_agents(self) -> List[Dict[str, Any]]:
        """Spawn all agents for the active demo."""
        if not self.active_demo:
            raise ValueError("No active demo scenario")
        
        spawned_agents = []
        
        for agent_persona in self.active_demo.agents:
            try:
                agent_id = await self.base_orchestrator.spawn_agent(
                    role=agent_persona.role,
                    agent_id=agent_persona.name
                )
                
                agent_info = {
                    "id": agent_id,
                    "persona": asdict(agent_persona),
                    "spawned_at": datetime.utcnow().isoformat(),
                    "status": "active"
                }
                
                spawned_agents.append(agent_info)
                self.demo_state["spawned_agents"].append(agent_info)
                self.demo_state["metrics"]["agents_spawned"] += 1
                
                logger.info(f"Spawned demo agent: {agent_persona.name}",
                           agent_id=agent_id,
                           role=agent_persona.role.value)
                
            except Exception as e:
                logger.error(f"Failed to spawn agent {agent_persona.name}: {e}")
        
        return spawned_agents
    
    async def get_next_tasks(self, agent_id: Optional[str] = None) -> List[DemoTask]:
        """Get next available tasks for assignment."""
        if not self.active_demo:
            return []
        
        available_tasks = []
        
        for task in self.task_queue:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if dependencies are satisfied
            deps_satisfied = all(
                any(completed.id == dep_id for completed in self.completed_tasks)
                for dep_id in task.dependencies
            )
            
            if not deps_satisfied:
                continue
            
            # Check if task matches agent if specified
            if agent_id:
                agent_info = next(
                    (a for a in self.demo_state["spawned_agents"] if a["id"] == agent_id),
                    None
                )
                if agent_info:
                    agent_role = agent_info["persona"]["role"]
                    if task.agent_role != agent_role.replace("AgentRole.", "").lower():
                        continue
            
            available_tasks.append(task)
        
        return available_tasks
    
    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        task = next((t for t in self.task_queue if t.id == task_id), None)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        task.assigned_agent_id = agent_id
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        self.demo_state["active_tasks"].append({
            "task_id": task_id,
            "agent_id": agent_id,
            "started_at": task.started_at.isoformat()
        })
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True
    
    async def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark a task as completed."""
        task = next((t for t in self.task_queue if t.id == task_id), None)
        if not task or task.status != TaskStatus.IN_PROGRESS:
            return False
        
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completed_at = datetime.utcnow()
        
        # Move to completed tasks
        self.completed_tasks.append(task)
        self.task_queue.remove(task)
        
        # Update metrics
        self.demo_state["metrics"]["tasks_completed"] += 1
        success_rate = len([t for t in self.completed_tasks if t.status == TaskStatus.COMPLETED]) / len(self.completed_tasks)
        self.demo_state["metrics"]["success_rate"] = success_rate
        
        # Remove from active tasks
        self.demo_state["active_tasks"] = [
            at for at in self.demo_state["active_tasks"] if at["task_id"] != task_id
        ]
        
        logger.info(f"Task {task_id} completed", success=success, agent_id=task.assigned_agent_id)
        return True
    
    async def get_demo_status(self) -> Dict[str, Any]:
        """Get comprehensive demo status."""
        if not self.active_demo:
            return {"active": False}
        
        # Calculate progress
        total_tasks = len(self.active_demo.tasks)
        completed_count = len(self.completed_tasks)
        progress_percentage = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
        
        # Calculate runtime
        start_time = self.demo_state.get("start_time")
        runtime = datetime.utcnow() - start_time if start_time else timedelta(0)
        
        # Determine current phase
        current_phase = self._determine_current_phase()
        
        return {
            "active": True,
            "scenario": {
                "id": self.active_demo.id,
                "name": self.active_demo.name,
                "description": self.active_demo.description
            },
            "progress": {
                "percentage": round(progress_percentage, 1),
                "tasks_completed": completed_count,
                "total_tasks": total_tasks,
                "current_phase": current_phase.value if current_phase else None
            },
            "agents": {
                "total": len(self.demo_state["spawned_agents"]),
                "active": len([a for a in self.demo_state["spawned_agents"] if a["status"] == "active"]),
                "details": self.demo_state["spawned_agents"]
            },
            "tasks": {
                "pending": len([t for t in self.task_queue if t.status == TaskStatus.PENDING]),
                "in_progress": len([t for t in self.task_queue if t.status == TaskStatus.IN_PROGRESS]),
                "completed": len(self.completed_tasks),
                "active_tasks": self.demo_state["active_tasks"]
            },
            "runtime": {
                "total_seconds": runtime.total_seconds(),
                "formatted": str(runtime).split('.')[0]
            },
            "metrics": self.demo_state["metrics"]
        }
    
    def _determine_current_phase(self) -> Optional[DemoPhase]:
        """Determine current phase based on completed tasks."""
        if not self.completed_tasks:
            return DemoPhase.SETUP
        
        phase_progress = {}
        for phase in DemoPhase:
            phase_tasks = [t for t in self.active_demo.tasks if t.phase == phase]
            completed_phase_tasks = [t for t in self.completed_tasks if t.phase == phase]
            
            if phase_tasks:
                phase_progress[phase] = len(completed_phase_tasks) / len(phase_tasks)
            else:
                phase_progress[phase] = 1.0
        
        # Find the current phase (first incomplete phase)
        for phase in DemoPhase:
            if phase_progress.get(phase, 0) < 1.0:
                return phase
        
        return DemoPhase.MONITORING  # All phases complete
    
    async def simulate_task_completion(self):
        """Simulate realistic task completion for demo purposes."""
        active_tasks = [t for t in self.task_queue if t.status == TaskStatus.IN_PROGRESS]
        
        for task in active_tasks:
            if not task.started_at:
                continue
            
            # Get agent persona for realistic timing
            agent_info = next(
                (a for a in self.demo_state["spawned_agents"] if a["id"] == task.assigned_agent_id),
                None
            )
            
            if agent_info:
                persona = agent_info["persona"]
                
                # Simulate realistic completion based on agent characteristics
                elapsed_minutes = (datetime.utcnow() - task.started_at).total_seconds() / 60
                progress_factor = self._calculate_progress_factor(persona, elapsed_minutes, task.estimated_duration)
                
                # Random completion chance based on progress and agent characteristics  
                completion_chance = progress_factor * (1 - persona["error_rate"])
                
                if random.random() < completion_chance * 0.1:  # 10% base chance per simulation cycle
                    success = random.random() > persona["error_rate"]
                    await self.complete_task(task.id, success)
    
    def _calculate_progress_factor(self, persona: Dict, elapsed_minutes: float, estimated_duration: int) -> float:
        """Calculate task progress factor based on agent persona."""
        productivity_pattern = persona.get("productivity_pattern", "steady")
        
        time_ratio = elapsed_minutes / estimated_duration
        
        if productivity_pattern == "burst":
            # Quick initial progress, then slower
            return min(1.0, 0.7 * time_ratio + 0.3 * (time_ratio ** 0.5))
        elif productivity_pattern == "slow-start":
            # Slow initial progress, then accelerates
            return min(1.0, 0.3 * time_ratio + 0.7 * (time_ratio ** 2))
        else:  # steady
            # Linear progress
            return min(1.0, time_ratio)
    
    async def stop_demo(self, save_results: bool = True) -> Dict[str, Any]:
        """Stop the active demo and cleanup."""
        if not self.active_demo:
            return {"stopped": False, "reason": "No active demo"}
        
        # Save results if requested
        results = {}
        if save_results:
            results = await self.get_demo_status()
        
        # Cleanup agents
        cleanup_results = []
        for agent_info in self.demo_state.get("spawned_agents", []):
            try:
                success = await self.base_orchestrator.shutdown_agent(
                    agent_info["id"], 
                    graceful=True
                )
                cleanup_results.append({"agent_id": agent_info["id"], "cleanup_success": success})
            except Exception as e:
                cleanup_results.append({"agent_id": agent_info["id"], "cleanup_error": str(e)})
        
        # Reset state
        self.active_demo = None
        self.demo_state = {}
        self.task_queue = []
        self.completed_tasks = []
        
        logger.info("Demo stopped and cleaned up")
        
        return {
            "stopped": True,
            "results": results if save_results else None,
            "cleanup": cleanup_results
        }
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available demo scenarios."""
        return [
            {
                "id": scenario.id,
                "name": scenario.name,
                "description": scenario.description,
                "industry": scenario.industry,
                "complexity": scenario.complexity,
                "duration_minutes": scenario.duration_minutes,
                "agents_count": len(scenario.agents),
                "tasks_count": len(scenario.tasks),
                "narrative": scenario.narrative
            }
            for scenario in self.scenarios.values()
        ]


def create_demo_orchestrator() -> DemoOrchestrator:
    """Factory function to create demo orchestrator."""
    return DemoOrchestrator()


# Export main classes
__all__ = [
    'DemoOrchestrator', 
    'DemoTask', 
    'AgentPersona', 
    'DemoScenario', 
    'DemoPhase',
    'create_demo_orchestrator'
]
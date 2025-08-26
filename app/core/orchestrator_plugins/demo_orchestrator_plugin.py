"""
Demo Orchestrator Plugin for LeanVibe Agent Hive 2.0

Epic 1 Phase 2.2: Consolidated plugin architecture
Consolidates demo_orchestrator.py capabilities into the unified plugin system.

Key Features:
- Realistic multi-agent development scenarios  
- Industry-specific demo patterns (E-commerce, Blog, API)
- Agent persona simulation with realistic behavior patterns
- Progressive workflow phases with dependencies
- Demo lifecycle management (start, monitor, stop)
- Performance metrics and success criteria tracking

Epic 1 Performance Targets:
- <100ms demo status retrieval
- <50ms scenario switching
- Memory-efficient scenario storage
- Lazy loading of demo data
"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from .base_plugin import OrchestratorPlugin, PluginMetadata, PluginError
from ..simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance
from ...models.task import TaskStatus, TaskPriority
from ..logging_service import get_component_logger

logger = get_component_logger("demo_orchestrator_plugin")


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


class DemoOrchestratorPlugin(OrchestratorPlugin):
    """
    Demo orchestrator plugin providing realistic multi-agent demonstrations.
    
    Epic 1 Phase 2.2: Consolidation of demo_orchestrator.py capabilities 
    into the unified plugin architecture for SimpleOrchestrator integration.
    """
    
    def __init__(self):
        super().__init__(
            metadata=PluginMetadata(
                name="demo_orchestrator",
                version="2.2.0",
                description="Realistic multi-agent development scenarios with personas",
                author="LeanVibe Agent Hive",
                capabilities=["demo_management", "scenario_execution", "persona_simulation"],
                dependencies=["simple_orchestrator"],
                epic_phase="Epic 1 Phase 2.2"
            )
        )
        
        # Demo state
        self.scenarios: Dict[str, DemoScenario] = {}
        self.active_demo: Optional[DemoScenario] = None
        self.demo_state: Dict[str, Any] = {}
        self.task_queue: List[DemoTask] = []
        self.completed_tasks: List[DemoTask] = []
        self.demo_metrics: Dict[str, Any] = {}
        
        # Performance tracking for Epic 1 targets
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_usage_mb: float = 0.0
        
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the demo orchestrator plugin."""
        await super().initialize(context)
        
        self.orchestrator = context.get("orchestrator")
        if not isinstance(self.orchestrator, SimpleOrchestrator):
            raise PluginError("DemoOrchestratorPlugin requires SimpleOrchestrator")
        
        # Initialize built-in scenarios
        await self._initialize_scenarios()
        
        logger.info("Demo Orchestrator Plugin initialized with realistic scenarios")
        
    async def _initialize_scenarios(self):
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
                role=AgentRole.QA_ENGINEER,
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
                role=AgentRole.META_AGENT,
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
            
            # Implementation Phase
            DemoTask(
                id="api-product-catalog",
                title="Product Catalog API",
                description="Build REST API for product management with search and filtering",
                agent_role="backend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=90,
                dependencies=["setup-api-auth"],
                skills_required=["REST API", "search optimization", "caching"],
                deliverables=["product endpoints", "search API", "filtering logic"]
            ),
            DemoTask(
                id="ui-product-listing",
                title="Product Listing Page",
                description="Create responsive product grid with search and filtering UI",
                agent_role="frontend_developer",
                phase=DemoPhase.IMPLEMENTATION,
                priority=TaskPriority.HIGH,
                estimated_duration=80,
                dependencies=["ui-component-library", "api-product-catalog"],
                skills_required=["React hooks", "responsive design", "API integration"],
                deliverables=["product grid", "search interface", "filter controls"]
            ),
            
            # Testing Phase
            DemoTask(
                id="e2e-testing",
                title="End-to-End Testing",
                description="Complete user journey testing with Playwright automation",
                agent_role="qa_engineer",
                phase=DemoPhase.TESTING,
                priority=TaskPriority.HIGH,
                estimated_duration=90,
                dependencies=["ui-product-listing"],
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
                dependencies=["e2e-testing"],
                skills_required=["Docker", "container orchestration", "environment config"],
                deliverables=["Dockerfile", "docker-compose.yml", "deployment guide"]
            ),
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
                role=AgentRole.QA_ENGINEER,
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
                agent_role="qa_engineer",
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
        """Start a demo scenario with Epic 1 performance tracking."""
        import time
        start_time_ms = time.time()
        
        try:
            if scenario_id not in self.scenarios:
                raise PluginError(f"Unknown scenario: {scenario_id}")
            
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
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("start_demo", operation_time_ms)
            
            logger.info(f"Started demo scenario: {scenario.name}", 
                       scenario_id=scenario_id, 
                       total_tasks=len(scenario.tasks),
                       operation_time_ms=operation_time_ms)
            
            return {
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "description": scenario.description,
                "agents_to_spawn": len(scenario.agents),
                "total_tasks": len(scenario.tasks),
                "estimated_duration": scenario.duration_minutes,
                "performance": {
                    "operation_time_ms": round(operation_time_ms, 2),
                    "epic1_compliant": operation_time_ms < 100.0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start demo: {e}")
            raise PluginError(f"Demo start failed: {e}")
    
    async def spawn_demo_agents(self) -> List[Dict[str, Any]]:
        """Spawn all agents for the active demo."""
        if not self.active_demo:
            raise PluginError("No active demo scenario")
        
        spawned_agents = []
        
        for agent_persona in self.active_demo.agents:
            try:
                agent_id = await self.orchestrator.spawn_agent(
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
    
    async def get_demo_status(self) -> Dict[str, Any]:
        """Get comprehensive demo status with Epic 1 performance tracking."""
        import time
        start_time_ms = time.time()
        
        try:
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
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("get_demo_status", operation_time_ms)
            
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
                "metrics": self.demo_state["metrics"],
                "performance": {
                    "operation_time_ms": round(operation_time_ms, 2),
                    "epic1_compliant": operation_time_ms < 100.0,
                    "memory_usage_mb": self.memory_usage_mb
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get demo status: {e}")
            return {"active": False, "error": str(e)}
    
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
                success = await self.orchestrator.shutdown_agent(
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
        import time
        start_time_ms = time.time()
        
        scenarios = [
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
        
        # Epic 1 Performance tracking
        operation_time_ms = (time.time() - start_time_ms) * 1000
        self._record_operation_time("get_available_scenarios", operation_time_ms)
        
        return scenarios
    
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
    
    def _record_operation_time(self, operation: str, time_ms: float) -> None:
        """Record operation time for Epic 1 performance monitoring."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        times = self.operation_times[operation]
        times.append(time_ms)
        
        # Keep only last 50 measurements for memory efficiency
        if len(times) > 50:
            times.pop(0)
        
        # Log performance warnings for Epic 1 targets
        if time_ms > 100.0:
            logger.warning("Demo operation slow", 
                         operation=operation,
                         operation_time_ms=time_ms,
                         target_ms=100.0)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics for Epic 1 monitoring."""
        metrics = {}
        
        for operation, times in self.operation_times.items():
            if times:
                import statistics
                metrics[operation] = {
                    "avg_ms": round(statistics.mean(times), 2),
                    "max_ms": round(max(times), 2),
                    "min_ms": round(min(times), 2),
                    "count": len(times),
                    "last_ms": round(times[-1], 2),
                    "epic1_compliant": statistics.mean(times) < 100.0
                }
        
        return {
            "operation_metrics": metrics,
            "scenarios_loaded": len(self.scenarios),
            "active_demo": self.active_demo.id if self.active_demo else None,
            "memory_usage_mb": self.memory_usage_mb
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        # Stop any active demo
        if self.active_demo:
            await self.stop_demo(save_results=False)
        
        # Clear scenarios and state
        self.scenarios.clear()
        self.demo_state.clear()
        self.task_queue.clear()
        self.completed_tasks.clear()
        self.demo_metrics.clear()
        self.operation_times.clear()
        
        await super().cleanup()
        
        logger.info("Demo Orchestrator Plugin cleanup complete")


def create_demo_orchestrator_plugin() -> DemoOrchestratorPlugin:
    """Factory function to create demo orchestrator plugin."""
    return DemoOrchestratorPlugin()


# Export for SimpleOrchestrator integration
__all__ = [
    'DemoOrchestratorPlugin',
    'DemoTask',
    'AgentPersona', 
    'DemoScenario',
    'DemoPhase',
    'create_demo_orchestrator_plugin'
]
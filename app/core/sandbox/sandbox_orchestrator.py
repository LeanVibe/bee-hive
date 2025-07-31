"""
Sandbox-Aware Orchestrator
Routes to mock services in sandbox mode while maintaining full functionality
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

from .sandbox_config import get_sandbox_config, is_sandbox_mode
from .mock_anthropic_client import MockAnthropicClient, create_mock_anthropic_client
from ..config import settings

logger = structlog.get_logger()


class SandboxAgentRole(Enum):
    """Agent roles for sandbox multi-agent coordination."""
    ARCHITECT = "architect"
    DEVELOPER = "developer" 
    TESTER = "tester"
    REVIEWER = "reviewer"
    DOCUMENTER = "documenter"


@dataclass
class SandboxAgent:
    """Mock agent for sandbox demonstrations."""
    id: str
    role: SandboxAgentRole
    name: str
    description: str
    status: str = "idle"
    current_task: Optional[str] = None
    completion_percentage: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "role": self.role.value,
            "name": self.name, 
            "description": self.description,
            "status": self.status,
            "current_task": self.current_task,
            "completion_percentage": self.completion_percentage
        }


@dataclass
class SandboxTask:
    """Mock task for sandbox demonstrations."""
    id: str
    title: str
    description: str
    assigned_agent: Optional[str] = None
    status: str = "pending"
    progress: int = 0
    artifacts: List[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "progress": self.progress,
            "artifacts": self.artifacts,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class SandboxOrchestrator:
    """
    Sandbox-aware orchestrator that provides full autonomous development 
    demonstrations using mock services.
    """
    
    def __init__(self):
        self.config = get_sandbox_config()
        self.anthropic_client = None
        self.agents: Dict[str, SandboxAgent] = {}
        self.tasks: Dict[str, SandboxTask] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize mock services if in sandbox mode
        if self.config.enabled:
            self._initialize_sandbox_services()
            self._create_demo_agents()
        
        logger.info("SandboxOrchestrator initialized", 
                   sandbox_mode=self.config.enabled,
                   mock_services=self._get_mock_services_list())
    
    def _initialize_sandbox_services(self):
        """Initialize mock services for sandbox mode."""
        if self.config.mock_anthropic:
            self.anthropic_client = create_mock_anthropic_client()
            logger.info("Mock Anthropic client initialized")
    
    def _get_mock_services_list(self) -> List[str]:
        """Get list of active mock services."""
        services = []
        if self.config.mock_anthropic:
            services.append("anthropic")
        if self.config.mock_openai:
            services.append("openai")
        if self.config.mock_github:
            services.append("github")
        return services
    
    def _create_demo_agents(self):
        """Create demonstration agents for sandbox mode."""
        demo_agents = [
            SandboxAgent(
                id="architect-001",
                role=SandboxAgentRole.ARCHITECT,
                name="System Architect",
                description="Designs system architecture and technical specifications"
            ),
            SandboxAgent(
                id="developer-001", 
                role=SandboxAgentRole.DEVELOPER,
                name="Senior Developer",
                description="Implements features with clean, maintainable code"
            ),
            SandboxAgent(
                id="tester-001",
                role=SandboxAgentRole.TESTER,
                name="QA Engineer", 
                description="Creates comprehensive tests and validates functionality"
            ),
            SandboxAgent(
                id="reviewer-001",
                role=SandboxAgentRole.REVIEWER,
                name="Code Reviewer",
                description="Reviews code quality and suggests improvements"
            ),
            SandboxAgent(
                id="documenter-001",
                role=SandboxAgentRole.DOCUMENTER,
                name="Technical Writer",
                description="Creates clear documentation and user guides"
            )
        ]
        
        for agent in demo_agents:
            self.agents[agent.id] = agent
        
        logger.info("Demo agents created", agent_count=len(self.agents))
    
    async def start_autonomous_development(
        self,
        session_id: str,
        task_description: str,
        requirements: List[str] = None,
        complexity: str = "simple"
    ) -> Dict[str, Any]:
        """
        Start autonomous development process in sandbox mode.
        
        Args:
            session_id: Unique session identifier
            task_description: Description of the development task
            requirements: Optional list of specific requirements
            complexity: Task complexity level (simple/moderate/complex)
            
        Returns:
            Development session information
        """
        if not self.config.enabled:
            raise ValueError("Sandbox mode not enabled")
        
        # Create development session
        session = {
            "id": session_id,
            "task_description": task_description,
            "requirements": requirements or [],
            "complexity": complexity,
            "status": "initializing",
            "progress": 0,
            "agents": [],
            "tasks": [],
            "artifacts": [],
            "started_at": datetime.utcnow(),
            "estimated_completion": datetime.utcnow() + timedelta(minutes=10)
        }
        
        self.active_sessions[session_id] = session
        
        # Create development tasks
        development_tasks = self._create_development_tasks(session_id, task_description, complexity)
        
        # Assign agents to tasks
        await self._assign_agents_to_tasks(session_id, development_tasks)
        
        # Start the development process
        asyncio.create_task(self._execute_development_workflow(session_id))
        
        logger.info("Autonomous development started",
                   session_id=session_id,
                   task_count=len(development_tasks),
                   complexity=complexity)
        
        return {
            "session_id": session_id,
            "status": "started",
            "estimated_duration_minutes": 10,
            "agents_assigned": len([task for task in development_tasks if task.assigned_agent]),
            "tasks_created": len(development_tasks)
        }
    
    def _create_development_tasks(
        self,
        session_id: str,
        description: str,
        complexity: str
    ) -> List[SandboxTask]:
        """Create development tasks based on complexity."""
        
        base_tasks = [
            SandboxTask(
                id=f"{session_id}-understanding",
                title="Requirements Analysis",
                description="Analyze requirements and create technical specification"
            ),
            SandboxTask(
                id=f"{session_id}-planning",
                title="Implementation Planning", 
                description="Design architecture and create implementation plan"
            ),
            SandboxTask(
                id=f"{session_id}-implementation",
                title="Code Implementation",
                description="Implement the core functionality with proper structure"
            ),
            SandboxTask(
                id=f"{session_id}-testing",
                title="Test Creation",
                description="Create comprehensive test suite"
            ),
            SandboxTask(
                id=f"{session_id}-documentation",
                title="Documentation",
                description="Create user documentation and code comments"
            )
        ]
        
        # Add complexity-specific tasks
        if complexity in ["moderate", "complex"]:
            base_tasks.extend([
                SandboxTask(
                    id=f"{session_id}-integration",
                    title="Integration Testing",
                    description="Test component integration and system behavior"
                ),
                SandboxTask(
                    id=f"{session_id}-review",
                    title="Code Review",
                    description="Review code quality and suggest improvements"
                )
            ])
        
        if complexity == "complex":
            base_tasks.extend([
                SandboxTask(
                    id=f"{session_id}-deployment",
                    title="Deployment Setup", 
                    description="Configure deployment and CI/CD pipeline"
                ),
                SandboxTask(
                    id=f"{session_id}-monitoring",
                    title="Monitoring Setup",
                    description="Set up logging and monitoring systems"
                )
            ])
        
        # Store tasks
        for task in base_tasks:
            self.tasks[task.id] = task
        
        return base_tasks
    
    async def _assign_agents_to_tasks(
        self,
        session_id: str,
        tasks: List[SandboxTask]
    ):
        """Assign appropriate agents to development tasks."""
        
        # Agent assignment strategy
        agent_assignments = {
            "understanding": "architect-001",
            "planning": "architect-001", 
            "implementation": "developer-001",
            "testing": "tester-001",
            "integration": "tester-001",
            "documentation": "documenter-001",
            "review": "reviewer-001",
            "deployment": "developer-001",
            "monitoring": "developer-001"
        }
        
        for task in tasks:
            # Extract task type from task ID
            task_type = task.id.split("-")[-1]
            agent_id = agent_assignments.get(task_type, "developer-001")
            
            if agent_id in self.agents:
                task.assigned_agent = agent_id
                self.agents[agent_id].status = "assigned"
        
        logger.info("Agents assigned to tasks",
                   session_id=session_id,
                   assignments=len([task for task in tasks if task.assigned_agent]))
    
    async def _execute_development_workflow(self, session_id: str):
        """Execute the complete development workflow with realistic timing."""
        
        session = self.active_sessions[session_id]
        session["status"] = "running"
        
        # Get tasks for this session
        session_tasks = [task for task in self.tasks.values() 
                        if task.id.startswith(session_id)]
        
        try:
            # Execute tasks in sequence with realistic delays
            for i, task in enumerate(session_tasks):
                await self._execute_task(task, session_id)
                
                # Update session progress
                session["progress"] = int(((i + 1) / len(session_tasks)) * 100)
                
                # Add realistic delay between tasks
                await asyncio.sleep(2.0)
            
            # Mark session as completed
            session["status"] = "completed"
            session["progress"] = 100
            session["completed_at"] = datetime.utcnow()
            
            logger.info("Development workflow completed", session_id=session_id)
            
        except Exception as e:
            # Handle errors gracefully
            session["status"] = "error"
            session["error"] = str(e)
            logger.error("Development workflow failed", session_id=session_id, error=str(e))
    
    async def _execute_task(self, task: SandboxTask, session_id: str):
        """Execute individual development task with AI interaction."""
        
        task.status = "running"
        task.started_at = datetime.utcnow()
        
        # Update assigned agent status
        if task.assigned_agent in self.agents:
            agent = self.agents[task.assigned_agent]
            agent.status = "working"
            agent.current_task = task.title
        
        # Simulate AI interaction for task execution
        if self.anthropic_client:
            await self._simulate_ai_task_execution(task, session_id)
        
        # Mark task as completed
        task.status = "completed"
        task.progress = 100
        task.completed_at = datetime.utcnow()
        
        # Update agent status
        if task.assigned_agent in self.agents:
            agent = self.agents[task.assigned_agent]
            agent.status = "idle"
            agent.current_task = None
            agent.completion_percentage = 100
        
        logger.info("Task completed", task_id=task.id, session_id=session_id)
    
    async def _simulate_ai_task_execution(self, task: SandboxTask, session_id: str):
        """Simulate AI interaction for realistic task execution."""
        
        # Create task-specific prompt
        prompt = f"""
Please help me with this development task:

Task: {task.title}
Description: {task.description}

Please provide a detailed response with implementation steps.
"""
        
        # Simulate AI interaction
        try:
            response = await self.anthropic_client.messages_create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            
            # Extract response content
            if response.content and len(response.content) > 0:
                ai_response = response.content[0].get("text", "Task completed successfully")
                
                # Create artifact from AI response
                artifact = {
                    "id": f"{task.id}-artifact",
                    "name": f"{task.title} Result",
                    "type": "ai_response",
                    "content": ai_response,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                task.artifacts.append(artifact)
        
        except Exception as e:
            logger.warning("AI simulation failed for task", task_id=task.id, error=str(e))
            
            # Create fallback artifact
            artifact = {
                "id": f"{task.id}-artifact",
                "name": f"{task.title} Result",
                "type": "completion_notice",
                "content": f"Task '{task.title}' completed successfully in sandbox mode.",
                "created_at": datetime.utcnow().isoformat()
            }
            
            task.artifacts.append(artifact)
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of development session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        session_tasks = [task.to_dict() for task in self.tasks.values() 
                        if task.id.startswith(session_id)]
        
        return {
            **session,
            "tasks": session_tasks,
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "sandbox_mode": True,
            "mock_services": self._get_mock_services_list()
        }
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all available agents."""
        return [agent.to_dict() for agent in self.agents.values()]
    
    def get_sandbox_status(self) -> Dict[str, Any]:
        """Get comprehensive sandbox status."""
        return {
            "enabled": self.config.enabled,
            "auto_detected": self.config.auto_detected,
            "reason": self.config.reason,
            "mock_services": self._get_mock_services_list(),
            "active_sessions": len(self.active_sessions),
            "available_agents": len(self.agents),
            "demo_features_enabled": self.config.demo_scenarios_enabled
        }


def create_sandbox_orchestrator() -> SandboxOrchestrator:
    """Create sandbox orchestrator instance."""
    return SandboxOrchestrator()
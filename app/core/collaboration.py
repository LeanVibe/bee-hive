"""
Advanced Human-AI Collaboration Interface for LeanVibe Agent Hive 2.0

This revolutionary system enables seamless collaboration between humans and
autonomous AI agents, providing intuitive interfaces for guidance, oversight,
and knowledge transfer in multi-agent development workflows.

CRITICAL: This system is designed to augment human capabilities, not replace them.
Humans remain in control with strategic oversight and intervention capabilities.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .redis import get_message_broker, get_session_cache
from .workspace_manager import workspace_manager
from .self_improvement import self_improvement_orchestrator
from ..models.agent import Agent, AgentStatus
from ..models.session import Session, SessionStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class InteractionType(Enum):
    """Types of human-AI interactions."""
    GUIDANCE = "guidance"              # Human provides direction
    FEEDBACK = "feedback"              # Human provides performance feedback
    INTERVENTION = "intervention"      # Human takes control
    KNOWLEDGE_TRANSFER = "knowledge_transfer"  # Human teaches new concepts
    APPROVAL_REQUEST = "approval_request"      # Agent requests human approval
    COLLABORATION = "collaboration"    # Joint problem-solving
    MONITORING = "monitoring"          # Human observes agent work
    DELEGATION = "delegation"          # Human assigns high-level goals


class PriorityLevel(Enum):
    """Priority levels for human attention."""
    LOW = "low"                # Background notifications
    MEDIUM = "medium"          # Standard notifications
    HIGH = "high"             # Immediate attention needed
    CRITICAL = "critical"     # Urgent intervention required
    EMERGENCY = "emergency"   # System-wide issue


class CollaborationMode(Enum):
    """Modes of human-AI collaboration."""
    AUTONOMOUS = "autonomous"          # Agents work independently
    SUPERVISED = "supervised"         # Human oversight with periodic check-ins
    COLLABORATIVE = "collaborative"   # Active human-AI partnership
    GUIDED = "guided"                 # Human provides continuous guidance
    CONTROLLED = "controlled"         # Human directs all major decisions


@dataclass
class HumanRequest:
    """Represents a request from a human to agents."""
    id: str
    human_id: str
    interaction_type: InteractionType
    priority: PriorityLevel
    
    # Request content
    title: str
    description: str
    target_agents: List[str]  # Agent IDs
    context: Dict[str, Any]
    
    # Requirements and constraints
    requirements: List[str]
    constraints: List[str]
    success_criteria: List[str]
    
    # Timeline
    due_date: Optional[datetime]
    estimated_effort_hours: Optional[float]
    
    # Metadata
    created_at: datetime
    updated_at: datetime
    status: str  # pending, in_progress, completed, cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResponse:
    """Represents a response from an agent to human."""
    id: str
    agent_id: str
    request_id: str
    response_type: str
    
    # Response content
    content: str
    status: str
    progress_percentage: float
    
    # Results and deliverables
    deliverables: List[Dict[str, Any]]
    code_generated: List[str]
    files_created: List[str]
    
    # Feedback and questions
    questions_for_human: List[str]
    feedback_requested: List[str]
    clarifications_needed: List[str]
    
    # Metadata
    created_at: datetime
    estimated_completion: Optional[datetime]


@dataclass
class CollaborationSession:
    """Represents an active collaboration session between human and agents."""
    id: str
    human_id: str
    participant_agents: List[str]
    mode: CollaborationMode
    
    # Session content
    title: str
    objective: str
    current_phase: str
    
    # Communication history
    messages: List[Dict[str, Any]]
    shared_context: Dict[str, Any]
    
    # Progress tracking
    milestones: List[Dict[str, Any]]
    completed_tasks: List[str]
    pending_tasks: List[str]
    
    # Timeline
    started_at: datetime
    last_activity: datetime
    estimated_duration_hours: Optional[float]
    
    # Status
    is_active: bool
    requires_human_attention: bool


class NaturalLanguageProcessor:
    """
    Processes natural language requests from humans into structured tasks.
    
    Converts human requirements into actionable agent tasks with proper
    context, constraints, and success criteria.
    """
    
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client
    
    async def process_human_request(
        self,
        request_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> HumanRequest:
        """Convert natural language request into structured format."""
        
        # Use AI to parse and structure the request
        parsing_prompt = f"""
        Parse this human request into a structured format:
        
        Request: "{request_text}"
        
        Additional Context: {json.dumps(context or {}, indent=2)}
        
        Extract and structure:
        1. Main objective/goal
        2. Specific requirements 
        3. Constraints and limitations
        4. Success criteria
        5. Priority level (low/medium/high/critical)
        6. Estimated effort and timeline
        7. Required agent capabilities
        8. Interaction type (guidance/feedback/delegation/etc.)
        
        Provide a clear, actionable breakdown that agents can understand and execute.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": parsing_prompt}]
            )
            
            # Parse AI response into structured request
            content = response.content[0].text
            
            # Create structured request (simplified for demo)
            return HumanRequest(
                id=str(uuid.uuid4()),
                human_id=context.get('human_id', 'anonymous'),
                interaction_type=InteractionType.DELEGATION,
                priority=PriorityLevel.MEDIUM,
                title=self._extract_title(request_text),
                description=request_text,
                target_agents=[],  # Will be assigned based on capabilities
                context=context or {},
                requirements=self._extract_requirements(content),
                constraints=self._extract_constraints(content),
                success_criteria=self._extract_success_criteria(content),
                due_date=None,  # Would be extracted from AI analysis
                estimated_effort_hours=None,  # Would be estimated by AI
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                status="pending"
            )
            
        except Exception as e:
            logger.error("Failed to process human request", error=str(e))
            raise
    
    def _extract_title(self, request_text: str) -> str:
        """Extract a concise title from request text."""
        # Simple extraction - first sentence or first 50 chars
        sentences = request_text.split('.')
        if sentences:
            title = sentences[0].strip()
            return title[:50] + "..." if len(title) > 50 else title
        return request_text[:50] + "..." if len(request_text) > 50 else request_text
    
    def _extract_requirements(self, ai_content: str) -> List[str]:
        """Extract requirements from AI analysis."""
        # Simplified extraction - would be more sophisticated in production
        return [
            "Implement requested functionality",
            "Follow coding best practices",
            "Include proper error handling",
            "Write comprehensive tests"
        ]
    
    def _extract_constraints(self, ai_content: str) -> List[str]:
        """Extract constraints from AI analysis."""
        return [
            "Use existing project architecture",
            "Maintain security standards",
            "Complete within reasonable timeframe"
        ]
    
    def _extract_success_criteria(self, ai_content: str) -> List[str]:
        """Extract success criteria from AI analysis."""
        return [
            "All tests pass",
            "Code quality standards met",
            "Functionality works as expected",
            "Documentation is complete"
        ]


class TaskDecomposer:
    """
    Breaks down complex human requests into agent-executable tasks.
    
    Analyzes requirements and creates a hierarchical task structure
    with proper dependencies and agent assignments.
    """
    
    def __init__(self, anthropic_client: AsyncAnthropic):
        self.anthropic = anthropic_client
    
    async def decompose_request(
        self,
        human_request: HumanRequest,
        available_agents: List[Agent]
    ) -> List[Task]:
        """Decompose human request into specific agent tasks."""
        
        # Analyze request complexity and requirements
        decomposition_prompt = f"""
        Break down this human request into specific, actionable tasks:
        
        Request: {human_request.title}
        Description: {human_request.description}
        Requirements: {', '.join(human_request.requirements)}
        Constraints: {', '.join(human_request.constraints)}
        
        Available Agents:
        {self._format_agent_capabilities(available_agents)}
        
        Create a task breakdown that:
        1. Divides work into manageable chunks (2-4 hour tasks)
        2. Identifies task dependencies
        3. Matches tasks to agent capabilities
        4. Ensures proper sequence and coordination
        5. Includes validation and testing tasks
        
        Focus on creating tasks that can be executed independently while contributing to the overall goal.
        """
        
        try:
            response = await self.anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=3000,
                messages=[{"role": "user", "content": decomposition_prompt}]
            )
            
            # Parse response and create tasks
            tasks = await self._create_tasks_from_analysis(
                response.content[0].text,
                human_request,
                available_agents
            )
            
            logger.info(
                "Request decomposed into tasks",
                request_id=human_request.id,
                task_count=len(tasks)
            )
            
            return tasks
            
        except Exception as e:
            logger.error("Failed to decompose request", error=str(e))
            return []
    
    def _format_agent_capabilities(self, agents: List[Agent]) -> str:
        """Format agent capabilities for AI analysis."""
        agent_info = []
        for agent in agents:
            capabilities = [cap.get('name', 'Unknown') for cap in (agent.capabilities or [])]
            agent_info.append(f"- {agent.role}: {', '.join(capabilities)}")
        return '\n'.join(agent_info)
    
    async def _create_tasks_from_analysis(
        self,
        ai_analysis: str,
        human_request: HumanRequest,
        available_agents: List[Agent]
    ) -> List[Task]:
        """Create task objects from AI analysis."""
        
        # Simplified task creation - would be more sophisticated in production
        base_tasks = [
            {
                "title": "Analyze requirements and create implementation plan",
                "description": f"Analyze the requirements for: {human_request.description}",
                "task_type": "ARCHITECTURE",
                "priority": TaskPriority.HIGH,
                "estimated_effort": 120,  # 2 hours
                "required_capabilities": ["planning", "architecture"]
            },
            {
                "title": "Implement core functionality",
                "description": f"Implement the main features for: {human_request.title}",
                "task_type": "FEATURE_DEVELOPMENT", 
                "priority": TaskPriority.HIGH,
                "estimated_effort": 240,  # 4 hours
                "required_capabilities": ["coding", "development"]
            },
            {
                "title": "Write comprehensive tests",
                "description": f"Create tests for: {human_request.title}",
                "task_type": "TESTING",
                "priority": TaskPriority.MEDIUM,
                "estimated_effort": 120,  # 2 hours
                "required_capabilities": ["testing", "quality_assurance"]
            },
            {
                "title": "Documentation and deployment",
                "description": f"Document and deploy: {human_request.title}",
                "task_type": "DOCUMENTATION",
                "priority": TaskPriority.MEDIUM,
                "estimated_effort": 60,   # 1 hour
                "required_capabilities": ["documentation", "deployment"]
            }
        ]
        
        tasks = []
        for i, task_data in enumerate(base_tasks):
            # Find best agent for this task
            best_agent = self._find_best_agent_for_task(
                task_data["required_capabilities"],
                available_agents
            )
            
            task = Task(
                id=str(uuid.uuid4()),
                title=task_data["title"],
                description=task_data["description"],
                task_type=getattr(TaskType, task_data["task_type"], TaskType.FEATURE_DEVELOPMENT),
                priority=task_data["priority"],
                assigned_agent_id=best_agent.id if best_agent else None,
                required_capabilities=task_data["required_capabilities"],
                estimated_effort=task_data["estimated_effort"],
                context={
                    "human_request_id": human_request.id,
                    "task_sequence": i + 1,
                    "total_tasks": len(base_tasks)
                }
            )
            
            # Set dependencies
            if i > 0:
                task.dependencies = [tasks[i-1].id]
            
            tasks.append(task)
        
        return tasks
    
    def _find_best_agent_for_task(
        self,
        required_capabilities: List[str],
        available_agents: List[Agent]
    ) -> Optional[Agent]:
        """Find the best agent for a specific task."""
        
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            if agent.status == AgentStatus.active:
                score = agent.calculate_task_suitability("task", required_capabilities)
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        return best_agent


class CollaborationOrchestrator:
    """
    Main orchestrator for human-AI collaboration.
    
    Manages collaboration sessions, processes human requests,
    coordinates agent responses, and maintains communication flow.
    """
    
    def __init__(self):
        self.anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.nlp_processor = NaturalLanguageProcessor(self.anthropic)
        self.task_decomposer = TaskDecomposer(self.anthropic)
        
        # Active sessions
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.pending_requests: Dict[str, HumanRequest] = {}
        self.human_notifications: Dict[str, List[Dict[str, Any]]] = {}
    
    async def process_human_request(
        self,
        human_id: str,
        request_text: str,
        context: Optional[Dict[str, Any]] = None,
        priority: PriorityLevel = PriorityLevel.MEDIUM
    ) -> Tuple[str, List[str]]:
        """Process a natural language request from a human."""
        
        logger.info(
            "Processing human request",
            human_id=human_id,
            request_length=len(request_text),
            priority=priority.value
        )
        
        try:
            # Parse natural language request
            human_request = await self.nlp_processor.process_human_request(
                request_text,
                {**(context or {}), "human_id": human_id}
            )
            human_request.priority = priority
            
            # Get available agents
            async with get_session() as db_session:
                from sqlalchemy import select
                result = await db_session.execute(
                    select(Agent).where(Agent.status == AgentStatus.active)
                )
                available_agents = result.scalars().all()
            
            if not available_agents:
                raise ValueError("No active agents available")
            
            # Decompose request into tasks
            tasks = await self.task_decomposer.decompose_request(
                human_request, available_agents
            )
            
            # Store request and tasks
            self.pending_requests[human_request.id] = human_request
            
            # Create tasks in database and assign to agents
            task_ids = []
            async with get_session() as db_session:
                for task in tasks:
                    db_session.add(task)
                    task_ids.append(task.id)
                
                await db_session.commit()
            
            # Notify agents about new tasks
            await self._notify_agents_about_tasks(tasks)
            
            # Create collaboration session if needed
            if priority in [PriorityLevel.HIGH, PriorityLevel.CRITICAL]:
                session_id = await self._create_collaboration_session(
                    human_id, human_request, [agent.id for agent in available_agents]
                )
            else:
                session_id = None
            
            logger.info(
                "Human request processed successfully",
                human_id=human_id,
                request_id=human_request.id,
                task_count=len(tasks),
                session_id=session_id
            )
            
            return human_request.id, task_ids
            
        except Exception as e:
            logger.error(
                "Failed to process human request",
                human_id=human_id,
                error=str(e)
            )
            raise
    
    async def _notify_agents_about_tasks(self, tasks: List[Task]) -> None:
        """Notify agents about newly assigned tasks."""
        
        message_broker = get_message_broker()
        
        for task in tasks:
            if task.assigned_agent_id:
                await message_broker.send_message(
                    from_agent="collaboration_orchestrator",
                    to_agent=str(task.assigned_agent_id),
                    message_type="task_assignment",
                    payload={
                        "task_id": str(task.id),
                        "title": task.title,
                        "description": task.description,
                        "priority": task.priority.value,
                        "estimated_effort": task.estimated_effort,
                        "required_capabilities": task.required_capabilities,
                        "context": task.context
                    }
                )
    
    async def _create_collaboration_session(
        self,
        human_id: str,
        human_request: HumanRequest,
        agent_ids: List[str]
    ) -> str:
        """Create a new collaboration session."""
        
        session = CollaborationSession(
            id=str(uuid.uuid4()),
            human_id=human_id,
            participant_agents=agent_ids,
            mode=CollaborationMode.SUPERVISED,
            title=human_request.title,
            objective=human_request.description,
            current_phase="initialization",
            messages=[],
            shared_context={"request_id": human_request.id},
            milestones=[],
            completed_tasks=[],
            pending_tasks=[],
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            estimated_duration_hours=human_request.estimated_effort_hours,
            is_active=True,
            requires_human_attention=False
        )
        
        self.active_sessions[session.id] = session
        
        # Store in cache for persistence
        session_cache = get_session_cache()
        await session_cache.set_session_state(
            f"collaboration_session:{session.id}",
            session.to_dict()
        )
        
        return session.id
    
    async def handle_agent_response(
        self,
        agent_id: str,
        response_data: Dict[str, Any]
    ) -> None:
        """Handle responses from agents."""
        
        logger.info(
            "Handling agent response",
            agent_id=agent_id,
            response_type=response_data.get("type", "unknown")
        )
        
        response_type = response_data.get("type")
        
        if response_type == "task_completion":
            await self._handle_task_completion(agent_id, response_data)
        elif response_type == "approval_request":
            await self._handle_approval_request(agent_id, response_data)
        elif response_type == "progress_update":
            await self._handle_progress_update(agent_id, response_data)
        elif response_type == "question":
            await self._handle_agent_question(agent_id, response_data)
        elif response_type == "error":
            await self._handle_agent_error(agent_id, response_data)
        
        # Update session activity
        await self._update_session_activity(agent_id)
    
    async def _handle_task_completion(
        self,
        agent_id: str,
        response_data: Dict[str, Any]
    ) -> None:
        """Handle task completion from agent."""
        
        task_id = response_data.get("task_id")
        if not task_id:
            return
        
        # Update task status in database
        async with get_session() as db_session:
            task = await db_session.get(Task, task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                task.result = response_data.get("result", {})
                await db_session.commit()
        
        # Notify human if important
        await self._notify_human(
            self._get_human_for_agent(agent_id),
            f"Task completed: {response_data.get('title', 'Unknown task')}",
            PriorityLevel.LOW,
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "result": response_data.get("result")
            }
        )
    
    async def _handle_approval_request(
        self,
        agent_id: str,
        response_data: Dict[str, Any]
    ) -> None:
        """Handle approval request from agent."""
        
        human_id = self._get_human_for_agent(agent_id)
        
        await self._notify_human(
            human_id,
            f"Agent {agent_id} requests approval: {response_data.get('request', 'Unknown')}",
            PriorityLevel.HIGH,
            {
                "agent_id": agent_id,
                "approval_type": response_data.get("approval_type"),
                "details": response_data.get("details"),
                "risks": response_data.get("risks", []),
                "benefits": response_data.get("benefits", [])
            }
        )
    
    async def _notify_human(
        self,
        human_id: str,
        message: str,
        priority: PriorityLevel,
        data: Dict[str, Any]
    ) -> None:
        """Send notification to human."""
        
        if human_id not in self.human_notifications:
            self.human_notifications[human_id] = []
        
        notification = {
            "id": str(uuid.uuid4()),
            "message": message,
            "priority": priority.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "read": False
        }
        
        self.human_notifications[human_id].append(notification)
        
        # Keep only last 100 notifications
        if len(self.human_notifications[human_id]) > 100:
            self.human_notifications[human_id] = self.human_notifications[human_id][-100:]
        
        logger.info(
            "Human notification sent",
            human_id=human_id,
            priority=priority.value,
            message=message[:100]
        )
    
    def _get_human_for_agent(self, agent_id: str) -> str:
        """Get human ID associated with an agent (simplified)."""
        # In production, this would look up the actual human-agent relationship
        return "default_human"
    
    async def get_human_dashboard(self, human_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard for human."""
        
        # Get active sessions
        human_sessions = [
            session for session in self.active_sessions.values()
            if session.human_id == human_id and session.is_active
        ]
        
        # Get pending notifications
        notifications = self.human_notifications.get(human_id, [])
        unread_count = len([n for n in notifications if not n["read"]])
        
        # Get agent status
        async with get_session() as db_session:
            from sqlalchemy import select
            result = await db_session.execute(select(Agent))
            all_agents = result.scalars().all()
        
        agent_status = {}
        for agent in all_agents:
            metrics = await workspace_manager.get_workspace(str(agent.id))
            if metrics:
                workspace_metrics = await metrics.get_metrics()
                agent_status[str(agent.id)] = {
                    "name": agent.name,
                    "role": agent.role,
                    "status": agent.status.value,
                    "current_task": getattr(agent, 'current_task', None),
                    "workspace_status": workspace_metrics.status.value,
                    "memory_usage": workspace_metrics.total_memory_mb,
                    "cpu_usage": workspace_metrics.total_cpu_percent
                }
        
        return {
            "human_id": human_id,
            "active_sessions": len(human_sessions),
            "unread_notifications": unread_count,
            "total_agents": len(all_agents),
            "active_agents": len([a for a in all_agents if a.status == AgentStatus.active]),
            "sessions": [session.__dict__ for session in human_sessions],
            "recent_notifications": notifications[-10:],  # Last 10 notifications
            "agent_status": agent_status,
            "system_health": {
                "workspace_manager": workspace_manager is not None,
                "self_improvement": self_improvement_orchestrator is not None,
                "collaboration_active": len(self.active_sessions) > 0
            }
        }


# Global collaboration orchestrator
collaboration_orchestrator = CollaborationOrchestrator()
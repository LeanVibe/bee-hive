"""
Collaborative Development Engine for LeanVibe Agent Hive 2.0

Advanced multi-agent coordination system providing shared project knowledge,
change coordination, conflict prevention, and collaborative planning for
optimal distributed development workflows.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent, AgentStatus
from ..models.project_index import ProjectIndex, FileEntry
from ..models.task import Task, TaskStatus
from .context_integration import AgentContextIntegration
from .task_router import IntelligentTaskRouter

logger = structlog.get_logger()


class CollaborationState(Enum):
    """States of collaboration sessions."""
    PLANNING = "planning"
    ACTIVE = "active"
    COORDINATING = "coordinating"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class ConflictType(Enum):
    """Types of collaboration conflicts."""
    FILE_CONFLICT = "file_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    RESOURCE_CONFLICT = "resource_conflict"
    LOGIC_CONFLICT = "logic_conflict"
    TIMING_CONFLICT = "timing_conflict"
    CAPABILITY_CONFLICT = "capability_conflict"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CollaborationRole(Enum):
    """Roles in collaborative development."""
    LEAD = "lead"
    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


@dataclass
class CollaborationSession:
    """Active collaboration session between agents."""
    session_id: str
    project_id: str
    task_id: str
    participants: List[str]  # Agent IDs
    lead_agent: str
    state: CollaborationState
    shared_context: Dict[str, Any]
    work_assignments: Dict[str, List[str]]  # agent_id -> file_paths
    communication_channel: str
    created_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "participants": self.participants,
            "lead_agent": self.lead_agent,
            "state": self.state.value,
            "shared_context": self.shared_context,
            "work_assignments": self.work_assignments,
            "communication_channel": self.communication_channel,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "actual_completion": self.actual_completion.isoformat() if self.actual_completion else None,
            "metadata": self.metadata
        }


@dataclass
class CollaborationConflict:
    """Detected or potential collaboration conflict."""
    conflict_id: str
    session_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    involved_agents: List[str]
    description: str
    affected_resources: List[str]
    detection_time: datetime
    resolution_suggestions: List[str]
    auto_resolvable: bool
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "session_id": self.session_id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity.value,
            "involved_agents": self.involved_agents,
            "description": self.description,
            "affected_resources": self.affected_resources,
            "detection_time": self.detection_time.isoformat(),
            "resolution_suggestions": self.resolution_suggestions,
            "auto_resolvable": self.auto_resolvable,
            "metadata": self.metadata
        }


@dataclass
class KnowledgeShare:
    """Shared knowledge between agents."""
    share_id: str
    source_agent: str
    target_agents: List[str]
    knowledge_type: str
    content: Dict[str, Any]
    project_id: Optional[str]
    shared_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "share_id": self.share_id,
            "source_agent": self.source_agent,
            "target_agents": self.target_agents,
            "knowledge_type": self.knowledge_type,
            "content": self.content,
            "project_id": self.project_id,
            "shared_at": self.shared_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "access_count": self.access_count
        }


class WorkInProgress(NamedTuple):
    """Current work-in-progress tracking."""
    agent_id: str
    file_paths: List[str]
    started_at: datetime
    estimated_completion: datetime
    progress_percentage: float


class CollaborativeDevelopmentEngine:
    """
    Advanced multi-agent coordination system for collaborative development.
    
    Provides shared project knowledge, change coordination, conflict prevention,
    and collaborative planning for optimal distributed development workflows.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration,
        task_router: IntelligentTaskRouter
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        self.task_router = task_router
        
        # Configuration
        self.session_timeout_hours = 24
        self.conflict_check_interval = 30  # seconds
        self.knowledge_share_ttl = 86400  # 24 hours
        self.wip_tracking_ttl = 3600  # 1 hour
        
        # Conflict detection settings
        self.file_lock_timeout = 300  # 5 minutes
        self.dependency_check_depth = 3
        self.auto_resolve_threshold = 0.8
    
    async def start_collaboration_session(
        self,
        project_id: str,
        task_id: str,
        participant_agents: List[str],
        lead_agent: Optional[str] = None
    ) -> CollaborationSession:
        """
        Start a new collaboration session for multi-agent development.
        
        Args:
            project_id: Project to collaborate on
            task_id: Task being collaborated on
            participant_agents: List of agent IDs participating
            lead_agent: Optional lead agent (auto-selected if not provided)
            
        Returns:
            CollaborationSession instance
        """
        session_id = str(uuid.uuid4())
        
        logger.info(
            "Starting collaboration session",
            session_id=session_id,
            project_id=project_id,
            task_id=task_id,
            participants=participant_agents
        )
        
        try:
            # Select lead agent if not provided
            if not lead_agent:
                lead_agent = await self._select_lead_agent(participant_agents, project_id)
            
            # Create communication channel
            comm_channel = f"collab_session:{session_id}"
            
            # Get shared context for the project
            shared_context = await self._build_shared_context(
                project_id, task_id, participant_agents
            )
            
            # Create initial work assignments
            work_assignments = await self._create_initial_assignments(
                project_id, task_id, participant_agents, shared_context
            )
            
            # Create collaboration session
            session = CollaborationSession(
                session_id=session_id,
                project_id=project_id,
                task_id=task_id,
                participants=participant_agents,
                lead_agent=lead_agent,
                state=CollaborationState.PLANNING,
                shared_context=shared_context,
                work_assignments=work_assignments,
                communication_channel=comm_channel,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(hours=4)
            )
            
            # Store session
            await self._store_collaboration_session(session)
            
            # Initialize session monitoring
            await self._initialize_session_monitoring(session)
            
            # Notify participants
            await self._notify_session_participants(session, "session_started")
            
            logger.info(
                "Collaboration session started",
                session_id=session_id,
                lead_agent=lead_agent,
                participants_count=len(participant_agents)
            )
            
            return session
            
        except Exception as e:
            logger.error(
                "Failed to start collaboration session",
                session_id=session_id,
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def coordinate_work_progress(
        self,
        session_id: str,
        agent_id: str,
        progress_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate work progress between collaborating agents.
        
        Args:
            session_id: Collaboration session ID
            agent_id: Agent reporting progress
            progress_update: Progress information
            
        Returns:
            Coordination response with instructions
        """
        try:
            # Get session
            session = await self._get_collaboration_session(session_id)
            if not session:
                raise ValueError(f"Collaboration session {session_id} not found")
            
            # Validate agent participation
            if agent_id not in session.participants:
                raise ValueError(f"Agent {agent_id} not part of session {session_id}")
            
            # Update work-in-progress tracking
            await self._update_wip_tracking(session_id, agent_id, progress_update)
            
            # Check for conflicts
            conflicts = await self._detect_conflicts(session, agent_id, progress_update)
            
            # Coordinate with other agents if needed
            coordination_actions = await self._coordinate_with_peers(
                session, agent_id, progress_update, conflicts
            )
            
            # Update session state
            await self._update_session_state(session, progress_update, conflicts)
            
            # Generate response
            response = {
                "session_id": session_id,
                "agent_id": agent_id,
                "coordination_status": "success",
                "conflicts_detected": len(conflicts),
                "coordination_actions": coordination_actions,
                "next_steps": await self._generate_next_steps(session, agent_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if conflicts:
                response["conflicts"] = [conflict.to_dict() for conflict in conflicts]
                response["conflict_resolutions"] = await self._suggest_conflict_resolutions(conflicts)
            
            logger.info(
                "Work progress coordinated",
                session_id=session_id,
                agent_id=agent_id,
                conflicts=len(conflicts),
                actions=len(coordination_actions)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Failed to coordinate work progress",
                session_id=session_id,
                agent_id=agent_id,
                error=str(e)
            )
            raise
    
    async def share_knowledge(
        self,
        source_agent: str,
        target_agents: List[str],
        knowledge_type: str,
        content: Dict[str, Any],
        project_id: Optional[str] = None,
        expires_hours: int = 24
    ) -> KnowledgeShare:
        """
        Share knowledge between agents for collaboration.
        
        Args:
            source_agent: Agent sharing knowledge
            target_agents: Agents to share with
            knowledge_type: Type of knowledge being shared
            content: Knowledge content
            project_id: Optional project context
            expires_hours: Hours until knowledge expires
            
        Returns:
            KnowledgeShare instance
        """
        share_id = str(uuid.uuid4())
        
        try:
            knowledge_share = KnowledgeShare(
                share_id=share_id,
                source_agent=source_agent,
                target_agents=target_agents,
                knowledge_type=knowledge_type,
                content=content,
                project_id=project_id,
                shared_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(hours=expires_hours)
            )
            
            # Store knowledge share
            share_key = f"knowledge_share:{share_id}"
            await self.redis.setex(
                share_key,
                expires_hours * 3600,
                json.dumps(knowledge_share.to_dict())
            )
            
            # Index by agent and project for discovery
            for target_agent in target_agents:
                index_key = f"agent_knowledge:{target_agent}"
                await self.redis.sadd(index_key, share_id)
                await self.redis.expire(index_key, expires_hours * 3600)
            
            if project_id:
                project_index_key = f"project_knowledge:{project_id}"
                await self.redis.sadd(project_index_key, share_id)
                await self.redis.expire(project_index_key, expires_hours * 3600)
            
            # Notify target agents
            await self._notify_knowledge_share(knowledge_share)
            
            logger.info(
                "Knowledge shared",
                share_id=share_id,
                source_agent=source_agent,
                target_agents=target_agents,
                knowledge_type=knowledge_type
            )
            
            return knowledge_share
            
        except Exception as e:
            logger.error(
                "Failed to share knowledge",
                source_agent=source_agent,
                target_agents=target_agents,
                error=str(e)
            )
            raise
    
    async def get_agent_knowledge(
        self,
        agent_id: str,
        knowledge_type: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[KnowledgeShare]:
        """
        Get available knowledge for an agent.
        
        Args:
            agent_id: Agent requesting knowledge
            knowledge_type: Optional filter by knowledge type
            project_id: Optional filter by project
            
        Returns:
            List of available knowledge shares
        """
        try:
            # Get agent's knowledge index
            index_key = f"agent_knowledge:{agent_id}"
            share_ids = await self.redis.smembers(index_key)
            
            if project_id:
                # Intersect with project knowledge
                project_key = f"project_knowledge:{project_id}"
                project_shares = await self.redis.smembers(project_key)
                share_ids = share_ids.intersection(project_shares)
            
            # Retrieve knowledge shares
            knowledge_shares = []
            for share_id in share_ids:
                share_key = f"knowledge_share:{share_id}"
                share_data = await self.redis.get(share_key)
                
                if share_data:
                    try:
                        share_dict = json.loads(share_data)
                        
                        # Filter by knowledge type if specified
                        if knowledge_type and share_dict.get("knowledge_type") != knowledge_type:
                            continue
                        
                        # Check if not expired
                        if share_dict.get("expires_at"):
                            expires_at = datetime.fromisoformat(share_dict["expires_at"])
                            if datetime.utcnow() > expires_at:
                                continue
                        
                        knowledge_share = KnowledgeShare(
                            share_id=share_dict["share_id"],
                            source_agent=share_dict["source_agent"],
                            target_agents=share_dict["target_agents"],
                            knowledge_type=share_dict["knowledge_type"],
                            content=share_dict["content"],
                            project_id=share_dict.get("project_id"),
                            shared_at=datetime.fromisoformat(share_dict["shared_at"]),
                            expires_at=datetime.fromisoformat(share_dict["expires_at"]) if share_dict.get("expires_at") else None,
                            access_count=share_dict.get("access_count", 0)
                        )
                        
                        knowledge_shares.append(knowledge_share)
                        
                        # Increment access count
                        await self._increment_access_count(share_id)
                        
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(
                            "Failed to parse knowledge share",
                            share_id=share_id,
                            error=str(e)
                        )
                        continue
            
            # Sort by recency
            knowledge_shares.sort(key=lambda x: x.shared_at, reverse=True)
            
            return knowledge_shares
            
        except Exception as e:
            logger.error(
                "Failed to get agent knowledge",
                agent_id=agent_id,
                error=str(e)
            )
            return []
    
    async def detect_potential_conflicts(
        self,
        project_id: str,
        time_window_minutes: int = 60
    ) -> List[CollaborationConflict]:
        """
        Proactively detect potential conflicts in ongoing work.
        
        Args:
            project_id: Project to check for conflicts
            time_window_minutes: Time window for conflict detection
            
        Returns:
            List of detected potential conflicts
        """
        try:
            # Get active sessions for the project
            active_sessions = await self._get_active_sessions(project_id)
            
            conflicts = []
            
            for session in active_sessions:
                # Check for file conflicts
                file_conflicts = await self._detect_file_conflicts(session)
                conflicts.extend(file_conflicts)
                
                # Check for dependency conflicts
                dep_conflicts = await self._detect_dependency_conflicts(session)
                conflicts.extend(dep_conflicts)
                
                # Check for resource conflicts
                resource_conflicts = await self._detect_resource_conflicts(session)
                conflicts.extend(resource_conflicts)
                
                # Check for timing conflicts
                timing_conflicts = await self._detect_timing_conflicts(session)
                conflicts.extend(timing_conflicts)
            
            # Sort by severity
            conflicts.sort(key=lambda x: self._severity_to_priority(x.severity), reverse=True)
            
            logger.info(
                "Conflict detection completed",
                project_id=project_id,
                conflicts_found=len(conflicts),
                time_window=time_window_minutes
            )
            
            return conflicts
            
        except Exception as e:
            logger.error(
                "Failed to detect conflicts",
                project_id=project_id,
                error=str(e)
            )
            return []
    
    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution_strategy: str,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve a detected collaboration conflict.
        
        Args:
            conflict_id: Conflict to resolve
            resolution_strategy: Strategy to use for resolution
            resolution_data: Data needed for resolution
            
        Returns:
            Resolution result
        """
        try:
            # Get conflict details
            conflict = await self._get_conflict(conflict_id)
            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")
            
            # Apply resolution strategy
            if resolution_strategy == "auto_merge":
                result = await self._auto_merge_resolution(conflict, resolution_data)
            elif resolution_strategy == "priority_assignment":
                result = await self._priority_assignment_resolution(conflict, resolution_data)
            elif resolution_strategy == "work_redistribution":
                result = await self._work_redistribution_resolution(conflict, resolution_data)
            elif resolution_strategy == "manual_intervention":
                result = await self._manual_intervention_resolution(conflict, resolution_data)
            else:
                raise ValueError(f"Unknown resolution strategy: {resolution_strategy}")
            
            # Update conflict status
            await self._mark_conflict_resolved(conflict_id, resolution_strategy, result)
            
            # Notify affected agents
            await self._notify_conflict_resolution(conflict, result)
            
            logger.info(
                "Conflict resolved",
                conflict_id=conflict_id,
                strategy=resolution_strategy,
                success=result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to resolve conflict",
                conflict_id=conflict_id,
                strategy=resolution_strategy,
                error=str(e)
            )
            raise
    
    async def get_collaboration_analytics(
        self,
        project_id: Optional[str] = None,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get collaboration analytics and metrics.
        
        Args:
            project_id: Optional project filter
            time_range_hours: Time range for analytics
            
        Returns:
            Collaboration analytics data
        """
        try:
            # Get sessions in time range
            sessions = await self._get_sessions_in_range(time_range_hours, project_id)
            
            # Get conflicts in time range
            conflicts = await self._get_conflicts_in_range(time_range_hours, project_id)
            
            # Calculate analytics
            analytics = {
                "time_range_hours": time_range_hours,
                "total_sessions": len(sessions),
                "active_sessions": len([s for s in sessions if s.state == CollaborationState.ACTIVE]),
                "completed_sessions": len([s for s in sessions if s.state == CollaborationState.COMPLETED]),
                "failed_sessions": len([s for s in sessions if s.state == CollaborationState.FAILED]),
                "total_conflicts": len(conflicts),
                "resolved_conflicts": len([c for c in conflicts if c.metadata.get("resolved", False)]),
                "conflict_types": self._analyze_conflict_types(conflicts),
                "collaboration_effectiveness": self._calculate_collaboration_effectiveness(sessions),
                "agent_collaboration_matrix": await self._build_collaboration_matrix(sessions),
                "knowledge_sharing_stats": await self._get_knowledge_sharing_stats(time_range_hours, project_id)
            }
            
            if project_id:
                analytics["project_id"] = project_id
                analytics["project_specific_metrics"] = await self._get_project_collaboration_metrics(project_id)
            
            return analytics
            
        except Exception as e:
            logger.error(
                "Failed to get collaboration analytics",
                project_id=project_id,
                error=str(e)
            )
            return {"error": str(e)}
    
    # ================== PRIVATE METHODS ==================
    
    async def _select_lead_agent(
        self,
        participant_agents: List[str],
        project_id: str
    ) -> str:
        """Select the best lead agent from participants."""
        if len(participant_agents) == 1:
            return participant_agents[0]
        
        # Score agents based on project familiarity and leadership capabilities
        scores = {}
        
        for agent_id in participant_agents:
            # Get agent
            stmt = select(Agent).where(Agent.id == agent_id)
            result = await self.session.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if not agent:
                scores[agent_id] = 0.0
                continue
            
            # Calculate leadership score
            familiarity_score = 0.0
            if project_id:
                history = await self.context_integration._get_agent_project_history(agent_id, project_id)
                familiarity_score = self.context_integration._calculate_familiarity_score(history)
            
            # Check for leadership capabilities
            leadership_score = 0.0
            if agent.capabilities:
                for cap in agent.capabilities:
                    if any(term in cap.get("name", "").lower() 
                          for term in ["lead", "coordin", "manage", "architect"]):
                        leadership_score += cap.get("confidence_level", 0.0) * 0.5
            
            # Performance score
            performance_score = float(agent.total_tasks_completed or 0) / max(
                float(agent.total_tasks_failed or 0) + float(agent.total_tasks_completed or 1), 1
            )
            
            total_score = (familiarity_score * 0.4 + leadership_score * 0.4 + performance_score * 0.2)
            scores[agent_id] = total_score
        
        # Return agent with highest score
        return max(scores, key=scores.get)
    
    async def _build_shared_context(
        self,
        project_id: str,
        task_id: str,
        participant_agents: List[str]
    ) -> Dict[str, Any]:
        """Build shared context for collaboration session."""
        # Get project information
        stmt = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await self.session.execute(stmt)
        project = result.scalar_one_or_none()
        
        # Get task information
        task_stmt = select(Task).where(Task.id == task_id)
        task_result = await self.session.execute(task_stmt)
        task = task_result.scalar_one_or_none()
        
        # Build context
        shared_context = {
            "project": {
                "id": project_id,
                "name": project.name if project else "Unknown",
                "description": project.description if project else "",
                "file_count": project.file_count if project else 0
            },
            "task": {
                "id": task_id,
                "description": task.description if task else "",
                "priority": task.priority.value if task else "normal"
            },
            "participants": [],
            "shared_resources": [],
            "communication_guidelines": {
                "update_frequency": "every_30_minutes",
                "conflict_escalation": "immediate",
                "knowledge_sharing": "encouraged"
            }
        }
        
        # Add participant information
        for agent_id in participant_agents:
            stmt = select(Agent).where(Agent.id == agent_id)
            result = await self.session.execute(stmt)
            agent = result.scalar_one_or_none()
            
            if agent:
                shared_context["participants"].append({
                    "agent_id": agent_id,
                    "name": agent.name,
                    "role": agent.role,
                    "capabilities": agent.capabilities or []
                })
        
        return shared_context
    
    async def _create_initial_assignments(
        self,
        project_id: str,
        task_id: str,
        participant_agents: List[str],
        shared_context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Create initial work assignments for agents."""
        assignments = {}
        
        # Get relevant files for the task
        # This is a simplified version - in practice, would use more sophisticated analysis
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.is_binary == False
            )
        ).limit(20)  # Limit for initial assignment
        
        result = await self.session.execute(stmt)
        files = result.scalars().all()
        
        # Distribute files among agents based on their capabilities
        file_paths = [f.relative_path for f in files]
        files_per_agent = len(file_paths) // len(participant_agents)
        
        for i, agent_id in enumerate(participant_agents):
            start_idx = i * files_per_agent
            end_idx = start_idx + files_per_agent if i < len(participant_agents) - 1 else len(file_paths)
            assignments[agent_id] = file_paths[start_idx:end_idx]
        
        return assignments
    
    async def _store_collaboration_session(self, session: CollaborationSession) -> None:
        """Store collaboration session in Redis."""
        session_key = f"collaboration_session:{session.session_id}"
        await self.redis.setex(
            session_key,
            self.session_timeout_hours * 3600,
            json.dumps(session.to_dict())
        )
        
        # Index by project
        project_sessions_key = f"project_sessions:{session.project_id}"
        await self.redis.sadd(project_sessions_key, session.session_id)
        await self.redis.expire(project_sessions_key, self.session_timeout_hours * 3600)
    
    async def _initialize_session_monitoring(self, session: CollaborationSession) -> None:
        """Initialize monitoring for the collaboration session."""
        # Set up conflict detection
        monitor_key = f"session_monitor:{session.session_id}"
        monitor_data = {
            "last_conflict_check": datetime.utcnow().isoformat(),
            "conflict_count": 0,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            monitor_key,
            self.session_timeout_hours * 3600,
            json.dumps(monitor_data)
        )
    
    async def _notify_session_participants(
        self,
        session: CollaborationSession,
        event_type: str
    ) -> None:
        """Notify session participants of events."""
        notification = {
            "type": event_type,
            "session_id": session.session_id,
            "project_id": session.project_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {
                "lead_agent": session.lead_agent,
                "participants": session.participants,
                "state": session.state.value
            }
        }
        
        # Send to each participant's notification channel
        for agent_id in session.participants:
            channel = f"agent_notifications:{agent_id}"
            await self.redis.lpush(channel, json.dumps(notification))
            await self.redis.expire(channel, 86400)  # 24 hours
    
    async def _get_collaboration_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session by ID."""
        session_key = f"collaboration_session:{session_id}"
        session_data = await self.redis.get(session_key)
        
        if not session_data:
            return None
        
        try:
            data = json.loads(session_data)
            return CollaborationSession(
                session_id=data["session_id"],
                project_id=data["project_id"],
                task_id=data["task_id"],
                participants=data["participants"],
                lead_agent=data["lead_agent"],
                state=CollaborationState(data["state"]),
                shared_context=data["shared_context"],
                work_assignments=data["work_assignments"],
                communication_channel=data["communication_channel"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                estimated_completion=datetime.fromisoformat(data["estimated_completion"]) if data.get("estimated_completion") else None,
                actual_completion=datetime.fromisoformat(data["actual_completion"]) if data.get("actual_completion") else None,
                metadata=data.get("metadata", {})
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    async def _update_wip_tracking(
        self,
        session_id: str,
        agent_id: str,
        progress_update: Dict[str, Any]
    ) -> None:
        """Update work-in-progress tracking for an agent."""
        wip_key = f"wip_tracking:{session_id}:{agent_id}"
        
        wip_data = {
            "agent_id": agent_id,
            "session_id": session_id,
            "current_files": progress_update.get("working_on_files", []),
            "completed_files": progress_update.get("completed_files", []),
            "progress_percentage": progress_update.get("progress_percentage", 0.0),
            "last_update": datetime.utcnow().isoformat(),
            "estimated_completion": progress_update.get("estimated_completion"),
            "status": progress_update.get("status", "active")
        }
        
        await self.redis.setex(wip_key, self.wip_tracking_ttl, json.dumps(wip_data))
    
    async def _detect_conflicts(
        self,
        session: CollaborationSession,
        agent_id: str,
        progress_update: Dict[str, Any]
    ) -> List[CollaborationConflict]:
        """Detect conflicts based on progress update."""
        conflicts = []
        
        # Check for file conflicts
        working_files = progress_update.get("working_on_files", [])
        for file_path in working_files:
            # Check if other agents are working on the same file
            other_agents_working = await self._get_agents_working_on_file(
                session.session_id, file_path, exclude_agent=agent_id
            )
            
            if other_agents_working:
                conflict = CollaborationConflict(
                    conflict_id=str(uuid.uuid4()),
                    session_id=session.session_id,
                    conflict_type=ConflictType.FILE_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    involved_agents=[agent_id] + other_agents_working,
                    description=f"Multiple agents working on file: {file_path}",
                    affected_resources=[file_path],
                    detection_time=datetime.utcnow(),
                    resolution_suggestions=[
                        "Coordinate file locking",
                        "Split file into sections",
                        "Sequential work assignment"
                    ],
                    auto_resolvable=True
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _get_agents_working_on_file(
        self,
        session_id: str,
        file_path: str,
        exclude_agent: Optional[str] = None
    ) -> List[str]:
        """Get agents currently working on a specific file."""
        agents = []
        
        # Check WIP tracking for all agents in session
        session = await self._get_collaboration_session(session_id)
        if not session:
            return agents
        
        for agent_id in session.participants:
            if exclude_agent and agent_id == exclude_agent:
                continue
            
            wip_key = f"wip_tracking:{session_id}:{agent_id}"
            wip_data = await self.redis.get(wip_key)
            
            if wip_data:
                try:
                    data = json.loads(wip_data)
                    current_files = data.get("current_files", [])
                    if file_path in current_files:
                        agents.append(agent_id)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        return agents
    
    async def _coordinate_with_peers(
        self,
        session: CollaborationSession,
        agent_id: str,
        progress_update: Dict[str, Any],
        conflicts: List[CollaborationConflict]
    ) -> List[Dict[str, Any]]:
        """Coordinate with peer agents based on progress and conflicts."""
        actions = []
        
        # If conflicts detected, create coordination actions
        for conflict in conflicts:
            if conflict.auto_resolvable and conflict.conflict_type == ConflictType.FILE_CONFLICT:
                # Suggest file locking coordination
                action = {
                    "type": "file_lock_coordination",
                    "description": f"Coordinate file access for {conflict.affected_resources[0]}",
                    "target_agents": [a for a in conflict.involved_agents if a != agent_id],
                    "suggested_resolution": "Request file lock before modification",
                    "priority": "medium"
                }
                actions.append(action)
        
        # Check for knowledge sharing opportunities
        if progress_update.get("insights") or progress_update.get("learnings"):
            action = {
                "type": "knowledge_sharing",
                "description": "Share insights with team",
                "target_agents": [a for a in session.participants if a != agent_id],
                "content": {
                    "insights": progress_update.get("insights", []),
                    "learnings": progress_update.get("learnings", [])
                },
                "priority": "low"
            }
            actions.append(action)
        
        return actions
    
    async def _update_session_state(
        self,
        session: CollaborationSession,
        progress_update: Dict[str, Any],
        conflicts: List[CollaborationConflict]
    ) -> None:
        """Update session state based on progress and conflicts."""
        # Determine new state
        new_state = session.state
        
        if conflicts and any(c.severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL] for c in conflicts):
            new_state = CollaborationState.COORDINATING
        elif session.state == CollaborationState.PLANNING:
            new_state = CollaborationState.ACTIVE
        
        # Update session
        session.state = new_state
        session.updated_at = datetime.utcnow()
        
        # Store updated session
        await self._store_collaboration_session(session)
    
    async def _generate_next_steps(
        self,
        session: CollaborationSession,
        agent_id: str
    ) -> List[str]:
        """Generate next steps for an agent in the collaboration."""
        next_steps = []
        
        # Get agent's assignment
        assignments = session.work_assignments.get(agent_id, [])
        
        if assignments:
            next_steps.append(f"Continue work on assigned files: {', '.join(assignments[:3])}")
        
        # Check if agent should coordinate with others
        if session.state == CollaborationState.COORDINATING:
            next_steps.append("Coordinate with team members to resolve conflicts")
        
        # Check for knowledge sharing opportunities
        next_steps.append("Share progress and insights with team")
        
        return next_steps
    
    async def _suggest_conflict_resolutions(
        self,
        conflicts: List[CollaborationConflict]
    ) -> List[Dict[str, Any]]:
        """Suggest resolutions for detected conflicts."""
        resolutions = []
        
        for conflict in conflicts:
            if conflict.conflict_type == ConflictType.FILE_CONFLICT:
                resolutions.append({
                    "conflict_id": conflict.conflict_id,
                    "strategy": "file_locking",
                    "description": "Implement file locking to prevent simultaneous modifications",
                    "auto_applicable": True,
                    "effort_level": "low"
                })
            elif conflict.conflict_type == ConflictType.DEPENDENCY_CONFLICT:
                resolutions.append({
                    "conflict_id": conflict.conflict_id,
                    "strategy": "dependency_coordination",
                    "description": "Coordinate dependency changes between agents",
                    "auto_applicable": False,
                    "effort_level": "medium"
                })
        
        return resolutions
    
    async def _notify_knowledge_share(self, knowledge_share: KnowledgeShare) -> None:
        """Notify target agents of new knowledge share."""
        notification = {
            "type": "knowledge_shared",
            "share_id": knowledge_share.share_id,
            "source_agent": knowledge_share.source_agent,
            "knowledge_type": knowledge_share.knowledge_type,
            "shared_at": knowledge_share.shared_at.isoformat(),
            "preview": str(knowledge_share.content)[:200] + "..." if len(str(knowledge_share.content)) > 200 else str(knowledge_share.content)
        }
        
        for target_agent in knowledge_share.target_agents:
            channel = f"agent_notifications:{target_agent}"
            await self.redis.lpush(channel, json.dumps(notification))
            await self.redis.expire(channel, 86400)
    
    async def _increment_access_count(self, share_id: str) -> None:
        """Increment access count for a knowledge share."""
        share_key = f"knowledge_share:{share_id}"
        share_data = await self.redis.get(share_key)
        
        if share_data:
            try:
                data = json.loads(share_data)
                data["access_count"] = data.get("access_count", 0) + 1
                
                # Get original TTL
                ttl = await self.redis.ttl(share_key)
                if ttl > 0:
                    await self.redis.setex(share_key, ttl, json.dumps(data))
                else:
                    await self.redis.set(share_key, json.dumps(data))
                    
            except (json.JSONDecodeError, KeyError):
                pass
    
    async def _get_active_sessions(self, project_id: str) -> List[CollaborationSession]:
        """Get active collaboration sessions for a project."""
        project_sessions_key = f"project_sessions:{project_id}"
        session_ids = await self.redis.smembers(project_sessions_key)
        
        sessions = []
        for session_id in session_ids:
            session = await self._get_collaboration_session(session_id)
            if session and session.state in [CollaborationState.ACTIVE, CollaborationState.COORDINATING]:
                sessions.append(session)
        
        return sessions
    
    async def _detect_file_conflicts(self, session: CollaborationSession) -> List[CollaborationConflict]:
        """Detect file-level conflicts in a session."""
        conflicts = []
        
        # Check for overlapping file assignments
        file_assignments = {}
        for agent_id, files in session.work_assignments.items():
            for file_path in files:
                if file_path not in file_assignments:
                    file_assignments[file_path] = []
                file_assignments[file_path].append(agent_id)
        
        # Find files assigned to multiple agents
        for file_path, agents in file_assignments.items():
            if len(agents) > 1:
                conflict = CollaborationConflict(
                    conflict_id=str(uuid.uuid4()),
                    session_id=session.session_id,
                    conflict_type=ConflictType.FILE_CONFLICT,
                    severity=ConflictSeverity.MEDIUM,
                    involved_agents=agents,
                    description=f"Multiple agents assigned to file: {file_path}",
                    affected_resources=[file_path],
                    detection_time=datetime.utcnow(),
                    resolution_suggestions=["Reassign file", "Split responsibilities", "Coordinate access"],
                    auto_resolvable=True
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_dependency_conflicts(self, session: CollaborationSession) -> List[CollaborationConflict]:
        """Detect dependency-related conflicts."""
        # This is a simplified implementation
        # In practice, would analyze actual dependency graphs
        return []
    
    async def _detect_resource_conflicts(self, session: CollaborationSession) -> List[CollaborationConflict]:
        """Detect resource allocation conflicts."""
        # This is a simplified implementation
        # In practice, would check for resource constraints
        return []
    
    async def _detect_timing_conflicts(self, session: CollaborationSession) -> List[CollaborationConflict]:
        """Detect timing and scheduling conflicts."""
        # This is a simplified implementation
        # In practice, would analyze task dependencies and timelines
        return []
    
    def _severity_to_priority(self, severity: ConflictSeverity) -> int:
        """Convert severity to numeric priority for sorting."""
        return {
            ConflictSeverity.CRITICAL: 4,
            ConflictSeverity.HIGH: 3,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 1
        }.get(severity, 0)
    
    async def _get_conflict(self, conflict_id: str) -> Optional[CollaborationConflict]:
        """Get conflict by ID."""
        # This would be implemented based on how conflicts are stored
        # For now, return None as placeholder
        return None
    
    async def _auto_merge_resolution(
        self,
        conflict: CollaborationConflict,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply auto-merge conflict resolution."""
        return {"success": True, "method": "auto_merge", "details": "Automatically merged changes"}
    
    async def _priority_assignment_resolution(
        self,
        conflict: CollaborationConflict,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply priority-based conflict resolution."""
        return {"success": True, "method": "priority_assignment", "details": "Assigned to highest priority agent"}
    
    async def _work_redistribution_resolution(
        self,
        conflict: CollaborationConflict,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply work redistribution conflict resolution."""
        return {"success": True, "method": "work_redistribution", "details": "Redistributed work among agents"}
    
    async def _manual_intervention_resolution(
        self,
        conflict: CollaborationConflict,
        resolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply manual intervention conflict resolution."""
        return {"success": True, "method": "manual_intervention", "details": "Escalated for manual resolution"}
    
    async def _mark_conflict_resolved(
        self,
        conflict_id: str,
        strategy: str,
        result: Dict[str, Any]
    ) -> None:
        """Mark a conflict as resolved."""
        # This would update the conflict record with resolution information
        pass
    
    async def _notify_conflict_resolution(
        self,
        conflict: CollaborationConflict,
        result: Dict[str, Any]
    ) -> None:
        """Notify agents of conflict resolution."""
        notification = {
            "type": "conflict_resolved",
            "conflict_id": conflict.conflict_id,
            "resolution_method": result.get("method"),
            "details": result.get("details"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for agent_id in conflict.involved_agents:
            channel = f"agent_notifications:{agent_id}"
            await self.redis.lpush(channel, json.dumps(notification))
            await self.redis.expire(channel, 86400)
    
    async def _get_sessions_in_range(
        self,
        time_range_hours: int,
        project_id: Optional[str] = None
    ) -> List[CollaborationSession]:
        """Get collaboration sessions in time range."""
        # This would query stored sessions
        # Placeholder implementation
        return []
    
    async def _get_conflicts_in_range(
        self,
        time_range_hours: int,
        project_id: Optional[str] = None
    ) -> List[CollaborationConflict]:
        """Get conflicts in time range."""
        # This would query stored conflicts
        # Placeholder implementation
        return []
    
    def _analyze_conflict_types(self, conflicts: List[CollaborationConflict]) -> Dict[str, int]:
        """Analyze distribution of conflict types."""
        types = {}
        for conflict in conflicts:
            conflict_type = conflict.conflict_type.value
            types[conflict_type] = types.get(conflict_type, 0) + 1
        return types
    
    def _calculate_collaboration_effectiveness(self, sessions: List[CollaborationSession]) -> float:
        """Calculate overall collaboration effectiveness."""
        if not sessions:
            return 0.0
        
        completed = len([s for s in sessions if s.state == CollaborationState.COMPLETED])
        return completed / len(sessions)
    
    async def _build_collaboration_matrix(self, sessions: List[CollaborationSession]) -> Dict[str, Any]:
        """Build agent collaboration compatibility matrix."""
        # This would analyze which agents work well together
        # Placeholder implementation
        return {}
    
    async def _get_knowledge_sharing_stats(
        self,
        time_range_hours: int,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get knowledge sharing statistics."""
        # This would analyze knowledge sharing patterns
        # Placeholder implementation
        return {"total_shares": 0, "average_access_count": 0.0}
    
    async def _get_project_collaboration_metrics(self, project_id: str) -> Dict[str, Any]:
        """Get project-specific collaboration metrics."""
        # This would provide project-specific insights
        # Placeholder implementation
        return {"agent_compatibility": {}, "common_conflict_patterns": []}


# Factory function for dependency injection
async def get_collaborative_development_engine(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None,
    task_router: IntelligentTaskRouter = None
) -> CollaborativeDevelopmentEngine:
    """Factory function to create CollaborativeDevelopmentEngine instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    if task_router is None:
        from .task_router import get_intelligent_task_router
        task_router = await get_intelligent_task_router(session, redis_client, context_integration)
    
    return CollaborativeDevelopmentEngine(session, redis_client, context_integration, task_router)
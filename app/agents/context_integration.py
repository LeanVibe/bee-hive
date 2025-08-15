"""
Agent Context Integration for LeanVibe Agent Hive 2.0

Provides intelligent context enhancement for agents working on specific projects,
enabling automatic context injection, task-aware optimization, and real-time
project understanding for enhanced agent coordination and effectiveness.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent, AgentStatus
from ..models.project_index import ProjectIndex, FileEntry, DependencyRelationship
from ..project_index.core import ProjectIndexer
from ..project_index.context_assembler import ContextAssembler
# from ..project_index.relevance_analyzer import RelevanceAnalyzer  # Commented out to avoid textstat dependency

logger = structlog.get_logger()


class ContextScope(Enum):
    """Scope of context for agent tasks."""
    FULL_PROJECT = "full_project"
    RELEVANT_FILES = "relevant_files"
    DEPENDENCIES = "dependencies"
    CHANGED_FILES = "changed_files"
    TASK_SPECIFIC = "task_specific"


class AgentTaskType(Enum):
    """Types of tasks agents can perform."""
    CODE_ANALYSIS = "code_analysis"
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIXING = "bug_fixing"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    ARCHITECTURE_REVIEW = "architecture_review"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


@dataclass
class ContextRequest:
    """Request for project context for an agent task."""
    agent_id: str
    project_id: str
    task_type: AgentTaskType
    task_description: str
    scope: ContextScope = ContextScope.RELEVANT_FILES
    max_files: int = 50
    max_context_size: int = 100000  # characters
    include_dependencies: bool = True
    include_history: bool = False
    focus_areas: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


@dataclass
class ContextResponse:
    """Response containing optimized context for agent."""
    request_id: str
    agent_id: str
    project_id: str
    context_data: Dict[str, Any]
    metadata: Dict[str, Any]
    cache_key: str
    created_at: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "project_id": self.project_id,
            "context_data": self.context_data,
            "metadata": self.metadata,
            "cache_key": self.cache_key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }


@dataclass
class AgentProjectHistory:
    """Historical data of agent work on a project."""
    agent_id: str
    project_id: str
    files_worked_on: Set[str]
    expertise_areas: Dict[str, float]  # area -> confidence score
    successful_patterns: List[str]
    common_issues: List[str]
    performance_metrics: Dict[str, float]
    last_worked_on: datetime


class AgentContextIntegration:
    """
    Core system for enhancing agents with intelligent project context.
    
    Provides automatic context injection, task-aware optimization, and 
    real-time project understanding to dramatically improve agent effectiveness.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        project_indexer: ProjectIndexer
    ):
        self.session = session
        self.redis = redis_client
        self.project_indexer = project_indexer
        self.context_assembler = ContextAssembler(session, redis_client)
        # self.relevance_analyzer = RelevanceAnalyzer()  # Commented out to avoid textstat dependency
        
        # Cache configurations
        self.context_cache_ttl = 3600  # 1 hour
        self.history_cache_ttl = 86400  # 24 hours
        
        # Performance limits
        self.max_concurrent_contexts = 20
        self.context_size_limits = {
            ContextScope.FULL_PROJECT: 200000,
            ContextScope.RELEVANT_FILES: 100000,
            ContextScope.DEPENDENCIES: 50000,
            ContextScope.CHANGED_FILES: 75000,
            ContextScope.TASK_SPECIFIC: 125000
        }
    
    async def request_context(
        self,
        request: ContextRequest
    ) -> ContextResponse:
        """
        Request optimized project context for an agent task.
        
        Args:
            request: Context request with task details and preferences
            
        Returns:
            ContextResponse with optimized context data
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(
            "Processing context request",
            request_id=request_id,
            agent_id=request.agent_id,
            project_id=request.project_id,
            task_type=request.task_type.value,
            scope=request.scope.value
        )
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_context = await self._get_cached_context(cache_key)
            
            if cached_context:
                logger.info(
                    "Context cache hit",
                    request_id=request_id,
                    cache_key=cache_key
                )
                return cached_context
            
            # Get project and agent information
            project = await self._get_project(request.project_id)
            agent = await self._get_agent(request.agent_id)
            
            if not project:
                raise ValueError(f"Project {request.project_id} not found")
            if not agent:
                raise ValueError(f"Agent {request.agent_id} not found")
            
            # Get agent's historical context with this project
            agent_history = await self._get_agent_project_history(
                request.agent_id, request.project_id
            )
            
            # Build context based on task type and scope
            context_data = await self._build_context_data(
                request, project, agent, agent_history
            )
            
            # Apply context optimizations
            optimized_context = await self._optimize_context(
                context_data, request, agent_history
            )
            
            # Create response
            response = ContextResponse(
                request_id=request_id,
                agent_id=request.agent_id,
                project_id=request.project_id,
                context_data=optimized_context,
                metadata={
                    "task_type": request.task_type.value,
                    "scope": request.scope.value,
                    "processing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    "context_size": len(json.dumps(optimized_context)),
                    "files_included": len(optimized_context.get("files", [])),
                    "dependencies_included": len(optimized_context.get("dependencies", [])),
                    "agent_familiarity_score": self._calculate_familiarity_score(agent_history),
                    "context_optimization_applied": True
                },
                cache_key=cache_key,
                created_at=start_time,
                expires_at=start_time + timedelta(seconds=self.context_cache_ttl)
            )
            
            # Cache the response
            await self._cache_context(cache_key, response)
            
            # Update agent history
            await self._update_agent_history(request.agent_id, request.project_id, request)
            
            logger.info(
                "Context request completed",
                request_id=request_id,
                processing_time_ms=response.metadata["processing_time_ms"],
                context_size=response.metadata["context_size"],
                files_included=response.metadata["files_included"]
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Context request failed",
                request_id=request_id,
                error=str(e),
                agent_id=request.agent_id,
                project_id=request.project_id
            )
            raise
    
    async def update_agent_context(
        self,
        agent_id: str,
        project_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """
        Update agent's context preferences for a project.
        
        Args:
            agent_id: Agent identifier
            project_id: Project identifier
            context_updates: Context preference updates
            
        Returns:
            Success status
        """
        try:
            # Update agent's context preferences
            preferences_key = f"agent_context_prefs:{agent_id}:{project_id}"
            current_prefs = await self.redis.hgetall(preferences_key) or {}
            
            # Merge updates
            updated_prefs = {**current_prefs, **context_updates}
            
            # Store updated preferences
            await self.redis.hmset(preferences_key, updated_prefs)
            await self.redis.expire(preferences_key, self.history_cache_ttl)
            
            # Invalidate related context cache
            await self._invalidate_agent_context_cache(agent_id, project_id)
            
            logger.info(
                "Agent context preferences updated",
                agent_id=agent_id,
                project_id=project_id,
                updates=list(context_updates.keys())
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update agent context",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
            return False
    
    async def get_agent_context_status(
        self,
        agent_id: str,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current context status for an agent.
        
        Args:
            agent_id: Agent identifier
            project_id: Optional project filter
            
        Returns:
            Context status information
        """
        try:
            if project_id:
                # Get status for specific project
                history = await self._get_agent_project_history(agent_id, project_id)
                cache_keys = await self._get_agent_cache_keys(agent_id, project_id)
                
                return {
                    "agent_id": agent_id,
                    "project_id": project_id,
                    "familiarity_score": self._calculate_familiarity_score(history),
                    "files_worked_on": len(history.files_worked_on) if history else 0,
                    "expertise_areas": history.expertise_areas if history else {},
                    "cached_contexts": len(cache_keys),
                    "last_worked_on": history.last_worked_on.isoformat() if history and history.last_worked_on else None
                }
            else:
                # Get status across all projects
                all_projects = await self._get_agent_all_projects(agent_id)
                
                status = {
                    "agent_id": agent_id,
                    "total_projects": len(all_projects),
                    "projects": []
                }
                
                for pid in all_projects:
                    project_status = await self.get_agent_context_status(agent_id, pid)
                    status["projects"].append(project_status)
                
                return status
                
        except Exception as e:
            logger.error(
                "Failed to get agent context status",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
            return {"error": str(e)}
    
    async def clear_agent_context(
        self,
        agent_id: str,
        project_id: Optional[str] = None
    ) -> bool:
        """
        Clear cached context for an agent.
        
        Args:
            agent_id: Agent identifier
            project_id: Optional project filter
            
        Returns:
            Success status
        """
        try:
            if project_id:
                # Clear for specific project
                await self._invalidate_agent_context_cache(agent_id, project_id)
                
                # Clear history
                history_key = f"agent_history:{agent_id}:{project_id}"
                await self.redis.delete(history_key)
                
                # Clear preferences
                prefs_key = f"agent_context_prefs:{agent_id}:{project_id}"
                await self.redis.delete(prefs_key)
                
            else:
                # Clear all projects for agent
                all_projects = await self._get_agent_all_projects(agent_id)
                
                for pid in all_projects:
                    await self.clear_agent_context(agent_id, pid)
            
            logger.info(
                "Agent context cleared",
                agent_id=agent_id,
                project_id=project_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to clear agent context",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
            return False
    
    # ================== PRIVATE METHODS ==================
    
    def _generate_cache_key(self, request: ContextRequest) -> str:
        """Generate cache key for context request."""
        key_parts = [
            "agent_context",
            request.agent_id,
            request.project_id,
            request.task_type.value,
            request.scope.value,
            str(request.max_files),
            str(request.max_context_size),
            str(request.include_dependencies),
            str(request.include_history)
        ]
        
        if request.focus_areas:
            key_parts.append("_".join(sorted(request.focus_areas)))
        
        if request.exclude_patterns:
            key_parts.append("_".join(sorted(request.exclude_patterns)))
        
        return ":".join(key_parts)
    
    async def _get_cached_context(self, cache_key: str) -> Optional[ContextResponse]:
        """Get cached context response."""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                
                # Check if still valid
                expires_at = datetime.fromisoformat(data["expires_at"])
                if datetime.utcnow() < expires_at:
                    return ContextResponse(
                        request_id=data["request_id"],
                        agent_id=data["agent_id"],
                        project_id=data["project_id"],
                        context_data=data["context_data"],
                        metadata=data["metadata"],
                        cache_key=data["cache_key"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        expires_at=expires_at
                    )
                else:
                    # Expired, remove from cache
                    await self.redis.delete(cache_key)
            
            return None
            
        except Exception as e:
            logger.warning("Failed to get cached context", cache_key=cache_key, error=str(e))
            return None
    
    async def _cache_context(self, cache_key: str, response: ContextResponse) -> None:
        """Cache context response."""
        try:
            await self.redis.setex(
                cache_key,
                self.context_cache_ttl,
                json.dumps(response.to_dict())
            )
        except Exception as e:
            logger.warning("Failed to cache context", cache_key=cache_key, error=str(e))
    
    async def _get_project(self, project_id: str) -> Optional[ProjectIndex]:
        """Get project by ID."""
        stmt = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        stmt = select(Agent).where(Agent.id == agent_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _get_agent_project_history(
        self,
        agent_id: str,
        project_id: str
    ) -> Optional[AgentProjectHistory]:
        """Get agent's historical work on a project."""
        try:
            history_key = f"agent_history:{agent_id}:{project_id}"
            history_data = await self.redis.hgetall(history_key)
            
            if not history_data:
                return None
            
            return AgentProjectHistory(
                agent_id=agent_id,
                project_id=project_id,
                files_worked_on=set(json.loads(history_data.get("files_worked_on", "[]"))),
                expertise_areas=json.loads(history_data.get("expertise_areas", "{}")),
                successful_patterns=json.loads(history_data.get("successful_patterns", "[]")),
                common_issues=json.loads(history_data.get("common_issues", "[]")),
                performance_metrics=json.loads(history_data.get("performance_metrics", "{}")),
                last_worked_on=datetime.fromisoformat(history_data["last_worked_on"]) if "last_worked_on" in history_data else datetime.utcnow()
            )
            
        except Exception as e:
            logger.warning(
                "Failed to get agent project history",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
            return None
    
    async def _build_context_data(
        self,
        request: ContextRequest,
        project: ProjectIndex,
        agent: Agent,
        history: Optional[AgentProjectHistory]
    ) -> Dict[str, Any]:
        """Build context data based on request parameters."""
        context = {
            "project": {
                "id": str(project.id),
                "name": project.name,
                "description": project.description,
                "root_path": project.root_path,
                "status": project.status.value,
                "file_count": project.file_count,
                "dependency_count": project.dependency_count
            },
            "agent": {
                "id": str(agent.id),
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities or []
            },
            "task": {
                "type": request.task_type.value,
                "description": request.task_description,
                "scope": request.scope.value
            },
            "files": [],
            "dependencies": [],
            "patterns": []
        }
        
        # Get relevant files based on scope and task type
        files = await self._get_relevant_files(request, project, history)
        context["files"] = [self._serialize_file_entry(f) for f in files[:request.max_files]]
        
        # Get dependencies if requested
        if request.include_dependencies:
            dependencies = await self._get_relevant_dependencies(request, project, files)
            context["dependencies"] = [self._serialize_dependency(d) for d in dependencies]
        
        # Add historical patterns if available
        if history and request.include_history:
            context["patterns"] = {
                "successful_patterns": history.successful_patterns,
                "common_issues": history.common_issues,
                "expertise_areas": history.expertise_areas
            }
        
        return context
    
    async def _get_relevant_files(
        self,
        request: ContextRequest,
        project: ProjectIndex,
        history: Optional[AgentProjectHistory]
    ) -> List[FileEntry]:
        """Get relevant files based on scope and task type."""
        if request.scope == ContextScope.FULL_PROJECT:
            stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        
        elif request.scope == ContextScope.RELEVANT_FILES:
            # Use relevance analyzer to find most relevant files
            stmt = select(FileEntry).where(
                and_(
                    FileEntry.project_id == project.id,
                    FileEntry.is_binary == False
                )
            )
            
            # Prioritize files agent has worked on before
            if history and history.files_worked_on:
                stmt = stmt.order_by(
                    FileEntry.relative_path.in_(list(history.files_worked_on)).desc(),
                    FileEntry.updated_at.desc()
                )
        
        elif request.scope == ContextScope.CHANGED_FILES:
            # Get recently modified files
            cutoff = datetime.utcnow() - timedelta(days=7)
            stmt = select(FileEntry).where(
                and_(
                    FileEntry.project_id == project.id,
                    FileEntry.updated_at >= cutoff
                )
            ).order_by(FileEntry.updated_at.desc())
        
        else:
            # Default to task-specific selection
            stmt = await self._get_task_specific_files(request, project)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _get_task_specific_files(
        self,
        request: ContextRequest,
        project: ProjectIndex
    ) -> any:
        """Get files specific to the task type."""
        base_stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        
        if request.task_type == AgentTaskType.TESTING:
            # Focus on test files and files being tested
            return base_stmt.where(
                or_(
                    FileEntry.relative_path.like("%test%"),
                    FileEntry.relative_path.like("%spec%"),
                    FileEntry.file_type == "test"
                )
            )
        
        elif request.task_type == AgentTaskType.DOCUMENTATION:
            # Focus on documentation files and source files that need docs
            return base_stmt.where(
                or_(
                    FileEntry.file_type == "documentation",
                    FileEntry.file_extension.in_([".md", ".rst", ".txt"])
                )
            )
        
        elif request.task_type == AgentTaskType.BUG_FIXING:
            # Focus on recently modified files and error-prone areas
            cutoff = datetime.utcnow() - timedelta(days=30)
            return base_stmt.where(
                FileEntry.updated_at >= cutoff
            ).order_by(FileEntry.updated_at.desc())
        
        else:
            # Default selection for other task types
            return base_stmt.where(
                FileEntry.file_type.in_(["source", "config"])
            ).order_by(FileEntry.updated_at.desc())
    
    async def _get_relevant_dependencies(
        self,
        request: ContextRequest,
        project: ProjectIndex,
        files: List[FileEntry]
    ) -> List[DependencyRelationship]:
        """Get relevant dependencies for the selected files."""
        if not files:
            return []
        
        file_ids = [f.id for f in files]
        
        stmt = select(DependencyRelationship).where(
            and_(
                DependencyRelationship.project_id == project.id,
                or_(
                    DependencyRelationship.source_file_id.in_(file_ids),
                    DependencyRelationship.target_file_id.in_(file_ids)
                )
            )
        )
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _optimize_context(
        self,
        context_data: Dict[str, Any],
        request: ContextRequest,
        history: Optional[AgentProjectHistory]
    ) -> Dict[str, Any]:
        """Apply context optimizations based on size limits and preferences."""
        # Get size limit for this scope
        size_limit = self.context_size_limits.get(request.scope, request.max_context_size)
        
        # Calculate current size
        current_size = len(json.dumps(context_data))
        
        if current_size <= size_limit:
            return context_data
        
        # Apply optimizations to reduce size
        optimized = context_data.copy()
        
        # 1. Reduce file content previews
        for file_data in optimized["files"]:
            if "content_preview" in file_data and file_data["content_preview"]:
                preview = file_data["content_preview"]
                if len(preview) > 1000:
                    file_data["content_preview"] = preview[:1000] + "... [truncated]"
        
        # 2. Prioritize files based on agent history
        if history and history.files_worked_on:
            prioritized_files = []
            other_files = []
            
            for file_data in optimized["files"]:
                if file_data["relative_path"] in history.files_worked_on:
                    prioritized_files.append(file_data)
                else:
                    other_files.append(file_data)
            
            # Keep prioritized files and trim others if needed
            max_files = request.max_files
            if len(prioritized_files) < max_files:
                remaining = max_files - len(prioritized_files)
                optimized["files"] = prioritized_files + other_files[:remaining]
            else:
                optimized["files"] = prioritized_files[:max_files]
        
        # 3. Check size again and apply further reductions if needed
        new_size = len(json.dumps(optimized))
        if new_size > size_limit:
            # Reduce dependencies
            max_deps = min(len(optimized["dependencies"]), 20)
            optimized["dependencies"] = optimized["dependencies"][:max_deps]
            
            # Further reduce files if still too large
            while len(json.dumps(optimized)) > size_limit and len(optimized["files"]) > 5:
                optimized["files"] = optimized["files"][:-1]
        
        return optimized
    
    def _serialize_file_entry(self, file_entry: FileEntry) -> Dict[str, Any]:
        """Serialize file entry for context."""
        return {
            "id": str(file_entry.id),
            "relative_path": file_entry.relative_path,
            "file_name": file_entry.file_name,
            "file_extension": file_entry.file_extension,
            "file_type": file_entry.file_type.value,
            "language": file_entry.language,
            "file_size": file_entry.file_size,
            "line_count": file_entry.line_count,
            "content_preview": file_entry.content_preview,
            "tags": file_entry.tags or [],
            "is_binary": file_entry.is_binary,
            "is_generated": file_entry.is_generated,
            "last_modified": file_entry.last_modified.isoformat() if file_entry.last_modified else None
        }
    
    def _serialize_dependency(self, dependency: DependencyRelationship) -> Dict[str, Any]:
        """Serialize dependency relationship for context."""
        return {
            "id": str(dependency.id),
            "source_file_id": str(dependency.source_file_id),
            "target_file_id": str(dependency.target_file_id) if dependency.target_file_id else None,
            "target_name": dependency.target_name,
            "dependency_type": dependency.dependency_type.value,
            "is_external": dependency.is_external,
            "confidence_score": dependency.confidence_score,
            "line_number": dependency.line_number,
            "source_text": dependency.source_text
        }
    
    def _calculate_familiarity_score(self, history: Optional[AgentProjectHistory]) -> float:
        """Calculate agent's familiarity score with the project."""
        if not history:
            return 0.0
        
        # Base score from files worked on
        files_score = min(len(history.files_worked_on) / 50.0, 1.0)
        
        # Expertise areas score
        expertise_score = sum(history.expertise_areas.values()) / max(len(history.expertise_areas), 1)
        
        # Recency score
        if history.last_worked_on:
            days_since = (datetime.utcnow() - history.last_worked_on).days
            recency_score = max(0.0, 1.0 - (days_since / 30.0))  # Decay over 30 days
        else:
            recency_score = 0.0
        
        # Performance score
        performance_score = history.performance_metrics.get("success_rate", 0.5)
        
        # Weighted average
        return (files_score * 0.3 + expertise_score * 0.3 + recency_score * 0.2 + performance_score * 0.2)
    
    async def _update_agent_history(
        self,
        agent_id: str,
        project_id: str,
        request: ContextRequest
    ) -> None:
        """Update agent's project history based on current request."""
        try:
            history_key = f"agent_history:{agent_id}:{project_id}"
            current_data = await self.redis.hgetall(history_key) or {}
            
            # Update last worked on time
            updates = {"last_worked_on": datetime.utcnow().isoformat()}
            
            # Track task types
            task_types = json.loads(current_data.get("task_types", "[]"))
            if request.task_type.value not in task_types:
                task_types.append(request.task_type.value)
                updates["task_types"] = json.dumps(task_types)
            
            # Update focus areas if provided
            if request.focus_areas:
                current_areas = json.loads(current_data.get("focus_areas", "[]"))
                for area in request.focus_areas:
                    if area not in current_areas:
                        current_areas.append(area)
                updates["focus_areas"] = json.dumps(current_areas)
            
            # Store updates
            await self.redis.hmset(history_key, updates)
            await self.redis.expire(history_key, self.history_cache_ttl)
            
        except Exception as e:
            logger.warning(
                "Failed to update agent history",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
    
    async def _invalidate_agent_context_cache(
        self,
        agent_id: str,
        project_id: str
    ) -> None:
        """Invalidate cached contexts for an agent-project combination."""
        try:
            # Find all cache keys for this agent-project
            pattern = f"agent_context:{agent_id}:{project_id}:*"
            cache_keys = await self.redis.keys(pattern)
            
            if cache_keys:
                await self.redis.delete(*cache_keys)
                
        except Exception as e:
            logger.warning(
                "Failed to invalidate agent context cache",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
    
    async def _get_agent_cache_keys(
        self,
        agent_id: str,
        project_id: str
    ) -> List[str]:
        """Get all cache keys for an agent-project combination."""
        try:
            pattern = f"agent_context:{agent_id}:{project_id}:*"
            return await self.redis.keys(pattern)
        except Exception as e:
            logger.warning(
                "Failed to get agent cache keys",
                agent_id=agent_id,
                project_id=project_id,
                error=str(e)
            )
            return []
    
    async def _get_agent_all_projects(self, agent_id: str) -> List[str]:
        """Get all project IDs that an agent has worked on."""
        try:
            pattern = f"agent_history:{agent_id}:*"
            keys = await self.redis.keys(pattern)
            
            # Extract project IDs from keys
            project_ids = []
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    project_ids.append(parts[2])
            
            return project_ids
            
        except Exception as e:
            logger.warning(
                "Failed to get agent projects",
                agent_id=agent_id,
                error=str(e)
            )
            return []


# Factory function for dependency injection
async def get_agent_context_integration(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    project_indexer: ProjectIndexer = None
) -> AgentContextIntegration:
    """Factory function to create AgentContextIntegration instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if project_indexer is None:
        from ..project_index.core import ProjectIndexer
        project_indexer = ProjectIndexer(session, redis_client)
    
    return AgentContextIntegration(session, redis_client, project_indexer)
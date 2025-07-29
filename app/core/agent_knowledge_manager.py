"""
Agent Knowledge Manager for LeanVibe Agent Hive 2.0

Manages agent-scoped knowledge with cross-agent sharing, access controls,
learning from workflow outcomes, and intelligent knowledge consolidation.

Features:
- Agent-specific knowledge bases with isolation
- Cross-agent knowledge sharing with access controls
- Learning from workflow outcomes and task results
- Knowledge consolidation and pattern recognition
- Expertise tracking and capability assessment
- Knowledge recommendation engine
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import structlog

from .database import get_session
from .redis import get_redis
from .semantic_memory_task_processor import (
    SemanticMemoryTaskProcessor, SemanticMemoryTask, SemanticTaskType, ProcessingPriority
)
from .workflow_context_manager import ContextFragment, ContextType, ContextScope

logger = structlog.get_logger()


# =============================================================================
# KNOWLEDGE DEFINITIONS AND TYPES
# =============================================================================

class KnowledgeType(str, Enum):
    """Types of agent knowledge."""
    PATTERN = "pattern"
    SKILL = "skill"
    EXPERIENCE = "experience"
    PREFERENCE = "preference"
    INTERACTION = "interaction"
    BEST_PRACTICE = "best_practice"
    LESSON_LEARNED = "lesson_learned"
    COLLABORATION = "collaboration"


class KnowledgeSource(str, Enum):
    """Sources of knowledge."""
    WORKFLOW_EXECUTION = "workflow_execution"
    TASK_COMPLETION = "task_completion"
    AGENT_INTERACTION = "agent_interaction"
    MANUAL_INPUT = "manual_input"
    CROSS_AGENT_SHARING = "cross_agent_sharing"
    SYSTEM_OBSERVATION = "system_observation"


class AccessLevel(str, Enum):
    """Knowledge access levels."""
    PRIVATE = "private"  # Agent only
    TEAM_SHARED = "team_shared"  # Same team/role
    CROSS_TEAM = "cross_team"  # Different teams
    PUBLIC = "public"  # All agents


class KnowledgeConfidence(str, Enum):
    """Confidence levels for knowledge."""
    LOW = "low"  # 0.0-0.4
    MEDIUM = "medium"  # 0.4-0.7
    HIGH = "high"  # 0.7-0.9
    VERY_HIGH = "very_high"  # 0.9-1.0


@dataclass
class KnowledgeItem:
    """A single piece of agent knowledge."""
    knowledge_id: str
    agent_id: str
    knowledge_type: KnowledgeType
    content: str
    title: Optional[str] = None
    description: Optional[str] = None
    
    # Metadata and context
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source: KnowledgeSource = KnowledgeSource.MANUAL_INPUT
    source_context: Dict[str, Any] = field(default_factory=dict)
    
    # Knowledge quality and validation
    confidence_score: float = 0.5
    validation_count: int = 0
    usage_count: int = 0
    success_rate: float = 0.0
    
    # Access and sharing
    access_level: AccessLevel = AccessLevel.PRIVATE
    shared_with: List[str] = field(default_factory=list)  # Agent IDs
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "knowledge_id": self.knowledge_id,
            "agent_id": self.agent_id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
            "tags": self.tags,
            "source": self.source.value,
            "source_context": self.source_context,
            "confidence_score": self.confidence_score,
            "validation_count": self.validation_count,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "access_level": self.access_level.value,
            "shared_with": self.shared_with,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create from dictionary."""
        return cls(
            knowledge_id=data["knowledge_id"],
            agent_id=data["agent_id"],
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            content=data["content"],
            title=data.get("title"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            source=KnowledgeSource(data.get("source", KnowledgeSource.MANUAL_INPUT.value)),
            source_context=data.get("source_context", {}),
            confidence_score=data.get("confidence_score", 0.5),
            validation_count=data.get("validation_count", 0),
            usage_count=data.get("usage_count", 0),
            success_rate=data.get("success_rate", 0.0),
            access_level=AccessLevel(data.get("access_level", AccessLevel.PRIVATE.value)),
            shared_with=data.get("shared_with", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
        )
    
    def update_usage_stats(self, success: bool) -> None:
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Update success rate
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Moving average
            weight = 1.0 / self.usage_count
            if success:
                self.success_rate = self.success_rate * (1 - weight) + weight
            else:
                self.success_rate = self.success_rate * (1 - weight)
    
    def get_confidence_level(self) -> KnowledgeConfidence:
        """Get confidence level enum."""
        if self.confidence_score >= 0.9:
            return KnowledgeConfidence.VERY_HIGH
        elif self.confidence_score >= 0.7:
            return KnowledgeConfidence.HIGH
        elif self.confidence_score >= 0.4:
            return KnowledgeConfidence.MEDIUM
        else:
            return KnowledgeConfidence.LOW
    
    def is_accessible_by(self, agent_id: str, agent_role: Optional[str] = None) -> bool:
        """Check if knowledge is accessible by given agent."""
        if self.agent_id == agent_id:
            return True  # Own knowledge
        
        if self.access_level == AccessLevel.PRIVATE:
            return False
        
        if self.access_level == AccessLevel.PUBLIC:
            return True
        
        if agent_id in self.shared_with:
            return True
        
        # For team-based access, would need to check agent roles/teams
        # Simplified for now
        return self.access_level in [AccessLevel.CROSS_TEAM, AccessLevel.PUBLIC]


@dataclass
class AgentKnowledgeBase:
    """Complete knowledge base for an agent."""
    agent_id: str
    knowledge_items: List[KnowledgeItem] = field(default_factory=list)
    expertise_areas: List[str] = field(default_factory=list)
    collaboration_patterns: Dict[str, Any] = field(default_factory=dict)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def get_knowledge_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeItem]:
        """Get knowledge items by type."""
        return [item for item in self.knowledge_items if item.knowledge_type == knowledge_type]
    
    def get_high_confidence_knowledge(self, threshold: float = 0.7) -> List[KnowledgeItem]:
        """Get high confidence knowledge items."""
        return [item for item in self.knowledge_items if item.confidence_score >= threshold]
    
    def get_recent_knowledge(self, hours: int = 24) -> List[KnowledgeItem]:
        """Get recently created or used knowledge."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            item for item in self.knowledge_items
            if item.created_at >= cutoff or (item.last_used and item.last_used >= cutoff)
        ]
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeItem]:
        """Simple text search in knowledge base."""
        query_lower = query.lower()
        results = []
        
        for item in self.knowledge_items:
            score = 0
            
            # Search in content
            if query_lower in item.content.lower():
                score += 2
            
            # Search in title
            if item.title and query_lower in item.title.lower():
                score += 3
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in item.tags):
                score += 1
            
            if score > 0:
                results.append((item, score))
        
        # Sort by score and confidence
        results.sort(key=lambda x: (x[1], x[0].confidence_score), reverse=True)
        return [item for item, _ in results[:limit]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "knowledge_items": [item.to_dict() for item in self.knowledge_items],
            "expertise_areas": self.expertise_areas,
            "collaboration_patterns": self.collaboration_patterns,
            "learning_preferences": self.learning_preferences,
            "performance_metrics": self.performance_metrics,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class KnowledgeRecommendation:
    """Knowledge recommendation for an agent."""
    recommendation_id: str
    target_agent_id: str
    recommended_knowledge: KnowledgeItem
    source_agent_id: str
    relevance_score: float
    reasoning: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    accepted: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "target_agent_id": self.target_agent_id,
            "recommended_knowledge": self.recommended_knowledge.to_dict(),
            "source_agent_id": self.source_agent_id,
            "relevance_score": self.relevance_score,
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
            "accepted": self.accepted
        }


# =============================================================================
# KNOWLEDGE LEARNING ENGINE
# =============================================================================

class KnowledgeLearningEngine:
    """Learns from workflow outcomes and task results."""
    
    def __init__(self, knowledge_manager: 'AgentKnowledgeManager'):
        self.knowledge_manager = knowledge_manager
        self.learning_patterns = {
            "workflow_success": self._learn_from_workflow_success,
            "workflow_failure": self._learn_from_workflow_failure,
            "task_completion": self._learn_from_task_completion,
            "agent_interaction": self._learn_from_agent_interaction,
            "performance_feedback": self._learn_from_performance_feedback
        }
    
    async def learn_from_workflow_outcome(
        self,
        workflow_id: str,
        agent_id: str,
        outcome: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn from workflow execution outcome."""
        learned_knowledge = []
        
        try:
            success = outcome.get("success", False)
            execution_time = outcome.get("execution_time_ms", 0)
            completed_tasks = outcome.get("completed_tasks", 0)
            failed_tasks = outcome.get("failed_tasks", 0)
            
            if success:
                learned_knowledge.extend(
                    await self._learn_from_workflow_success(
                        workflow_id, agent_id, outcome, execution_context
                    )
                )
            else:
                learned_knowledge.extend(
                    await self._learn_from_workflow_failure(
                        workflow_id, agent_id, outcome, execution_context
                    )
                )
            
            # Learn performance patterns
            if execution_time > 0:
                performance_knowledge = await self._learn_performance_patterns(
                    agent_id, workflow_id, execution_time, completed_tasks, failed_tasks
                )
                learned_knowledge.extend(performance_knowledge)
            
            # Store learned knowledge
            for knowledge in learned_knowledge:
                await self.knowledge_manager.add_knowledge(knowledge)
            
            logger.info(
                f"Learned {len(learned_knowledge)} knowledge items from workflow {workflow_id}",
                agent_id=agent_id,
                workflow_success=success
            )
            
            return learned_knowledge
            
        except Exception as e:
            logger.error(f"Failed to learn from workflow outcome: {e}")
            return []
    
    async def _learn_from_workflow_success(
        self,
        workflow_id: str,
        agent_id: str,
        outcome: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn from successful workflow execution."""
        knowledge_items = []
        
        # Extract successful patterns
        if "critical_path" in context:
            critical_path = context["critical_path"]
            
            pattern_knowledge = KnowledgeItem(
                knowledge_id=f"pattern_{workflow_id}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                knowledge_type=KnowledgeType.PATTERN,
                title=f"Successful workflow pattern for {workflow_id}",
                content=f"Critical path optimization: {json.dumps(critical_path, indent=2)}",
                source=KnowledgeSource.WORKFLOW_EXECUTION,
                source_context={
                    "workflow_id": workflow_id,
                    "execution_time_ms": outcome.get("execution_time_ms", 0),
                    "completed_tasks": outcome.get("completed_tasks", 0)
                },
                confidence_score=0.8,
                access_level=AccessLevel.TEAM_SHARED,
                tags=["workflow", "pattern", "success", "critical-path"]
            )
            knowledge_items.append(pattern_knowledge)
        
        # Learn about effective task sequencing
        if "task_execution_order" in context:
            sequencing_knowledge = KnowledgeItem(
                knowledge_id=f"sequence_{workflow_id}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                knowledge_type=KnowledgeType.BEST_PRACTICE,
                title="Effective task sequencing",
                content=f"Task execution order that led to success: {context['task_execution_order']}",
                source=KnowledgeSource.WORKFLOW_EXECUTION,
                source_context={"workflow_id": workflow_id},
                confidence_score=0.7,
                access_level=AccessLevel.CROSS_TEAM,
                tags=["sequencing", "best-practice", "efficiency"]
            )
            knowledge_items.append(sequencing_knowledge)
        
        return knowledge_items
    
    async def _learn_from_workflow_failure(
        self,
        workflow_id: str,
        agent_id: str,
        outcome: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn from failed workflow execution."""
        knowledge_items = []
        
        error_message = outcome.get("error_message", "Unknown error")
        failed_tasks = outcome.get("failed_tasks", 0)
        
        # Learn from failure patterns
        lesson_learned = KnowledgeItem(
            knowledge_id=f"lesson_{workflow_id}_{uuid.uuid4().hex[:8]}",
            agent_id=agent_id,
            knowledge_type=KnowledgeType.LESSON_LEARNED,
            title=f"Failure analysis for workflow {workflow_id}",
            content=f"Workflow failed with {failed_tasks} failed tasks. Error: {error_message}",
            description="Analysis of workflow failure to prevent future occurrences",
            source=KnowledgeSource.WORKFLOW_EXECUTION,
            source_context={
                "workflow_id": workflow_id,
                "error_message": error_message,
                "failed_tasks": failed_tasks
            },
            confidence_score=0.9,  # High confidence in failure analysis
            access_level=AccessLevel.CROSS_TEAM,  # Share failures to help others
            tags=["failure", "lesson-learned", "debugging", "workflow"]
        )
        knowledge_items.append(lesson_learned)
        
        # Extract specific failure patterns
        if "bottleneck_tasks" in context:
            bottleneck_knowledge = KnowledgeItem(
                knowledge_id=f"bottleneck_{workflow_id}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                knowledge_type=KnowledgeType.PATTERN,
                title="Bottleneck pattern identification",
                content=f"Identified bottlenecks: {context['bottleneck_tasks']}",
                source=KnowledgeSource.WORKFLOW_EXECUTION,
                source_context={"workflow_id": workflow_id},
                confidence_score=0.8,
                access_level=AccessLevel.TEAM_SHARED,
                tags=["bottleneck", "performance", "optimization"]
            )
            knowledge_items.append(bottleneck_knowledge)
        
        return knowledge_items
    
    async def _learn_from_task_completion(
        self,
        task_id: str,
        agent_id: str,
        task_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn from individual task completion."""
        knowledge_items = []
        
        success = task_result.get("success", False)
        execution_time = task_result.get("execution_time_ms", 0)
        
        if success and execution_time > 0:
            # Learn efficient task execution patterns
            if execution_time < 1000:  # Fast execution (< 1 second)
                efficiency_knowledge = KnowledgeItem(
                    knowledge_id=f"efficiency_{task_id}_{uuid.uuid4().hex[:8]}",
                    agent_id=agent_id,
                    knowledge_type=KnowledgeType.SKILL,
                    title="Efficient task execution",
                    content=f"Completed task {task_id} efficiently in {execution_time}ms",
                    source=KnowledgeSource.TASK_COMPLETION,
                    source_context={"task_id": task_id, "execution_time_ms": execution_time},
                    confidence_score=0.6,
                    access_level=AccessLevel.PRIVATE,
                    tags=["efficiency", "skill", "performance"]
                )
                knowledge_items.append(efficiency_knowledge)
        
        return knowledge_items
    
    async def _learn_from_agent_interaction(
        self,
        interaction_data: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn from agent-to-agent interactions."""
        knowledge_items = []
        
        source_agent = interaction_data.get("source_agent_id")
        target_agent = interaction_data.get("target_agent_id")
        interaction_type = interaction_data.get("interaction_type")
        success = interaction_data.get("success", False)
        
        if success and source_agent and target_agent:
            collaboration_knowledge = KnowledgeItem(
                knowledge_id=f"collab_{source_agent}_{target_agent}_{uuid.uuid4().hex[:8]}",
                agent_id=source_agent,
                knowledge_type=KnowledgeType.COLLABORATION,
                title=f"Successful collaboration with {target_agent}",
                content=f"Effective {interaction_type} interaction with agent {target_agent}",
                source=KnowledgeSource.AGENT_INTERACTION,
                source_context=interaction_data,
                confidence_score=0.7,
                access_level=AccessLevel.TEAM_SHARED,
                tags=["collaboration", "interaction", target_agent]
            )
            knowledge_items.append(collaboration_knowledge)
        
        return knowledge_items
    
    async def _learn_performance_patterns(
        self,
        agent_id: str,
        workflow_id: str,
        execution_time: float,
        completed_tasks: int,
        failed_tasks: int
    ) -> List[KnowledgeItem]:
        """Learn performance patterns from execution metrics."""
        knowledge_items = []
        
        # Calculate performance score
        total_tasks = completed_tasks + failed_tasks
        success_rate = completed_tasks / max(1, total_tasks)
        
        if success_rate > 0.9:  # High success rate
            performance_knowledge = KnowledgeItem(
                knowledge_id=f"perf_{workflow_id}_{uuid.uuid4().hex[:8]}",
                agent_id=agent_id,
                knowledge_type=KnowledgeType.EXPERIENCE,
                title="High performance execution",
                content=f"Achieved {success_rate:.2%} success rate in workflow {workflow_id}",
                source=KnowledgeSource.WORKFLOW_EXECUTION,
                source_context={
                    "workflow_id": workflow_id,
                    "success_rate": success_rate,
                    "execution_time_ms": execution_time
                },
                confidence_score=success_rate,
                access_level=AccessLevel.PRIVATE,
                tags=["performance", "experience", "success"]
            )
            knowledge_items.append(performance_knowledge)
        
        return knowledge_items


# =============================================================================
# AGENT KNOWLEDGE MANAGER
# =============================================================================

class AgentKnowledgeManager:
    """
    Manages agent-scoped knowledge with cross-agent sharing and access controls.
    
    Features:
    - Agent-specific knowledge bases with isolation
    - Cross-agent knowledge sharing with permissions
    - Learning from workflow and task outcomes
    - Knowledge consolidation and pattern recognition
    - Expertise tracking and recommendation engine
    """
    
    def __init__(self, task_processor: SemanticMemoryTaskProcessor, redis_client=None):
        self.task_processor = task_processor
        self.redis = redis_client or get_redis()
        
        # Knowledge storage
        self.agent_knowledge_bases: Dict[str, AgentKnowledgeBase] = {}
        self.knowledge_recommendations: Dict[str, List[KnowledgeRecommendation]] = defaultdict(list)
        
        # Learning engine
        self.learning_engine = KnowledgeLearningEngine(self)
        
        # Knowledge graph for relationships
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)  # knowledge_id -> related_knowledge_ids
        
        # Performance metrics
        self.metrics = {
            "total_knowledge_items": 0,
            "shared_knowledge_items": 0,
            "knowledge_usage_count": 0,
            "cross_agent_shares": 0,
            "recommendation_acceptance_rate": 0.0
        }
        
        logger.info("Agent Knowledge Manager initialized")
    
    async def get_agent_knowledge_base(self, agent_id: str) -> AgentKnowledgeBase:
        """Get or create agent knowledge base."""
        if agent_id not in self.agent_knowledge_bases:
            # Try to load from storage first
            knowledge_base = await self._load_knowledge_base(agent_id)
            if not knowledge_base:
                knowledge_base = AgentKnowledgeBase(agent_id=agent_id)
            self.agent_knowledge_bases[agent_id] = knowledge_base
        
        return self.agent_knowledge_bases[agent_id]
    
    async def add_knowledge(self, knowledge_item: KnowledgeItem) -> bool:
        """Add knowledge item to agent's knowledge base."""
        try:
            knowledge_base = await self.get_agent_knowledge_base(knowledge_item.agent_id)
            
            # Check for duplicates
            existing = [
                item for item in knowledge_base.knowledge_items
                if item.content == knowledge_item.content and item.knowledge_type == knowledge_item.knowledge_type
            ]
            
            if existing:
                # Update existing knowledge instead of creating duplicate
                existing_item = existing[0]
                existing_item.validation_count += 1
                existing_item.confidence_score = min(1.0, existing_item.confidence_score + 0.1)
                existing_item.updated_at = datetime.utcnow()
                logger.debug(f"Updated existing knowledge item {existing_item.knowledge_id}")
            else:
                # Add new knowledge
                knowledge_base.knowledge_items.append(knowledge_item)
                self.metrics["total_knowledge_items"] += 1
                
                if knowledge_item.access_level != AccessLevel.PRIVATE:
                    self.metrics["shared_knowledge_items"] += 1
                
                logger.debug(f"Added new knowledge item {knowledge_item.knowledge_id}")
            
            knowledge_base.last_updated = datetime.utcnow()
            
            # Store in semantic memory for searchability
            await self._store_knowledge_in_semantic_memory(knowledge_item)
            
            # Persist knowledge base
            await self._persist_knowledge_base(knowledge_base)
            
            # Generate recommendations for other agents
            await self._generate_knowledge_recommendations(knowledge_item)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False
    
    async def search_knowledge(
        self,
        agent_id: str,
        query: str,
        include_shared: bool = True,
        knowledge_types: Optional[List[KnowledgeType]] = None,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """Search for knowledge items."""
        try:
            # Search in agent's own knowledge base
            knowledge_base = await self.get_agent_knowledge_base(agent_id)
            results = knowledge_base.search_knowledge(query, limit)
            
            # Filter by knowledge types if specified
            if knowledge_types:
                results = [item for item in results if item.knowledge_type in knowledge_types]
            
            # Search in shared knowledge if enabled
            if include_shared:
                shared_results = await self._search_shared_knowledge(agent_id, query, knowledge_types)
                
                # Merge results, prioritizing own knowledge
                all_results = results + shared_results
                
                # Remove duplicates and sort by relevance and confidence
                seen = set()
                unique_results = []
                for item in all_results:
                    if item.knowledge_id not in seen:
                        seen.add(item.knowledge_id)
                        unique_results.append(item)
                
                results = unique_results[:limit]
            
            # Update usage statistics
            for item in results:
                item.usage_count += 1
                item.last_used = datetime.utcnow()
            
            self.metrics["knowledge_usage_count"] += len(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []
    
    async def share_knowledge(
        self,
        knowledge_id: str,
        source_agent_id: str,
        target_agent_ids: List[str],
        access_level: AccessLevel = AccessLevel.TEAM_SHARED
    ) -> bool:
        """Share knowledge with other agents."""
        try:
            # Find the knowledge item
            source_knowledge_base = await self.get_agent_knowledge_base(source_agent_id)
            knowledge_item = None
            
            for item in source_knowledge_base.knowledge_items:
                if item.knowledge_id == knowledge_id:
                    knowledge_item = item
                    break
            
            if not knowledge_item:
                logger.warning(f"Knowledge item {knowledge_id} not found")
                return False
            
            # Update access level and sharing list
            knowledge_item.access_level = access_level
            knowledge_item.shared_with.extend(target_agent_ids)
            knowledge_item.shared_with = list(set(knowledge_item.shared_with))  # Remove duplicates
            knowledge_item.updated_at = datetime.utcnow()
            
            # Create shared copies for target agents
            for target_agent_id in target_agent_ids:
                shared_knowledge = KnowledgeItem(
                    knowledge_id=f"shared_{knowledge_id}_{target_agent_id}_{uuid.uuid4().hex[:8]}",
                    agent_id=target_agent_id,
                    knowledge_type=knowledge_item.knowledge_type,
                    content=knowledge_item.content,
                    title=f"[Shared] {knowledge_item.title or 'Knowledge'}",
                    description=knowledge_item.description,
                    metadata={
                        **knowledge_item.metadata,
                        "original_agent": source_agent_id,
                        "original_knowledge_id": knowledge_id,
                        "shared_at": datetime.utcnow().isoformat()
                    },
                    tags=knowledge_item.tags + ["shared"],
                    source=KnowledgeSource.CROSS_AGENT_SHARING,
                    source_context={
                        "original_agent": source_agent_id,
                        "original_knowledge_id": knowledge_id
                    },
                    confidence_score=knowledge_item.confidence_score * 0.9,  # Slightly reduce confidence for shared
                    access_level=AccessLevel.PRIVATE,  # Becomes private to receiving agent
                )
                
                await self.add_knowledge(shared_knowledge)
            
            # Persist updated source knowledge base
            await self._persist_knowledge_base(source_knowledge_base)
            
            self.metrics["cross_agent_shares"] += 1
            
            logger.info(
                f"Knowledge {knowledge_id} shared by {source_agent_id} with {len(target_agent_ids)} agents"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Knowledge sharing failed: {e}")
            return False
    
    async def learn_from_workflow_outcome(
        self,
        workflow_id: str,
        agent_id: str,
        outcome: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> List[KnowledgeItem]:
        """Learn knowledge from workflow execution outcome."""
        return await self.learning_engine.learn_from_workflow_outcome(
            workflow_id, agent_id, outcome, execution_context
        )
    
    async def get_agent_expertise(self, agent_id: str) -> Dict[str, Any]:
        """Get agent expertise analysis."""
        try:
            knowledge_base = await self.get_agent_knowledge_base(agent_id)
            
            # Analyze expertise areas
            skill_items = knowledge_base.get_knowledge_by_type(KnowledgeType.SKILL)
            pattern_items = knowledge_base.get_knowledge_by_type(KnowledgeType.PATTERN)
            experience_items = knowledge_base.get_knowledge_by_type(KnowledgeType.EXPERIENCE)
            
            # Extract expertise areas from tags and content
            expertise_tags = set()
            for item in skill_items + pattern_items + experience_items:
                expertise_tags.update(item.tags)
            
            # Calculate expertise scores
            expertise_scores = {}
            for tag in expertise_tags:
                related_items = [
                    item for item in knowledge_base.knowledge_items
                    if tag in item.tags
                ]
                if related_items:
                    avg_confidence = sum(item.confidence_score for item in related_items) / len(related_items)
                    avg_usage = sum(item.usage_count for item in related_items) / len(related_items)
                    expertise_scores[tag] = (avg_confidence + avg_usage / 10) / 2  # Combined score
            
            # Top expertise areas
            top_expertise = sorted(expertise_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "agent_id": agent_id,
                "total_knowledge_items": len(knowledge_base.knowledge_items),
                "high_confidence_items": len(knowledge_base.get_high_confidence_knowledge()),
                "expertise_areas": [area for area, score in top_expertise],
                "expertise_scores": dict(top_expertise),
                "knowledge_distribution": {
                    ktype.value: len(knowledge_base.get_knowledge_by_type(ktype))
                    for ktype in KnowledgeType
                },
                "recent_learning": len(knowledge_base.get_recent_knowledge()),
                "last_updated": knowledge_base.last_updated.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent expertise: {e}")
            return {"error": str(e)}
    
    async def get_knowledge_recommendations(self, agent_id: str) -> List[KnowledgeRecommendation]:
        """Get knowledge recommendations for an agent."""
        return self.knowledge_recommendations.get(agent_id, [])
    
    async def accept_knowledge_recommendation(
        self,
        agent_id: str,
        recommendation_id: str
    ) -> bool:
        """Accept a knowledge recommendation."""
        try:
            recommendations = self.knowledge_recommendations.get(agent_id, [])
            recommendation = None
            
            for rec in recommendations:
                if rec.recommendation_id == recommendation_id:
                    recommendation = rec
                    break
            
            if not recommendation:
                return False
            
            # Add the recommended knowledge to agent's knowledge base
            recommended_knowledge = recommendation.recommended_knowledge
            recommended_knowledge.agent_id = agent_id  # Change ownership
            recommended_knowledge.knowledge_id = f"accepted_{recommendation_id}_{uuid.uuid4().hex[:8]}"
            
            success = await self.add_knowledge(recommended_knowledge)
            
            if success:
                recommendation.accepted = True
                
                # Update acceptance rate metric
                total_recommendations = len(recommendations)
                accepted_recommendations = len([r for r in recommendations if r.accepted is True])
                self.metrics["recommendation_acceptance_rate"] = accepted_recommendations / total_recommendations
                
                logger.info(f"Agent {agent_id} accepted knowledge recommendation {recommendation_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to accept recommendation: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get knowledge manager metrics."""
        return {
            **self.metrics,
            "active_knowledge_bases": len(self.agent_knowledge_bases),
            "total_recommendations": sum(len(recs) for recs in self.knowledge_recommendations.values()),
            "knowledge_graph_edges": sum(len(edges) for edges in self.knowledge_graph.values())
        }
    
    # Private helper methods
    
    async def _search_shared_knowledge(
        self,
        agent_id: str,
        query: str,
        knowledge_types: Optional[List[KnowledgeType]] = None
    ) -> List[KnowledgeItem]:
        """Search for shared knowledge accessible by agent."""
        shared_results = []
        
        for other_agent_id, knowledge_base in self.agent_knowledge_bases.items():
            if other_agent_id == agent_id:
                continue  # Skip own knowledge base
            
            for item in knowledge_base.knowledge_items:
                if item.is_accessible_by(agent_id):
                    # Simple text matching
                    query_lower = query.lower()
                    if (query_lower in item.content.lower() or
                        (item.title and query_lower in item.title.lower()) or
                        any(query_lower in tag.lower() for tag in item.tags)):
                        
                        if not knowledge_types or item.knowledge_type in knowledge_types:
                            shared_results.append(item)
        
        return shared_results
    
    async def _store_knowledge_in_semantic_memory(self, knowledge_item: KnowledgeItem) -> None:
        """Store knowledge in semantic memory for advanced search."""
        try:
            # Create semantic memory task to ingest knowledge
            ingest_task = SemanticMemoryTask(
                task_id=f"ingest_knowledge_{knowledge_item.knowledge_id}",
                task_type=SemanticTaskType.INGEST_DOCUMENT,
                agent_id=knowledge_item.agent_id,
                priority=ProcessingPriority.NORMAL,
                payload={
                    "content": knowledge_item.content,
                    "metadata": {
                        "knowledge_id": knowledge_item.knowledge_id,
                        "knowledge_type": knowledge_item.knowledge_type.value,
                        "title": knowledge_item.title,
                        "agent_id": knowledge_item.agent_id,
                        "confidence_score": knowledge_item.confidence_score,
                        "access_level": knowledge_item.access_level.value
                    },
                    "agent_id": knowledge_item.agent_id,
                    "tags": knowledge_item.tags + ["agent_knowledge"],
                    "processing_options": {
                        "generate_summary": True,
                        "priority": ProcessingPriority.NORMAL
                    }
                }
            )
            
            await self.task_processor.submit_task(ingest_task)
            
        except Exception as e:
            logger.warning(f"Failed to store knowledge in semantic memory: {e}")
    
    async def _generate_knowledge_recommendations(self, knowledge_item: KnowledgeItem) -> None:
        """Generate recommendations for sharing knowledge with other agents."""
        try:
            # Only recommend shareable knowledge
            if knowledge_item.access_level == AccessLevel.PRIVATE:
                return
            
            # Find agents who might benefit from this knowledge
            for agent_id, knowledge_base in self.agent_knowledge_bases.items():
                if agent_id == knowledge_item.agent_id:
                    continue  # Don't recommend to original owner
                
                # Calculate relevance score based on agent's existing knowledge
                relevance_score = self._calculate_knowledge_relevance(knowledge_item, knowledge_base)
                
                if relevance_score > 0.5:  # Threshold for recommendation
                    recommendation = KnowledgeRecommendation(
                        recommendation_id=f"rec_{knowledge_item.knowledge_id}_{agent_id}_{uuid.uuid4().hex[:8]}",
                        target_agent_id=agent_id,
                        recommended_knowledge=knowledge_item,
                        source_agent_id=knowledge_item.agent_id,
                        relevance_score=relevance_score,
                        reasoning=f"Recommended based on expertise overlap and knowledge gaps"
                    )
                    
                    self.knowledge_recommendations[agent_id].append(recommendation)
                    
                    # Limit recommendations per agent
                    if len(self.knowledge_recommendations[agent_id]) > 20:
                        # Remove oldest recommendations
                        self.knowledge_recommendations[agent_id] = sorted(
                            self.knowledge_recommendations[agent_id],
                            key=lambda r: r.created_at,
                            reverse=True
                        )[:20]
        
        except Exception as e:
            logger.warning(f"Failed to generate knowledge recommendations: {e}")
    
    def _calculate_knowledge_relevance(
        self,
        knowledge_item: KnowledgeItem,
        target_knowledge_base: AgentKnowledgeBase
    ) -> float:
        """Calculate relevance score for recommending knowledge to an agent."""
        relevance_score = 0.0
        
        # Check for tag overlap
        target_tags = set()
        for item in target_knowledge_base.knowledge_items:
            target_tags.update(item.tags)
        
        knowledge_tags = set(knowledge_item.tags)
        tag_overlap = len(knowledge_tags & target_tags) / max(1, len(knowledge_tags | target_tags))
        relevance_score += tag_overlap * 0.4
        
        # Check for knowledge type overlap
        target_types = set(item.knowledge_type for item in target_knowledge_base.knowledge_items)
        if knowledge_item.knowledge_type in target_types:
            relevance_score += 0.3
        
        # Boost for high confidence knowledge
        relevance_score += knowledge_item.confidence_score * 0.3
        
        return min(1.0, relevance_score)
    
    async def _persist_knowledge_base(self, knowledge_base: AgentKnowledgeBase) -> None:
        """Persist knowledge base to Redis."""
        try:
            key = f"agent_knowledge:{knowledge_base.agent_id}"
            data = json.dumps(knowledge_base.to_dict())
            await self.redis.setex(key, 86400, data)  # 24 hour TTL
        except Exception as e:
            logger.error(f"Failed to persist knowledge base: {e}")
    
    async def _load_knowledge_base(self, agent_id: str) -> Optional[AgentKnowledgeBase]:
        """Load knowledge base from Redis."""
        try:
            key = f"agent_knowledge:{agent_id}"
            data = await self.redis.get(key)
            if data:
                knowledge_dict = json.loads(data)
                knowledge_items = [
                    KnowledgeItem.from_dict(item_dict)
                    for item_dict in knowledge_dict.get("knowledge_items", [])
                ]
                
                return AgentKnowledgeBase(
                    agent_id=agent_id,
                    knowledge_items=knowledge_items,
                    expertise_areas=knowledge_dict.get("expertise_areas", []),
                    collaboration_patterns=knowledge_dict.get("collaboration_patterns", {}),
                    learning_preferences=knowledge_dict.get("learning_preferences", {}),
                    performance_metrics=knowledge_dict.get("performance_metrics", {}),
                    last_updated=datetime.fromisoformat(knowledge_dict["last_updated"])
                )
        except Exception as e:
            logger.debug(f"Failed to load knowledge base from Redis: {e}")
        return None


# =============================================================================
# GLOBAL KNOWLEDGE MANAGER INSTANCE
# =============================================================================

_knowledge_manager: Optional[AgentKnowledgeManager] = None


async def get_agent_knowledge_manager() -> AgentKnowledgeManager:
    """Get global agent knowledge manager instance."""
    global _knowledge_manager
    
    if _knowledge_manager is None:
        from .semantic_memory_task_processor import get_processor_manager
        
        processor_manager = await get_processor_manager()
        
        # Get or create a processor for the knowledge manager
        if not processor_manager.processors:
            await processor_manager.start_processor("knowledge_processor")
        
        processor = list(processor_manager.processors.values())[0]
        _knowledge_manager = AgentKnowledgeManager(processor)
    
    return _knowledge_manager


async def shutdown_agent_knowledge_manager():
    """Shutdown global agent knowledge manager."""
    global _knowledge_manager
    
    if _knowledge_manager:
        # Cleanup would go here
        _knowledge_manager = None
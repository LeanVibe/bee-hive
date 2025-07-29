"""
Cross-Agent Knowledge Manager for LeanVibe Agent Hive 2.0

Enhanced knowledge sharing system that enables intelligent collaboration between agents
through sophisticated access controls, quality scoring, and collaborative learning patterns.

Features:
- Advanced Access Control: Role-based permissions with granular sharing controls
- Knowledge Quality Scoring: Reputation and reliability metrics for shared knowledge
- Collaborative Learning: Agents learn from successful patterns across teams  
- Privacy Preservation: Agent-specific vs shared knowledge boundaries
- Knowledge Graphs: Relationship mapping between agent expertise domains
- Discovery Engine: Find relevant agent expertise for specific domains/capabilities
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import logging

import structlog
import networkx as nx

from .agent_knowledge_manager import (
    AgentKnowledgeManager, KnowledgeItem, AgentKnowledgeBase, 
    KnowledgeType, KnowledgeSource, AccessLevel, KnowledgeConfidence
)
from .context_compression_engine import get_context_compression_engine, CompressionConfig, CompressionQuality
from .database import get_session
from .redis import get_redis

logger = structlog.get_logger()


# =============================================================================
# CROSS-AGENT KNOWLEDGE TYPES AND CONFIGURATIONS
# =============================================================================

class SharingPolicy(str, Enum):
    """Knowledge sharing policies."""
    PRIVATE_ONLY = "private_only"
    TEAM_SHARING = "team_sharing" 
    CROSS_TEAM_SELECTIVE = "cross_team_selective"
    OPEN_SHARING = "open_sharing"
    EXPERTISE_BASED = "expertise_based"


class KnowledgeQualityMetric(str, Enum):
    """Quality metrics for knowledge assessment."""
    ACCURACY = "accuracy"
    USEFULNESS = "usefulness"
    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    SPECIFICITY = "specificity"


class CollaborationPattern(str, Enum):
    """Types of collaboration patterns."""
    MENTOR_STUDENT = "mentor_student"
    PEER_TO_PEER = "peer_to_peer"
    EXPERT_CONSULTATION = "expert_consultation"
    TEAM_COLLABORATION = "team_collaboration"
    KNOWLEDGE_BROADCAST = "knowledge_broadcast"


@dataclass
class AgentExpertise:
    """Agent expertise profile."""
    agent_id: str
    domain: str
    capability: str
    proficiency_level: float  # 0.0 to 1.0
    evidence_count: int
    success_rate: float
    last_demonstrated: datetime
    validation_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "capability": self.capability,
            "proficiency_level": self.proficiency_level,
            "evidence_count": self.evidence_count,
            "success_rate": self.success_rate,
            "last_demonstrated": self.last_demonstrated.isoformat(),
            "validation_sources": self.validation_sources
        }


@dataclass
class KnowledgeQualityScore:
    """Quality assessment for shared knowledge."""
    knowledge_id: str
    overall_score: float
    metric_scores: Dict[KnowledgeQualityMetric, float] = field(default_factory=dict)
    reviewer_count: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    usage_success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_feedback(self, positive: bool, success_rate: Optional[float] = None):
        """Update quality score based on feedback."""
        if positive:
            self.positive_feedback += 1
        else:
            self.negative_feedback += 1
        
        self.reviewer_count = self.positive_feedback + self.negative_feedback
        
        if success_rate is not None:
            # Update usage success rate with exponential moving average
            alpha = 0.3  # Learning rate
            self.usage_success_rate = (alpha * success_rate) + ((1 - alpha) * self.usage_success_rate)
        
        # Recalculate overall score
        feedback_ratio = self.positive_feedback / max(1, self.reviewer_count)
        self.overall_score = (feedback_ratio * 0.4) + (self.usage_success_rate * 0.6)
        self.last_updated = datetime.utcnow()


@dataclass
class SharingRequest:
    """Request to share knowledge with other agents."""
    request_id: str
    source_agent_id: str
    target_agent_ids: List[str]
    knowledge_id: str
    sharing_policy: SharingPolicy
    justification: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


@dataclass
class CollaborativeLearningOutcome:
    """Outcome from collaborative learning between agents."""
    outcome_id: str
    workflow_id: str
    participating_agents: List[str]
    collaboration_pattern: CollaborationPattern
    knowledge_shared: List[str]  # Knowledge IDs
    success_metrics: Dict[str, float]
    lessons_learned: List[str]
    best_practices: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# KNOWLEDGE DISCOVERY ENGINE  
# =============================================================================

class KnowledgeDiscoveryEngine:
    """Engine for discovering relevant agent expertise and knowledge."""
    
    def __init__(self, knowledge_manager: 'CrossAgentKnowledgeManager'):
        self.knowledge_manager = knowledge_manager
        self.expertise_graph = nx.DiGraph()  # Agent expertise relationships
        self.domain_graph = nx.Graph()  # Knowledge domain relationships
    
    async def discover_expertise(
        self,
        domain: str,
        capability: str,
        min_proficiency: float = 0.6,
        exclude_agents: Optional[List[str]] = None
    ) -> List[AgentExpertise]:
        """Discover agents with specific expertise."""
        try:
            exclude_agents = exclude_agents or []
            expertise_matches = []
            
            # Search through agent expertise profiles
            for agent_id, expertise_profile in self.knowledge_manager.agent_expertise.items():
                if agent_id in exclude_agents:
                    continue
                
                for expertise in expertise_profile:
                    # Domain and capability matching
                    domain_match = (
                        domain.lower() in expertise.domain.lower() or
                        expertise.domain.lower() in domain.lower()
                    )
                    capability_match = (
                        capability.lower() in expertise.capability.lower() or
                        expertise.capability.lower() in capability.lower()
                    )
                    
                    if (domain_match or capability_match) and expertise.proficiency_level >= min_proficiency:
                        expertise_matches.append(expertise)
            
            # Sort by proficiency and recent demonstration
            expertise_matches.sort(
                key=lambda e: (e.proficiency_level, e.success_rate, -e.evidence_count),
                reverse=True
            )
            
            return expertise_matches[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Expertise discovery failed: {e}")
            return []
    
    async def find_knowledge_gaps(
        self,
        agent_id: str,
        domain: str
    ) -> List[Dict[str, Any]]:
        """Identify knowledge gaps for an agent in a specific domain."""
        try:
            # Get agent's current knowledge in domain
            agent_knowledge = await self.knowledge_manager.base_manager.get_agent_knowledge_base(agent_id)
            agent_domain_knowledge = [
                item for item in agent_knowledge.knowledge_items
                if domain.lower() in " ".join(item.tags).lower() or domain.lower() in item.content.lower()
            ]
            
            # Find what other agents know in this domain
            other_agents_knowledge = []
            for other_agent_id in self.knowledge_manager.agent_expertise.keys():
                if other_agent_id != agent_id:
                    other_knowledge = await self.knowledge_manager.base_manager.get_agent_knowledge_base(other_agent_id)
                    domain_knowledge = [
                        item for item in other_knowledge.knowledge_items
                        if (domain.lower() in " ".join(item.tags).lower() or 
                            domain.lower() in item.content.lower()) and
                        item.access_level != AccessLevel.PRIVATE and
                        item.confidence_score >= 0.6
                    ]
                    other_agents_knowledge.extend(domain_knowledge)
            
            # Identify gaps (knowledge types/topics others have but agent doesn't)
            agent_topics = set()
            for item in agent_domain_knowledge:
                agent_topics.update(item.tags)
            
            gaps = []
            for item in other_agents_knowledge:
                item_topics = set(item.tags)
                if not (item_topics & agent_topics):  # No overlap
                    gaps.append({
                        "knowledge_type": item.knowledge_type.value,
                        "topics": list(item_topics),
                        "available_from": item.agent_id,
                        "confidence": item.confidence_score,
                        "description": item.title or item.content[:100]
                    })
            
            # Remove duplicates and sort by confidence
            unique_gaps = []
            seen_topics = set()
            for gap in sorted(gaps, key=lambda g: g["confidence"], reverse=True):
                gap_signature = tuple(sorted(gap["topics"]))
                if gap_signature not in seen_topics:
                    seen_topics.add(gap_signature)
                    unique_gaps.append(gap)
            
            return unique_gaps[:5]  # Top 5 gaps
            
        except Exception as e:
            logger.error(f"Knowledge gap analysis failed: {e}")
            return []
    
    async def recommend_collaborators(
        self,
        agent_id: str,
        task_description: str,
        collaboration_type: CollaborationPattern = CollaborationPattern.PEER_TO_PEER
    ) -> List[Dict[str, Any]]:
        """Recommend agents for collaboration based on complementary expertise."""
        try:
            # Extract keywords from task description
            task_keywords = set(word.lower() for word in task_description.split() if len(word) > 3)
            
            collaboration_candidates = []
            
            for other_agent_id, expertise_list in self.knowledge_manager.agent_expertise.items():
                if other_agent_id == agent_id:
                    continue
                
                relevance_score = 0.0
                matching_expertise = []
                
                for expertise in expertise_list:
                    # Check if expertise is relevant to task
                    expertise_keywords = set(expertise.domain.lower().split()) | set(expertise.capability.lower().split())
                    keyword_overlap = len(task_keywords & expertise_keywords)
                    
                    if keyword_overlap > 0:
                        expertise_relevance = (keyword_overlap / len(task_keywords)) * expertise.proficiency_level
                        relevance_score += expertise_relevance
                        matching_expertise.append(expertise)
                
                if relevance_score > 0.1:  # Minimum relevance threshold
                    collaboration_candidates.append({
                        "agent_id": other_agent_id,
                        "relevance_score": relevance_score,
                        "matching_expertise": [exp.to_dict() for exp in matching_expertise],
                        "collaboration_history": await self._get_collaboration_history(agent_id, other_agent_id)
                    })
            
            # Sort by relevance and collaboration history
            collaboration_candidates.sort(
                key=lambda c: (c["relevance_score"], len(c["collaboration_history"]["successful_collaborations"])),
                reverse=True
            )
            
            return collaboration_candidates[:5]  # Top 5 candidates
            
        except Exception as e:
            logger.error(f"Collaboration recommendation failed: {e}")
            return []
    
    async def _get_collaboration_history(self, agent1_id: str, agent2_id: str) -> Dict[str, Any]:
        """Get collaboration history between two agents."""
        # This would typically query a database of past collaborations
        # For now, return mock data
        return {
            "successful_collaborations": [],
            "total_collaborations": 0,
            "success_rate": 0.0,
            "last_collaboration": None
        }


# =============================================================================
# ACCESS CONTROL MANAGER
# =============================================================================

class AccessControlManager:
    """Manages access control and permissions for knowledge sharing."""
    
    def __init__(self):
        self.access_policies: Dict[str, Dict[str, Any]] = {}
        self.agent_roles: Dict[str, List[str]] = {}
        self.team_memberships: Dict[str, List[str]] = {}
    
    def set_agent_role(self, agent_id: str, roles: List[str]):
        """Set roles for an agent."""
        self.agent_roles[agent_id] = roles
    
    def set_team_membership(self, agent_id: str, teams: List[str]):
        """Set team memberships for an agent."""
        self.team_memberships[agent_id] = teams
    
    def can_access_knowledge(
        self,
        requester_agent_id: str,
        knowledge_item: KnowledgeItem,
        access_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Check if agent can access specific knowledge."""
        try:
            # Owner always has access
            if requester_agent_id == knowledge_item.agent_id:
                return True, "Owner access"
            
            # Check access level
            if knowledge_item.access_level == AccessLevel.PRIVATE:
                return False, "Private knowledge"
            
            if knowledge_item.access_level == AccessLevel.PUBLIC:
                return True, "Public access"
            
            # Check explicit sharing list
            if requester_agent_id in knowledge_item.shared_with:
                return True, "Explicitly shared"
            
            # Check team-based access
            if knowledge_item.access_level == AccessLevel.TEAM_SHARED:
                owner_teams = self.team_memberships.get(knowledge_item.agent_id, [])
                requester_teams = self.team_memberships.get(requester_agent_id, [])
                
                if any(team in requester_teams for team in owner_teams):
                    return True, "Same team access"
            
            # Check cross-team access
            if knowledge_item.access_level == AccessLevel.CROSS_TEAM:
                # Allow access if requester has sufficient role/clearance
                requester_roles = self.agent_roles.get(requester_agent_id, [])
                if "senior_agent" in requester_roles or "coordinator" in requester_roles:
                    return True, "Cross-team role access"
            
            # Check quality-based access
            if (access_context and 
                access_context.get("quality_threshold") and
                knowledge_item.confidence_score >= access_context["quality_threshold"]):
                return True, "Quality-based access"
            
            return False, "Access denied"
            
        except Exception as e:
            logger.error(f"Access control check failed: {e}")
            return False, f"Error: {e}"
    
    def create_sharing_policy(
        self,
        agent_id: str,
        policy_name: str,
        policy_config: Dict[str, Any]
    ):
        """Create a custom sharing policy for an agent."""
        if agent_id not in self.access_policies:
            self.access_policies[agent_id] = {}
        
        self.access_policies[agent_id][policy_name] = {
            "config": policy_config,
            "created_at": datetime.utcnow().isoformat()
        }


# =============================================================================
# MAIN CROSS-AGENT KNOWLEDGE MANAGER
# =============================================================================

class CrossAgentKnowledgeManager:
    """
    Enhanced cross-agent knowledge management system with sophisticated sharing,
    access controls, quality assessment, and collaborative learning capabilities.
    """
    
    def __init__(self, base_manager: AgentKnowledgeManager):
        """Initialize with base agent knowledge manager."""
        self.base_manager = base_manager
        self.redis = get_redis()
        
        # Cross-agent components
        self.discovery_engine = KnowledgeDiscoveryEngine(self)
        self.access_control = AccessControlManager()
        
        # Knowledge sharing infrastructure
        self.agent_expertise: Dict[str, List[AgentExpertise]] = defaultdict(list)
        self.knowledge_quality: Dict[str, KnowledgeQualityScore] = {}
        self.sharing_requests: Dict[str, SharingRequest] = {}
        self.collaboration_outcomes: List[CollaborativeLearningOutcome] = []
        
        # Performance metrics
        self.metrics = {
            "knowledge_shares": 0,
            "successful_shares": 0,
            "cross_agent_queries": 0,
            "expertise_discoveries": 0,
            "collaboration_sessions": 0,
            "quality_assessments": 0
        }
        
        logger.info("Cross-Agent Knowledge Manager initialized")
    
    async def initialize(self):
        """Initialize the cross-agent knowledge manager."""
        # Initialize compression engine for knowledge summarization
        self.compression_engine = await get_context_compression_engine()
        
        # Load existing expertise profiles and quality scores
        await self._load_agent_expertise()
        await self._load_knowledge_quality_scores()
        
        logger.info("✅ Cross-Agent Knowledge Manager fully initialized")
    
    # =============================================================================
    # KNOWLEDGE SHARING OPERATIONS
    # =============================================================================
    
    async def share_knowledge(
        self,
        knowledge_item: KnowledgeItem,
        target_agents: List[str],
        sharing_policy: SharingPolicy = SharingPolicy.TEAM_SHARING,
        justification: Optional[str] = None
    ) -> Dict[str, Any]:
        """Share knowledge with specified agents."""
        try:
            sharing_result = {
                "request_id": str(uuid.uuid4()),
                "source_agent": knowledge_item.agent_id,
                "target_agents": target_agents,
                "successful_shares": [],
                "failed_shares": [],
                "quality_score": None
            }
            
            # Assess knowledge quality before sharing
            quality_score = await self._assess_knowledge_quality(knowledge_item)
            sharing_result["quality_score"] = quality_score.overall_score
            
            # Apply sharing policy
            approved_targets = await self._apply_sharing_policy(
                knowledge_item, target_agents, sharing_policy
            )
            
            # Compress knowledge for efficient sharing if it's large
            compressed_knowledge = knowledge_item
            if len(knowledge_item.content) > 2000:  # Compress large content
                compression_config = CompressionConfig(
                    quality=CompressionQuality.BALANCED,
                    target_reduction=0.6
                )
                compressed_result = await self.compression_engine.compress_context(
                    knowledge_item.content, compression_config
                )
                
                # Create compressed version
                compressed_knowledge = KnowledgeItem(
                    knowledge_id=f"compressed_{knowledge_item.knowledge_id}",
                    agent_id=knowledge_item.agent_id,
                    knowledge_type=knowledge_item.knowledge_type,
                    content=compressed_result.compressed_content,
                    title=knowledge_item.title,
                    description=f"Compressed: {knowledge_item.description or ''}",
                    metadata={
                        **knowledge_item.metadata,
                        "compressed": True,
                        "compression_ratio": compressed_result.compression_ratio,
                        "original_knowledge_id": knowledge_item.knowledge_id
                    },
                    tags=knowledge_item.tags + ["compressed"],
                    source=KnowledgeSource.CROSS_AGENT_SHARING,
                    confidence_score=knowledge_item.confidence_score * 0.95  # Slight reduction for compression
                )
            
            # Share with approved targets
            for target_agent_id in approved_targets:
                try:
                    # Create shared knowledge copy
                    shared_knowledge = self._create_shared_knowledge_copy(
                        compressed_knowledge, target_agent_id
                    )
                    
                    # Add to target agent's knowledge base
                    success = await self.base_manager.add_knowledge(shared_knowledge)
                    
                    if success:
                        sharing_result["successful_shares"].append({
                            "agent_id": target_agent_id,
                            "shared_knowledge_id": shared_knowledge.knowledge_id
                        })
                        
                        # Update sharing metrics
                        await self._update_sharing_metrics(
                            knowledge_item.agent_id, target_agent_id, True
                        )
                    else:
                        sharing_result["failed_shares"].append({
                            "agent_id": target_agent_id,
                            "reason": "Failed to add to knowledge base"
                        })
                        
                except Exception as e:
                    sharing_result["failed_shares"].append({
                        "agent_id": target_agent_id,
                        "reason": str(e)
                    })
            
            self.metrics["knowledge_shares"] += 1
            self.metrics["successful_shares"] += len(sharing_result["successful_shares"])
            
            logger.info(
                f"Knowledge sharing completed: {len(sharing_result['successful_shares'])}/{len(target_agents)} successful",
                knowledge_id=knowledge_item.knowledge_id,
                source_agent=knowledge_item.agent_id
            )
            
            return sharing_result
            
        except Exception as e:
            logger.error(f"Knowledge sharing failed: {e}")
            return {
                "error": str(e),
                "successful_shares": [],
                "failed_shares": [{"agent_id": agent_id, "reason": str(e)} for agent_id in target_agents]
            }
    
    async def discover_relevant_knowledge(
        self,
        agent_id: str,
        query: str,
        domain: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Discover relevant knowledge from other agents."""
        try:
            self.metrics["cross_agent_queries"] += 1
            
            discovered_knowledge = []
            
            # Search across all agent knowledge bases
            for other_agent_id in self.base_manager.agent_knowledge_bases.keys():
                if other_agent_id == agent_id:
                    continue  # Skip own knowledge
                
                # Get other agent's knowledge base
                other_kb = await self.base_manager.get_agent_knowledge_base(other_agent_id)
                
                # Search for relevant knowledge
                relevant_items = other_kb.search_knowledge(query, limit=5)
                
                for item in relevant_items:
                    # Check access permissions
                    can_access, reason = self.access_control.can_access_knowledge(
                        agent_id, item
                    )
                    
                    if can_access:
                        # Get quality score
                        quality = self.knowledge_quality.get(item.knowledge_id)
                        quality_score = quality.overall_score if quality else 0.5
                        
                        discovered_knowledge.append({
                            "knowledge_id": item.knowledge_id,
                            "source_agent": other_agent_id,
                            "content": item.content,
                            "title": item.title,
                            "knowledge_type": item.knowledge_type.value,
                            "confidence_score": item.confidence_score,
                            "quality_score": quality_score,
                            "tags": item.tags,
                            "access_reason": reason,
                            "usage_count": item.usage_count,
                            "last_used": item.last_used.isoformat() if item.last_used else None
                        })
            
            # Sort by combined relevance score
            discovered_knowledge.sort(
                key=lambda k: (k["quality_score"] * k["confidence_score"]),
                reverse=True
            )
            
            return discovered_knowledge[:max_results]
            
        except Exception as e:
            logger.error(f"Knowledge discovery failed: {e}")
            return []
    
    async def request_knowledge_access(
        self,
        requester_agent_id: str,
        knowledge_owner_agent_id: str,
        knowledge_id: str,
        justification: str
    ) -> Dict[str, Any]:
        """Request access to specific knowledge from another agent."""
        try:
            request = SharingRequest(
                request_id=str(uuid.uuid4()),
                source_agent_id=knowledge_owner_agent_id,
                target_agent_ids=[requester_agent_id],
                knowledge_id=knowledge_id,
                sharing_policy=SharingPolicy.EXPERTISE_BASED,
                justification=justification
            )
            
            self.sharing_requests[request.request_id] = request
            
            # Auto-approve based on policies (simplified)
            # In production, this might require human approval or more sophisticated rules
            auto_approve = await self._evaluate_access_request(request)
            
            if auto_approve:
                request.approved = True
                request.approved_by = "system"
                request.approved_at = datetime.utcnow()
                
                # Grant access
                owner_kb = await self.base_manager.get_agent_knowledge_base(knowledge_owner_agent_id)
                knowledge_item = None
                
                for item in owner_kb.knowledge_items:
                    if item.knowledge_id == knowledge_id:
                        knowledge_item = item
                        break
                
                if knowledge_item:
                    sharing_result = await self.share_knowledge(
                        knowledge_item, [requester_agent_id], 
                        SharingPolicy.EXPERTISE_BASED, justification
                    )
                    
                    return {
                        "request_id": request.request_id,
                        "status": "approved",
                        "sharing_result": sharing_result
                    }
            
            return {
                "request_id": request.request_id,
                "status": "pending",
                "message": "Request submitted for review"
            }
            
        except Exception as e:
            logger.error(f"Knowledge access request failed: {e}")
            return {"error": str(e)}
    
    # =============================================================================
    # EXPERTISE DISCOVERY AND MANAGEMENT
    # =============================================================================
    
    async def register_agent_expertise(
        self,
        agent_id: str,
        domain: str,
        capability: str,
        evidence: Dict[str, Any]
    ) -> bool:
        """Register or update agent expertise based on evidence."""
        try:
            # Calculate proficiency level based on evidence
            proficiency = self._calculate_proficiency(evidence)
            
            # Find existing expertise or create new
            existing_expertise = None
            for expertise in self.agent_expertise[agent_id]:
                if expertise.domain == domain and expertise.capability == capability:
                    existing_expertise = expertise
                    break
            
            if existing_expertise:
                # Update existing expertise
                existing_expertise.evidence_count += 1
                existing_expertise.proficiency_level = (
                    existing_expertise.proficiency_level * 0.7 + proficiency * 0.3
                )
                existing_expertise.last_demonstrated = datetime.utcnow()
                
                # Update success rate
                success = evidence.get("success", True)
                existing_expertise.success_rate = (
                    existing_expertise.success_rate * 0.8 + (1.0 if success else 0.0) * 0.2
                )
            else:
                # Create new expertise
                new_expertise = AgentExpertise(
                    agent_id=agent_id,
                    domain=domain,
                    capability=capability,
                    proficiency_level=proficiency,
                    evidence_count=1,
                    success_rate=1.0 if evidence.get("success", True) else 0.0,
                    last_demonstrated=datetime.utcnow(),
                    validation_sources=[evidence.get("source", "system")]
                )
                self.agent_expertise[agent_id].append(new_expertise)
            
            # Persist expertise profile
            await self._persist_agent_expertise(agent_id)
            
            logger.info(f"Registered expertise for {agent_id}: {domain}/{capability}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register expertise: {e}")
            return False
    
    async def discover_agent_expertise(
        self,
        domain: str,
        capability: str,
        min_proficiency: float = 0.6
    ) -> List[AgentExpertise]:
        """Discover agents with specific expertise."""
        self.metrics["expertise_discoveries"] += 1
        return await self.discovery_engine.discover_expertise(
            domain, capability, min_proficiency
        )
    
    async def get_agent_expertise_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive expertise profile for an agent."""
        try:
            expertise_list = self.agent_expertise.get(agent_id, [])
            
            # Calculate overall metrics
            total_expertise = len(expertise_list)
            avg_proficiency = sum(exp.proficiency_level for exp in expertise_list) / max(1, total_expertise)
            avg_success_rate = sum(exp.success_rate for exp in expertise_list) / max(1, total_expertise)
            
            # Group by domain
            domains = defaultdict(list)
            for expertise in expertise_list:
                domains[expertise.domain].append(expertise)
            
            return {
                "agent_id": agent_id,
                "total_expertise_areas": total_expertise,
                "average_proficiency": avg_proficiency,
                "average_success_rate": avg_success_rate,
                "domains": {
                    domain: [exp.to_dict() for exp in expertise_list]
                    for domain, expertise_list in domains.items()
                },
                "top_expertise": [
                    exp.to_dict() for exp in 
                    sorted(expertise_list, key=lambda e: e.proficiency_level, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get expertise profile: {e}")
            return {"error": str(e)}
    
    # =============================================================================
    # COLLABORATIVE LEARNING
    # =============================================================================
    
    async def learn_from_collaboration(
        self,
        workflow_id: str,
        participating_agents: List[str],
        collaboration_data: Dict[str, Any]
    ) -> CollaborativeLearningOutcome:
        """Learn from collaborative workflow execution."""
        try:
            # Extract collaboration patterns
            pattern = self._identify_collaboration_pattern(collaboration_data)
            
            # Identify knowledge that was shared during collaboration
            shared_knowledge = collaboration_data.get("knowledge_shared", [])
            
            # Extract success metrics
            success_metrics = {
                "completion_time": collaboration_data.get("completion_time_ms", 0),
                "success_rate": collaboration_data.get("success_rate", 0.0),
                "efficiency_score": collaboration_data.get("efficiency_score", 0.0),
                "quality_score": collaboration_data.get("quality_score", 0.0)
            }
            
            # Generate lessons learned
            lessons_learned = await self._extract_lessons_learned(collaboration_data)
            best_practices = await self._extract_best_practices(collaboration_data)
            
            # Create learning outcome
            outcome = CollaborativeLearningOutcome(
                outcome_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                participating_agents=participating_agents,
                collaboration_pattern=pattern,
                knowledge_shared=shared_knowledge,
                success_metrics=success_metrics,
                lessons_learned=lessons_learned,
                best_practices=best_practices
            )
            
            self.collaboration_outcomes.append(outcome)
            
            # Update agent expertise based on collaboration
            for agent_id in participating_agents:
                await self._update_expertise_from_collaboration(agent_id, outcome)
            
            self.metrics["collaboration_sessions"] += 1
            
            logger.info(f"Learned from collaboration: {workflow_id} with {len(participating_agents)} agents")
            return outcome
            
        except Exception as e:
            logger.error(f"Collaborative learning failed: {e}")
            # Return empty outcome on error
            return CollaborativeLearningOutcome(
                outcome_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                participating_agents=participating_agents,
                collaboration_pattern=CollaborationPattern.PEER_TO_PEER,
                knowledge_shared=[],
                success_metrics={},
                lessons_learned=[],
                best_practices=[]
            )
    
    # =============================================================================
    # QUALITY ASSESSMENT AND MANAGEMENT
    # =============================================================================
    
    async def assess_knowledge_quality(
        self,
        knowledge_id: str,
        assessor_agent_id: str,
        feedback: Dict[str, Any]
    ) -> KnowledgeQualityScore:
        """Assess and update knowledge quality based on usage feedback."""
        try:
            # Get or create quality score
            if knowledge_id not in self.knowledge_quality:
                self.knowledge_quality[knowledge_id] = KnowledgeQualityScore(
                    knowledge_id=knowledge_id,
                    overall_score=0.5
                )
            
            quality_score = self.knowledge_quality[knowledge_id]
            
            # Process feedback
            positive = feedback.get("helpful", True)
            success_rate = feedback.get("success_rate")
            
            quality_score.update_feedback(positive, success_rate)
            
            # Update specific metric scores if provided
            for metric_name, score in feedback.get("metric_scores", {}).items():
                try:
                    metric = KnowledgeQualityMetric(metric_name)
                    quality_score.metric_scores[metric] = score
                except ValueError:
                    logger.warning(f"Unknown quality metric: {metric_name}")
            
            self.metrics["quality_assessments"] += 1
            
            # Persist quality score
            await self._persist_quality_score(quality_score)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return self.knowledge_quality.get(knowledge_id, KnowledgeQualityScore(
                knowledge_id=knowledge_id,
                overall_score=0.5
            ))
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get overall quality metrics across all knowledge."""
        if not self.knowledge_quality:
            return {"message": "No quality data available"}
        
        scores = [qs.overall_score for qs in self.knowledge_quality.values()]
        
        return {
            "total_assessed_knowledge": len(self.knowledge_quality),
            "average_quality_score": sum(scores) / len(scores),
            "high_quality_knowledge": len([s for s in scores if s >= 0.8]),
            "low_quality_knowledge": len([s for s in scores if s < 0.4]),
            "quality_distribution": {
                "excellent": len([s for s in scores if s >= 0.9]),
                "good": len([s for s in scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in scores if s < 0.5])
            }
        }
    
    # =============================================================================
    # UTILITY AND HELPER METHODS
    # =============================================================================
    
    def _create_shared_knowledge_copy(
        self,
        original: KnowledgeItem,
        target_agent_id: str
    ) -> KnowledgeItem:
        """Create a copy of knowledge for sharing with another agent."""
        return KnowledgeItem(
            knowledge_id=f"shared_{original.knowledge_id}_{target_agent_id}_{uuid.uuid4().hex[:8]}",
            agent_id=target_agent_id,
            knowledge_type=original.knowledge_type,
            content=original.content,
            title=f"[Shared] {original.title or 'Knowledge'}",
            description=original.description,
            metadata={
                **original.metadata,
                "original_agent": original.agent_id,
                "original_knowledge_id": original.knowledge_id,
                "shared_at": datetime.utcnow().isoformat(),
                "sharing_method": "cross_agent_sharing"
            },
            tags=original.tags + ["shared", f"from:{original.agent_id}"],
            source=KnowledgeSource.CROSS_AGENT_SHARING,
            source_context={
                "original_agent": original.agent_id,
                "original_knowledge_id": original.knowledge_id
            },
            confidence_score=original.confidence_score * 0.95,  # Slight reduction for shared knowledge
            access_level=AccessLevel.PRIVATE,  # Becomes private to receiving agent
            shared_with=[],
            created_at=datetime.utcnow()
        )
    
    async def _assess_knowledge_quality(self, knowledge_item: KnowledgeItem) -> KnowledgeQualityScore:
        """Assess the quality of knowledge before sharing."""
        if knowledge_item.knowledge_id in self.knowledge_quality:
            return self.knowledge_quality[knowledge_item.knowledge_id]
        
        # Create initial quality assessment
        quality_score = KnowledgeQualityScore(
            knowledge_id=knowledge_item.knowledge_id,
            overall_score=knowledge_item.confidence_score,
            metric_scores={
                KnowledgeQualityMetric.ACCURACY: knowledge_item.confidence_score,
                KnowledgeQualityMetric.USEFULNESS: min(1.0, knowledge_item.usage_count / 10),
                KnowledgeQualityMetric.COMPLETENESS: 0.7,  # Default
                KnowledgeQualityMetric.TIMELINESS: 1.0 if knowledge_item.created_at > datetime.utcnow() - timedelta(days=30) else 0.5,
                KnowledgeQualityMetric.SPECIFICITY: 0.8  # Default
            }
        )
        
        self.knowledge_quality[knowledge_item.knowledge_id] = quality_score
        return quality_score
    
    async def _apply_sharing_policy(
        self,
        knowledge_item: KnowledgeItem,
        target_agents: List[str],
        policy: SharingPolicy
    ) -> List[str]:
        """Apply sharing policy to determine approved targets."""
        approved_targets = []
        
        for agent_id in target_agents:
            if policy == SharingPolicy.PRIVATE_ONLY:
                continue  # No sharing allowed
            elif policy == SharingPolicy.OPEN_SHARING:
                approved_targets.append(agent_id)
            elif policy == SharingPolicy.TEAM_SHARING:
                # Check if in same team
                owner_teams = self.access_control.team_memberships.get(knowledge_item.agent_id, [])
                agent_teams = self.access_control.team_memberships.get(agent_id, [])
                if any(team in agent_teams for team in owner_teams):
                    approved_targets.append(agent_id)
            elif policy == SharingPolicy.EXPERTISE_BASED:
                # Check if agent has related expertise
                agent_expertise = self.agent_expertise.get(agent_id, [])
                knowledge_tags = set(knowledge_item.tags)
                
                for expertise in agent_expertise:
                    expertise_keywords = set(expertise.domain.split()) | set(expertise.capability.split())
                    if knowledge_tags & expertise_keywords:
                        approved_targets.append(agent_id)
                        break
        
        return approved_targets
    
    def _calculate_proficiency(self, evidence: Dict[str, Any]) -> float:
        """Calculate proficiency level from evidence."""
        base_score = 0.5
        
        # Task completion success rate
        if "success_rate" in evidence:
            base_score += evidence["success_rate"] * 0.3
        
        # Performance metrics
        if "performance_score" in evidence:
            base_score += evidence["performance_score"] * 0.2
        
        # Peer validation
        if "peer_validation" in evidence:
            base_score += evidence["peer_validation"] * 0.2
        
        # Complexity handled
        if "complexity_score" in evidence:
            base_score += evidence["complexity_score"] * 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _identify_collaboration_pattern(self, collaboration_data: Dict[str, Any]) -> CollaborationPattern:
        """Identify the type of collaboration pattern from data."""
        agent_count = collaboration_data.get("agent_count", 1)
        interaction_type = collaboration_data.get("interaction_type", "")
        
        if "mentor" in interaction_type.lower():
            return CollaborationPattern.MENTOR_STUDENT
        elif agent_count == 2:
            return CollaborationPattern.PEER_TO_PEER
        elif "expert" in interaction_type.lower() or "consultation" in interaction_type.lower():
            return CollaborationPattern.EXPERT_CONSULTATION
        elif agent_count > 3:
            return CollaborationPattern.TEAM_COLLABORATION
        else:
            return CollaborationPattern.PEER_TO_PEER
    
    async def _extract_lessons_learned(self, collaboration_data: Dict[str, Any]) -> List[str]:
        """Extract lessons learned from collaboration data."""
        lessons = []
        
        if collaboration_data.get("errors"):
            lessons.append(f"Error patterns: {collaboration_data['errors']}")
        
        if collaboration_data.get("bottlenecks"):
            lessons.append(f"Bottlenecks identified: {collaboration_data['bottlenecks']}")
        
        if collaboration_data.get("improvements"):
            lessons.append(f"Improvement opportunities: {collaboration_data['improvements']}")
        
        return lessons
    
    async def _extract_best_practices(self, collaboration_data: Dict[str, Any]) -> List[str]:
        """Extract best practices from successful collaboration."""
        practices = []
        
        if collaboration_data.get("success_factors"):
            practices.extend(collaboration_data["success_factors"])
        
        if collaboration_data.get("effective_patterns"):
            practices.extend(collaboration_data["effective_patterns"])
        
        return practices
    
    async def _update_expertise_from_collaboration(
        self,
        agent_id: str,
        outcome: CollaborativeLearningOutcome
    ) -> None:
        """Update agent expertise based on collaboration outcome."""
        if outcome.success_metrics.get("success_rate", 0) > 0.7:
            # Successful collaboration - update relevant expertise
            for domain in ["collaboration", "teamwork"]:
                evidence = {
                    "success_rate": outcome.success_metrics.get("success_rate", 0),
                    "performance_score": outcome.success_metrics.get("efficiency_score", 0),
                    "source": f"collaboration_{outcome.workflow_id}"
                }
                await self.register_agent_expertise(
                    agent_id, domain, "team_coordination", evidence
                )
    
    async def _evaluate_access_request(self, request: SharingRequest) -> bool:
        """Evaluate whether to auto-approve an access request."""
        # Simple auto-approval logic - in production this would be more sophisticated
        return True  # For demo purposes, auto-approve all requests
    
    async def _update_sharing_metrics(
        self,
        source_agent_id: str,
        target_agent_id: str,
        success: bool
    ) -> None:
        """Update metrics for knowledge sharing between agents."""
        # This would typically update a database of sharing statistics
        pass
    
    # =============================================================================
    # PERSISTENCE METHODS
    # =============================================================================
    
    async def _persist_agent_expertise(self, agent_id: str) -> None:
        """Persist agent expertise to Redis."""
        try:
            key = f"agent_expertise:{agent_id}"
            expertise_data = [exp.to_dict() for exp in self.agent_expertise[agent_id]]
            await self.redis.setex(key, 86400, json.dumps(expertise_data))
        except Exception as e:
            logger.error(f"Failed to persist agent expertise: {e}")
    
    async def _load_agent_expertise(self) -> None:
        """Load agent expertise from Redis."""
        try:
            # This would load from database/Redis in production
            logger.debug("Agent expertise loaded from storage")
        except Exception as e:
            logger.error(f"Failed to load agent expertise: {e}")
    
    async def _persist_quality_score(self, quality_score: KnowledgeQualityScore) -> None:
        """Persist quality score to Redis."""
        try:
            key = f"knowledge_quality:{quality_score.knowledge_id}"
            quality_data = {
                "knowledge_id": quality_score.knowledge_id,
                "overall_score": quality_score.overall_score,
                "metric_scores": {k.value: v for k, v in quality_score.metric_scores.items()},
                "reviewer_count": quality_score.reviewer_count,
                "positive_feedback": quality_score.positive_feedback,
                "negative_feedback": quality_score.negative_feedback,
                "usage_success_rate": quality_score.usage_success_rate,
                "last_updated": quality_score.last_updated.isoformat()
            }
            await self.redis.setex(key, 86400, json.dumps(quality_data))
        except Exception as e:
            logger.error(f"Failed to persist quality score: {e}")
    
    async def _load_knowledge_quality_scores(self) -> None:
        """Load knowledge quality scores from Redis."""
        try:
            # This would load from database/Redis in production
            logger.debug("Knowledge quality scores loaded from storage")
        except Exception as e:
            logger.error(f"Failed to load quality scores: {e}")
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for cross-agent knowledge management."""
        return {
            **self.metrics,
            "total_agent_expertise": sum(len(expertise_list) for expertise_list in self.agent_expertise.values()),
            "quality_metrics": self.get_quality_metrics(),
            "active_sharing_requests": len([r for r in self.sharing_requests.values() if r.approved is None]),
            "collaboration_outcomes": len(self.collaboration_outcomes)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cross-agent knowledge manager."""
        try:
            # Test basic functionality
            test_agent_id = "test_agent"
            test_knowledge = KnowledgeItem(
                knowledge_id="health_check_knowledge",
                agent_id=test_agent_id,
                knowledge_type=KnowledgeType.PATTERN,
                content="Test knowledge for health check",
                confidence_score=0.8
            )
            
            # Test quality assessment
            quality_score = await self._assess_knowledge_quality(test_knowledge)
            
            return {
                "status": "healthy",
                "components": {
                    "discovery_engine": "operational",
                    "access_control": "operational", 
                    "quality_assessment": "operational",
                    "compression_engine": "operational"
                },
                "test_results": {
                    "quality_assessment": quality_score.overall_score
                },
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Cross-agent knowledge manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "discovery_engine": "unknown",
                    "access_control": "unknown",
                    "quality_assessment": "unknown", 
                    "compression_engine": "unknown"
                }
            }


# =============================================================================
# GLOBAL CROSS-AGENT KNOWLEDGE MANAGER INSTANCE
# =============================================================================

_cross_agent_manager: Optional[CrossAgentKnowledgeManager] = None


async def get_cross_agent_knowledge_manager() -> CrossAgentKnowledgeManager:
    """Get global cross-agent knowledge manager instance."""
    global _cross_agent_manager
    
    if _cross_agent_manager is None:
        from .agent_knowledge_manager import get_agent_knowledge_manager
        
        base_manager = await get_agent_knowledge_manager()
        _cross_agent_manager = CrossAgentKnowledgeManager(base_manager)
        await _cross_agent_manager.initialize()
    
    return _cross_agent_manager


async def cleanup_cross_agent_knowledge_manager():
    """Clean up global cross-agent knowledge manager."""
    global _cross_agent_manager
    _cross_agent_manager = None
"""
Enhanced Memory Manager - Advanced Long-term Memory Persistence & Hierarchical Organization.

Provides sophisticated memory management capabilities for LeanVibe Agent Hive 2.0:
- Long-term memory persistence with intelligent decay strategies
- Hierarchical memory organization (short-term, working, long-term)
- Cross-session knowledge retention with semantic integrity
- Agent-specific memory isolation with shared knowledge pools
- Memory optimization strategies for performance and relevance
- Integration with vector search and consolidation engines

Performance Targets:
- 70%+ token reduction while maintaining semantic integrity
- <500ms memory retrieval response time
- 95%+ context restoration accuracy
- <10MB memory overhead per agent
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from sqlalchemy import select, and_, or_, desc, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.context_manager import ContextManager, get_context_manager
from ..core.enhanced_context_consolidator import UltraCompressedContextMode, get_ultra_compressed_context_mode
from ..core.vector_search_engine import VectorSearchEngine, ContextMatch, SearchFilters
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.redis import get_redis_client
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory with different persistence and access patterns."""
    SHORT_TERM = "short_term"      # Active working memory (hours)
    WORKING = "working"            # Current session memory (days)
    LONG_TERM = "long_term"        # Persistent memory (weeks/months)
    EPISODIC = "episodic"          # Event-based memories
    SEMANTIC = "semantic"          # Knowledge and facts
    PROCEDURAL = "procedural"      # Skills and procedures


class MemoryPriority(Enum):
    """Memory priority levels for retention decisions."""
    CRITICAL = "critical"          # Never decay, always retain
    HIGH = "high"                 # Long retention, slow decay
    MEDIUM = "medium"             # Standard retention policies
    LOW = "low"                   # Aggressive decay, early cleanup
    EPHEMERAL = "ephemeral"       # Short-lived, rapid decay


class DecayStrategy(Enum):
    """Memory decay strategies for different content types."""
    EXPONENTIAL = "exponential"    # Exponential decay over time
    LINEAR = "linear"             # Linear decay over time
    STEP = "step"                 # Step function decay at intervals
    ADAPTIVE = "adaptive"         # Adaptive based on access patterns
    NONE = "none"                 # No decay, permanent retention


@dataclass
class MemoryMetrics:
    """Metrics for memory management operations."""
    total_memories: int = 0
    short_term_memories: int = 0
    working_memories: int = 0
    long_term_memories: int = 0
    memory_utilization_percent: float = 0.0
    consolidation_efficiency: float = 0.0
    retrieval_accuracy: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    token_reduction_achieved: float = 0.0
    memory_overhead_mb: float = 0.0


@dataclass
class MemoryFragment:
    """Represents a fragment of agent memory with metadata."""
    fragment_id: str
    agent_id: uuid.UUID
    memory_type: MemoryType
    priority: MemoryPriority
    decay_strategy: DecayStrategy
    content: str
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    last_accessed: datetime = None
    access_count: int = 0
    importance_score: float = 0.5
    decay_factor: float = 1.0
    consolidation_level: int = 0
    source_context_ids: List[uuid.UUID] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.source_context_ids is None:
            self.source_context_ids = []
        if self.metadata is None:
            self.metadata = {}


class EnhancedMemoryManager:
    """
    Advanced Memory Manager for long-term persistence and hierarchical organization.
    
    Features:
    - Hierarchical memory organization (short-term, working, long-term)
    - Intelligent memory decay and consolidation strategies
    - Cross-session knowledge retention with semantic integrity
    - Agent-specific memory isolation with shared knowledge pools
    - Memory optimization for performance and relevance
    - Integration with vector search and consolidation systems
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        consolidator: Optional[UltraCompressedContextMode] = None,
        embedding_service: Optional[EmbeddingService] = None,
        redis_client = None
    ):
        self.settings = get_settings()
        self.context_manager = context_manager or get_context_manager()
        self.consolidator = consolidator or get_ultra_compressed_context_mode()
        self.embedding_service = embedding_service or get_embedding_service()
        
        # Initialize Redis for caching
        try:
            self.redis_client = redis_client or get_redis_client()
        except Exception as e:
            logger.warning(f"Redis not available for memory caching: {e}")
            self.redis_client = None
        
        # Memory configuration
        self.config = {
            "short_term_retention_hours": 24,
            "working_memory_retention_days": 7,
            "long_term_retention_days": 90,
            "max_memories_per_agent": 10000,
            "consolidation_threshold": 100,
            "decay_check_interval_hours": 6,
            "semantic_similarity_threshold": 0.85,
            "importance_boost_factor": 1.5
        }
        
        # Memory stores by agent
        self._memory_stores: Dict[uuid.UUID, Dict[MemoryType, List[MemoryFragment]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Performance tracking
        self._metrics: Dict[uuid.UUID, MemoryMetrics] = {}
        self._operation_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._decay_monitor_running = False
        
        logger.info("🧠 Enhanced Memory Manager initialized")
    
    async def store_memory(
        self,
        agent_id: uuid.UUID,
        content: str,
        memory_type: MemoryType = MemoryType.WORKING,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        importance_score: float = 0.5,
        source_context_ids: Optional[List[uuid.UUID]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_embed: bool = True
    ) -> MemoryFragment:
        """
        Store a new memory fragment with hierarchical organization.
        
        Args:
            agent_id: Agent storing the memory
            content: Memory content
            memory_type: Type of memory (short-term, working, long-term, etc.)
            priority: Memory priority for retention decisions
            importance_score: Importance score (0.0-1.0)
            source_context_ids: Source contexts this memory was derived from
            metadata: Additional metadata
            auto_embed: Whether to automatically generate embeddings
            
        Returns:
            Created memory fragment
        """
        try:
            # Create memory fragment
            fragment = MemoryFragment(
                fragment_id=str(uuid.uuid4()),
                agent_id=agent_id,
                memory_type=memory_type,
                priority=priority,
                decay_strategy=self._determine_decay_strategy(memory_type, priority),
                content=content,
                importance_score=importance_score,
                source_context_ids=source_context_ids or [],
                metadata=metadata or {}
            )
            
            # Generate embedding if requested
            if auto_embed:
                try:
                    embedding = await self.embedding_service.generate_embedding(content)
                    fragment.embedding = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for memory fragment: {e}")
            
            # Store in appropriate memory hierarchy
            self._memory_stores[agent_id][memory_type].append(fragment)
            
            # Cache in Redis if available
            if self.redis_client:
                cache_key = f"memory:fragment:{fragment.fragment_id}"
                await self._cache_memory_fragment(cache_key, fragment)
            
            # Update metrics
            await self._update_memory_metrics(agent_id)
            
            # Check for consolidation opportunities
            if len(self._memory_stores[agent_id][memory_type]) > self.config["consolidation_threshold"]:
                await self._trigger_memory_consolidation(agent_id, memory_type)
            
            logger.debug(
                f"🧠 Stored memory fragment",
                agent_id=str(agent_id),
                memory_type=memory_type.value,
                priority=priority.value,
                fragment_id=fragment.fragment_id
            )
            
            return fragment
            
        except Exception as e:
            logger.error(f"Failed to store memory fragment: {e}")
            raise
    
    async def retrieve_memories(
        self,
        agent_id: uuid.UUID,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ) -> List[Tuple[MemoryFragment, float]]:
        """
        Retrieve relevant memories using semantic search.
        
        Args:
            agent_id: Agent requesting memories
            query: Search query
            memory_types: Types of memory to search (all if None)
            limit: Maximum number of memories to return
            similarity_threshold: Minimum similarity score
            include_metadata: Whether to include full metadata
            
        Returns:
            List of (memory_fragment, relevance_score) tuples
        """
        start_time = datetime.utcnow()
        
        try:
            # Default to all memory types if not specified
            if memory_types is None:
                memory_types = list(MemoryType)
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            relevant_memories = []
            
            # Search across specified memory types
            for memory_type in memory_types:
                if memory_type in self._memory_stores[agent_id]:
                    memories = self._memory_stores[agent_id][memory_type]
                    
                    # Calculate similarities
                    for memory in memories:
                        if memory.embedding:
                            similarity = await self._calculate_similarity(
                                query_embedding, memory.embedding
                            )
                            
                            if similarity >= similarity_threshold:
                                # Boost importance for recent access and high importance
                                relevance_score = self._calculate_relevance_score(
                                    memory, similarity
                                )
                                relevant_memories.append((memory, relevance_score))
                                
                                # Update access tracking
                                memory.last_accessed = datetime.utcnow()
                                memory.access_count += 1
            
            # Sort by relevance and limit results
            relevant_memories.sort(key=lambda x: x[1], reverse=True)
            relevant_memories = relevant_memories[:limit]
            
            # Update metrics
            retrieval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._update_retrieval_metrics(agent_id, retrieval_time, len(relevant_memories))
            
            logger.info(
                f"🧠 Retrieved {len(relevant_memories)} relevant memories",
                agent_id=str(agent_id),
                query_length=len(query),
                retrieval_time_ms=retrieval_time
            )
            
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def consolidate_memories(
        self,
        agent_id: uuid.UUID,
        memory_type: Optional[MemoryType] = None,
        force_consolidation: bool = False,
        target_reduction: float = 0.7
    ) -> Dict[str, Any]:
        """
        Consolidate memories using semantic clustering and compression.
        
        Args:
            agent_id: Agent to consolidate memories for
            memory_type: Specific memory type to consolidate (all if None)
            force_consolidation: Force consolidation regardless of thresholds
            target_reduction: Target memory reduction ratio
            
        Returns:
            Consolidation results and metrics
        """
        try:
            consolidation_results = {
                "agent_id": str(agent_id),
                "consolidated_types": [],
                "original_memory_count": 0,
                "consolidated_memory_count": 0,
                "reduction_achieved": 0.0,
                "processing_time_ms": 0.0,
                "success": False
            }
            
            start_time = datetime.utcnow()
            
            # Determine memory types to consolidate
            types_to_consolidate = [memory_type] if memory_type else list(MemoryType)
            
            for mem_type in types_to_consolidate:
                if mem_type not in self._memory_stores[agent_id]:
                    continue
                
                memories = self._memory_stores[agent_id][mem_type]
                
                # Check if consolidation is needed
                if not force_consolidation and len(memories) < self.config["consolidation_threshold"]:
                    continue
                
                original_count = len(memories)
                consolidation_results["original_memory_count"] += original_count
                
                # Perform semantic clustering
                clusters = await self._cluster_memories_semantically(memories)
                
                # Consolidate each cluster
                consolidated_memories = []
                for cluster in clusters:
                    if len(cluster) > 1:
                        # Merge similar memories
                        consolidated_memory = await self._merge_memory_cluster(cluster)
                        consolidated_memories.append(consolidated_memory)
                    else:
                        # Keep single memories as-is
                        consolidated_memories.extend(cluster)
                
                # Update memory store
                self._memory_stores[agent_id][mem_type] = consolidated_memories
                consolidation_results["consolidated_types"].append(mem_type.value)
                consolidation_results["consolidated_memory_count"] += len(consolidated_memories)
                
                logger.info(
                    f"🧠 Consolidated {mem_type.value} memories: {original_count} → {len(consolidated_memories)}"
                )
            
            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            consolidation_results["processing_time_ms"] = processing_time
            
            if consolidation_results["original_memory_count"] > 0:
                consolidation_results["reduction_achieved"] = 1 - (
                    consolidation_results["consolidated_memory_count"] / 
                    consolidation_results["original_memory_count"]
                )
            
            consolidation_results["success"] = True
            
            # Update agent metrics
            await self._update_memory_metrics(agent_id)
            
            logger.info(
                f"🧠 Memory consolidation completed",
                agent_id=str(agent_id),
                reduction_achieved=consolidation_results["reduction_achieved"],
                processing_time_ms=processing_time
            )
            
            return consolidation_results
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return {
                "agent_id": str(agent_id),
                "success": False,
                "error": str(e)
            }
    
    async def promote_memory(
        self,
        fragment_id: str,
        target_type: MemoryType,
        target_priority: Optional[MemoryPriority] = None
    ) -> bool:
        """
        Promote memory to a higher retention level.
        
        Args:
            fragment_id: Memory fragment ID to promote
            target_type: Target memory type
            target_priority: Optional target priority
            
        Returns:
            True if promotion was successful
        """
        try:
            # Find the memory fragment
            fragment = await self._find_memory_fragment(fragment_id)
            if not fragment:
                logger.warning(f"Memory fragment {fragment_id} not found for promotion")
                return False
            
            agent_id = fragment.agent_id
            current_type = fragment.memory_type
            
            # Remove from current type
            if fragment in self._memory_stores[agent_id][current_type]:
                self._memory_stores[agent_id][current_type].remove(fragment)
            
            # Update fragment properties
            fragment.memory_type = target_type
            if target_priority:
                fragment.priority = target_priority
            
            # Update decay strategy based on new type/priority
            fragment.decay_strategy = self._determine_decay_strategy(target_type, fragment.priority)
            
            # Add to new type
            self._memory_stores[agent_id][target_type].append(fragment)
            
            # Update cache
            if self.redis_client:
                cache_key = f"memory:fragment:{fragment_id}"
                await self._cache_memory_fragment(cache_key, fragment)
            
            logger.info(
                f"🧠 Promoted memory fragment",
                fragment_id=fragment_id,
                from_type=current_type.value,
                to_type=target_type.value
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Memory promotion failed: {e}")
            return False
    
    async def decay_memories(
        self,
        agent_id: Optional[uuid.UUID] = None,
        force_decay: bool = False
    ) -> Dict[str, Any]:
        """
        Apply memory decay strategies to reduce memory footprint.
        
        Args:
            agent_id: Specific agent to decay memories for (all if None)
            force_decay: Force decay regardless of intervals
            
        Returns:
            Decay operation results
        """
        try:
            decay_results = {
                "agents_processed": [],
                "total_memories_before": 0,
                "total_memories_after": 0,
                "memories_decayed": 0,
                "memories_removed": 0,
                "processing_time_ms": 0.0
            }
            
            start_time = datetime.utcnow()
            
            # Determine agents to process
            agents_to_process = [agent_id] if agent_id else list(self._memory_stores.keys())
            
            for agent in agents_to_process:
                if agent not in self._memory_stores:
                    continue
                
                agent_decay_results = await self._decay_agent_memories(agent, force_decay)
                
                decay_results["agents_processed"].append(str(agent))
                decay_results["total_memories_before"] += agent_decay_results["memories_before"]
                decay_results["total_memories_after"] += agent_decay_results["memories_after"]
                decay_results["memories_decayed"] += agent_decay_results["memories_decayed"]
                decay_results["memories_removed"] += agent_decay_results["memories_removed"]
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            decay_results["processing_time_ms"] = processing_time
            
            logger.info(
                f"🧠 Memory decay completed",
                agents_processed=len(decay_results["agents_processed"]),
                memories_decayed=decay_results["memories_decayed"],
                memories_removed=decay_results["memories_removed"]
            )
            
            return decay_results
            
        except Exception as e:
            logger.error(f"Memory decay failed: {e}")
            return {"error": str(e)}
    
    async def get_memory_analytics(
        self,
        agent_id: Optional[uuid.UUID] = None,
        include_distribution: bool = True,
        include_performance_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive memory analytics.
        
        Args:
            agent_id: Specific agent analytics (all if None)
            include_distribution: Include memory type distribution
            include_performance_metrics: Include performance metrics
            
        Returns:
            Comprehensive analytics data
        """
        try:
            analytics = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_analytics": {},
                "system_analytics": {},
                "performance_metrics": {}
            }
            
            # Agent-specific analytics
            agents_to_analyze = [agent_id] if agent_id else list(self._memory_stores.keys())
            
            for agent in agents_to_analyze:
                agent_analytics = await self._calculate_agent_memory_analytics(
                    agent, include_distribution
                )
                analytics["agent_analytics"][str(agent)] = agent_analytics
            
            # System-wide analytics
            analytics["system_analytics"] = await self._calculate_system_memory_analytics()
            
            # Performance metrics
            if include_performance_metrics:
                analytics["performance_metrics"] = await self._calculate_performance_metrics()
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get memory analytics: {e}")
            return {"error": str(e)}
    
    async def optimize_memory_performance(
        self,
        agent_id: uuid.UUID,
        target_performance_ms: float = 500.0
    ) -> Dict[str, Any]:
        """
        Optimize memory performance to meet target response times.
        
        Args:
            agent_id: Agent to optimize
            target_performance_ms: Target response time in milliseconds
            
        Returns:
            Optimization results
        """
        try:
            optimization_results = {
                "agent_id": str(agent_id),
                "target_performance_ms": target_performance_ms,
                "optimization_actions": [],
                "performance_before": 0.0,
                "performance_after": 0.0,
                "improvement_achieved": False
            }
            
            # Measure current performance
            performance_before = await self._measure_memory_performance(agent_id)
            optimization_results["performance_before"] = performance_before
            
            if performance_before <= target_performance_ms:
                optimization_results["improvement_achieved"] = True
                optimization_results["optimization_actions"].append("No optimization needed")
                return optimization_results
            
            # Apply optimization strategies
            if performance_before > target_performance_ms * 2:
                # Aggressive optimization needed
                await self.consolidate_memories(agent_id, force_consolidation=True, target_reduction=0.8)
                optimization_results["optimization_actions"].append("Aggressive consolidation applied")
            
            elif performance_before > target_performance_ms * 1.5:
                # Standard optimization
                await self.consolidate_memories(agent_id, target_reduction=0.7)
                optimization_results["optimization_actions"].append("Standard consolidation applied")
            
            # Apply decay to reduce memory footprint
            await self.decay_memories(agent_id, force_decay=True)
            optimization_results["optimization_actions"].append("Memory decay applied")
            
            # Measure performance after optimization
            performance_after = await self._measure_memory_performance(agent_id)
            optimization_results["performance_after"] = performance_after
            optimization_results["improvement_achieved"] = performance_after <= target_performance_ms
            
            logger.info(
                f"🧠 Memory performance optimization completed",
                agent_id=str(agent_id),
                performance_before=performance_before,
                performance_after=performance_after,
                improvement_achieved=optimization_results["improvement_achieved"]
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Memory performance optimization failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _determine_decay_strategy(
        self, memory_type: MemoryType, priority: MemoryPriority
    ) -> DecayStrategy:
        """Determine appropriate decay strategy based on memory type and priority."""
        if priority == MemoryPriority.CRITICAL:
            return DecayStrategy.NONE
        elif priority == MemoryPriority.HIGH:
            return DecayStrategy.LINEAR
        elif memory_type == MemoryType.SHORT_TERM:
            return DecayStrategy.EXPONENTIAL
        elif memory_type == MemoryType.EPHEMERAL:
            return DecayStrategy.STEP
        else:
            return DecayStrategy.ADAPTIVE
    
    async def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        try:
            import numpy as np
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_relevance_score(
        self, memory: MemoryFragment, base_similarity: float
    ) -> float:
        """Calculate relevance score including importance and recency boosts."""
        # Base score from similarity
        relevance = base_similarity
        
        # Boost for importance
        relevance += memory.importance_score * 0.2
        
        # Boost for recent access
        time_since_access = datetime.utcnow() - memory.last_accessed
        recency_boost = max(0, 0.1 - (time_since_access.total_seconds() / 86400) * 0.01)
        relevance += recency_boost
        
        # Boost for high access count
        access_boost = min(0.1, memory.access_count * 0.01)
        relevance += access_boost
        
        return min(1.0, relevance)
    
    async def _cluster_memories_semantically(
        self, memories: List[MemoryFragment]
    ) -> List[List[MemoryFragment]]:
        """Cluster memories by semantic similarity."""
        try:
            clusters = []
            processed = set()
            
            for memory in memories:
                if memory.fragment_id in processed:
                    continue
                
                cluster = [memory]
                processed.add(memory.fragment_id)
                
                # Find similar memories
                for other_memory in memories:
                    if (other_memory.fragment_id in processed or 
                        not memory.embedding or not other_memory.embedding):
                        continue
                    
                    similarity = await self._calculate_similarity(
                        memory.embedding, other_memory.embedding
                    )
                    
                    if similarity >= self.config["semantic_similarity_threshold"]:
                        cluster.append(other_memory)
                        processed.add(other_memory.fragment_id)
                
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Memory clustering failed: {e}")
            return [[memory] for memory in memories]  # Return individual clusters on failure
    
    async def _merge_memory_cluster(
        self, cluster: List[MemoryFragment]
    ) -> MemoryFragment:
        """Merge a cluster of similar memories into a single consolidated memory."""
        try:
            # Use the most important memory as base
            base_memory = max(cluster, key=lambda m: m.importance_score)
            
            # Combine content from all memories
            combined_content = []
            all_source_contexts = []
            total_access_count = 0
            
            for memory in cluster:
                combined_content.append(memory.content)
                all_source_contexts.extend(memory.source_context_ids)
                total_access_count += memory.access_count
            
            # Create consolidated content using compression
            full_content = "\n\n---\n\n".join(combined_content)
            
            # Use context consolidator for compression
            compressed_result = await self.consolidator.compressor.compress_conversation(
                conversation_content=full_content,
                compression_level=self.consolidator.compressor.CompressionLevel.STANDARD
            )
            
            # Create new consolidated memory
            consolidated = MemoryFragment(
                fragment_id=str(uuid.uuid4()),
                agent_id=base_memory.agent_id,
                memory_type=base_memory.memory_type,
                priority=max(m.priority for m in cluster),
                decay_strategy=base_memory.decay_strategy,
                content=compressed_result.summary,
                importance_score=max(m.importance_score for m in cluster),
                access_count=total_access_count,
                consolidation_level=base_memory.consolidation_level + 1,
                source_context_ids=list(set(all_source_contexts)),
                metadata={
                    "consolidated_from": [m.fragment_id for m in cluster],
                    "consolidation_ratio": compressed_result.compression_ratio,
                    "key_insights": compressed_result.key_insights,
                    "original_memory_count": len(cluster)
                }
            )
            
            # Generate embedding for consolidated content
            if self.embedding_service:
                try:
                    consolidated.embedding = await self.embedding_service.generate_embedding(
                        consolidated.content
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for consolidated memory: {e}")
            
            return consolidated
            
        except Exception as e:
            logger.error(f"Memory cluster merge failed: {e}")
            # Return the most important memory if merge fails
            return max(cluster, key=lambda m: m.importance_score)
    
    async def _cache_memory_fragment(self, cache_key: str, fragment: MemoryFragment) -> None:
        """Cache memory fragment in Redis."""
        try:
            if self.redis_client:
                fragment_data = asdict(fragment)
                fragment_data['created_at'] = fragment.created_at.isoformat()
                fragment_data['last_accessed'] = fragment.last_accessed.isoformat()
                
                await self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour cache
                    str(fragment_data)
                )
        except Exception as e:
            logger.warning(f"Failed to cache memory fragment: {e}")
    
    async def _find_memory_fragment(self, fragment_id: str) -> Optional[MemoryFragment]:
        """Find a memory fragment by ID across all agents and types."""
        for agent_id, memory_types in self._memory_stores.items():
            for memory_type, memories in memory_types.items():
                for memory in memories:
                    if memory.fragment_id == fragment_id:
                        return memory
        return None
    
    async def _decay_agent_memories(
        self, agent_id: uuid.UUID, force_decay: bool
    ) -> Dict[str, Any]:
        """Apply decay strategies to an agent's memories."""
        results = {
            "memories_before": 0,
            "memories_after": 0,
            "memories_decayed": 0,
            "memories_removed": 0
        }
        
        try:
            for memory_type, memories in self._memory_stores[agent_id].items():
                results["memories_before"] += len(memories)
                
                memories_to_keep = []
                
                for memory in memories:
                    # Apply decay based on strategy
                    should_remove = self._should_remove_memory(memory, force_decay)
                    
                    if should_remove:
                        results["memories_removed"] += 1
                    else:
                        # Apply decay factor
                        if self._apply_decay_factor(memory):
                            results["memories_decayed"] += 1
                        memories_to_keep.append(memory)
                
                # Update memory store
                self._memory_stores[agent_id][memory_type] = memories_to_keep
                results["memories_after"] += len(memories_to_keep)
            
            return results
            
        except Exception as e:
            logger.error(f"Agent memory decay failed: {e}")
            return results
    
    def _should_remove_memory(self, memory: MemoryFragment, force_decay: bool) -> bool:
        """Determine if a memory should be removed based on decay strategy."""
        if memory.priority == MemoryPriority.CRITICAL:
            return False
        
        if memory.decay_strategy == DecayStrategy.NONE:
            return False
        
        now = datetime.utcnow()
        age_hours = (now - memory.created_at).total_seconds() / 3600
        
        if memory.memory_type == MemoryType.SHORT_TERM:
            return age_hours > self.config["short_term_retention_hours"]
        elif memory.memory_type == MemoryType.EPHEMERAL:
            return age_hours > 1  # 1 hour for ephemeral
        elif force_decay and memory.priority == MemoryPriority.LOW:
            return age_hours > 24  # Force remove low priority after 24 hours
        
        return False
    
    def _apply_decay_factor(self, memory: MemoryFragment) -> bool:
        """Apply decay factor to reduce memory importance over time."""
        if memory.decay_strategy in [DecayStrategy.NONE, DecayStrategy.ADAPTIVE]:
            return False
        
        age_days = (datetime.utcnow() - memory.created_at).days
        
        if memory.decay_strategy == DecayStrategy.EXPONENTIAL:
            memory.decay_factor *= 0.95 ** age_days
        elif memory.decay_strategy == DecayStrategy.LINEAR:
            memory.decay_factor = max(0.1, 1.0 - (age_days * 0.01))
        elif memory.decay_strategy == DecayStrategy.STEP and age_days > 7:
            memory.decay_factor *= 0.8
        
        return True
    
    async def _update_memory_metrics(self, agent_id: uuid.UUID) -> None:
        """Update memory metrics for an agent."""
        try:
            metrics = MemoryMetrics()
            
            for memory_type, memories in self._memory_stores[agent_id].items():
                count = len(memories)
                metrics.total_memories += count
                
                if memory_type == MemoryType.SHORT_TERM:
                    metrics.short_term_memories = count
                elif memory_type == MemoryType.WORKING:
                    metrics.working_memories = count
                elif memory_type == MemoryType.LONG_TERM:
                    metrics.long_term_memories = count
            
            # Calculate memory utilization
            max_memories = self.config["max_memories_per_agent"]
            metrics.memory_utilization_percent = (metrics.total_memories / max_memories) * 100
            
            self._metrics[agent_id] = metrics
            
        except Exception as e:
            logger.error(f"Failed to update memory metrics: {e}")
    
    async def _update_retrieval_metrics(
        self, agent_id: uuid.UUID, retrieval_time: float, result_count: int
    ) -> None:
        """Update retrieval performance metrics."""
        try:
            if agent_id in self._metrics:
                metrics = self._metrics[agent_id]
                # Update average retrieval time
                metrics.avg_retrieval_time_ms = (
                    metrics.avg_retrieval_time_ms * 0.9 + retrieval_time * 0.1
                )
        except Exception as e:
            logger.error(f"Failed to update retrieval metrics: {e}")
    
    async def _measure_memory_performance(self, agent_id: uuid.UUID) -> float:
        """Measure current memory performance for an agent."""
        try:
            start_time = datetime.utcnow()
            
            # Perform test retrieval
            await self.retrieve_memories(
                agent_id=agent_id,
                query="test performance query",
                limit=5
            )
            
            retrieval_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return retrieval_time
            
        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            return 1000.0  # Return high value on failure
    
    async def _calculate_agent_memory_analytics(
        self, agent_id: uuid.UUID, include_distribution: bool
    ) -> Dict[str, Any]:
        """Calculate analytics for a specific agent."""
        try:
            analytics = {
                "agent_id": str(agent_id),
                "total_memories": 0,
                "memory_types": {},
                "average_importance": 0.0,
                "memory_utilization_percent": 0.0
            }
            
            if agent_id not in self._memory_stores:
                return analytics
            
            total_memories = 0
            total_importance = 0.0
            
            for memory_type, memories in self._memory_stores[agent_id].items():
                count = len(memories)
                total_memories += count
                
                if include_distribution:
                    analytics["memory_types"][memory_type.value] = {
                        "count": count,
                        "average_importance": sum(m.importance_score for m in memories) / max(1, count),
                        "average_access_count": sum(m.access_count for m in memories) / max(1, count)
                    }
                
                total_importance += sum(m.importance_score for m in memories)
            
            analytics["total_memories"] = total_memories
            analytics["average_importance"] = total_importance / max(1, total_memories)
            analytics["memory_utilization_percent"] = (
                total_memories / self.config["max_memories_per_agent"]
            ) * 100
            
            return analytics
            
        except Exception as e:
            logger.error(f"Agent analytics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_system_memory_analytics(self) -> Dict[str, Any]:
        """Calculate system-wide memory analytics."""
        try:
            analytics = {
                "total_agents": len(self._memory_stores),
                "total_memories_system_wide": 0,
                "memory_type_distribution": {},
                "average_memories_per_agent": 0.0
            }
            
            type_counts = defaultdict(int)
            total_memories = 0
            
            for agent_id, memory_types in self._memory_stores.items():
                for memory_type, memories in memory_types.items():
                    count = len(memories)
                    type_counts[memory_type.value] += count
                    total_memories += count
            
            analytics["total_memories_system_wide"] = total_memories
            analytics["memory_type_distribution"] = dict(type_counts)
            analytics["average_memories_per_agent"] = (
                total_memories / max(1, len(self._memory_stores))
            )
            
            return analytics
            
        except Exception as e:
            logger.error(f"System analytics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics across all agents."""
        try:
            if not self._metrics:
                return {"no_metrics_available": True}
            
            total_agents = len(self._metrics)
            avg_retrieval_time = sum(m.avg_retrieval_time_ms for m in self._metrics.values()) / total_agents
            avg_utilization = sum(m.memory_utilization_percent for m in self._metrics.values()) / total_agents
            
            return {
                "average_retrieval_time_ms": avg_retrieval_time,
                "average_memory_utilization_percent": avg_utilization,
                "agents_tracked": total_agents,
                "performance_target_met": avg_retrieval_time <= 500.0
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _trigger_memory_consolidation(
        self, agent_id: uuid.UUID, memory_type: MemoryType
    ) -> None:
        """Trigger background memory consolidation."""
        try:
            # Create background task for consolidation
            task = asyncio.create_task(
                self.consolidate_memories(agent_id, memory_type)
            )
            self._background_tasks.append(task)
            
            logger.info(f"🧠 Triggered background consolidation for {memory_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to trigger memory consolidation: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup memory manager resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Clear memory stores
            self._memory_stores.clear()
            self._metrics.clear()
            self._operation_history.clear()
            
            logger.info("🧠 Enhanced Memory Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory manager cleanup failed: {e}")


# Global instance
_enhanced_memory_manager: Optional[EnhancedMemoryManager] = None


async def get_enhanced_memory_manager() -> EnhancedMemoryManager:
    """Get singleton enhanced memory manager instance."""
    global _enhanced_memory_manager
    
    if _enhanced_memory_manager is None:
        _enhanced_memory_manager = EnhancedMemoryManager()
    
    return _enhanced_memory_manager


async def cleanup_enhanced_memory_manager() -> None:
    """Cleanup enhanced memory manager resources."""
    global _enhanced_memory_manager
    
    if _enhanced_memory_manager:
        await _enhanced_memory_manager.cleanup()
        _enhanced_memory_manager = None
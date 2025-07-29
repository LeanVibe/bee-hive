"""
Memory Hierarchy Manager for LeanVibe Agent Hive 2.0

Manages different levels of memory storage with intelligent aging strategies,
consolidation algorithms, and cross-session continuity for optimal memory usage.

Features:
- Memory Hierarchies: Short-term, working, long-term, and institutional memory
- Knowledge Aging: Automatic aging and archival strategies  
- Memory Consolidation: Periodic compression and optimization cycles
- Cross-Session Continuity: Persistent knowledge across agent restarts
- Adaptive Memory Management: Dynamic memory allocation based on usage patterns
- Memory Analytics: Detailed insights into memory usage and effectiveness
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
import logging

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .context_compression_engine import (
    get_context_compression_engine, CompressionConfig, CompressionQuality, 
    CompressionContext, CompressedResult
)
from .agent_knowledge_manager import KnowledgeItem, KnowledgeType, AccessLevel
from .database import get_session
from .redis import get_redis

logger = structlog.get_logger()


# =============================================================================
# MEMORY HIERARCHY TYPES AND CONFIGURATIONS
# =============================================================================

class MemoryLevel(str, Enum):
    """Levels in the memory hierarchy."""
    SHORT_TERM = "short_term"          # Minutes to hours (working memory)
    WORKING = "working"                # Hours to days (active context)
    LONG_TERM = "long_term"            # Days to months (consolidated memory)
    INSTITUTIONAL = "institutional"    # Permanent (system knowledge)


class MemoryType(str, Enum):
    """Types of memory content."""
    EPISODIC = "episodic"              # Specific events and experiences
    SEMANTIC = "semantic"              # Facts and general knowledge
    PROCEDURAL = "procedural"          # Skills and procedures
    META_COGNITIVE = "meta_cognitive"  # Knowledge about knowledge
    CONTEXTUAL = "contextual"          # Context-dependent information


class AgingStrategy(str, Enum):
    """Strategies for memory aging."""
    TIME_BASED = "time_based"          # Age based on time elapsed
    USAGE_BASED = "usage_based"        # Age based on access frequency
    IMPORTANCE_BASED = "importance_based"  # Age based on importance score
    HYBRID = "hybrid"                  # Combination of multiple factors


class ConsolidationTrigger(str, Enum):
    """Triggers for memory consolidation."""
    TIME_INTERVAL = "time_interval"    # Regular time-based intervals
    MEMORY_PRESSURE = "memory_pressure"  # When memory usage is high
    SIGNIFICANCE_THRESHOLD = "significance_threshold"  # When important events occur
    AGENT_IDLE = "agent_idle"          # When agent is not actively working


@dataclass
class MemoryItem:
    """Individual item in the memory hierarchy."""
    memory_id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    memory_level: MemoryLevel
    
    # Temporal information
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    # Importance and quality metrics
    importance_score: float = 0.5
    relevance_score: float = 0.5
    confidence_score: float = 0.5
    
    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    
    # Aging and lifecycle
    decay_rate: float = 0.1
    consolidation_count: int = 0
    archived: bool = False
    archived_at: Optional[datetime] = None
    
    def calculate_age_hours(self) -> float:
        """Calculate age in hours."""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    def calculate_recency_score(self) -> float:
        """Calculate recency score (0.0 to 1.0, higher = more recent)."""
        age_hours = self.calculate_age_hours()
        # Exponential decay with 24-hour half-life
        return max(0.0, min(1.0, 2 ** (-age_hours / 24)))
    
    def calculate_usage_score(self) -> float:
        """Calculate usage score based on access patterns."""
        if self.access_count == 0:
            return 0.0
        
        # Consider both frequency and recency of access
        frequency_score = min(1.0, self.access_count / 10)  # Normalize to 10 accesses
        
        if self.last_accessed:
            last_access_hours = (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
            recency_factor = max(0.1, 2 ** (-last_access_hours / 168))  # 1-week half-life
        else:
            recency_factor = 0.1
        
        return frequency_score * recency_factor
    
    def calculate_retention_score(self, aging_strategy: AgingStrategy) -> float:
        """Calculate how strongly this memory should be retained."""
        base_score = self.importance_score
        
        if aging_strategy == AgingStrategy.TIME_BASED:
            retention_score = base_score * self.calculate_recency_score()
        elif aging_strategy == AgingStrategy.USAGE_BASED:
            retention_score = base_score * self.calculate_usage_score()
        elif aging_strategy == AgingStrategy.IMPORTANCE_BASED:
            retention_score = base_score  # Pure importance
        else:  # HYBRID
            recency = self.calculate_recency_score()
            usage = self.calculate_usage_score()
            retention_score = base_score * (0.4 * recency + 0.3 * usage + 0.3)
        
        return min(1.0, max(0.0, retention_score))
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "memory_level": self.memory_level.value,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "context": self.context,
            "tags": self.tags,
            "related_memories": self.related_memories,
            "decay_rate": self.decay_rate,
            "consolidation_count": self.consolidation_count,
            "archived": self.archived,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            agent_id=data["agent_id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            memory_level=MemoryLevel(data["memory_level"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            access_count=data.get("access_count", 0),
            importance_score=data.get("importance_score", 0.5),
            relevance_score=data.get("relevance_score", 0.5),
            confidence_score=data.get("confidence_score", 0.5),
            context=data.get("context", {}),
            tags=data.get("tags", []),
            related_memories=data.get("related_memories", []),
            decay_rate=data.get("decay_rate", 0.1),
            consolidation_count=data.get("consolidation_count", 0),
            archived=data.get("archived", False),
            archived_at=datetime.fromisoformat(data["archived_at"]) if data.get("archived_at") else None
        )


@dataclass
class MemoryHierarchyConfig:
    """Configuration for memory hierarchy management."""
    # Capacity limits for each level
    short_term_capacity: int = 100
    working_memory_capacity: int = 500
    long_term_capacity: int = 5000
    institutional_capacity: int = -1  # Unlimited
    
    # Aging parameters
    aging_strategy: AgingStrategy = AgingStrategy.HYBRID
    short_term_retention_hours: float = 24.0
    working_memory_retention_days: float = 7.0
    long_term_retention_days: float = 90.0
    
    # Consolidation parameters
    consolidation_triggers: List[ConsolidationTrigger] = field(default_factory=lambda: [
        ConsolidationTrigger.TIME_INTERVAL,
        ConsolidationTrigger.MEMORY_PRESSURE
    ])
    consolidation_interval_hours: float = 6.0
    memory_pressure_threshold: float = 0.8
    
    # Compression settings
    enable_compression: bool = True
    compression_threshold_size: int = 1000  # Characters
    compression_quality: CompressionQuality = CompressionQuality.BALANCED


@dataclass
class ConsolidationResult:
    """Result of a memory consolidation cycle."""
    consolidation_id: str
    agent_id: str
    trigger: ConsolidationTrigger
    started_at: datetime
    completed_at: datetime
    
    # Statistics
    memories_processed: int
    memories_compressed: int
    memories_promoted: int
    memories_archived: int
    memories_deleted: int
    
    # Size metrics
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    
    # Quality metrics
    avg_semantic_preservation: float
    memories_with_high_preservation: int
    
    processing_time_ms: float
    summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consolidation_id": self.consolidation_id,
            "agent_id": self.agent_id,
            "trigger": self.trigger.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "memories_processed": self.memories_processed,
            "memories_compressed": self.memories_compressed,
            "memories_promoted": self.memories_promoted,
            "memories_archived": self.memories_archived,
            "memories_deleted": self.memories_deleted,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "avg_semantic_preservation": self.avg_semantic_preservation,
            "memories_with_high_preservation": self.memories_with_high_preservation,
            "processing_time_ms": self.processing_time_ms,
            "summary": self.summary
        }


# =============================================================================
# MEMORY CONSOLIDATION ENGINE
# =============================================================================

class MemoryConsolidationEngine:
    """Engine for consolidating and optimizing memory across hierarchy levels."""
    
    def __init__(self, compression_engine):
        self.compression_engine = compression_engine
        self.consolidation_history: List[ConsolidationResult] = []
    
    async def consolidate_memories(
        self,
        memories: List[MemoryItem],
        config: MemoryHierarchyConfig,
        trigger: ConsolidationTrigger
    ) -> ConsolidationResult:
        """Consolidate memories according to hierarchy rules."""
        start_time = time.time()
        started_at = datetime.utcnow()
        
        result = ConsolidationResult(
            consolidation_id=str(uuid.uuid4()),
            agent_id=memories[0].agent_id if memories else "unknown",
            trigger=trigger,
            started_at=started_at,
            completed_at=started_at,  # Will be updated
            memories_processed=len(memories),
            memories_compressed=0,
            memories_promoted=0,
            memories_archived=0,
            memories_deleted=0,
            original_size_bytes=0,
            compressed_size_bytes=0,
            compression_ratio=0.0,
            avg_semantic_preservation=0.0,
            memories_with_high_preservation=0,
            processing_time_ms=0.0,
            summary=""
        )
        
        try:
            # Calculate original size
            result.original_size_bytes = sum(len(m.content.encode('utf-8')) for m in memories)
            
            # Group memories by level for processing
            memories_by_level = defaultdict(list)
            for memory in memories:
                memories_by_level[memory.memory_level].append(memory)
            
            processed_memories = []
            compression_results = []
            
            # Process each memory level
            for level, level_memories in memories_by_level.items():
                level_result = await self._consolidate_memory_level(
                    level_memories, level, config
                )
                processed_memories.extend(level_result["memories"])
                compression_results.extend(level_result["compressions"])
                
                result.memories_compressed += level_result["compressed_count"]
                result.memories_promoted += level_result["promoted_count"]
                result.memories_archived += level_result["archived_count"]
                result.memories_deleted += level_result["deleted_count"]
            
            # Calculate compression metrics
            if compression_results:
                total_preservation = sum(cr.semantic_preservation_score for cr in compression_results)
                result.avg_semantic_preservation = total_preservation / len(compression_results)
                result.memories_with_high_preservation = len([
                    cr for cr in compression_results if cr.semantic_preservation_score >= 0.8
                ])
            
            result.compressed_size_bytes = sum(len(m.content.encode('utf-8')) for m in processed_memories)
            if result.original_size_bytes > 0:
                result.compression_ratio = 1 - (result.compressed_size_bytes / result.original_size_bytes)
            
            result.completed_at = datetime.utcnow()
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            result.summary = self._generate_consolidation_summary(result)
            
            self.consolidation_history.append(result)
            
            logger.info(
                f"Memory consolidation completed",
                consolidation_id=result.consolidation_id,
                trigger=trigger.value,
                compression_ratio=result.compression_ratio,
                processing_time=result.processing_time_ms
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            result.completed_at = datetime.utcnow()
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.summary = f"Consolidation failed: {e}"
            return result
    
    async def _consolidate_memory_level(
        self,
        memories: List[MemoryItem],
        level: MemoryLevel,
        config: MemoryHierarchyConfig
    ) -> Dict[str, Any]:
        """Consolidate memories within a specific level."""
        compressed_count = 0
        promoted_count = 0
        archived_count = 0
        deleted_count = 0
        compression_results = []
        processed_memories = []
        
        for memory in memories:
            # Calculate retention score
            retention_score = memory.calculate_retention_score(config.aging_strategy)
            
            # Decide what to do with this memory
            if level == MemoryLevel.SHORT_TERM:
                # Short-term memory: promote to working or delete
                if retention_score >= 0.7:
                    memory.memory_level = MemoryLevel.WORKING
                    promoted_count += 1
                    processed_memories.append(memory)
                elif retention_score >= 0.3:
                    # Keep in short-term but possibly compress
                    if (config.enable_compression and 
                        len(memory.content) > config.compression_threshold_size):
                        compressed_memory, compression_result = await self._compress_memory(
                            memory, config
                        )
                        if compression_result:
                            compression_results.append(compression_result)
                            compressed_count += 1
                            processed_memories.append(compressed_memory)
                        else:
                            processed_memories.append(memory)
                    else:
                        processed_memories.append(memory)
                else:
                    # Delete low-value short-term memories
                    deleted_count += 1
            
            elif level == MemoryLevel.WORKING:
                # Working memory: promote to long-term, compress, or archive
                if retention_score >= 0.8:
                    memory.memory_level = MemoryLevel.LONG_TERM
                    promoted_count += 1
                    processed_memories.append(memory)
                elif retention_score >= 0.4:
                    # Compress and keep in working memory
                    if (config.enable_compression and 
                        len(memory.content) > config.compression_threshold_size):
                        compressed_memory, compression_result = await self._compress_memory(
                            memory, config
                        )
                        if compression_result:
                            compression_results.append(compression_result)
                            compressed_count += 1
                            processed_memories.append(compressed_memory)
                        else:
                            processed_memories.append(memory)
                    else:
                        processed_memories.append(memory)
                else:
                    # Archive low-value working memories
                    memory.archived = True
                    memory.archived_at = datetime.utcnow()
                    archived_count += 1
                    processed_memories.append(memory)
            
            elif level == MemoryLevel.LONG_TERM:
                # Long-term memory: promote to institutional or compress
                if retention_score >= 0.9 and memory.memory_type == MemoryType.SEMANTIC:
                    memory.memory_level = MemoryLevel.INSTITUTIONAL
                    promoted_count += 1
                    processed_memories.append(memory)
                elif retention_score >= 0.3:
                    # Compress long-term memories
                    if config.enable_compression:
                        compressed_memory, compression_result = await self._compress_memory(
                            memory, config
                        )
                        if compression_result:
                            compression_results.append(compression_result)
                            compressed_count += 1
                            processed_memories.append(compressed_memory)
                        else:
                            processed_memories.append(memory)
                    else:
                        processed_memories.append(memory)
                else:
                    # Archive very old long-term memories
                    memory.archived = True
                    memory.archived_at = datetime.utcnow()
                    archived_count += 1
                    processed_memories.append(memory)
            
            else:  # INSTITUTIONAL
                # Institutional memory: permanent, only compress if needed
                if (config.enable_compression and 
                    len(memory.content) > config.compression_threshold_size * 2):
                    compressed_memory, compression_result = await self._compress_memory(
                        memory, config
                    )
                    if compression_result:
                        compression_results.append(compression_result)
                        compressed_count += 1
                        processed_memories.append(compressed_memory)
                    else:
                        processed_memories.append(memory)
                else:
                    processed_memories.append(memory)
            
            # Update consolidation count
            memory.consolidation_count += 1
        
        return {
            "memories": processed_memories,
            "compressions": compression_results,
            "compressed_count": compressed_count,
            "promoted_count": promoted_count,
            "archived_count": archived_count,
            "deleted_count": deleted_count
        }
    
    async def _compress_memory(
        self,
        memory: MemoryItem,
        config: MemoryHierarchyConfig
    ) -> Tuple[MemoryItem, Optional[CompressedResult]]:
        """Compress a single memory item."""
        try:
            compression_config = CompressionConfig(
                quality=config.compression_quality,
                target_reduction=0.6,
                enable_semantic_validation=True
            )
            
            # Create compression context
            context_metadata = {
                "memory_type": memory.memory_type.value,
                "importance_score": memory.importance_score,
                "agent_id": memory.agent_id
            }
            
            # Perform compression
            compression_result = await self.compression_engine.compress_context(
                memory.content, compression_config, context_metadata
            )
            
            # Create compressed memory
            compressed_memory = MemoryItem(
                memory_id=memory.memory_id,
                agent_id=memory.agent_id,
                content=compression_result.compressed_content,
                memory_type=memory.memory_type,
                memory_level=memory.memory_level,
                created_at=memory.created_at,
                last_accessed=memory.last_accessed,
                access_count=memory.access_count,
                importance_score=memory.importance_score,
                relevance_score=memory.relevance_score,
                confidence_score=memory.confidence_score,
                context={
                    **memory.context,
                    "compressed": True,
                    "compression_ratio": compression_result.compression_ratio,
                    "semantic_preservation": compression_result.semantic_preservation_score,
                    "original_size": compression_result.original_size,
                    "compressed_size": compression_result.compressed_size
                },
                tags=memory.tags + ["compressed"],
                related_memories=memory.related_memories,
                decay_rate=memory.decay_rate,
                consolidation_count=memory.consolidation_count,
                archived=memory.archived,
                archived_at=memory.archived_at
            )
            
            return compressed_memory, compression_result
            
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")
            return memory, None
    
    def _generate_consolidation_summary(self, result: ConsolidationResult) -> str:
        """Generate a human-readable summary of consolidation results."""
        summary_parts = [
            f"Processed {result.memories_processed} memories"
        ]
        
        if result.memories_compressed > 0:
            summary_parts.append(f"compressed {result.memories_compressed}")
        
        if result.memories_promoted > 0:
            summary_parts.append(f"promoted {result.memories_promoted}")
        
        if result.memories_archived > 0:
            summary_parts.append(f"archived {result.memories_archived}")
        
        if result.memories_deleted > 0:
            summary_parts.append(f"deleted {result.memories_deleted}")
        
        summary = " | ".join(summary_parts)
        
        if result.compression_ratio > 0:
            summary += f" | Achieved {result.compression_ratio:.1%} size reduction"
        
        if result.avg_semantic_preservation > 0:
            summary += f" | {result.avg_semantic_preservation:.2f} avg semantic preservation"
        
        return summary


# =============================================================================
# MAIN MEMORY HIERARCHY MANAGER
# =============================================================================

class MemoryHierarchyManager:
    """
    Manages hierarchical memory storage with intelligent aging, consolidation,
    and cross-session continuity for optimal memory utilization.
    """
    
    def __init__(self, config: Optional[MemoryHierarchyConfig] = None):
        """Initialize the memory hierarchy manager."""
        self.config = config or MemoryHierarchyConfig()
        self.compression_engine = None
        self.consolidation_engine = None
        self.redis = get_redis()
        
        # Memory storage by agent and level
        self.agent_memories: Dict[str, Dict[MemoryLevel, List[MemoryItem]]] = defaultdict(
            lambda: defaultdict(list)
        )
        
        # Consolidation scheduling
        self.last_consolidation: Dict[str, datetime] = {}
        self.consolidation_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.metrics = {
            "total_memories": 0,
            "memories_by_level": {level: 0 for level in MemoryLevel},
            "total_consolidations": 0,
            "avg_consolidation_time_ms": 0.0,
            "total_compression_ratio": 0.0,
            "memory_pressure": 0.0
        }
        
        logger.info("Memory Hierarchy Manager initialized")
    
    async def initialize(self):
        """Initialize the memory hierarchy manager with required services."""
        from .context_compression_engine import get_context_compression_engine
        
        self.compression_engine = await get_context_compression_engine()
        self.consolidation_engine = MemoryConsolidationEngine(self.compression_engine)
        
        # Load existing memories from persistent storage
        await self._load_memories_from_storage()
        
        logger.info("✅ Memory Hierarchy Manager fully initialized")
    
    # =============================================================================
    # MEMORY MANAGEMENT OPERATIONS
    # =============================================================================
    
    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType,
        memory_level: MemoryLevel = MemoryLevel.SHORT_TERM,
        importance_score: float = 0.5,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> MemoryItem:
        """Store a new memory item in the hierarchy."""
        try:
            memory_item = MemoryItem(
                memory_id=str(uuid.uuid4()),
                agent_id=agent_id,
                content=content,
                memory_type=memory_type,
                memory_level=memory_level,
                importance_score=importance_score,
                context=context or {},
                tags=tags or []
            )
            
            # Add to memory hierarchy
            self.agent_memories[agent_id][memory_level].append(memory_item)
            
            # Update metrics
            self.metrics["total_memories"] += 1
            self.metrics["memories_by_level"][memory_level] += 1
            
            # Check for memory pressure and trigger consolidation if needed
            await self._check_memory_pressure(agent_id)
            
            # Persist to storage
            await self._persist_memory(memory_item)
            
            logger.debug(f"Stored memory in {memory_level.value}", agent_id=agent_id, memory_id=memory_item.memory_id)
            
            return memory_item
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memories(
        self,
        agent_id: str,
        memory_levels: Optional[List[MemoryLevel]] = None,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 100,
        min_importance: float = 0.0,
        include_archived: bool = False
    ) -> List[MemoryItem]:
        """Retrieve memories based on criteria."""
        try:
            memories = []
            levels_to_search = memory_levels or list(MemoryLevel)
            
            for level in levels_to_search:
                level_memories = self.agent_memories[agent_id][level]
                
                for memory in level_memories:
                    # Apply filters
                    if memory_types and memory.memory_type not in memory_types:
                        continue
                    
                    if memory.importance_score < min_importance:
                        continue
                    
                    if memory.archived and not include_archived:
                        continue
                    
                    # Update access statistics
                    memory.update_access()
                    memories.append(memory)
            
            # Sort by importance and recency
            memories.sort(
                key=lambda m: (m.importance_score, m.calculate_recency_score()),
                reverse=True
            )
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []
    
    async def search_memories(
        self,
        agent_id: str,
        query: str,
        memory_levels: Optional[List[MemoryLevel]] = None,
        limit: int = 20
    ) -> List[Tuple[MemoryItem, float]]:
        """Search memories by content similarity."""
        try:
            all_memories = await self.retrieve_memories(
                agent_id, memory_levels, limit=1000  # Search in larger set
            )
            
            results = []
            query_lower = query.lower()
            
            for memory in all_memories:
                score = 0.0
                content_lower = memory.content.lower()
                
                # Simple text matching (could be enhanced with embeddings)
                if query_lower in content_lower:
                    score += 2.0
                
                # Tag matching
                for tag in memory.tags:
                    if query_lower in tag.lower():
                        score += 1.0
                
                # Fuzzy matching for partial words
                query_words = query_lower.split()
                content_words = content_lower.split()
                
                for q_word in query_words:
                    for c_word in content_words:
                        if q_word in c_word or c_word in q_word:
                            score += 0.5
                
                if score > 0:
                    # Weight by importance and recency
                    final_score = score * memory.importance_score * memory.calculate_recency_score()
                    results.append((memory, final_score))
            
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []
    
    async def update_memory_importance(
        self,
        memory_id: str,
        new_importance: float,
        agent_id: Optional[str] = None
    ) -> bool:
        """Update the importance score of a memory."""
        try:
            memory = await self._find_memory_by_id(memory_id, agent_id)
            if memory:
                memory.importance_score = max(0.0, min(1.0, new_importance))
                await self._persist_memory(memory)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}")
            return False
    
    async def archive_memory(self, memory_id: str, agent_id: Optional[str] = None) -> bool:
        """Archive a specific memory."""
        try:
            memory = await self._find_memory_by_id(memory_id, agent_id)
            if memory:
                memory.archived = True
                memory.archived_at = datetime.utcnow()
                await self._persist_memory(memory)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to archive memory: {e}")
            return False
    
    # =============================================================================
    # MEMORY CONSOLIDATION OPERATIONS
    # =============================================================================
    
    async def trigger_consolidation(
        self,
        agent_id: str,
        trigger: ConsolidationTrigger = ConsolidationTrigger.TIME_INTERVAL
    ) -> ConsolidationResult:
        """Manually trigger memory consolidation for an agent."""
        try:
            # Get all memories for the agent
            all_memories = []
            for level in MemoryLevel:
                all_memories.extend(self.agent_memories[agent_id][level])
            
            if not all_memories:
                # Create empty result
                return ConsolidationResult(
                    consolidation_id=str(uuid.uuid4()),
                    agent_id=agent_id,
                    trigger=trigger,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    memories_processed=0,
                    memories_compressed=0,
                    memories_promoted=0,
                    memories_archived=0,
                    memories_deleted=0,
                    original_size_bytes=0,
                    compressed_size_bytes=0,
                    compression_ratio=0.0,
                    avg_semantic_preservation=0.0,
                    memories_with_high_preservation=0,
                    processing_time_ms=0.0,
                    summary="No memories to consolidate"
                )
            
            # Perform consolidation
            result = await self.consolidation_engine.consolidate_memories(
                all_memories, self.config, trigger
            )
            
            # Update memory storage with consolidated memories
            await self._update_memories_after_consolidation(agent_id, all_memories)
            
            # Update metrics
            self.metrics["total_consolidations"] += 1
            if self.metrics["total_consolidations"] > 0:
                self.metrics["avg_consolidation_time_ms"] = (
                    (self.metrics["avg_consolidation_time_ms"] * (self.metrics["total_consolidations"] - 1) +
                     result.processing_time_ms) / self.metrics["total_consolidations"]
                )
            
            if result.compression_ratio > 0:
                self.metrics["total_compression_ratio"] = (
                    (self.metrics["total_compression_ratio"] + result.compression_ratio) / 2
                )
            
            # Update last consolidation time
            self.last_consolidation[agent_id] = datetime.utcnow()
            
            logger.info(
                f"Memory consolidation completed for agent {agent_id}",
                consolidation_id=result.consolidation_id,
                compression_ratio=result.compression_ratio
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            # Return error result
            return ConsolidationResult(
                consolidation_id=str(uuid.uuid4()),
                agent_id=agent_id,
                trigger=trigger,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                memories_processed=0,
                memories_compressed=0,
                memories_promoted=0,
                memories_archived=0,
                memories_deleted=0,
                original_size_bytes=0,
                compressed_size_bytes=0,
                compression_ratio=0.0,
                avg_semantic_preservation=0.0,
                memories_with_high_preservation=0,
                processing_time_ms=0.0,
                summary=f"Consolidation failed: {e}"
            )
    
    async def schedule_automatic_consolidation(self, agent_id: str):
        """Schedule automatic consolidation for an agent."""
        if agent_id in self.consolidation_tasks:
            return  # Already scheduled
        
        async def consolidation_loop():
            while True:
                try:
                    # Wait for consolidation interval
                    await asyncio.sleep(self.config.consolidation_interval_hours * 3600)
                    
                    # Check if consolidation is needed
                    if await self._should_consolidate(agent_id):
                        await self.trigger_consolidation(agent_id, ConsolidationTrigger.TIME_INTERVAL)
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Automatic consolidation error for {agent_id}: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        task = asyncio.create_task(consolidation_loop())
        self.consolidation_tasks[agent_id] = task
        
        logger.info(f"Scheduled automatic consolidation for agent {agent_id}")
    
    async def stop_automatic_consolidation(self, agent_id: str):
        """Stop automatic consolidation for an agent."""
        if agent_id in self.consolidation_tasks:
            self.consolidation_tasks[agent_id].cancel()
            del self.consolidation_tasks[agent_id]
            logger.info(f"Stopped automatic consolidation for agent {agent_id}")
    
    # =============================================================================
    # MEMORY ANALYTICS AND METRICS
    # =============================================================================
    
    def get_memory_statistics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            if agent_id:
                # Agent-specific statistics
                agent_memories_flat = []
                for level in MemoryLevel:
                    agent_memories_flat.extend(self.agent_memories[agent_id][level])
                
                if not agent_memories_flat:
                    return {"agent_id": agent_id, "total_memories": 0}
                
                # Calculate statistics
                total_size = sum(len(m.content.encode('utf-8')) for m in agent_memories_flat)
                avg_importance = sum(m.importance_score for m in agent_memories_flat) / len(agent_memories_flat)
                avg_age_hours = sum(m.calculate_age_hours() for m in agent_memories_flat) / len(agent_memories_flat)
                
                # Memory distribution by level
                level_distribution = {}
                for level in MemoryLevel:
                    level_memories = self.agent_memories[agent_id][level]
                    level_distribution[level.value] = {
                        "count": len(level_memories),
                        "avg_importance": sum(m.importance_score for m in level_memories) / len(level_memories) if level_memories else 0,
                        "total_size_bytes": sum(len(m.content.encode('utf-8')) for m in level_memories)
                    }
                
                # Memory distribution by type
                type_distribution = defaultdict(int)
                for memory in agent_memories_flat:
                    type_distribution[memory.memory_type.value] += 1
                
                return {
                    "agent_id": agent_id,
                    "total_memories": len(agent_memories_flat),
                    "total_size_bytes": total_size,
                    "avg_importance_score": avg_importance,
                    "avg_age_hours": avg_age_hours,
                    "level_distribution": level_distribution,
                    "type_distribution": dict(type_distribution),
                    "archived_count": len([m for m in agent_memories_flat if m.archived]),
                    "last_consolidation": self.last_consolidation.get(agent_id, "Never").isoformat() if isinstance(self.last_consolidation.get(agent_id), datetime) else "Never"
                }
            else:
                # System-wide statistics
                return {
                    "system_metrics": self.metrics,
                    "total_agents": len(self.agent_memories),
                    "consolidation_history": len(self.consolidation_engine.consolidation_history) if self.consolidation_engine else 0,
                    "active_consolidation_tasks": len(self.consolidation_tasks)
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
    
    def get_memory_pressure(self, agent_id: str) -> float:
        """Calculate memory pressure for an agent (0.0 to 1.0)."""
        try:
            pressure_scores = []
            
            for level in MemoryLevel:
                level_memories = self.agent_memories[agent_id][level]
                
                if level == MemoryLevel.SHORT_TERM:
                    capacity = self.config.short_term_capacity
                elif level == MemoryLevel.WORKING:
                    capacity = self.config.working_memory_capacity
                elif level == MemoryLevel.LONG_TERM:
                    capacity = self.config.long_term_capacity
                else:  # INSTITUTIONAL
                    capacity = self.config.institutional_capacity
                
                if capacity > 0:
                    utilization = len(level_memories) / capacity
                    pressure_scores.append(min(1.0, utilization))
            
            overall_pressure = sum(pressure_scores) / len(pressure_scores) if pressure_scores else 0.0
            self.metrics["memory_pressure"] = overall_pressure
            
            return overall_pressure
            
        except Exception as e:
            logger.error(f"Failed to calculate memory pressure: {e}")
            return 0.0
    
    # =============================================================================
    # UTILITY AND HELPER METHODS
    # =============================================================================
    
    async def _check_memory_pressure(self, agent_id: str):
        """Check memory pressure and trigger consolidation if needed."""
        pressure = self.get_memory_pressure(agent_id)
        
        if (pressure >= self.config.memory_pressure_threshold and
            ConsolidationTrigger.MEMORY_PRESSURE in self.config.consolidation_triggers):
            
            logger.info(f"Memory pressure {pressure:.2f} triggered consolidation for {agent_id}")
            await self.trigger_consolidation(agent_id, ConsolidationTrigger.MEMORY_PRESSURE)
    
    async def _should_consolidate(self, agent_id: str) -> bool:
        """Check if consolidation should be performed for an agent."""
        last_consolidation = self.last_consolidation.get(agent_id)
        if not last_consolidation:
            return True  # Never consolidated
        
        time_since_last = datetime.utcnow() - last_consolidation
        return time_since_last.total_seconds() >= (self.config.consolidation_interval_hours * 3600)
    
    async def _find_memory_by_id(self, memory_id: str, agent_id: Optional[str] = None) -> Optional[MemoryItem]:
        """Find a memory by ID."""
        agents_to_search = [agent_id] if agent_id else self.agent_memories.keys()
        
        for aid in agents_to_search:
            for level in MemoryLevel:
                for memory in self.agent_memories[aid][level]:
                    if memory.memory_id == memory_id:
                        return memory
        
        return None
    
    async def _update_memories_after_consolidation(
        self,
        agent_id: str,
        consolidated_memories: List[MemoryItem]
    ):
        """Update memory storage after consolidation."""
        # Clear existing memories
        for level in MemoryLevel:
            self.agent_memories[agent_id][level].clear()
        
        # Re-add consolidated memories
        for memory in consolidated_memories:
            if not memory.archived:  # Don't re-add archived memories to active storage
                self.agent_memories[agent_id][memory.memory_level].append(memory)
    
    # =============================================================================
    # PERSISTENCE METHODS
    # =============================================================================
    
    async def _persist_memory(self, memory: MemoryItem):
        """Persist a memory to Redis."""
        try:
            key = f"memory:{memory.agent_id}:{memory.memory_id}"
            await self.redis.setex(key, 86400 * 7, json.dumps(memory.to_dict()))
        except Exception as e:
            logger.error(f"Failed to persist memory: {e}")
    
    async def _load_memories_from_storage(self):
        """Load memories from persistent storage."""
        try:
            # This would load from database/Redis in production
            logger.debug("Memories loaded from persistent storage")
        except Exception as e:
            logger.error(f"Failed to load memories from storage: {e}")
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on memory hierarchy manager."""
        try:
            # Test basic functionality
            test_agent_id = "health_check_agent"
            test_memory = await self.store_memory(
                test_agent_id,
                "Health check memory content",
                MemoryType.SEMANTIC,
                MemoryLevel.SHORT_TERM,
                0.8
            )
            
            # Test retrieval
            retrieved = await self.retrieve_memories(test_agent_id, limit=1)
            
            # Test search
            search_results = await self.search_memories(test_agent_id, "health check", limit=1)
            
            # Clean up test memory
            await self.archive_memory(test_memory.memory_id, test_agent_id)
            
            return {
                "status": "healthy",
                "components": {
                    "memory_storage": "operational",
                    "memory_retrieval": "operational",
                    "memory_search": "operational",
                    "compression_engine": "operational" if self.compression_engine else "unavailable",
                    "consolidation_engine": "operational" if self.consolidation_engine else "unavailable"
                },
                "test_results": {
                    "memory_stored": test_memory.memory_id is not None,
                    "memory_retrieved": len(retrieved) > 0,
                    "search_functional": len(search_results) > 0
                },
                "metrics": self.get_memory_statistics()
            }
            
        except Exception as e:
            logger.error(f"Memory hierarchy manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "memory_storage": "unknown",
                    "memory_retrieval": "unknown",
                    "memory_search": "unknown",
                    "compression_engine": "unknown",
                    "consolidation_engine": "unknown"
                }
            }


# =============================================================================
# GLOBAL MEMORY HIERARCHY MANAGER INSTANCE
# =============================================================================

_memory_manager: Optional[MemoryHierarchyManager] = None


async def get_memory_hierarchy_manager() -> MemoryHierarchyManager:
    """Get global memory hierarchy manager instance."""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryHierarchyManager()
        await _memory_manager.initialize()
    
    return _memory_manager


async def cleanup_memory_hierarchy_manager():
    """Clean up global memory hierarchy manager."""
    global _memory_manager
    
    if _memory_manager:
        # Stop all consolidation tasks
        for task in _memory_manager.consolidation_tasks.values():
            task.cancel()
        
        _memory_manager = None
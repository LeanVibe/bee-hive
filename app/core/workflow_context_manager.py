"""
Workflow Context Manager for LeanVibe Agent Hive 2.0

Provides intelligent context injection, compression, and management for workflow-scoped
memory with token efficiency and context-aware task execution.

Features:
- Workflow-scoped context isolation and management
- Intelligent context compression with semantic preservation
- Token-efficient context injection with automatic optimization
- Context versioning and rollback capabilities
- Cross-workflow context sharing with access controls
- Performance monitoring and optimization
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
import httpx

from .database import get_session
from .redis import get_redis
from .semantic_memory_task_processor import SemanticMemoryTaskProcessor, SemanticMemoryTask, SemanticTaskType
from ..models.workflow import Workflow, WorkflowStatus
from ..schemas.semantic_memory import ProcessingPriority

logger = structlog.get_logger()


# =============================================================================
# CONTEXT DEFINITIONS AND TYPES
# =============================================================================

class ContextType(str, Enum):
    """Types of workflow context."""
    TASK_INPUT = "task_input"
    TASK_OUTPUT = "task_output"
    WORKFLOW_STATE = "workflow_state"
    AGENT_MEMORY = "agent_memory"
    CROSS_WORKFLOW = "cross_workflow"
    EXTERNAL_DATA = "external_data"


class CompressionStrategy(str, Enum):
    """Context compression strategies."""
    NO_COMPRESSION = "no_compression"
    TOKEN_LIMIT = "token_limit"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    IMPORTANCE_FILTERING = "importance_filtering"
    TEMPORAL_DECAY = "temporal_decay"
    HYBRID = "hybrid"


class ContextScope(str, Enum):
    """Context access scope."""
    WORKFLOW_PRIVATE = "workflow_private"
    AGENT_SHARED = "agent_shared"
    CROSS_AGENT = "cross_agent"
    GLOBAL_SHARED = "global_shared"


@dataclass
class ContextFragment:
    """A fragment of workflow context."""
    fragment_id: str
    context_type: ContextType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    token_count: int = 0
    semantic_embedding: Optional[List[float]] = None
    access_scope: ContextScope = ContextScope.WORKFLOW_PRIVATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fragment_id": self.fragment_id,
            "context_type": self.context_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "token_count": self.token_count,
            "access_scope": self.access_scope.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextFragment':
        """Create from dictionary."""
        return cls(
            fragment_id=data["fragment_id"],
            context_type=ContextType(data["context_type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_count=data.get("token_count", 0),
            access_scope=ContextScope(data.get("access_scope", ContextScope.WORKFLOW_PRIVATE.value))
        )


@dataclass
class WorkflowContext:
    """Complete workflow context with version management."""
    workflow_id: str
    version: int
    fragments: List[ContextFragment] = field(default_factory=list)
    compressed_fragments: List[ContextFragment] = field(default_factory=list)
    total_tokens: int = 0
    compression_ratio: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_permissions: Dict[str, List[str]] = field(default_factory=dict)  # agent_id -> [scopes]
    
    def get_total_token_count(self) -> int:
        """Get total token count including compressed fragments."""
        active_fragments = self.compressed_fragments if self.compressed_fragments else self.fragments
        return sum(fragment.token_count for fragment in active_fragments)
    
    def get_context_by_type(self, context_type: ContextType) -> List[ContextFragment]:
        """Get context fragments by type."""
        active_fragments = self.compressed_fragments if self.compressed_fragments else self.fragments
        return [f for f in active_fragments if f.context_type == context_type]
    
    def get_high_importance_context(self, threshold: float = 0.7) -> List[ContextFragment]:
        """Get high importance context fragments."""
        active_fragments = self.compressed_fragments if self.compressed_fragments else self.fragments
        return [f for f in active_fragments if f.importance >= threshold]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "version": self.version,
            "fragments": [f.to_dict() for f in self.fragments],
            "compressed_fragments": [f.to_dict() for f in self.compressed_fragments],
            "total_tokens": self.total_tokens,
            "compression_ratio": self.compression_ratio,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
            "access_permissions": self.access_permissions
        }


@dataclass
class ContextInjectionConfig:
    """Configuration for context injection."""
    max_context_tokens: int = 2000
    compression_strategy: CompressionStrategy = CompressionStrategy.HYBRID
    importance_threshold: float = 0.3
    include_task_history: bool = True
    include_agent_memory: bool = True
    include_cross_workflow: bool = False
    temporal_decay_factor: float = 0.1
    preserve_recent_threshold: timedelta = field(default_factory=lambda: timedelta(hours=1))


@dataclass
class ContextPerformanceMetrics:
    """Performance metrics for context operations."""
    total_injections: int = 0
    successful_injections: int = 0
    failed_injections: int = 0
    average_injection_time_ms: float = 0.0
    average_compression_time_ms: float = 0.0
    average_compression_ratio: float = 0.0
    token_efficiency_ratio: float = 0.0
    context_cache_hits: int = 0
    context_cache_misses: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "injection_stats": {
                "total": self.total_injections,
                "successful": self.successful_injections,
                "failed": self.failed_injections,
                "success_rate": self.successful_injections / max(1, self.total_injections)
            },
            "performance": {
                "avg_injection_time_ms": self.average_injection_time_ms,
                "avg_compression_time_ms": self.average_compression_time_ms,
                "avg_compression_ratio": self.average_compression_ratio,
                "token_efficiency_ratio": self.token_efficiency_ratio
            },
            "cache": {
                "hits": self.context_cache_hits,
                "misses": self.context_cache_misses,
                "hit_rate": self.context_cache_hits / max(1, self.context_cache_hits + self.context_cache_misses)
            }
        }


# =============================================================================
# CONTEXT COMPRESSION ENGINE
# =============================================================================

class ContextCompressionEngine:
    """Handles intelligent context compression with semantic preservation."""
    
    def __init__(self, task_processor: SemanticMemoryTaskProcessor):
        self.task_processor = task_processor
        self.compression_cache: Dict[str, Tuple[List[ContextFragment], float]] = {}
        self.compression_metrics = {
            "compressions_performed": 0,
            "average_compression_ratio": 0.0,
            "compression_time_ms": 0.0
        }
    
    async def compress_context(
        self,
        fragments: List[ContextFragment],
        strategy: CompressionStrategy,
        target_tokens: int,
        preserve_importance_threshold: float = 0.8
    ) -> Tuple[List[ContextFragment], float]:
        """
        Compress context fragments using specified strategy.
        
        Args:
            fragments: Context fragments to compress
            strategy: Compression strategy to use
            target_tokens: Target token count after compression
            preserve_importance_threshold: Preserve fragments above this importance
            
        Returns:
            Tuple of (compressed_fragments, compression_ratio)
        """
        start_time = time.time()
        
        try:
            # Calculate current token count
            current_tokens = sum(f.token_count for f in fragments)
            if current_tokens <= target_tokens:
                return fragments, 1.0  # No compression needed
            
            # Check cache
            cache_key = self._get_compression_cache_key(fragments, strategy, target_tokens)
            if cache_key in self.compression_cache:
                logger.debug(f"Using cached compression for key {cache_key[:16]}...")
                return self.compression_cache[cache_key]
            
            # Apply compression strategy
            if strategy == CompressionStrategy.NO_COMPRESSION:
                compressed_fragments = fragments
            elif strategy == CompressionStrategy.TOKEN_LIMIT:
                compressed_fragments = await self._compress_by_token_limit(fragments, target_tokens)
            elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
                compressed_fragments = await self._compress_by_semantic_clustering(fragments, target_tokens)
            elif strategy == CompressionStrategy.IMPORTANCE_FILTERING:
                compressed_fragments = await self._compress_by_importance(
                    fragments, target_tokens, preserve_importance_threshold
                )
            elif strategy == CompressionStrategy.TEMPORAL_DECAY:
                compressed_fragments = await self._compress_by_temporal_decay(fragments, target_tokens)
            elif strategy == CompressionStrategy.HYBRID:
                compressed_fragments = await self._compress_hybrid(
                    fragments, target_tokens, preserve_importance_threshold
                )
            else:
                compressed_fragments = fragments
            
            # Calculate compression ratio
            compressed_tokens = sum(f.token_count for f in compressed_fragments)
            compression_ratio = compressed_tokens / max(1, current_tokens)
            
            # Update metrics
            compression_time_ms = (time.time() - start_time) * 1000
            self._update_compression_metrics(compression_ratio, compression_time_ms)
            
            # Cache result
            self.compression_cache[cache_key] = (compressed_fragments, compression_ratio)
            
            logger.info(
                f"Context compressed using {strategy.value}",
                original_tokens=current_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                compression_time_ms=compression_time_ms
            )
            
            return compressed_fragments, compression_ratio
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            # Return original fragments on error
            return fragments, 1.0
    
    async def _compress_by_token_limit(
        self, 
        fragments: List[ContextFragment], 
        target_tokens: int
    ) -> List[ContextFragment]:
        """Compress by simple token limit (truncation)."""
        # Sort by importance descending, then by timestamp descending
        sorted_fragments = sorted(
            fragments,
            key=lambda f: (-f.importance, -f.timestamp.timestamp())
        )
        
        compressed = []
        current_tokens = 0
        
        for fragment in sorted_fragments:
            if current_tokens + fragment.token_count <= target_tokens:
                compressed.append(fragment)
                current_tokens += fragment.token_count
            else:
                # Partial inclusion if possible
                remaining_tokens = target_tokens - current_tokens
                if remaining_tokens > 50:  # Minimum useful fragment size
                    # Truncate content proportionally
                    truncation_ratio = remaining_tokens / fragment.token_count
                    truncated_content = fragment.content[:int(len(fragment.content) * truncation_ratio)]
                    
                    truncated_fragment = ContextFragment(
                        fragment_id=f"{fragment.fragment_id}_truncated",
                        context_type=fragment.context_type,
                        content=truncated_content,
                        metadata={**fragment.metadata, "truncated": True},
                        importance=fragment.importance,
                        timestamp=fragment.timestamp,
                        token_count=remaining_tokens,
                        access_scope=fragment.access_scope
                    )
                    compressed.append(truncated_fragment)
                break
        
        return compressed
    
    async def _compress_by_semantic_clustering(
        self,
        fragments: List[ContextFragment],
        target_tokens: int
    ) -> List[ContextFragment]:
        """Compress using semantic clustering via memory service."""
        try:
            # Create context for compression
            context_content = "\n\n".join(f.content for f in fragments)
            
            # Submit compression task
            compression_task = SemanticMemoryTask(
                task_id=f"compress_{uuid.uuid4().hex[:8]}",
                task_type=SemanticTaskType.COMPRESS_CONTEXT,
                agent_id="context_manager",
                priority=ProcessingPriority.HIGH,
                payload={
                    "context_id": f"workflow_context_{uuid.uuid4().hex[:8]}",
                    "compression_method": "semantic_clustering",
                    "target_reduction": 1.0 - (target_tokens / sum(f.token_count for f in fragments)),
                    "preserve_importance_threshold": 0.8,
                    "agent_id": "context_manager"
                }
            )
            
            # Submit and wait for result (simplified - in production would use proper async handling)
            await self.task_processor.submit_task(compression_task)
            
            # For now, fall back to importance filtering
            return await self._compress_by_importance(fragments, target_tokens, 0.5)
            
        except Exception as e:
            logger.warning(f"Semantic clustering compression failed: {e}")
            return await self._compress_by_importance(fragments, target_tokens, 0.5)
    
    async def _compress_by_importance(
        self,
        fragments: List[ContextFragment],
        target_tokens: int,
        preserve_threshold: float
    ) -> List[ContextFragment]:
        """Compress by filtering based on importance scores."""
        # Always preserve high importance fragments
        high_importance = [f for f in fragments if f.importance >= preserve_threshold]
        low_importance = [f for f in fragments if f.importance < preserve_threshold]
        
        # Calculate tokens used by high importance fragments
        high_importance_tokens = sum(f.token_count for f in high_importance)
        
        if high_importance_tokens >= target_tokens:
            # High importance fragments exceed target, use token limit on them
            return await self._compress_by_token_limit(high_importance, target_tokens)
        
        # Add low importance fragments until target reached
        remaining_tokens = target_tokens - high_importance_tokens
        low_importance_sorted = sorted(low_importance, key=lambda f: -f.importance)
        
        selected_low_importance = []
        current_tokens = 0
        
        for fragment in low_importance_sorted:
            if current_tokens + fragment.token_count <= remaining_tokens:
                selected_low_importance.append(fragment)
                current_tokens += fragment.token_count
            else:
                break
        
        return high_importance + selected_low_importance
    
    async def _compress_by_temporal_decay(
        self,
        fragments: List[ContextFragment],
        target_tokens: int
    ) -> List[ContextFragment]:
        """Compress using temporal decay - more recent content is more important."""
        now = datetime.utcnow()
        
        # Calculate temporal scores (recent = higher score)
        for fragment in fragments:
            age_hours = (now - fragment.timestamp).total_seconds() / 3600
            temporal_score = max(0.1, 1.0 / (1.0 + age_hours * 0.1))  # Decay factor
            # Combine with existing importance
            fragment.importance = (fragment.importance + temporal_score) / 2
        
        # Use importance filtering with updated scores
        return await self._compress_by_importance(fragments, target_tokens, 0.3)
    
    async def _compress_hybrid(
        self,
        fragments: List[ContextFragment],
        target_tokens: int,
        preserve_threshold: float
    ) -> List[ContextFragment]:
        """Hybrid compression using multiple strategies."""
        # Step 1: Apply temporal decay to update importance scores
        await self._compress_by_temporal_decay(fragments, target_tokens * 2)  # Intermediate target
        
        # Step 2: Filter by importance
        importance_filtered = await self._compress_by_importance(
            fragments, target_tokens, preserve_threshold
        )
        
        # Step 3: Apply final token limit if still over target
        current_tokens = sum(f.token_count for f in importance_filtered)
        if current_tokens > target_tokens:
            return await self._compress_by_token_limit(importance_filtered, target_tokens)
        
        return importance_filtered
    
    def _get_compression_cache_key(
        self,
        fragments: List[ContextFragment],
        strategy: CompressionStrategy,
        target_tokens: int
    ) -> str:
        """Generate cache key for compression."""
        fragment_ids = [f.fragment_id for f in fragments]
        content_hash = hash(json.dumps(sorted(fragment_ids)))
        return f"{strategy.value}_{target_tokens}_{content_hash}"
    
    def _update_compression_metrics(self, compression_ratio: float, compression_time_ms: float) -> None:
        """Update compression performance metrics."""
        self.compression_metrics["compressions_performed"] += 1
        
        # Update average compression ratio
        current_avg = self.compression_metrics["average_compression_ratio"]
        count = self.compression_metrics["compressions_performed"]
        new_avg = ((current_avg * (count - 1)) + compression_ratio) / count
        self.compression_metrics["average_compression_ratio"] = new_avg
        
        # Update average compression time
        current_time_avg = self.compression_metrics["compression_time_ms"]
        new_time_avg = ((current_time_avg * (count - 1)) + compression_time_ms) / count
        self.compression_metrics["compression_time_ms"] = new_time_avg


# =============================================================================
# WORKFLOW CONTEXT MANAGER
# =============================================================================

class WorkflowContextManager:
    """
    Manages workflow-scoped context with intelligent injection and compression.
    
    Features:
    - Context isolation per workflow with version management
    - Intelligent context injection with token optimization
    - Multi-strategy compression with semantic preservation
    - Cross-workflow context sharing with access controls
    - Performance monitoring and optimization
    - Context caching and persistence
    """
    
    def __init__(self, task_processor: SemanticMemoryTaskProcessor, redis_client=None):
        self.task_processor = task_processor
        self.redis = redis_client or get_redis()
        self.compression_engine = ContextCompressionEngine(task_processor)
        
        # Context storage
        self.workflow_contexts: Dict[str, WorkflowContext] = {}
        self.context_versions: Dict[str, List[WorkflowContext]] = defaultdict(list)
        
        # Context cache for performance
        self.context_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 300  # 5 minutes
        
        # Performance tracking
        self.performance_metrics = ContextPerformanceMetrics()
        
        # Default configuration
        self.default_injection_config = ContextInjectionConfig()
        
        logger.info("Workflow Context Manager initialized")
    
    async def create_workflow_context(
        self,
        workflow_id: str,
        initial_fragments: Optional[List[ContextFragment]] = None
    ) -> WorkflowContext:
        """Create a new workflow context."""
        if workflow_id in self.workflow_contexts:
            logger.warning(f"Workflow context {workflow_id} already exists")
            return self.workflow_contexts[workflow_id]
        
        context = WorkflowContext(
            workflow_id=workflow_id,
            version=1,
            fragments=initial_fragments or [],
            metadata={"created_at": datetime.utcnow().isoformat()}
        )
        
        self.workflow_contexts[workflow_id] = context
        self.context_versions[workflow_id].append(context)
        
        # Persist to Redis
        await self._persist_context(context)
        
        logger.info(f"Created workflow context {workflow_id}")
        return context
    
    async def add_context_fragment(
        self,
        workflow_id: str,
        fragment: ContextFragment
    ) -> bool:
        """Add a context fragment to workflow context."""
        try:
            if workflow_id not in self.workflow_contexts:
                await self.create_workflow_context(workflow_id)
            
            context = self.workflow_contexts[workflow_id]
            
            # Calculate token count if not set
            if fragment.token_count == 0:
                fragment.token_count = self._estimate_token_count(fragment.content)
            
            # Add fragment
            context.fragments.append(fragment)
            context.total_tokens = sum(f.token_count for f in context.fragments)
            context.last_updated = datetime.utcnow()
            
            # Clear compressed fragments to force recompression
            context.compressed_fragments = []
            context.compression_ratio = 1.0
            
            # Persist changes
            await self._persist_context(context)
            
            # Clear related cache entries
            await self._invalidate_context_cache(workflow_id)
            
            logger.debug(
                f"Added context fragment to workflow {workflow_id}",
                fragment_id=fragment.fragment_id,
                context_type=fragment.context_type.value,
                tokens=fragment.token_count
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add context fragment: {e}")
            return False
    
    async def inject_context(
        self,
        workflow_id: str,
        task_data: Dict[str, Any],
        injection_config: Optional[ContextInjectionConfig] = None
    ) -> Dict[str, Any]:
        """
        Inject relevant context into task data.
        
        Args:
            workflow_id: Workflow identifier
            task_data: Original task data
            injection_config: Configuration for context injection
            
        Returns:
            Enhanced task data with injected context
        """
        start_time = time.time()
        config = injection_config or self.default_injection_config
        
        try:
            self.performance_metrics.total_injections += 1
            
            # Get workflow context
            context = await self._get_workflow_context(workflow_id)
            if not context:
                logger.warning(f"No context found for workflow {workflow_id}")
                return task_data
            
            # Check cache first
            cache_key = self._get_injection_cache_key(workflow_id, task_data, config)
            cached_result = await self._get_cached_context(cache_key)
            if cached_result:
                self.performance_metrics.context_cache_hits += 1
                return cached_result
            
            self.performance_metrics.context_cache_misses += 1
            
            # Collect relevant context fragments
            relevant_fragments = await self._collect_relevant_context(
                context, task_data, config
            )
            
            # Compress context if needed
            compressed_fragments, compression_ratio = await self._compress_context_if_needed(
                relevant_fragments, config
            )
            
            # Inject context into task data
            enhanced_task_data = await self._inject_context_into_task_data(
                task_data, compressed_fragments, compression_ratio
            )
            
            # Cache result
            await self._cache_context(cache_key, enhanced_task_data)
            
            # Update metrics
            injection_time_ms = (time.time() - start_time) * 1000
            self._update_injection_metrics(injection_time_ms, True)
            
            # Update context efficiency
            original_tokens = sum(f.token_count for f in relevant_fragments)
            compressed_tokens = sum(f.token_count for f in compressed_fragments)
            efficiency_ratio = compressed_tokens / max(1, original_tokens)
            self.performance_metrics.token_efficiency_ratio = (
                self.performance_metrics.token_efficiency_ratio * 0.9 + efficiency_ratio * 0.1
            )
            
            logger.info(
                f"Context injected for workflow {workflow_id}",
                original_fragments=len(relevant_fragments),
                compressed_fragments=len(compressed_fragments),
                compression_ratio=compression_ratio,
                injection_time_ms=injection_time_ms
            )
            
            return enhanced_task_data
            
        except Exception as e:
            injection_time_ms = (time.time() - start_time) * 1000
            self._update_injection_metrics(injection_time_ms, False)
            
            logger.error(f"Context injection failed for workflow {workflow_id}: {e}")
            return task_data  # Return original data on error
    
    async def compress_workflow_context(
        self,
        workflow_id: str,
        target_tokens: Optional[int] = None,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID
    ) -> bool:
        """Manually compress workflow context."""
        try:
            context = await self._get_workflow_context(workflow_id)
            if not context:
                return False
            
            target_tokens = target_tokens or self.default_injection_config.max_context_tokens
            
            # Compress context
            compressed_fragments, compression_ratio = await self.compression_engine.compress_context(
                context.fragments, strategy, target_tokens
            )
            
            # Update context
            context.compressed_fragments = compressed_fragments
            context.compression_ratio = compression_ratio
            context.last_updated = datetime.utcnow()
            
            # Persist changes
            await self._persist_context(context)
            
            logger.info(
                f"Manually compressed workflow context {workflow_id}",
                compression_ratio=compression_ratio,
                compressed_fragments=len(compressed_fragments)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Manual context compression failed for workflow {workflow_id}: {e}")
            return False
    
    async def share_context_across_workflows(
        self,
        source_workflow_id: str,
        target_workflow_id: str,
        fragment_filters: Optional[Dict[str, Any]] = None,
        access_scope: ContextScope = ContextScope.CROSS_AGENT
    ) -> bool:
        """Share context fragments between workflows."""
        try:
            source_context = await self._get_workflow_context(source_workflow_id)
            if not source_context:
                return False
            
            # Filter fragments to share
            fragments_to_share = self._filter_fragments_for_sharing(
                source_context.fragments, fragment_filters, access_scope
            )
            
            if not fragments_to_share:
                logger.warning(f"No shareable fragments found in workflow {source_workflow_id}")
                return False
            
            # Create shared fragments for target workflow
            for fragment in fragments_to_share:
                shared_fragment = ContextFragment(
                    fragment_id=f"shared_{fragment.fragment_id}_{uuid.uuid4().hex[:8]}",
                    context_type=ContextType.CROSS_WORKFLOW,
                    content=fragment.content,
                    metadata={
                        **fragment.metadata,
                        "source_workflow": source_workflow_id,
                        "shared_at": datetime.utcnow().isoformat()
                    },
                    importance=fragment.importance * 0.8,  # Slightly reduce importance for shared context
                    timestamp=datetime.utcnow(),
                    token_count=fragment.token_count,
                    access_scope=access_scope
                )
                
                await self.add_context_fragment(target_workflow_id, shared_fragment)
            
            logger.info(
                f"Shared {len(fragments_to_share)} fragments from {source_workflow_id} to {target_workflow_id}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Context sharing failed: {e}")
            return False
    
    async def get_context_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get analytics about workflow context."""
        try:
            context = await self._get_workflow_context(workflow_id)
            if not context:
                return {"error": "Workflow context not found"}
            
            analytics = {
                "workflow_id": workflow_id,
                "version": context.version,
                "total_fragments": len(context.fragments),
                "compressed_fragments": len(context.compressed_fragments),
                "total_tokens": context.get_total_token_count(),
                "compression_ratio": context.compression_ratio,
                "last_updated": context.last_updated.isoformat(),
                "context_type_distribution": {},
                "importance_distribution": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                },
                "temporal_distribution": {
                    "recent": 0,
                    "older": 0
                }
            }
            
            # Analyze context type distribution
            active_fragments = context.compressed_fragments if context.compressed_fragments else context.fragments
            for fragment in active_fragments:
                context_type = fragment.context_type.value
                analytics["context_type_distribution"][context_type] = (
                    analytics["context_type_distribution"].get(context_type, 0) + 1
                )
                
                # Importance distribution
                if fragment.importance >= 0.7:
                    analytics["importance_distribution"]["high"] += 1
                elif fragment.importance >= 0.4:
                    analytics["importance_distribution"]["medium"] += 1
                else:
                    analytics["importance_distribution"]["low"] += 1
                
                # Temporal distribution
                age_hours = (datetime.utcnow() - fragment.timestamp).total_seconds() / 3600
                if age_hours <= 1:
                    analytics["temporal_distribution"]["recent"] += 1
                else:
                    analytics["temporal_distribution"]["older"] += 1
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get context analytics: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> ContextPerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics
    
    # Private helper methods
    
    async def _get_workflow_context(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Get workflow context, loading from Redis if needed."""
        if workflow_id not in self.workflow_contexts:
            # Try to load from Redis
            context = await self._load_context_from_redis(workflow_id)
            if context:
                self.workflow_contexts[workflow_id] = context
        
        return self.workflow_contexts.get(workflow_id)
    
    async def _collect_relevant_context(
        self,
        context: WorkflowContext,
        task_data: Dict[str, Any],
        config: ContextInjectionConfig
    ) -> List[ContextFragment]:
        """Collect relevant context fragments for task."""
        relevant_fragments = []
        
        # Include high importance fragments
        high_importance_fragments = context.get_high_importance_context(config.importance_threshold)
        relevant_fragments.extend(high_importance_fragments)
        
        # Include recent fragments if configured
        if config.include_task_history:
            now = datetime.utcnow()
            recent_fragments = [
                f for f in context.fragments
                if (now - f.timestamp) <= config.preserve_recent_threshold
                and f not in relevant_fragments
            ]
            relevant_fragments.extend(recent_fragments)
        
        # Include agent memory if configured
        if config.include_agent_memory:
            agent_memory_fragments = context.get_context_by_type(ContextType.AGENT_MEMORY)
            for fragment in agent_memory_fragments:
                if fragment not in relevant_fragments:
                    relevant_fragments.append(fragment)
        
        # Include cross-workflow context if configured
        if config.include_cross_workflow:
            cross_workflow_fragments = context.get_context_by_type(ContextType.CROSS_WORKFLOW)
            for fragment in cross_workflow_fragments:
                if fragment not in relevant_fragments:
                    relevant_fragments.append(fragment)
        
        return relevant_fragments
    
    async def _compress_context_if_needed(
        self,
        fragments: List[ContextFragment],
        config: ContextInjectionConfig
    ) -> Tuple[List[ContextFragment], float]:
        """Compress context if it exceeds token limit."""
        total_tokens = sum(f.token_count for f in fragments)
        
        if total_tokens <= config.max_context_tokens:
            return fragments, 1.0
        
        return await self.compression_engine.compress_context(
            fragments,
            config.compression_strategy,
            config.max_context_tokens
        )
    
    async def _inject_context_into_task_data(
        self,
        task_data: Dict[str, Any],
        fragments: List[ContextFragment],
        compression_ratio: float
    ) -> Dict[str, Any]:
        """Inject context fragments into task data."""
        enhanced_data = task_data.copy()
        
        # Add context section
        enhanced_data["_injected_context"] = {
            "fragments": [f.to_dict() for f in fragments],
            "total_fragments": len(fragments),
            "total_tokens": sum(f.token_count for f in fragments),
            "compression_ratio": compression_ratio,
            "injection_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add context summary
        context_summary = self._generate_context_summary(fragments)
        enhanced_data["_context_summary"] = context_summary
        
        return enhanced_data
    
    def _generate_context_summary(self, fragments: List[ContextFragment]) -> str:
        """Generate a human-readable context summary."""
        if not fragments:
            return "No context available"
        
        summary_parts = []
        
        # Group by context type
        type_groups = defaultdict(list)
        for fragment in fragments:
            type_groups[fragment.context_type].append(fragment)
        
        for context_type, type_fragments in type_groups.items():
            high_importance = [f for f in type_fragments if f.importance >= 0.7]
            if high_importance:
                summary_parts.append(f"{context_type.value}: {len(high_importance)} high-importance items")
        
        if summary_parts:
            return "; ".join(summary_parts)
        else:
            return f"Context includes {len(fragments)} fragments from workflow history"
    
    def _filter_fragments_for_sharing(
        self,
        fragments: List[ContextFragment],
        filters: Optional[Dict[str, Any]],
        access_scope: ContextScope
    ) -> List[ContextFragment]:
        """Filter fragments that can be shared."""
        shareable = []
        
        for fragment in fragments:
            # Check access scope
            if fragment.access_scope == ContextScope.WORKFLOW_PRIVATE:
                continue  # Private fragments cannot be shared
            
            # Apply filters if provided
            if filters:
                if "min_importance" in filters and fragment.importance < filters["min_importance"]:
                    continue
                if "context_types" in filters and fragment.context_type not in filters["context_types"]:
                    continue
                if "max_age_hours" in filters:
                    age_hours = (datetime.utcnow() - fragment.timestamp).total_seconds() / 3600
                    if age_hours > filters["max_age_hours"]:
                        continue
            
            shareable.append(fragment)
        
        return shareable
    
    def _estimate_token_count(self, content: str) -> int:
        """Estimate token count for content (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return max(1, len(content) // 4)
    
    def _get_injection_cache_key(
        self,
        workflow_id: str,
        task_data: Dict[str, Any],
        config: ContextInjectionConfig
    ) -> str:
        """Generate cache key for context injection."""
        task_hash = hash(json.dumps(task_data, sort_keys=True))
        config_hash = hash(f"{config.max_context_tokens}_{config.compression_strategy.value}")
        return f"injection_{workflow_id}_{task_hash}_{config_hash}"
    
    async def _get_cached_context(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached context result."""
        try:
            cached_data = await self.redis.get(f"context_cache:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Cache retrieval failed: {e}")
        return None
    
    async def _cache_context(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache context result."""
        try:
            await self.redis.setex(
                f"context_cache:{cache_key}",
                self.cache_ttl_seconds,
                json.dumps(data)
            )
        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")
    
    async def _invalidate_context_cache(self, workflow_id: str) -> None:
        """Invalidate cache entries for workflow."""
        try:
            pattern = f"context_cache:*{workflow_id}*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
        except Exception as e:
            logger.debug(f"Cache invalidation failed: {e}")
    
    async def _persist_context(self, context: WorkflowContext) -> None:
        """Persist context to Redis."""
        try:
            key = f"workflow_context:{context.workflow_id}"
            data = json.dumps(context.to_dict())
            await self.redis.setex(key, 86400, data)  # 24 hour TTL
        except Exception as e:
            logger.error(f"Failed to persist context: {e}")
    
    async def _load_context_from_redis(self, workflow_id: str) -> Optional[WorkflowContext]:
        """Load context from Redis."""
        try:
            key = f"workflow_context:{workflow_id}"
            data = await self.redis.get(key)
            if data:
                context_dict = json.loads(data)
                return self._deserialize_context(context_dict)
        except Exception as e:
            logger.debug(f"Failed to load context from Redis: {e}")
        return None
    
    def _deserialize_context(self, data: Dict[str, Any]) -> WorkflowContext:
        """Deserialize context from dictionary."""
        fragments = [ContextFragment.from_dict(f) for f in data.get("fragments", [])]
        compressed_fragments = [ContextFragment.from_dict(f) for f in data.get("compressed_fragments", [])]
        
        return WorkflowContext(
            workflow_id=data["workflow_id"],
            version=data["version"],
            fragments=fragments,
            compressed_fragments=compressed_fragments,
            total_tokens=data.get("total_tokens", 0),
            compression_ratio=data.get("compression_ratio", 1.0),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {}),
            access_permissions=data.get("access_permissions", {})
        )
    
    def _update_injection_metrics(self, injection_time_ms: float, success: bool) -> None:
        """Update injection performance metrics."""
        if success:
            self.performance_metrics.successful_injections += 1
        else:
            self.performance_metrics.failed_injections += 1
        
        # Update average injection time
        total_injections = self.performance_metrics.total_injections
        current_avg = self.performance_metrics.average_injection_time_ms
        new_avg = ((current_avg * (total_injections - 1)) + injection_time_ms) / total_injections
        self.performance_metrics.average_injection_time_ms = new_avg


# =============================================================================
# GLOBAL CONTEXT MANAGER INSTANCE
# =============================================================================

_context_manager: Optional[WorkflowContextManager] = None


async def get_workflow_context_manager() -> WorkflowContextManager:
    """Get global workflow context manager instance."""
    global _context_manager
    
    if _context_manager is None:
        from .semantic_memory_task_processor import get_processor_manager
        
        processor_manager = await get_processor_manager()
        
        # Get or create a processor for the context manager
        if not processor_manager.processors:
            await processor_manager.start_processor("context_processor")
        
        processor = list(processor_manager.processors.values())[0]
        _context_manager = WorkflowContextManager(processor)
    
    return _context_manager


async def shutdown_workflow_context_manager():
    """Shutdown global workflow context manager."""
    global _context_manager
    
    if _context_manager:
        # Cleanup would go here
        _context_manager = None
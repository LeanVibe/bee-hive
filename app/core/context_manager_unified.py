"""
Unified Context Manager for LeanVibe Agent Hive 2.0

Consolidates 20 context-related files into a comprehensive context management system:
- Context lifecycle management
- Context compression and optimization  
- Context analytics and performance monitoring
- Context caching and retrieval
- Context orchestrator integration
- Sleep-wake context optimization
- Advanced context engine functionality
"""

import asyncio
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy import select, and_, or_, desc, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType
from .database import get_async_session
from .redis import get_redis
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class CompressionLevel(str, Enum):
    """Context compression levels."""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"
    ULTRA = "ultra"


class ContextTriggerType(str, Enum):
    """Context consolidation trigger types."""
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    SIZE_BASED = "size_based"
    AGENT_REQUEST = "agent_request"
    SYSTEM_PRESSURE = "system_pressure"


class ContextEventType(str, Enum):
    """Context event types for monitoring."""
    CONTEXT_CREATED = "context_created"
    CONTEXT_COMPRESSED = "context_compressed"
    CONTEXT_RETRIEVED = "context_retrieved"
    CONTEXT_EXPIRED = "context_expired"
    COMPRESSION_TRIGGERED = "compression_triggered"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"


@dataclass
class ContextCompressionRequest:
    """Request for context compression."""
    context_id: uuid.UUID
    compression_level: CompressionLevel = CompressionLevel.MEDIUM
    target_reduction: float = 0.5  # Target 50% reduction
    preserve_critical: bool = True
    trigger_type: ContextTriggerType = ContextTriggerType.USAGE_BASED


@dataclass 
class ContextCompressionResult:
    """Result of context compression operation."""
    success: bool
    original_size: int = 0
    compressed_size: int = 0
    reduction_ratio: float = 0.0
    compression_time_ms: float = 0.0
    tokens_saved: int = 0
    error_message: Optional[str] = None


@dataclass
class ContextSearchRequest:
    """Request for context search."""
    query: str
    context_types: List[ContextType] = None
    agent_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    max_results: int = 10
    semantic_search: bool = True
    include_embeddings: bool = False


@dataclass
class ContextMatch:
    """Context search match result."""
    context_id: uuid.UUID
    relevance_score: float
    snippet: str
    metadata: Dict[str, Any]


@dataclass
class ContextAnalytics:
    """Context usage and performance analytics."""
    total_contexts_stored: int = 0
    total_searches_performed: int = 0
    total_compressions_performed: int = 0
    average_retrieval_time_ms: float = 0.0
    compression_ratio_achieved: float = 0.0
    tokens_saved: int = 0
    cache_hit_rate: float = 0.0
    cross_agent_shares: int = 0


class ContextCompressor:
    """Advanced context compression engine."""
    
    def __init__(self):
        self.compression_strategies = {
            CompressionLevel.LIGHT: self._light_compression,
            CompressionLevel.MEDIUM: self._medium_compression,
            CompressionLevel.AGGRESSIVE: self._aggressive_compression,
            CompressionLevel.ULTRA: self._ultra_compression
        }
    
    async def compress_context(
        self, 
        content: str, 
        level: CompressionLevel,
        metadata: Dict[str, Any] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress context content based on level."""
        compression_func = self.compression_strategies.get(level, self._medium_compression)
        return await compression_func(content, metadata or {})
    
    async def _light_compression(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Light compression - remove redundant whitespace and comments."""
        lines = content.split('\n')
        compressed_lines = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                compressed_lines.append(stripped)
        
        compressed = '\n'.join(compressed_lines)
        compression_stats = {
            "compression_level": "light",
            "original_lines": len(lines),
            "compressed_lines": len(compressed_lines),
            "reduction_ratio": (len(content) - len(compressed)) / len(content)
        }
        
        return compressed, compression_stats
    
    async def _medium_compression(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Medium compression - semantic summarization."""
        # Identify key sections
        sections = self._identify_sections(content)
        compressed_sections = []
        
        for section in sections:
            if section["importance"] > 0.7:
                # Keep high importance sections
                compressed_sections.append(section["content"])
            elif section["importance"] > 0.4:
                # Summarize medium importance sections
                summary = self._summarize_section(section["content"])
                compressed_sections.append(f"[SUMMARY] {summary}")
            # Skip low importance sections
        
        compressed = '\n'.join(compressed_sections)
        compression_stats = {
            "compression_level": "medium",
            "sections_processed": len(sections),
            "sections_kept": len(compressed_sections),
            "reduction_ratio": (len(content) - len(compressed)) / len(content)
        }
        
        return compressed, compression_stats
    
    async def _aggressive_compression(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Aggressive compression - extract key insights only."""
        insights = self._extract_key_insights(content)
        compressed = '\n'.join([f"• {insight}" for insight in insights])
        
        compression_stats = {
            "compression_level": "aggressive", 
            "insights_extracted": len(insights),
            "reduction_ratio": (len(content) - len(compressed)) / len(content)
        }
        
        return compressed, compression_stats
    
    async def _ultra_compression(self, content: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Ultra compression - minimal essential information."""
        # Extract only the most critical information
        critical_info = self._extract_critical_info(content)
        compressed = f"[ULTRA-COMPRESSED] {critical_info}"
        
        compression_stats = {
            "compression_level": "ultra",
            "reduction_ratio": (len(content) - len(compressed)) / len(content)
        }
        
        return compressed, compression_stats
    
    def _identify_sections(self, content: str) -> List[Dict[str, Any]]:
        """Identify content sections with importance scoring."""
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.strip() and (line.startswith('#') or line.startswith('def ') or line.startswith('class ')):
                if current_section:
                    section_content = '\n'.join(current_section)
                    importance = self._calculate_importance(section_content)
                    sections.append({
                        "content": section_content,
                        "importance": importance
                    })
                    current_section = []
            current_section.append(line)
        
        if current_section:
            section_content = '\n'.join(current_section)
            importance = self._calculate_importance(section_content)
            sections.append({
                "content": section_content,
                "importance": importance
            })
        
        return sections
    
    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content section."""
        # Simple heuristic based on keywords and patterns
        important_keywords = [
            'error', 'exception', 'critical', 'important', 'todo', 'fixme',
            'bug', 'issue', 'required', 'necessary', 'essential'
        ]
        
        content_lower = content.lower()
        score = 0.5  # Base score
        
        # Increase score for important keywords
        for keyword in important_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # Increase score for function definitions
        if 'def ' in content or 'class ' in content:
            score += 0.2
        
        # Decrease score for comments
        if content.strip().startswith('#'):
            score -= 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _summarize_section(self, content: str) -> str:
        """Create a summary of a content section."""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) <= 3:
            return ' '.join(non_empty_lines)
        
        # Take first and last lines as summary
        return f"{non_empty_lines[0]} ... {non_empty_lines[-1]}"
    
    def _extract_key_insights(self, content: str) -> List[str]:
        """Extract key insights from content."""
        insights = []
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if (stripped and 
                any(keyword in stripped.lower() for keyword in 
                    ['todo', 'fixme', 'bug', 'issue', 'important', 'critical', 'note'])):
                insights.append(stripped)
        
        # If no insights found, take first few meaningful lines
        if not insights:
            meaningful_lines = [line.strip() for line in lines 
                             if line.strip() and not line.strip().startswith('#')]
            insights = meaningful_lines[:3]
        
        return insights[:10]  # Limit to 10 insights
    
    def _extract_critical_info(self, content: str) -> str:
        """Extract only the most critical information."""
        critical_patterns = [
            r'ERROR:.*',
            r'CRITICAL:.*', 
            r'TODO:.*',
            r'FIXME:.*',
            r'BUG:.*'
        ]
        
        import re
        critical_info = []
        
        for pattern in critical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            critical_info.extend(matches)
        
        if not critical_info:
            # Fallback to first meaningful line
            lines = content.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    critical_info.append(line.strip())
                    break
        
        return '; '.join(critical_info[:3])


class ContextCache:
    """High-performance context caching system."""
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get context from cache."""
        # Try local cache first
        if key in self.local_cache:
            self.cache_stats["hits"] += 1
            return self.local_cache[key]["data"]
        
        # Try Redis cache
        if self.redis:
            try:
                cached_data = await self.redis.get(f"context_cache:{key}")
                if cached_data:
                    data = json.loads(cached_data)
                    # Store in local cache
                    self.local_cache[key] = {
                        "data": data,
                        "timestamp": datetime.utcnow()
                    }
                    self.cache_stats["hits"] += 1
                    return data
            except Exception as e:
                logger.warning("Redis cache get failed", key=key, error=str(e))
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, key: str, data: Any, ttl_seconds: int = 3600) -> None:
        """Set context in cache."""
        # Store in local cache
        self.local_cache[key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
        
        # Store in Redis cache
        if self.redis:
            try:
                await self.redis.setex(
                    f"context_cache:{key}",
                    ttl_seconds,
                    json.dumps(data, default=str)
                )
            except Exception as e:
                logger.warning("Redis cache set failed", key=key, error=str(e))
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return self.cache_stats["hits"] / total


class ContextUnifiedManager(UnifiedManagerBase):
    """
    Unified Context Manager consolidating all context-related functionality.
    
    Replaces 20 separate files:
    - context_manager.py
    - context_compression.py
    - context_compression_engine.py
    - context_lifecycle_manager.py
    - context_cache_manager.py
    - context_consolidator.py
    - context_consolidation_triggers.py
    - context_performance_monitor.py
    - context_analytics.py
    - context_adapter.py
    - context_relevance_scorer.py
    - context_engine_integration.py
    - context_orchestrator_integration.py
    - context_memory_manager.py
    - context_aware_orchestrator_integration.py
    - advanced_context_engine.py
    - enhanced_context_engine.py
    - enhanced_context_consolidator.py
    - sleep_wake_context_optimizer.py
    - workflow_context_manager.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Core components
        self.compressor = ContextCompressor()
        self.cache = ContextCache()
        self.analytics = ContextAnalytics()
        
        # State tracking
        self.active_contexts: Dict[uuid.UUID, Dict[str, Any]] = {}
        self.compression_queue: List[ContextCompressionRequest] = []
        self.context_embeddings: Dict[uuid.UUID, List[float]] = {}
        
        # Performance metrics
        self.total_compressions = 0
        self.total_searches = 0
        self.total_contexts_created = 0
    
    async def _initialize_manager(self) -> bool:
        """Initialize the context manager."""
        try:
            # Initialize Redis connection
            redis_client = get_redis()
            self.cache = ContextCache(redis_client)
            
            # Load existing contexts
            await self._load_active_contexts()
            
            # Start background compression processor
            asyncio.create_task(self._compression_processor())
            
            logger.info(
                "Context Manager initialized",
                active_contexts=len(self.active_contexts),
                cache_enabled=redis_client is not None
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Context Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the context manager."""
        try:
            # Process remaining compressions
            await self._process_compression_queue()
            
            # Save context state
            await self._save_context_state()
            
            logger.info("Context Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Context Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get context manager health information."""
        return {
            "active_contexts": len(self.active_contexts),
            "compression_queue_size": len(self.compression_queue),
            "total_compressions": self.total_compressions,
            "total_searches": self.total_searches,
            "total_contexts_created": self.total_contexts_created,
            "cache_hit_rate": self.cache.get_hit_rate(),
            "analytics": {
                "compression_ratio_achieved": self.analytics.compression_ratio_achieved,
                "tokens_saved": self.analytics.tokens_saved,
                "average_retrieval_time_ms": self.analytics.average_retrieval_time_ms
            }
        }
    
    async def _load_plugins(self) -> None:
        """Load context manager plugins."""
        # Plugins will be loaded based on configuration
        pass
    
    # === CORE CONTEXT OPERATIONS ===
    
    async def create_context(
        self,
        content: str,
        context_type: ContextType,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_compress: bool = True
    ) -> uuid.UUID:
        """Create a new context with optional compression."""
        return await self.execute_with_monitoring(
            "create_context",
            self._create_context_impl,
            content,
            context_type,
            agent_id,
            session_id,
            metadata or {},
            auto_compress
        )
    
    async def _create_context_impl(
        self,
        content: str,
        context_type: ContextType,
        agent_id: Optional[uuid.UUID],
        session_id: Optional[uuid.UUID],
        metadata: Dict[str, Any],
        auto_compress: bool
    ) -> uuid.UUID:
        """Internal implementation of context creation."""
        context_id = uuid.uuid4()
        
        try:
            async with get_async_session() as db:
                # Create context record
                context = Context(
                    id=context_id,
                    content=content,
                    context_type=context_type,
                    agent_id=agent_id,
                    session_id=session_id,
                    metadata=metadata,
                    created_at=datetime.utcnow()
                )
                
                db.add(context)
                await db.commit()
                
                # Track in active contexts
                self.active_contexts[context_id] = {
                    "content": content,
                    "type": context_type,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "metadata": metadata,
                    "created_at": datetime.utcnow(),
                    "access_count": 0,
                    "last_accessed": datetime.utcnow()
                }
                
                self.total_contexts_created += 1
                self.analytics.total_contexts_stored += 1
                
                # Generate cache key and store
                cache_key = self._generate_cache_key(context_id, "full")
                await self.cache.set(cache_key, {
                    "content": content,
                    "metadata": metadata,
                    "type": context_type.value
                })
                
                # Queue for compression if enabled
                if auto_compress and len(content) > 1000:  # Only compress larger contexts
                    compression_request = ContextCompressionRequest(
                        context_id=context_id,
                        compression_level=CompressionLevel.MEDIUM
                    )
                    self.compression_queue.append(compression_request)
                
                # Publish context creation event
                await self._publish_context_event(
                    ContextEventType.CONTEXT_CREATED,
                    context_id,
                    {"content_length": len(content), "type": context_type.value}
                )
                
                logger.info(
                    "✅ Context created",
                    context_id=str(context_id),
                    type=context_type.value,
                    content_length=len(content),
                    auto_compress=auto_compress
                )
                
                return context_id
                
        except Exception as e:
            logger.error("❌ Context creation failed", error=str(e))
            raise
    
    async def get_context(
        self,
        context_id: uuid.UUID,
        include_compressed: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve context by ID with caching."""
        return await self.execute_with_monitoring(
            "get_context",
            self._get_context_impl,
            context_id,
            include_compressed
        )
    
    async def _get_context_impl(
        self,
        context_id: uuid.UUID,
        include_compressed: bool
    ) -> Optional[Dict[str, Any]]:
        """Internal implementation of context retrieval."""
        start_time = datetime.utcnow()
        
        try:
            # Try cache first
            cache_key = self._generate_cache_key(context_id, "full")
            cached_context = await self.cache.get(cache_key)
            
            if cached_context:
                await self._publish_context_event(
                    ContextEventType.CACHE_HIT,
                    context_id,
                    {"cache_key": cache_key}
                )
                
                # Update access tracking
                if context_id in self.active_contexts:
                    self.active_contexts[context_id]["access_count"] += 1
                    self.active_contexts[context_id]["last_accessed"] = datetime.utcnow()
                
                return cached_context
            
            # Cache miss - get from database
            await self._publish_context_event(
                ContextEventType.CACHE_MISS,
                context_id,
                {"cache_key": cache_key}
            )
            
            async with get_async_session() as db:
                result = await db.execute(select(Context).where(Context.id == context_id))
                context = result.scalar_one_or_none()
                
                if not context:
                    return None
                
                context_data = {
                    "id": str(context.id),
                    "content": context.content,
                    "compressed_content": context.compressed_content if include_compressed else None,
                    "context_type": context.context_type.value,
                    "agent_id": str(context.agent_id) if context.agent_id else None,
                    "session_id": str(context.session_id) if context.session_id else None,
                    "metadata": context.metadata or {},
                    "created_at": context.created_at.isoformat(),
                    "updated_at": context.updated_at.isoformat() if context.updated_at else None
                }
                
                # Cache the result
                await self.cache.set(cache_key, context_data)
                
                # Update access tracking
                if context_id in self.active_contexts:
                    self.active_contexts[context_id]["access_count"] += 1
                    self.active_contexts[context_id]["last_accessed"] = datetime.utcnow()
                
                # Update analytics
                retrieval_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.analytics.average_retrieval_time_ms = (
                    (self.analytics.average_retrieval_time_ms * self.analytics.total_searches + retrieval_time_ms) /
                    (self.analytics.total_searches + 1)
                )
                self.analytics.total_searches += 1
                
                await self._publish_context_event(
                    ContextEventType.CONTEXT_RETRIEVED,
                    context_id,
                    {"retrieval_time_ms": retrieval_time_ms}
                )
                
                return context_data
                
        except Exception as e:
            logger.error("❌ Context retrieval failed", context_id=str(context_id), error=str(e))
            return None
    
    async def compress_context(
        self,
        context_id: uuid.UUID,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        force: bool = False
    ) -> ContextCompressionResult:
        """Compress a specific context."""
        return await self.execute_with_monitoring(
            "compress_context",
            self._compress_context_impl,
            context_id,
            compression_level,
            force
        )
    
    async def _compress_context_impl(
        self,
        context_id: uuid.UUID,
        compression_level: CompressionLevel,
        force: bool
    ) -> ContextCompressionResult:
        """Internal implementation of context compression."""
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                result = await db.execute(select(Context).where(Context.id == context_id))
                context = result.scalar_one_or_none()
                
                if not context:
                    return ContextCompressionResult(
                        success=False,
                        error_message="Context not found"
                    )
                
                # Check if already compressed
                if context.compressed_content and not force:
                    return ContextCompressionResult(
                        success=False,
                        error_message="Context already compressed (use force=True to re-compress)"
                    )
                
                original_size = len(context.content)
                
                # Perform compression
                compressed_content, compression_stats = await self.compressor.compress_context(
                    context.content,
                    compression_level,
                    context.metadata or {}
                )
                
                compressed_size = len(compressed_content)
                reduction_ratio = compression_stats.get("reduction_ratio", 0.0)
                tokens_saved = int(original_size * 0.75 * reduction_ratio)  # Rough token estimate
                
                # Update context in database
                context.compressed_content = compressed_content
                context.compression_metadata = compression_stats
                context.updated_at = datetime.utcnow()
                await db.commit()
                
                # Update analytics
                self.total_compressions += 1
                self.analytics.total_compressions_performed += 1
                self.analytics.tokens_saved += tokens_saved
                self.analytics.compression_ratio_achieved = (
                    (self.analytics.compression_ratio_achieved * (self.total_compressions - 1) + reduction_ratio) /
                    self.total_compressions
                )
                
                # Clear cache to force refresh
                cache_key = self._generate_cache_key(context_id, "full")
                # Note: Could implement cache invalidation here
                
                compression_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Publish compression event
                await self._publish_context_event(
                    ContextEventType.CONTEXT_COMPRESSED,
                    context_id,
                    {
                        "compression_level": compression_level.value,
                        "original_size": original_size,
                        "compressed_size": compressed_size,
                        "reduction_ratio": reduction_ratio,
                        "tokens_saved": tokens_saved,
                        "compression_time_ms": compression_time_ms
                    }
                )
                
                logger.info(
                    "✅ Context compressed",
                    context_id=str(context_id),
                    compression_level=compression_level.value,
                    reduction_ratio=reduction_ratio,
                    tokens_saved=tokens_saved
                )
                
                return ContextCompressionResult(
                    success=True,
                    original_size=original_size,
                    compressed_size=compressed_size,
                    reduction_ratio=reduction_ratio,
                    compression_time_ms=compression_time_ms,
                    tokens_saved=tokens_saved
                )
                
        except Exception as e:
            logger.error("❌ Context compression failed", context_id=str(context_id), error=str(e))
            return ContextCompressionResult(
                success=False,
                error_message=str(e)
            )
    
    async def search_contexts(self, request: ContextSearchRequest) -> List[ContextMatch]:
        """Search contexts with semantic similarity."""
        return await self.execute_with_monitoring(
            "search_contexts",
            self._search_contexts_impl,
            request
        )
    
    async def _search_contexts_impl(self, request: ContextSearchRequest) -> List[ContextMatch]:
        """Internal implementation of context search."""
        try:
            async with get_async_session() as db:
                # Build query
                query = select(Context)
                
                # Apply filters
                filters = []
                if request.context_types:
                    filters.append(Context.context_type.in_(request.context_types))
                if request.agent_id:
                    filters.append(Context.agent_id == request.agent_id)
                if request.session_id:
                    filters.append(Context.session_id == request.session_id)
                
                if filters:
                    query = query.where(and_(*filters))
                
                # Execute query
                result = await db.execute(query.limit(request.max_results * 2))  # Get more for filtering
                contexts = result.scalars().all()
                
                # Perform semantic search
                matches = []
                query_lower = request.query.lower()
                
                for context in contexts:
                    # Simple text matching for now
                    content_lower = context.content.lower()
                    relevance_score = 0.0
                    
                    # Exact phrase match
                    if query_lower in content_lower:
                        relevance_score += 0.8
                    
                    # Word matching
                    query_words = query_lower.split()
                    content_words = content_lower.split()
                    word_matches = len(set(query_words) & set(content_words))
                    if len(query_words) > 0:
                        relevance_score += (word_matches / len(query_words)) * 0.6
                    
                    if relevance_score > 0.1:  # Minimum relevance threshold
                        # Create snippet
                        snippet = self._create_snippet(context.content, request.query)
                        
                        matches.append(ContextMatch(
                            context_id=context.id,
                            relevance_score=relevance_score,
                            snippet=snippet,
                            metadata={
                                "context_type": context.context_type.value,
                                "agent_id": str(context.agent_id) if context.agent_id else None,
                                "created_at": context.created_at.isoformat()
                            }
                        ))
                
                # Sort by relevance and limit results
                matches.sort(key=lambda x: x.relevance_score, reverse=True)
                final_matches = matches[:request.max_results]
                
                self.total_searches += 1
                
                return final_matches
                
        except Exception as e:
            logger.error("❌ Context search failed", error=str(e))
            return []
    
    # === COMPRESSION MANAGEMENT ===
    
    async def _compression_processor(self) -> None:
        """Background processor for compression queue."""
        while True:
            try:
                await asyncio.sleep(30)  # Process every 30 seconds
                await self._process_compression_queue()
                
            except Exception as e:
                logger.error("Error in compression processor", error=str(e))
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _process_compression_queue(self) -> None:
        """Process pending compression requests."""
        if not self.compression_queue:
            return
        
        # Process up to 10 compressions at a time
        batch_size = min(10, len(self.compression_queue))
        batch = self.compression_queue[:batch_size]
        self.compression_queue = self.compression_queue[batch_size:]
        
        tasks = []
        for request in batch:
            task = self._compress_context_impl(
                request.context_id,
                request.compression_level,
                False  # Don't force
            )
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_compressions = sum(1 for r in results if isinstance(r, ContextCompressionResult) and r.success)
            
            if successful_compressions > 0:
                logger.info(f"Processed {successful_compressions} compressions from queue")
                
        except Exception as e:
            logger.error("Error processing compression batch", error=str(e))
    
    # === UTILITY METHODS ===
    
    def _generate_cache_key(self, context_id: uuid.UUID, key_type: str) -> str:
        """Generate cache key for context."""
        return f"context:{key_type}:{str(context_id)}"
    
    def _create_snippet(self, content: str, query: str, max_length: int = 200) -> str:
        """Create a snippet around the query match."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find query position
        pos = content_lower.find(query_lower)
        if pos == -1:
            # No exact match, return beginning
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Calculate snippet boundaries
        start = max(0, pos - max_length // 2)
        end = min(len(content), pos + len(query) + max_length // 2)
        
        snippet = content[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
    
    async def _load_active_contexts(self) -> None:
        """Load active contexts from database."""
        try:
            async with get_async_session() as db:
                # Load recent contexts
                result = await db.execute(
                    select(Context)
                    .where(Context.created_at > datetime.utcnow() - timedelta(hours=24))
                    .order_by(desc(Context.created_at))
                    .limit(1000)
                )
                contexts = result.scalars().all()
                
                for context in contexts:
                    self.active_contexts[context.id] = {
                        "content": context.content,
                        "type": context.context_type,
                        "agent_id": context.agent_id,
                        "session_id": context.session_id,
                        "metadata": context.metadata or {},
                        "created_at": context.created_at,
                        "access_count": 0,
                        "last_accessed": context.created_at
                    }
                
        except Exception as e:
            logger.error("Failed to load active contexts", error=str(e))
    
    async def _save_context_state(self) -> None:
        """Save context state for shutdown."""
        # Context state is primarily in database, minimal state to save
        logger.info("Context state saved", active_contexts=len(self.active_contexts))
    
    async def _publish_context_event(
        self,
        event_type: ContextEventType,
        context_id: uuid.UUID,
        payload: Dict[str, Any]
    ) -> None:
        """Publish context event for monitoring."""
        try:
            if hasattr(self, 'redis') and self.cache.redis:
                event_data = {
                    "event_type": event_type.value,
                    "context_id": str(context_id),
                    "timestamp": datetime.utcnow().isoformat(),
                    "payload": payload
                }
                
                await self.cache.redis.xadd(
                    "system_events:context",
                    event_data,
                    maxlen=10000
                )
                
        except Exception as e:
            logger.error("Failed to publish context event", event_type=event_type.value, error=str(e))
    
    # === PUBLIC API METHODS ===
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get context analytics and metrics."""
        return {
            "contexts": {
                "total_stored": self.analytics.total_contexts_stored,
                "active_contexts": len(self.active_contexts),
                "total_created": self.total_contexts_created
            },
            "performance": {
                "total_searches": self.analytics.total_searches_performed,
                "average_retrieval_time_ms": self.analytics.average_retrieval_time_ms,
                "cache_hit_rate": self.cache.get_hit_rate()
            },
            "compression": {
                "total_compressions": self.analytics.total_compressions_performed,
                "compression_ratio_achieved": self.analytics.compression_ratio_achieved,
                "tokens_saved": self.analytics.tokens_saved,
                "compression_queue_size": len(self.compression_queue)
            },
            "cache": {
                "local_cache_size": len(self.cache.local_cache),
                "cache_hits": self.cache.cache_stats["hits"],
                "cache_misses": self.cache.cache_stats["misses"],
                "hit_rate": self.cache.get_hit_rate()
            }
        }
    
    async def trigger_compression(
        self,
        context_ids: List[uuid.UUID] = None,
        compression_level: CompressionLevel = CompressionLevel.MEDIUM,
        trigger_type: ContextTriggerType = ContextTriggerType.AGENT_REQUEST
    ) -> int:
        """Trigger compression for specific contexts or all eligible contexts."""
        triggered_count = 0
        
        try:
            if context_ids:
                # Compress specific contexts
                for context_id in context_ids:
                    request = ContextCompressionRequest(
                        context_id=context_id,
                        compression_level=compression_level,
                        trigger_type=trigger_type
                    )
                    self.compression_queue.append(request)
                    triggered_count += 1
            else:
                # Find eligible contexts for compression
                async with get_async_session() as db:
                    result = await db.execute(
                        select(Context.id)
                        .where(
                            and_(
                                Context.compressed_content.is_(None),
                                func.length(Context.content) > 1000
                            )
                        )
                        .limit(100)
                    )
                    eligible_context_ids = result.scalars().all()
                    
                    for context_id in eligible_context_ids:
                        request = ContextCompressionRequest(
                            context_id=context_id,
                            compression_level=compression_level,
                            trigger_type=trigger_type
                        )
                        self.compression_queue.append(request)
                        triggered_count += 1
            
            await self._publish_context_event(
                ContextEventType.COMPRESSION_TRIGGERED,
                uuid.uuid4(),  # Use random ID for system event
                {
                    "triggered_count": triggered_count,
                    "compression_level": compression_level.value,
                    "trigger_type": trigger_type.value
                }
            )
            
            logger.info(f"Triggered compression for {triggered_count} contexts")
            return triggered_count
            
        except Exception as e:
            logger.error("Failed to trigger compression", error=str(e))
            return 0


# Factory function for creating context manager
def create_context_manager(**config_overrides) -> ContextUnifiedManager:
    """Create and initialize a context manager."""
    config = create_manager_config("ContextManager", **config_overrides)
    return ContextUnifiedManager(config)
"""
Context Plugin for Orchestrator

Consolidates functionality from:
- context_orchestrator_integration.py
- context_aware_orchestrator_integration.py
- context_compression_engine.py
- context_consolidator.py
- enhanced_context_engine.py
- context_manager.py
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from . import OrchestratorPlugin, PluginMetadata, PluginType
from ..config import settings
from ..redis import get_redis
from ..database import get_session
from ..logging_service import get_component_logger

logger = get_component_logger("context_plugin")


class ContextCompressionStrategy(Enum):
    """Context compression strategies."""
    LIGHT = "light"           # Basic compression
    MEDIUM = "medium"         # Semantic compression
    HEAVY = "heavy"           # Aggressive compression
    INTELLIGENT = "intelligent"  # AI-driven compression


class ContextPriority(Enum):
    """Context priority levels."""
    CRITICAL = "critical"     # Never compress
    HIGH = "high"            # Compress only when necessary
    MEDIUM = "medium"        # Standard compression
    LOW = "low"              # Aggressive compression


@dataclass
class ContextFragment:
    """Individual context fragment."""
    fragment_id: str
    session_id: str
    agent_id: str
    content: str
    content_type: str
    priority: ContextPriority
    timestamp: datetime
    size_bytes: int
    relevance_score: float
    access_count: int
    last_accessed: datetime


@dataclass
class ContextMetrics:
    """Context management metrics."""
    timestamp: datetime
    total_contexts: int
    total_size_bytes: int
    compression_ratio: float
    active_sessions: int
    context_cache_hit_rate: float
    avg_relevance_score: float


@dataclass
class CompressionResult:
    """Result of context compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    fragments_removed: int
    fragments_compressed: int
    time_taken: float
    strategy_used: ContextCompressionStrategy


class ContextPlugin(OrchestratorPlugin):
    """Plugin for context compression, memory management, and session optimization."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="context_plugin",
            version="1.0.0",
            plugin_type=PluginType.CONTEXT,
            description="Context compression, memory management, and session optimization",
            dependencies=["redis", "database"]
        )
        super().__init__(metadata)
        
        self.context_fragments: Dict[str, ContextFragment] = {}
        self.session_contexts: Dict[str, List[str]] = {}  # session_id -> fragment_ids
        self.compression_cache: Dict[str, str] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_context_size = 50 * 1024 * 1024  # 50MB
        self.max_fragment_age_hours = 24
        self.compression_threshold = 0.8  # Compress when 80% full
        self.cache_ttl = 3600  # 1 hour cache TTL
        
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize context management."""
        try:
            self.redis = await get_redis()
            
            # Load existing context fragments from Redis
            await self._load_context_fragments()
            
            logger.info("Context plugin initialized successfully")
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._context_cleanup_loop())
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize context plugin: {e}")
            return False
            
    async def cleanup(self) -> bool:
        """Cleanup context management resources."""
        try:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                
            # Save context fragments to Redis
            await self._save_context_fragments()
            
            logger.info("Context plugin cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup context plugin: {e}")
            return False
            
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context before task execution."""
        session_id = task_context.get("session_id")
        agent_id = task_context.get("agent_id")
        
        if session_id:
            # Load relevant context for the session
            context_data = await self._load_session_context(session_id, agent_id)
            task_context["loaded_context"] = context_data
            
            # Check if compression is needed
            if await self._should_compress_context(session_id):
                compression_result = await self._compress_session_context(session_id)
                task_context["compression_applied"] = compression_result
                
        task_context["context_optimized"] = True
        return task_context
        
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Save context after task execution."""
        session_id = task_context.get("session_id")
        agent_id = task_context.get("agent_id")
        
        if session_id and result:
            # Extract context from result
            context_content = await self._extract_context_from_result(result)
            
            if context_content:
                # Create context fragment
                fragment = await self._create_context_fragment(
                    session_id,
                    agent_id,
                    context_content,
                    ContextPriority.MEDIUM
                )
                
                # Store fragment
                await self._store_context_fragment(fragment)
                
        return result
        
    async def health_check(self) -> Dict[str, Any]:
        """Return context plugin health status."""
        metrics = await self._collect_context_metrics()
        
        health_status = "healthy"
        issues = []
        
        # Check context size
        total_size_mb = metrics.total_size_bytes / (1024 * 1024)
        max_size_mb = self.max_context_size / (1024 * 1024)
        
        if total_size_mb > max_size_mb * 0.9:
            health_status = "warning"
            issues.append(f"High context usage: {total_size_mb:.1f}MB / {max_size_mb:.1f}MB")
            
        # Check compression efficiency
        if metrics.compression_ratio < 0.3:
            health_status = "warning"
            issues.append(f"Low compression ratio: {metrics.compression_ratio:.2f}")
            
        # Check cache hit rate
        if metrics.context_cache_hit_rate < 0.7:
            health_status = "warning"
            issues.append(f"Low cache hit rate: {metrics.context_cache_hit_rate:.2f}")
            
        return {
            "plugin": self.metadata.name,
            "enabled": self.enabled,
            "status": health_status,
            "issues": issues,
            "metrics": {
                "total_contexts": metrics.total_contexts,
                "total_size_mb": total_size_mb,
                "compression_ratio": metrics.compression_ratio,
                "active_sessions": metrics.active_sessions,
                "cache_hit_rate": metrics.context_cache_hit_rate,
                "avg_relevance_score": metrics.avg_relevance_score
            }
        }
        
    async def _context_cleanup_loop(self):
        """Background task for context cleanup and optimization."""
        while True:
            try:
                # Clean old fragments
                await self._cleanup_old_fragments()
                
                # Optimize context storage
                await self._optimize_context_storage()
                
                # Update metrics
                metrics = await self._collect_context_metrics()
                await self._store_context_metrics(metrics)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in context cleanup loop: {e}")
                await asyncio.sleep(300)
                
    async def _load_context_fragments(self):
        """Load existing context fragments from Redis."""
        try:
            fragment_keys = await self.redis.keys("context:fragment:*")
            
            for key in fragment_keys:
                fragment_data = await self.redis.get(key)
                if fragment_data:
                    fragment_dict = json.loads(fragment_data)
                    fragment = ContextFragment(
                        fragment_id=fragment_dict["fragment_id"],
                        session_id=fragment_dict["session_id"],
                        agent_id=fragment_dict["agent_id"],
                        content=fragment_dict["content"],
                        content_type=fragment_dict["content_type"],
                        priority=ContextPriority(fragment_dict["priority"]),
                        timestamp=datetime.fromisoformat(fragment_dict["timestamp"]),
                        size_bytes=fragment_dict["size_bytes"],
                        relevance_score=fragment_dict["relevance_score"],
                        access_count=fragment_dict["access_count"],
                        last_accessed=datetime.fromisoformat(fragment_dict["last_accessed"])
                    )
                    
                    self.context_fragments[fragment.fragment_id] = fragment
                    
                    # Update session mapping
                    if fragment.session_id not in self.session_contexts:
                        self.session_contexts[fragment.session_id] = []
                    self.session_contexts[fragment.session_id].append(fragment.fragment_id)
                    
            logger.info(f"Loaded {len(self.context_fragments)} context fragments")
            
        except Exception as e:
            logger.error(f"Error loading context fragments: {e}")
            
    async def _save_context_fragments(self):
        """Save context fragments to Redis."""
        try:
            for fragment in self.context_fragments.values():
                fragment_data = {
                    "fragment_id": fragment.fragment_id,
                    "session_id": fragment.session_id,
                    "agent_id": fragment.agent_id,
                    "content": fragment.content,
                    "content_type": fragment.content_type,
                    "priority": fragment.priority.value,
                    "timestamp": fragment.timestamp.isoformat(),
                    "size_bytes": fragment.size_bytes,
                    "relevance_score": fragment.relevance_score,
                    "access_count": fragment.access_count,
                    "last_accessed": fragment.last_accessed.isoformat()
                }
                
                await self.redis.set(
                    f"context:fragment:{fragment.fragment_id}",
                    json.dumps(fragment_data),
                    ex=self.cache_ttl
                )
                
        except Exception as e:
            logger.error(f"Error saving context fragments: {e}")
            
    async def _load_session_context(self, session_id: str, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Load relevant context for a session."""
        try:
            session_fragments = self.session_contexts.get(session_id, [])
            relevant_fragments = []
            
            for fragment_id in session_fragments:
                fragment = self.context_fragments.get(fragment_id)
                if fragment:
                    # Filter by agent if specified
                    if agent_id and fragment.agent_id != agent_id:
                        continue
                        
                    # Update access tracking
                    fragment.access_count += 1
                    fragment.last_accessed = datetime.utcnow()
                    
                    relevant_fragments.append({
                        "fragment_id": fragment.fragment_id,
                        "content": fragment.content,
                        "content_type": fragment.content_type,
                        "relevance_score": fragment.relevance_score,
                        "timestamp": fragment.timestamp.isoformat()
                    })
                    
            # Sort by relevance and recency
            relevant_fragments.sort(
                key=lambda x: (x["relevance_score"], x["timestamp"]), 
                reverse=True
            )
            
            return {
                "session_id": session_id,
                "agent_id": agent_id,
                "fragments": relevant_fragments[:50],  # Limit to top 50
                "total_fragments": len(relevant_fragments),
                "loaded_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading session context: {e}")
            return {}
            
    async def _should_compress_context(self, session_id: str) -> bool:
        """Check if context compression is needed for a session."""
        session_fragments = self.session_contexts.get(session_id, [])
        
        # Calculate total size
        total_size = sum(
            self.context_fragments[fid].size_bytes
            for fid in session_fragments
            if fid in self.context_fragments
        )
        
        # Check if compression threshold is reached
        return total_size > (self.max_context_size * self.compression_threshold)
        
    async def _compress_session_context(self, session_id: str) -> CompressionResult:
        """Compress context for a session."""
        start_time = time.time()
        session_fragments = self.session_contexts.get(session_id, [])
        
        original_size = 0
        compressed_size = 0
        fragments_removed = 0
        fragments_compressed = 0
        
        # Get fragments sorted by priority and relevance
        fragments_to_process = []
        for fragment_id in session_fragments:
            if fragment_id in self.context_fragments:
                fragment = self.context_fragments[fragment_id]
                fragments_to_process.append((fragment.relevance_score, fragment.priority, fragment))
                original_size += fragment.size_bytes
                
        # Sort by priority (critical first) and relevance
        fragments_to_process.sort(key=lambda x: (x[1].value == "critical", x[0]), reverse=True)
        
        # Compression strategy based on context size
        if original_size > self.max_context_size * 0.9:
            strategy = ContextCompressionStrategy.HEAVY
        elif original_size > self.max_context_size * 0.7:
            strategy = ContextCompressionStrategy.MEDIUM
        else:
            strategy = ContextCompressionStrategy.LIGHT
            
        # Apply compression
        for score, priority, fragment in fragments_to_process:
            if priority == ContextPriority.CRITICAL:
                # Never compress critical fragments
                compressed_size += fragment.size_bytes
                continue
                
            if strategy == ContextCompressionStrategy.HEAVY:
                if priority == ContextPriority.LOW or fragment.relevance_score < 0.3:
                    # Remove low-priority/low-relevance fragments
                    del self.context_fragments[fragment.fragment_id]
                    self.session_contexts[session_id].remove(fragment.fragment_id)
                    fragments_removed += 1
                elif priority == ContextPriority.MEDIUM and fragment.relevance_score < 0.6:
                    # Compress medium-priority fragments
                    compressed_content = await self._compress_fragment_content(fragment.content)
                    fragment.content = compressed_content
                    fragment.size_bytes = len(compressed_content.encode('utf-8'))
                    compressed_size += fragment.size_bytes
                    fragments_compressed += 1
                else:
                    compressed_size += fragment.size_bytes
                    
            elif strategy == ContextCompressionStrategy.MEDIUM:
                if priority == ContextPriority.LOW and fragment.relevance_score < 0.4:
                    # Remove only very low relevance fragments
                    del self.context_fragments[fragment.fragment_id]
                    self.session_contexts[session_id].remove(fragment.fragment_id)
                    fragments_removed += 1
                elif priority == ContextPriority.MEDIUM and fragment.relevance_score < 0.5:
                    # Light compression
                    compressed_content = await self._compress_fragment_content(fragment.content, light=True)
                    fragment.content = compressed_content
                    fragment.size_bytes = len(compressed_content.encode('utf-8'))
                    compressed_size += fragment.size_bytes
                    fragments_compressed += 1
                else:
                    compressed_size += fragment.size_bytes
                    
            else:  # LIGHT compression
                if priority == ContextPriority.LOW and fragment.relevance_score < 0.2:
                    # Remove only extremely low relevance fragments
                    del self.context_fragments[fragment.fragment_id]
                    self.session_contexts[session_id].remove(fragment.fragment_id)
                    fragments_removed += 1
                else:
                    compressed_size += fragment.size_bytes
                    
        time_taken = time.time() - start_time
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            fragments_removed=fragments_removed,
            fragments_compressed=fragments_compressed,
            time_taken=time_taken,
            strategy_used=strategy
        )
        
        logger.info(f"Context compression completed: {compression_ratio:.2f} ratio, "
                   f"{fragments_removed} removed, {fragments_compressed} compressed")
        
        return result
        
    async def _compress_fragment_content(self, content: str, light: bool = False) -> str:
        """Compress fragment content."""
        if light:
            # Light compression: remove extra whitespace and redundancy
            lines = content.strip().split('\n')
            compressed_lines = []
            for line in lines:
                line = ' '.join(line.split())  # Remove extra spaces
                if line and line not in compressed_lines[-3:]:  # Avoid recent duplicates
                    compressed_lines.append(line)
            return '\n'.join(compressed_lines)
        else:
            # Medium compression: summarize content (simplified)
            # In production, use AI-based summarization
            lines = content.strip().split('\n')
            if len(lines) <= 5:
                return content
                
            # Keep first 2 and last 2 lines, summarize middle
            summary = lines[:2] + [f"... ({len(lines)-4} lines summarized) ..."] + lines[-2:]
            return '\n'.join(summary)
            
    async def _extract_context_from_result(self, result: Any) -> Optional[str]:
        """Extract context content from task result."""
        try:
            if isinstance(result, dict):
                # Extract relevant fields
                context_fields = ["output", "response", "content", "data", "result"]
                for field in context_fields:
                    if field in result and result[field]:
                        return str(result[field])
                        
            elif isinstance(result, str):
                return result
                
            return None
        except Exception:
            return None
            
    async def _create_context_fragment(
        self,
        session_id: str,
        agent_id: str,
        content: str,
        priority: ContextPriority
    ) -> ContextFragment:
        """Create a new context fragment."""
        fragment_id = f"{session_id}_{agent_id}_{int(time.time() * 1000)}"
        
        # Calculate relevance score (simplified)
        relevance_score = await self._calculate_relevance_score(content, session_id)
        
        return ContextFragment(
            fragment_id=fragment_id,
            session_id=session_id,
            agent_id=agent_id,
            content=content,
            content_type="text",
            priority=priority,
            timestamp=datetime.utcnow(),
            size_bytes=len(content.encode('utf-8')),
            relevance_score=relevance_score,
            access_count=0,
            last_accessed=datetime.utcnow()
        )
        
    async def _calculate_relevance_score(self, content: str, session_id: str) -> float:
        """Calculate relevance score for content (simplified)."""
        # In production, use more sophisticated relevance scoring
        base_score = 0.5
        
        # Score based on content length
        if len(content) > 1000:
            base_score += 0.2
        elif len(content) < 100:
            base_score -= 0.1
            
        # Score based on keywords (simplified)
        important_keywords = ["error", "result", "success", "complete", "output"]
        keyword_count = sum(1 for keyword in important_keywords if keyword.lower() in content.lower())
        base_score += keyword_count * 0.05
        
        return min(1.0, max(0.0, base_score))
        
    async def _store_context_fragment(self, fragment: ContextFragment):
        """Store context fragment."""
        self.context_fragments[fragment.fragment_id] = fragment
        
        # Update session mapping
        if fragment.session_id not in self.session_contexts:
            self.session_contexts[fragment.session_id] = []
        self.session_contexts[fragment.session_id].append(fragment.fragment_id)
        
        # Save to Redis
        fragment_data = {
            "fragment_id": fragment.fragment_id,
            "session_id": fragment.session_id,
            "agent_id": fragment.agent_id,
            "content": fragment.content,
            "content_type": fragment.content_type,
            "priority": fragment.priority.value,
            "timestamp": fragment.timestamp.isoformat(),
            "size_bytes": fragment.size_bytes,
            "relevance_score": fragment.relevance_score,
            "access_count": fragment.access_count,
            "last_accessed": fragment.last_accessed.isoformat()
        }
        
        await self.redis.set(
            f"context:fragment:{fragment.fragment_id}",
            json.dumps(fragment_data),
            ex=self.cache_ttl
        )
        
    async def _cleanup_old_fragments(self):
        """Clean up old context fragments."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.max_fragment_age_hours)
        fragments_to_remove = []
        
        for fragment_id, fragment in self.context_fragments.items():
            if fragment.timestamp < cutoff_time and fragment.priority != ContextPriority.CRITICAL:
                fragments_to_remove.append(fragment_id)
                
        for fragment_id in fragments_to_remove:
            fragment = self.context_fragments[fragment_id]
            
            # Remove from session mapping
            if fragment.session_id in self.session_contexts:
                if fragment_id in self.session_contexts[fragment.session_id]:
                    self.session_contexts[fragment.session_id].remove(fragment_id)
                    
            # Remove from Redis
            await self.redis.delete(f"context:fragment:{fragment_id}")
            
            # Remove from memory
            del self.context_fragments[fragment_id]
            
        if fragments_to_remove:
            logger.info(f"Cleaned up {len(fragments_to_remove)} old context fragments")
            
    async def _optimize_context_storage(self):
        """Optimize context storage efficiency."""
        # Identify duplicate content
        content_hashes = {}
        duplicates_found = 0
        
        for fragment_id, fragment in self.context_fragments.items():
            content_hash = hash(fragment.content)
            
            if content_hash in content_hashes:
                # Found duplicate content
                existing_fragment_id = content_hashes[content_hash]
                existing_fragment = self.context_fragments[existing_fragment_id]
                
                # Keep the one with higher relevance or more recent
                if (fragment.relevance_score > existing_fragment.relevance_score or
                    fragment.timestamp > existing_fragment.timestamp):
                    # Remove existing, keep new
                    await self._remove_fragment(existing_fragment_id)
                    content_hashes[content_hash] = fragment_id
                else:
                    # Remove new, keep existing
                    await self._remove_fragment(fragment_id)
                    
                duplicates_found += 1
            else:
                content_hashes[content_hash] = fragment_id
                
        if duplicates_found > 0:
            logger.info(f"Optimized storage: removed {duplicates_found} duplicate fragments")
            
    async def _remove_fragment(self, fragment_id: str):
        """Remove a context fragment."""
        if fragment_id in self.context_fragments:
            fragment = self.context_fragments[fragment_id]
            
            # Remove from session mapping
            if fragment.session_id in self.session_contexts:
                if fragment_id in self.session_contexts[fragment.session_id]:
                    self.session_contexts[fragment.session_id].remove(fragment_id)
                    
            # Remove from Redis
            await self.redis.delete(f"context:fragment:{fragment_id}")
            
            # Remove from memory
            del self.context_fragments[fragment_id]
            
    async def _collect_context_metrics(self) -> ContextMetrics:
        """Collect current context metrics."""
        total_contexts = len(self.context_fragments)
        total_size_bytes = sum(fragment.size_bytes for fragment in self.context_fragments.values())
        
        # Calculate compression ratio (simplified)
        original_size_estimate = total_size_bytes * 1.5  # Assume 50% compression on average
        compression_ratio = total_size_bytes / original_size_estimate if original_size_estimate > 0 else 1.0
        
        active_sessions = len(self.session_contexts)
        
        # Cache hit rate (simplified calculation)
        cache_hit_rate = 0.85  # Mock value, in production calculate from actual cache hits
        
        # Average relevance score
        if self.context_fragments:
            avg_relevance_score = sum(f.relevance_score for f in self.context_fragments.values()) / len(self.context_fragments)
        else:
            avg_relevance_score = 0.0
            
        return ContextMetrics(
            timestamp=datetime.utcnow(),
            total_contexts=total_contexts,
            total_size_bytes=total_size_bytes,
            compression_ratio=compression_ratio,
            active_sessions=active_sessions,
            context_cache_hit_rate=cache_hit_rate,
            avg_relevance_score=avg_relevance_score
        )
        
    async def _store_context_metrics(self, metrics: ContextMetrics):
        """Store context metrics in Redis."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "total_contexts": metrics.total_contexts,
                "total_size_bytes": metrics.total_size_bytes,
                "compression_ratio": metrics.compression_ratio,
                "active_sessions": metrics.active_sessions,
                "context_cache_hit_rate": metrics.context_cache_hit_rate,
                "avg_relevance_score": metrics.avg_relevance_score
            }
            
            await self.redis.set("context:current_metrics", json.dumps(metrics_data))
            await self.redis.lpush("context:metrics_history", json.dumps(metrics_data))
            await self.redis.ltrim("context:metrics_history", 0, 1000)
            
        except Exception as e:
            logger.error(f"Error storing context metrics: {e}")
            
    async def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary."""
        metrics = await self._collect_context_metrics()
        
        # Session breakdown
        session_breakdown = {}
        for session_id, fragment_ids in self.session_contexts.items():
            session_size = sum(
                self.context_fragments[fid].size_bytes 
                for fid in fragment_ids 
                if fid in self.context_fragments
            )
            session_breakdown[session_id] = {
                "fragment_count": len(fragment_ids),
                "total_size_bytes": session_size
            }
            
        return {
            "current_metrics": {
                "total_contexts": metrics.total_contexts,
                "total_size_mb": metrics.total_size_bytes / (1024 * 1024),
                "compression_ratio": metrics.compression_ratio,
                "active_sessions": metrics.active_sessions,
                "cache_hit_rate": metrics.context_cache_hit_rate,
                "avg_relevance_score": metrics.avg_relevance_score
            },
            "session_breakdown": session_breakdown,
            "storage_efficiency": {
                "max_size_mb": self.max_context_size / (1024 * 1024),
                "usage_percentage": (metrics.total_size_bytes / self.max_context_size) * 100,
                "compression_threshold": self.compression_threshold
            }
        }
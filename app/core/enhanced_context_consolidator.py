"""
Enhanced Context Consolidator - UltraCompressed Context Mode Implementation.

Provides advanced context consolidation with real-time compression achieving 70% token reduction
through intelligent semantic clustering, hierarchical compression, and adaptive context merging.

Key Features:
- Real-time context compression with 70% target reduction
- Semantic similarity clustering for intelligent merging
- Hierarchical compression levels based on context importance
- Adaptive threshold management based on usage patterns
- Memory-efficient streaming consolidation
- Integration with existing context infrastructure
"""

import asyncio
import logging
import statistics
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncIterator
from uuid import UUID, uuid4
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete, desc
from sqlalchemy.orm import selectinload

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.context_manager import ContextManager
from ..core.context_consolidator import ContextConsolidator, ConsolidationResult, get_context_consolidator
from ..core.context_compression import ContextCompressor, CompressionLevel, get_context_compressor
from ..core.vector_search_engine import VectorSearchEngine, SearchFilters
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Advanced compression strategies for different context types."""
    AGGRESSIVE_MERGE = "aggressive_merge"        # 80-90% reduction, high information loss
    SEMANTIC_CLUSTER = "semantic_cluster"        # 70-80% reduction, semantic preservation
    HIERARCHICAL_COMPRESS = "hierarchical_compress"  # 60-70% reduction, structure preservation
    INTELLIGENT_SUMMARY = "intelligent_summary"  # 50-60% reduction, meaning preservation
    LIGHT_OPTIMIZATION = "light_optimization"   # 30-40% reduction, minimal loss


class ContextPriority(Enum):
    """Context priority levels for compression decisions."""
    CRITICAL = "critical"      # Never compress aggressively
    HIGH = "high"             # Conservative compression only
    MEDIUM = "medium"         # Standard compression
    LOW = "low"              # Aggressive compression allowed
    DISPOSABLE = "disposable" # Can be heavily compressed or archived


@dataclass
class CompressionMetrics:
    """Metrics for compression operations."""
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 0.0
    semantic_similarity_loss: float = 0.0
    processing_time_ms: float = 0.0
    contexts_merged: int = 0
    contexts_archived: int = 0
    strategy_used: CompressionStrategy = CompressionStrategy.LIGHT_OPTIMIZATION
    
    def calculate_efficiency_score(self) -> float:
        """Calculate compression efficiency score (0.0-1.0)."""
        compression_score = self.compression_ratio * 0.4
        speed_score = max(0, 1.0 - (self.processing_time_ms / 10000)) * 0.3
        quality_score = max(0, 1.0 - self.semantic_similarity_loss) * 0.3
        return min(1.0, compression_score + speed_score + quality_score)


@dataclass
class ContextCluster:
    """Represents a cluster of semantically similar contexts."""
    cluster_id: str
    representative_context: Context
    similar_contexts: List[Context]
    similarity_scores: List[float]
    cluster_embedding: Optional[List[float]] = None
    priority_level: ContextPriority = ContextPriority.MEDIUM
    
    @property
    def size(self) -> int:
        return len(self.similar_contexts) + 1
    
    @property
    def avg_similarity(self) -> float:
        return statistics.mean(self.similarity_scores) if self.similarity_scores else 0.0
    
    @property
    def total_tokens(self) -> int:
        total = len(self.representative_context.content or "")
        total += sum(len(ctx.content or "") for ctx in self.similar_contexts)
        return total


class UltraCompressedContextMode:
    """
    Advanced context consolidation system achieving 70% token reduction.
    
    Features:
    - Real-time semantic clustering and compression
    - Adaptive compression strategies based on context importance
    - Hierarchical compression with quality preservation
    - Memory-efficient streaming processing
    - Integration with existing infrastructure
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        context_consolidator: Optional[ContextConsolidator] = None,
        context_compressor: Optional[ContextCompressor] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.settings = get_settings()
        # Initialize dependencies lazily to handle missing dependencies gracefully
        self.context_manager = context_manager
        self.consolidator = context_consolidator
        self.compressor = context_compressor
        self.embedding_service = embedding_service
        
        # Configuration
        self.target_compression_ratio = 0.70  # 70% reduction target
        self.similarity_threshold = 0.85      # For clustering
        self.max_cluster_size = 10            # Max contexts per cluster
        self.max_processing_time = 300        # 5 minutes max
        self.min_context_age_hours = 1        # Don't compress very recent contexts
        
        # Performance tracking
        self._compression_history: deque = deque(maxlen=1000)
        self._performance_metrics: Dict[str, Any] = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "avg_compression_ratio": 0.0,
            "avg_processing_time": 0.0,
            "strategy_distribution": defaultdict(int)
        }
        
        # Adaptive thresholds
        self._adaptive_thresholds = {
            "similarity_threshold": 0.85,
            "cluster_size_limit": 10,
            "compression_aggressiveness": 0.7
        }
        
        # Initialization flag
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the consolidator dependencies lazily."""
        if self._initialized:
            return
            
        try:
            if not self.context_manager:
                self.context_manager = ContextManager()
            
            if not self.consolidator:
                self.consolidator = get_context_consolidator()
            
            if not self.compressor:
                self.compressor = get_context_compressor()
            
            if not self.embedding_service:
                self.embedding_service = get_embedding_service()
            
            self._initialized = True
            logger.info("Enhanced Context Consolidator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Context Consolidator: {e}")
            raise
    
    async def ultra_compress_agent_contexts(
        self,
        agent_id: UUID,
        target_reduction: float = 0.70,
        preserve_critical: bool = True
    ) -> CompressionMetrics:
        """
        Perform ultra compression on agent's contexts.
        
        Args:
            agent_id: Agent ID to compress contexts for
            target_reduction: Target compression ratio (0.0-1.0)
            preserve_critical: Whether to preserve critical contexts
            
        Returns:
            CompressionMetrics with detailed results
        """
        # Ensure initialization
        await self.initialize()
        
        start_time = datetime.utcnow()
        metrics = CompressionMetrics()
        
        try:
            logger.info(f"Starting ultra compression for agent {agent_id} (target: {target_reduction:.1%})")
            
            # Phase 1: Analyze and categorize contexts
            contexts = await self._get_compressible_contexts(agent_id)
            if not contexts:
                logger.info(f"No contexts found for compression for agent {agent_id}")
                return metrics
            
            metrics.original_token_count = sum(
                len(ctx.content or "") for ctx in contexts
            )
            
            # Phase 2: Build semantic clusters
            clusters = await self._build_semantic_clusters(contexts)
            
            # Phase 3: Apply compression strategies
            compression_results = await self._apply_compression_strategies(
                clusters, target_reduction, preserve_critical
            )
            
            # Phase 4: Update database and generate new contexts
            await self._commit_compression_results(agent_id, compression_results)
            
            # Calculate final metrics
            end_time = datetime.utcnow()
            metrics.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Calculate compressed token count from results
            metrics.compressed_token_count = sum(
                result["compressed_tokens"] for result in compression_results
            )
            
            metrics.compression_ratio = 1 - (
                metrics.compressed_token_count / max(1, metrics.original_token_count)
            )
            
            metrics.contexts_merged = sum(
                result["contexts_merged"] for result in compression_results
            )
            
            metrics.contexts_archived = sum(
                result["contexts_archived"] for result in compression_results
            )
            
            # Update performance tracking
            self._update_performance_metrics(metrics)
            
            logger.info(
                f"Ultra compression completed for agent {agent_id}: "
                f"{metrics.original_token_count} → {metrics.compressed_token_count} tokens "
                f"({metrics.compression_ratio:.1%} reduction) in {metrics.processing_time_ms:.0f}ms"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ultra compression failed for agent {agent_id}: {e}")
            metrics.processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return metrics
    
    async def real_time_compression_stream(
        self,
        agent_id: UUID,
        compression_interval_minutes: int = 30
    ) -> AsyncIterator[CompressionMetrics]:
        """
        Continuous real-time compression stream.
        
        Args:
            agent_id: Agent ID to monitor
            compression_interval_minutes: How often to compress
            
        Yields:
            CompressionMetrics for each compression cycle
        """
        try:
            logger.info(f"Starting real-time compression stream for agent {agent_id}")
            
            while True:
                try:
                    # Check if compression is needed
                    if await self._should_compress(agent_id):
                        metrics = await self.ultra_compress_agent_contexts(agent_id)
                        yield metrics
                    
                    # Wait for next interval
                    await asyncio.sleep(compression_interval_minutes * 60)
                    
                except asyncio.CancelledError:
                    logger.info(f"Real-time compression stream cancelled for agent {agent_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in compression stream for agent {agent_id}: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Fatal error in compression stream for agent {agent_id}: {e}")
    
    async def adaptive_threshold_optimization(self, agent_id: UUID) -> Dict[str, float]:
        """
        Optimize compression thresholds based on agent behavior.
        
        Args:
            agent_id: Agent ID to optimize for
            
        Returns:
            Dictionary of optimized thresholds
        """
        try:
            # Analyze recent compression history
            recent_compressions = [
                metrics for metrics in self._compression_history
                if metrics.processing_time_ms > 0  # Valid compressions only
            ][-10:]  # Last 10 compressions
            
            if not recent_compressions:
                return self._adaptive_thresholds.copy()
            
            # Calculate optimization metrics
            avg_efficiency = statistics.mean(
                metrics.calculate_efficiency_score() for metrics in recent_compressions
            )
            avg_compression_ratio = statistics.mean(
                metrics.compression_ratio for metrics in recent_compressions
            )
            avg_processing_time = statistics.mean(
                metrics.processing_time_ms for metrics in recent_compressions
            )
            
            # Adjust thresholds based on performance
            new_thresholds = self._adaptive_thresholds.copy()
            
            # If efficiency is low, reduce aggressiveness
            if avg_efficiency < 0.6:
                new_thresholds["similarity_threshold"] = min(0.95, new_thresholds["similarity_threshold"] + 0.05)
                new_thresholds["compression_aggressiveness"] = max(0.4, new_thresholds["compression_aggressiveness"] - 0.1)
            
            # If compression ratio is below target, increase aggressiveness
            elif avg_compression_ratio < self.target_compression_ratio:
                new_thresholds["similarity_threshold"] = max(0.75, new_thresholds["similarity_threshold"] - 0.05)
                new_thresholds["compression_aggressiveness"] = min(0.9, new_thresholds["compression_aggressiveness"] + 0.1)
            
            # If processing time is too high, reduce cluster size
            if avg_processing_time > 60000:  # 60 seconds
                new_thresholds["cluster_size_limit"] = max(5, new_thresholds["cluster_size_limit"] - 1)
            elif avg_processing_time < 10000:  # 10 seconds
                new_thresholds["cluster_size_limit"] = min(15, new_thresholds["cluster_size_limit"] + 1)
            
            self._adaptive_thresholds = new_thresholds
            
            logger.info(f"Optimized thresholds for agent {agent_id}: {new_thresholds}")
            return new_thresholds
            
        except Exception as e:
            logger.error(f"Error optimizing thresholds for agent {agent_id}: {e}")
            return self._adaptive_thresholds.copy()
    
    async def get_compression_analytics(self, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get detailed compression analytics."""
        try:
            analytics = {
                "performance_metrics": self._performance_metrics.copy(),
                "adaptive_thresholds": self._adaptive_thresholds.copy(),
                "recent_compressions": []
            }
            
            # Add recent compression details
            recent_compressions = list(self._compression_history)[-20:]  # Last 20
            for metrics in recent_compressions:
                analytics["recent_compressions"].append({
                    "compression_ratio": metrics.compression_ratio,
                    "efficiency_score": metrics.calculate_efficiency_score(),
                    "processing_time_ms": metrics.processing_time_ms,
                    "contexts_merged": metrics.contexts_merged,
                    "strategy_used": metrics.strategy_used.value
                })
            
            # Agent-specific analytics if requested
            if agent_id:
                async with get_async_session() as session:
                    # Get context statistics
                    total_contexts = await session.scalar(
                        select(func.count(Context.id)).where(Context.agent_id == agent_id)
                    )
                    
                    compressed_contexts = await session.scalar(
                        select(func.count(Context.id)).where(
                            and_(
                                Context.agent_id == agent_id,
                                Context.is_consolidated == True
                            )
                        )
                    )
                    
                    analytics["agent_specific"] = {
                        "agent_id": str(agent_id),
                        "total_contexts": total_contexts or 0,
                        "compressed_contexts": compressed_contexts or 0,
                        "compression_coverage": (compressed_contexts or 0) / max(1, total_contexts or 1)
                    }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting compression analytics: {e}")
            return {"error": str(e)}
    
    async def _get_compressible_contexts(self, agent_id: UUID) -> List[Context]:
        """Get contexts that are candidates for compression."""
        try:
            async with get_async_session() as session:
                # Get contexts older than minimum age and not already compressed
                cutoff_time = datetime.utcnow() - timedelta(hours=self.min_context_age_hours)
                
                result = await session.execute(
                    select(Context).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.created_at < cutoff_time,
                            Context.is_consolidated == False,
                            Context.is_archived == False,
                            Context.content.isnot(None),
                            func.length(Context.content) > 100  # Only compress substantial content
                        )
                    ).order_by(desc(Context.created_at))
                    .limit(500)  # Process in batches
                )
                
                return list(result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting compressible contexts for agent {agent_id}: {e}")
            return []
    
    async def _build_semantic_clusters(self, contexts: List[Context]) -> List[ContextCluster]:
        """Build semantic clusters from contexts."""
        try:
            clusters = []
            processed_ids = set()
            
            for i, context in enumerate(contexts):
                if context.id in processed_ids:
                    continue
                
                # Create new cluster with this context as representative
                cluster = ContextCluster(
                    cluster_id=str(uuid4()),
                    representative_context=context,
                    similar_contexts=[],
                    similarity_scores=[]
                )
                
                # Find similar contexts
                for j, other_context in enumerate(contexts[i+1:], i+1):
                    if other_context.id in processed_ids:
                        continue
                    
                    if len(cluster.similar_contexts) >= self._adaptive_thresholds["cluster_size_limit"]:
                        break
                    
                    # Calculate semantic similarity
                    similarity = await self._calculate_semantic_similarity(context, other_context)
                    
                    if similarity >= self._adaptive_thresholds["similarity_threshold"]:
                        cluster.similar_contexts.append(other_context)
                        cluster.similarity_scores.append(similarity)
                        processed_ids.add(other_context.id)
                
                # Determine cluster priority
                cluster.priority_level = self._determine_context_priority(cluster)
                
                if cluster.size >= 2:  # Only include clusters with multiple contexts
                    clusters.append(cluster)
                    processed_ids.add(context.id)
            
            logger.info(f"Built {len(clusters)} semantic clusters from {len(contexts)} contexts")
            return clusters
            
        except Exception as e:
            logger.error(f"Error building semantic clusters: {e}")
            return []
    
    async def _calculate_semantic_similarity(self, context1: Context, context2: Context) -> float:
        """Calculate semantic similarity between two contexts."""
        try:
            if not context1.content or not context2.content:
                return 0.0
            
            # Use embeddings for semantic similarity if available
            if context1.embedding and context2.embedding:
                import numpy as np
                embedding1 = np.array(context1.embedding)
                embedding2 = np.array(context2.embedding)
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                return float(similarity)
            
            # Fallback to content-based similarity
            return self._calculate_content_similarity(context1.content, context2.content)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using text analysis."""
        try:
            # Simple word overlap similarity
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    def _determine_context_priority(self, cluster: ContextCluster) -> ContextPriority:
        """Determine priority level for a context cluster."""
        try:
            representative = cluster.representative_context
            
            # Check for critical context types
            if representative.context_type in [ContextType.DECISION, ContextType.ERROR_RESOLUTION]:
                return ContextPriority.CRITICAL
            
            # Check importance score
            if representative.importance_score and representative.importance_score > 0.8:
                return ContextPriority.HIGH
            
            # Check recent access
            if representative.access_count and representative.access_count > 10:
                return ContextPriority.HIGH
            
            # Check age
            if representative.created_at:
                age_days = (datetime.utcnow() - representative.created_at).days
                if age_days > 30:
                    return ContextPriority.LOW
                elif age_days > 7:
                    return ContextPriority.MEDIUM
            
            return ContextPriority.MEDIUM
            
        except Exception as e:
            logger.error(f"Error determining context priority: {e}")
            return ContextPriority.MEDIUM
    
    async def _apply_compression_strategies(
        self,
        clusters: List[ContextCluster],
        target_reduction: float,
        preserve_critical: bool
    ) -> List[Dict[str, Any]]:
        """Apply appropriate compression strategies to clusters."""
        try:
            results = []
            
            for cluster in clusters:
                strategy = self._select_compression_strategy(cluster, target_reduction, preserve_critical)
                
                if strategy == CompressionStrategy.AGGRESSIVE_MERGE:
                    result = await self._aggressive_merge_cluster(cluster)
                elif strategy == CompressionStrategy.SEMANTIC_CLUSTER:
                    result = await self._semantic_cluster_compress(cluster)
                elif strategy == CompressionStrategy.HIERARCHICAL_COMPRESS:
                    result = await self._hierarchical_compress_cluster(cluster)
                elif strategy == CompressionStrategy.INTELLIGENT_SUMMARY:
                    result = await self._intelligent_summary_compress(cluster)
                else:  # LIGHT_OPTIMIZATION
                    result = await self._light_optimization_compress(cluster)
                
                result["strategy_used"] = strategy
                result["cluster_id"] = cluster.cluster_id
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying compression strategies: {e}")
            return []
    
    def _select_compression_strategy(
        self,
        cluster: ContextCluster,
        target_reduction: float,
        preserve_critical: bool
    ) -> CompressionStrategy:
        """Select appropriate compression strategy for a cluster."""
        try:
            # Never compress critical contexts aggressively
            if preserve_critical and cluster.priority_level == ContextPriority.CRITICAL:
                return CompressionStrategy.LIGHT_OPTIMIZATION
            
            # High priority contexts get conservative compression
            if cluster.priority_level == ContextPriority.HIGH:
                return CompressionStrategy.INTELLIGENT_SUMMARY
            
            # Consider cluster characteristics
            if cluster.size >= 8 and cluster.avg_similarity >= 0.9:
                return CompressionStrategy.AGGRESSIVE_MERGE
            elif cluster.size >= 5 and cluster.avg_similarity >= 0.85:
                return CompressionStrategy.SEMANTIC_CLUSTER
            elif cluster.size >= 3:
                return CompressionStrategy.HIERARCHICAL_COMPRESS
            else:
                return CompressionStrategy.INTELLIGENT_SUMMARY
                
        except Exception as e:
            logger.error(f"Error selecting compression strategy: {e}")
            return CompressionStrategy.LIGHT_OPTIMIZATION
    
    async def _aggressive_merge_cluster(self, cluster: ContextCluster) -> Dict[str, Any]:
        """Aggressively merge cluster contexts (80-90% reduction)."""
        try:
            all_contexts = [cluster.representative_context] + cluster.similar_contexts
            
            # Extract key information from all contexts
            key_points = []
            decisions = []
            
            for context in all_contexts:
                if context.content:
                    # Simple extraction - in production, use LLM
                    lines = context.content.split('\n')
                    key_points.extend([line.strip() for line in lines if len(line.strip()) > 50])
            
            # Create ultra-compressed summary
            compressed_content = f"Consolidated {len(all_contexts)} related contexts:\n"
            compressed_content += "\n".join(key_points[:5])  # Top 5 key points
            
            original_tokens = sum(len(ctx.content or "") for ctx in all_contexts)
            compressed_tokens = len(compressed_content)
            
            return {
                "compressed_content": compressed_content,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "contexts_merged": len(all_contexts),
                "contexts_archived": 0,
                "compression_ratio": 1 - (compressed_tokens / max(1, original_tokens))
            }
            
        except Exception as e:
            logger.error(f"Error in aggressive merge: {e}")
            return {"compressed_content": "", "original_tokens": 0, "compressed_tokens": 0, "contexts_merged": 0, "contexts_archived": 0, "compression_ratio": 0.0}
    
    async def _semantic_cluster_compress(self, cluster: ContextCluster) -> Dict[str, Any]:
        """Compress cluster using semantic analysis (70-80% reduction)."""
        try:
            all_contexts = [cluster.representative_context] + cluster.similar_contexts
            
            # Combine all content
            combined_content = "\n\n---\n\n".join(
                ctx.content for ctx in all_contexts if ctx.content
            )
            
            # Use existing compressor for semantic compression
            compressed_result = await self.compressor.compress_conversation(
                conversation_content=combined_content,
                compression_level=CompressionLevel.AGGRESSIVE,
                context_type=cluster.representative_context.context_type
            )
            
            return {
                "compressed_content": compressed_result.summary,
                "original_tokens": compressed_result.original_token_count,
                "compressed_tokens": compressed_result.compressed_token_count,
                "contexts_merged": len(all_contexts),
                "contexts_archived": 0,
                "compression_ratio": compressed_result.compression_ratio,
                "key_insights": compressed_result.key_insights,
                "decisions_made": compressed_result.decisions_made
            }
            
        except Exception as e:
            logger.error(f"Error in semantic cluster compression: {e}")
            return {"compressed_content": "", "original_tokens": 0, "compressed_tokens": 0, "contexts_merged": 0, "contexts_archived": 0, "compression_ratio": 0.0}
    
    async def _hierarchical_compress_cluster(self, cluster: ContextCluster) -> Dict[str, Any]:
        """Compress cluster hierarchically (60-70% reduction)."""
        try:
            all_contexts = [cluster.representative_context] + cluster.similar_contexts
            
            # Group contexts by importance
            high_importance = [ctx for ctx in all_contexts if (ctx.importance_score or 0) > 0.7]
            medium_importance = [ctx for ctx in all_contexts if 0.3 <= (ctx.importance_score or 0) <= 0.7]
            low_importance = [ctx for ctx in all_contexts if (ctx.importance_score or 0) < 0.3]
            
            compressed_parts = []
            
            # Preserve high importance contexts with light compression
            for ctx in high_importance:
                if ctx.content:
                    light_compressed = await self.compressor.compress_conversation(
                        ctx.content, CompressionLevel.LIGHT, ctx.context_type
                    )
                    compressed_parts.append(f"[HIGH] {light_compressed.summary}")
            
            # Medium compression for medium importance
            if medium_importance:
                combined_medium = "\n".join(ctx.content for ctx in medium_importance if ctx.content)
                if combined_medium:
                    medium_compressed = await self.compressor.compress_conversation(
                        combined_medium, CompressionLevel.STANDARD
                    )
                    compressed_parts.append(f"[MEDIUM] {medium_compressed.summary}")
            
            # Aggressive compression for low importance
            if low_importance:
                combined_low = "\n".join(ctx.content for ctx in low_importance if ctx.content)
                if combined_low:
                    low_compressed = await self.compressor.compress_conversation(
                        combined_low, CompressionLevel.AGGRESSIVE
                    )
                    compressed_parts.append(f"[LOW] {low_compressed.summary}")
            
            compressed_content = "\n\n".join(compressed_parts)
            original_tokens = sum(len(ctx.content or "") for ctx in all_contexts)
            compressed_tokens = len(compressed_content)
            
            return {
                "compressed_content": compressed_content,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "contexts_merged": len(all_contexts),
                "contexts_archived": 0,
                "compression_ratio": 1 - (compressed_tokens / max(1, original_tokens))
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical compression: {e}")
            return {"compressed_content": "", "original_tokens": 0, "compressed_tokens": 0, "contexts_merged": 0, "contexts_archived": 0, "compression_ratio": 0.0}
    
    async def _intelligent_summary_compress(self, cluster: ContextCluster) -> Dict[str, Any]:
        """Compress cluster using intelligent summarization (50-60% reduction)."""
        try:
            all_contexts = [cluster.representative_context] + cluster.similar_contexts
            
            # Use standard compression level for balance
            combined_content = "\n\n".join(ctx.content for ctx in all_contexts if ctx.content)
            
            compressed_result = await self.compressor.compress_conversation(
                conversation_content=combined_content,
                compression_level=CompressionLevel.STANDARD,
                context_type=cluster.representative_context.context_type,
                preserve_decisions=True,
                preserve_patterns=True
            )
            
            return {
                "compressed_content": compressed_result.summary,
                "original_tokens": compressed_result.original_token_count,
                "compressed_tokens": compressed_result.compressed_token_count,
                "contexts_merged": len(all_contexts),
                "contexts_archived": 0,
                "compression_ratio": compressed_result.compression_ratio,
                "key_insights": compressed_result.key_insights,
                "decisions_made": compressed_result.decisions_made,
                "patterns_identified": compressed_result.patterns_identified
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent summary compression: {e}")
            return {"compressed_content": "", "original_tokens": 0, "compressed_tokens": 0, "contexts_merged": 0, "contexts_archived": 0, "compression_ratio": 0.0}
    
    async def _light_optimization_compress(self, cluster: ContextCluster) -> Dict[str, Any]:
        """Light optimization compression (30-40% reduction)."""
        try:
            all_contexts = [cluster.representative_context] + cluster.similar_contexts
            
            # Light compression preserving most content
            combined_content = "\n\n".join(ctx.content for ctx in all_contexts if ctx.content)
            
            compressed_result = await self.compressor.compress_conversation(
                conversation_content=combined_content,
                compression_level=CompressionLevel.LIGHT,
                context_type=cluster.representative_context.context_type
            )
            
            return {
                "compressed_content": compressed_result.summary,
                "original_tokens": compressed_result.original_token_count,
                "compressed_tokens": compressed_result.compressed_token_count,
                "contexts_merged": len(all_contexts),
                "contexts_archived": 0,
                "compression_ratio": compressed_result.compression_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in light optimization compression: {e}")
            return {"compressed_content": "", "original_tokens": 0, "compressed_tokens": 0, "contexts_merged": 0, "contexts_archived": 0, "compression_ratio": 0.0}
    
    async def _commit_compression_results(
        self,
        agent_id: UUID,
        compression_results: List[Dict[str, Any]]
    ) -> None:
        """Commit compression results to database."""
        try:
            async with get_async_session() as session:
                for result in compression_results:
                    if result["contexts_merged"] > 0:
                        # Create new compressed context
                        compressed_context = Context(
                            agent_id=agent_id,
                            content=result["compressed_content"],
                            context_type=ContextType.CONVERSATION,  # Default type
                            is_consolidated=True,
                            importance_score=0.8,  # High importance for compressed content
                            metadata={
                                "compression": {
                                    "strategy": result.get("strategy_used", "unknown"),
                                    "original_tokens": result["original_tokens"],
                                    "compressed_tokens": result["compressed_tokens"],
                                    "compression_ratio": result["compression_ratio"],
                                    "contexts_merged": result["contexts_merged"],
                                    "compressed_at": datetime.utcnow().isoformat(),
                                    "cluster_id": result.get("cluster_id")
                                }
                            }
                        )
                        
                        # Add insights if available
                        if "key_insights" in result:
                            compressed_context.metadata["key_insights"] = result["key_insights"]
                        if "decisions_made" in result:
                            compressed_context.metadata["decisions_made"] = result["decisions_made"]
                        if "patterns_identified" in result:
                            compressed_context.metadata["patterns_identified"] = result["patterns_identified"]
                        
                        session.add(compressed_context)
                        await session.flush()
                        
                        # Generate embedding for compressed context
                        await self.context_manager.generate_embedding(compressed_context.id)
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error committing compression results: {e}")
            await session.rollback()
    
    async def _should_compress(self, agent_id: UUID) -> bool:
        """Determine if compression is needed for an agent."""
        try:
            async with get_async_session() as session:
                # Count uncompressed contexts
                uncompressed_count = await session.scalar(
                    select(func.count(Context.id)).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == False,
                            Context.created_at < datetime.utcnow() - timedelta(hours=self.min_context_age_hours)
                        )
                    )
                )
                
                # Compress if there are enough uncompressed contexts
                return (uncompressed_count or 0) >= 10
                
        except Exception as e:
            logger.error(f"Error checking compression need for agent {agent_id}: {e}")
            return False
    
    def _update_performance_metrics(self, metrics: CompressionMetrics) -> None:
        """Update performance tracking metrics."""
        try:
            self._compression_history.append(metrics)
            
            # Update aggregated metrics
            self._performance_metrics["total_compressions"] += 1
            self._performance_metrics["total_tokens_saved"] += (
                metrics.original_token_count - metrics.compressed_token_count
            )
            
            # Update averages
            total_compressions = self._performance_metrics["total_compressions"]
            old_avg_ratio = self._performance_metrics["avg_compression_ratio"]
            old_avg_time = self._performance_metrics["avg_processing_time"]
            
            self._performance_metrics["avg_compression_ratio"] = (
                (old_avg_ratio * (total_compressions - 1) + metrics.compression_ratio) / total_compressions
            )
            
            self._performance_metrics["avg_processing_time"] = (
                (old_avg_time * (total_compressions - 1) + metrics.processing_time_ms) / total_compressions
            )
            
            # Update strategy distribution
            self._performance_metrics["strategy_distribution"][metrics.strategy_used.value] += 1
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")


# Global instance for application use
_ultra_compressed_context_mode: Optional[UltraCompressedContextMode] = None


def get_ultra_compressed_context_mode() -> UltraCompressedContextMode:
    """Get singleton ultra compressed context mode instance."""
    global _ultra_compressed_context_mode
    if _ultra_compressed_context_mode is None:
        _ultra_compressed_context_mode = UltraCompressedContextMode()
    return _ultra_compressed_context_mode


async def ultra_compress_agent_contexts(
    agent_id: UUID,
    target_reduction: float = 0.70
) -> CompressionMetrics:
    """Convenience function for ultra compressing agent contexts."""
    compressor = get_ultra_compressed_context_mode()
    return await compressor.ultra_compress_agent_contexts(agent_id, target_reduction)


async def start_real_time_compression(
    agent_id: UUID,
    interval_minutes: int = 30
) -> AsyncIterator[CompressionMetrics]:
    """Convenience function for starting real-time compression."""
    compressor = get_ultra_compressed_context_mode()
    async for metrics in compressor.real_time_compression_stream(agent_id, interval_minutes):
        yield metrics
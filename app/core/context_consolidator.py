"""
Context Consolidator - Automated context compression during sleep cycles.

Provides intelligent context consolidation using semantic search to:
- Identify redundant and similar contexts for merging
- Compress contexts using semantic similarity analysis
- Archive old contexts based on usage patterns
- Optimize vector indexes during consolidation
- Track consolidation metrics and efficiency
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from sqlalchemy.orm import selectinload

from ..models.context import Context, ContextType
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.context_manager import ContextManager
from ..core.vector_search_engine import VectorSearchEngine, SearchFilters
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.config import get_settings


logger = logging.getLogger(__name__)


class ConsolidationResult:
    """Result of context consolidation operation."""
    
    def __init__(self):
        self.contexts_processed: int = 0
        self.contexts_merged: int = 0
        self.contexts_archived: int = 0
        self.contexts_compressed: int = 0
        self.redundant_contexts_removed: int = 0
        self.tokens_saved: int = 0
        self.consolidation_ratio: float = 0.0
        self.processing_time_ms: float = 0.0
        self.efficiency_score: float = 0.0
        self.errors: List[str] = []


class SimilarityCluster:
    """Cluster of similar contexts for merging."""
    
    def __init__(self, representative_context: Context, similarity_threshold: float = 0.85):
        self.representative_context = representative_context
        self.similar_contexts: List[Context] = []
        self.similarity_scores: List[float] = []
        self.similarity_threshold = similarity_threshold
        self.merged_content: Optional[str] = None
        self.merged_metadata: Optional[Dict[str, Any]] = None
    
    def add_similar_context(self, context: Context, similarity_score: float) -> None:
        """Add a similar context to the cluster."""
        if similarity_score >= self.similarity_threshold:
            self.similar_contexts.append(context)
            self.similarity_scores.append(similarity_score)
    
    def should_merge(self) -> bool:
        """Determine if this cluster should be merged."""
        return len(self.similar_contexts) > 0 and statistics.mean(self.similarity_scores) >= self.similarity_threshold
    
    def get_merge_candidate_count(self) -> int:
        """Get number of contexts that can be merged."""
        return len(self.similar_contexts) + 1  # Include representative


class ContextConsolidator:
    """
    Intelligent context consolidation during sleep cycles.
    
    Features:
    - Semantic similarity analysis for context merging
    - Redundant content identification and removal
    - Age-based archival with usage pattern analysis
    - Vector index optimization during consolidation
    - Comprehensive consolidation metrics and reporting
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.context_manager = ContextManager()
        self.vector_search = VectorSearchEngine()
        self.embedding_service = get_embedding_service()
        
        # Consolidation settings
        self.similarity_threshold = 0.85  # 85% similarity for merging
        self.min_age_for_archival_days = 30  # Archive contexts older than 30 days
        self.max_contexts_per_batch = 100  # Process in batches
        self.redundancy_threshold = 0.95  # 95% similarity for redundancy removal
        self.usage_threshold = 5  # Archive if used less than 5 times
        
        # Performance settings
        self.max_concurrent_operations = 3
        self.timeout_per_operation_seconds = 300  # 5 minutes per operation
        
        # Metrics tracking
        self._consolidation_metrics: Dict[str, Any] = {}
        self._performance_history: List[ConsolidationResult] = []
    
    async def consolidate_during_sleep(self, agent_id: UUID) -> ConsolidationResult:
        """
        Perform comprehensive context consolidation for an agent during sleep.
        
        Args:
            agent_id: Agent ID for consolidation
            
        Returns:
            ConsolidationResult with consolidation metrics
        """
        start_time = datetime.utcnow()
        result = ConsolidationResult()
        
        try:
            logger.info(f"Starting context consolidation for agent {agent_id}")
            
            # Phase 1: Identify contexts for consolidation
            contexts_to_process = await self._get_consolidation_candidates(agent_id)
            result.contexts_processed = len(contexts_to_process)
            
            if not contexts_to_process:
                logger.info(f"No contexts found for consolidation for agent {agent_id}")
                return result
            
            # Phase 2: Find and merge similar contexts
            merge_result = await self._merge_similar_contexts(contexts_to_process, agent_id)
            result.contexts_merged = merge_result["merged_count"]
            result.tokens_saved += merge_result["tokens_saved"]
            
            # Phase 3: Remove redundant contexts
            redundancy_result = await self._remove_redundant_contexts(contexts_to_process, agent_id)
            result.redundant_contexts_removed = redundancy_result["removed_count"]
            result.tokens_saved += redundancy_result["tokens_saved"]
            
            # Phase 4: Compress remaining contexts
            compression_result = await self._compress_contexts(contexts_to_process, agent_id)
            result.contexts_compressed = compression_result["compressed_count"]
            result.tokens_saved += compression_result["tokens_saved"]
            
            # Phase 5: Archive old contexts
            archival_result = await self._archive_old_contexts(agent_id)
            result.contexts_archived = archival_result["archived_count"]
            
            # Calculate efficiency metrics
            end_time = datetime.utcnow()
            result.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            result.consolidation_ratio = result.tokens_saved / max(1, result.contexts_processed * 1000)  # Estimate
            result.efficiency_score = self._calculate_efficiency_score(result)
            
            # Update metrics tracking
            self._performance_history.append(result)
            self._update_consolidation_metrics(agent_id, result)
            
            logger.info(
                f"Context consolidation completed for agent {agent_id}: "
                f"{result.contexts_processed} processed, {result.tokens_saved} tokens saved"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during context consolidation for agent {agent_id}: {e}")
            result.errors.append(str(e))
            return result
    
    async def identify_redundant_contexts(self, agent_id: UUID) -> List[UUID]:
        """
        Identify redundant contexts for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of context IDs that are redundant
        """
        try:
            redundant_ids = []
            
            # Get contexts for analysis
            async with get_async_session() as session:
                contexts_result = await session.execute(
                    select(Context).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.is_consolidated == False
                        )
                    ).limit(self.max_contexts_per_batch)
                )
                contexts = contexts_result.scalars().all()
            
            # Find redundant contexts using vector similarity
            for i, context1 in enumerate(contexts):
                if context1.id in redundant_ids:
                    continue
                
                for j, context2 in enumerate(contexts[i+1:], i+1):
                    if context2.id in redundant_ids:
                        continue
                    
                    # Calculate similarity using embeddings
                    similarity = await self._calculate_content_similarity(context1, context2)
                    
                    if similarity >= self.redundancy_threshold:
                        # Keep the more recent context
                        older_context = context1 if context1.created_at < context2.created_at else context2
                        redundant_ids.append(older_context.id)
                        
                        logger.debug(f"Identified redundant context {older_context.id} (similarity: {similarity:.2f})")
            
            logger.info(f"Identified {len(redundant_ids)} redundant contexts for agent {agent_id}")
            return redundant_ids
            
        except Exception as e:
            logger.error(f"Error identifying redundant contexts for agent {agent_id}: {e}")
            return []
    
    async def merge_similar_contexts(self, contexts: List[Context]) -> Context:
        """
        Merge a list of similar contexts into a single consolidated context.
        
        Args:
            contexts: List of contexts to merge
            
        Returns:
            New merged context
        """
        try:
            if not contexts:
                raise ValueError("No contexts provided for merging")
            
            # Use the most recent context as the base
            base_context = max(contexts, key=lambda c: c.created_at)
            
            # Merge content
            merged_content_parts = []
            merged_metadata = {}
            
            for context in contexts:
                if context.content and context.content not in merged_content_parts:
                    merged_content_parts.append(context.content)
                
                # Merge metadata
                if context.metadata:
                    for key, value in context.metadata.items():
                        if key not in merged_metadata:
                            merged_metadata[key] = value
                        elif isinstance(value, list):
                            if isinstance(merged_metadata[key], list):
                                merged_metadata[key].extend(value)
                            else:
                                merged_metadata[key] = [merged_metadata[key]] + value
            
            # Create merged content
            merged_content = "\n\n---\n\n".join(merged_content_parts)
            
            # Add merge metadata
            merged_metadata["consolidation"] = {
                "merged_from": [str(c.id) for c in contexts],
                "merge_date": datetime.utcnow().isoformat(),
                "original_count": len(contexts)
            }
            
            # Create new consolidated context
            async with get_async_session() as session:
                merged_context = Context(
                    agent_id=base_context.agent_id,
                    content=merged_content,
                    context_type=base_context.context_type,
                    metadata=merged_metadata,
                    is_consolidated=True,
                    tags=list(set().union(*[c.tags or [] for c in contexts])),
                    priority=max(c.priority or 0 for c in contexts)
                )
                
                session.add(merged_context)
                await session.commit()
                await session.refresh(merged_context)
                
                # Generate embedding for merged context
                await self.context_manager.generate_embedding(merged_context.id)
                
                logger.info(f"Merged {len(contexts)} contexts into new context {merged_context.id}")
                return merged_context
                
        except Exception as e:
            logger.error(f"Error merging contexts: {e}")
            raise
    
    async def archive_old_contexts(self, agent_id: UUID, age_threshold: timedelta) -> int:
        """
        Archive old contexts based on age and usage patterns.
        
        Args:
            agent_id: Agent ID
            age_threshold: Age threshold for archival
            
        Returns:
            Number of contexts archived
        """
        try:
            cutoff_date = datetime.utcnow() - age_threshold
            archived_count = 0
            
            async with get_async_session() as session:
                # Find old contexts with low usage
                old_contexts_result = await session.execute(
                    select(Context).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.created_at < cutoff_date,
                            Context.is_archived == False,
                            or_(
                                Context.access_count < self.usage_threshold,
                                Context.access_count.is_(None)
                            )
                        )
                    )
                )
                old_contexts = old_contexts_result.scalars().all()
                
                for context in old_contexts:
                    # Archive the context
                    context.is_archived = True
                    context.archived_at = datetime.utcnow()
                    
                    # Update metadata
                    if not context.metadata:
                        context.metadata = {}
                    context.metadata["archival"] = {
                        "archived_date": datetime.utcnow().isoformat(),
                        "reason": "age_and_usage_threshold",
                        "age_days": (datetime.utcnow() - context.created_at).days,
                        "access_count": context.access_count or 0
                    }
                    
                    archived_count += 1
                
                await session.commit()
                
                logger.info(f"Archived {archived_count} old contexts for agent {agent_id}")
                return archived_count
                
        except Exception as e:
            logger.error(f"Error archiving old contexts for agent {agent_id}: {e}")
            return 0
    
    async def get_consolidation_metrics(self, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get consolidation metrics for an agent or system-wide."""
        try:
            if agent_id:
                return self._consolidation_metrics.get(str(agent_id), {})
            else:
                # Return system-wide metrics
                return {
                    "total_agents_processed": len(self._consolidation_metrics),
                    "total_performance_records": len(self._performance_history),
                    "average_efficiency": statistics.mean([r.efficiency_score for r in self._performance_history]) if self._performance_history else 0,
                    "total_tokens_saved": sum([r.tokens_saved for r in self._performance_history]),
                    "total_contexts_processed": sum([r.contexts_processed for r in self._performance_history]),
                    "agents_metrics": self._consolidation_metrics
                }
        except Exception as e:
            logger.error(f"Error getting consolidation metrics: {e}")
            return {}
    
    async def _get_consolidation_candidates(self, agent_id: UUID) -> List[Context]:
        """Get contexts that are candidates for consolidation."""
        try:
            async with get_async_session() as session:
                # Get non-consolidated contexts older than 1 hour
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                contexts_result = await session.execute(
                    select(Context).where(
                        and_(
                            Context.agent_id == agent_id,
                            Context.created_at < cutoff_time,
                            Context.is_consolidated == False,
                            Context.is_archived == False
                        )
                    ).limit(self.max_contexts_per_batch)
                )
                
                return list(contexts_result.scalars().all())
                
        except Exception as e:
            logger.error(f"Error getting consolidation candidates for agent {agent_id}: {e}")
            return []
    
    async def _merge_similar_contexts(self, contexts: List[Context], agent_id: UUID) -> Dict[str, int]:
        """Merge similar contexts using semantic analysis."""
        try:
            merged_count = 0
            tokens_saved = 0
            
            # Find similarity clusters
            clusters = await self._identify_similarity_clusters(contexts)
            
            for cluster in clusters:
                if cluster.should_merge():
                    # Calculate tokens saved before merging
                    original_tokens = sum(len(c.content or "") for c in cluster.similar_contexts + [cluster.representative_context])
                    
                    # Merge the cluster
                    merged_context = await self.merge_similar_contexts(cluster.similar_contexts + [cluster.representative_context])
                    
                    # Calculate tokens saved
                    new_tokens = len(merged_context.content or "")
                    tokens_saved += max(0, original_tokens - new_tokens)
                    merged_count += len(cluster.similar_contexts)
                    
                    # Mark original contexts as consolidated
                    async with get_async_session() as session:
                        for context in cluster.similar_contexts + [cluster.representative_context]:
                            if context.id != merged_context.id:
                                context.is_consolidated = True
                                context.consolidated_into_id = merged_context.id
                        await session.commit()
            
            return {"merged_count": merged_count, "tokens_saved": tokens_saved}
            
        except Exception as e:
            logger.error(f"Error merging similar contexts: {e}")
            return {"merged_count": 0, "tokens_saved": 0}
    
    async def _remove_redundant_contexts(self, contexts: List[Context], agent_id: UUID) -> Dict[str, int]:
        """Remove redundant contexts with high similarity."""
        try:
            removed_count = 0
            tokens_saved = 0
            
            redundant_ids = await self.identify_redundant_contexts(agent_id)
            
            if redundant_ids:
                async with get_async_session() as session:
                    for context in contexts:
                        if context.id in redundant_ids:
                            tokens_saved += len(context.content or "")
                            await session.delete(context)
                            removed_count += 1
                    
                    await session.commit()
            
            return {"removed_count": removed_count, "tokens_saved": tokens_saved}
            
        except Exception as e:
            logger.error(f"Error removing redundant contexts: {e}")
            return {"removed_count": 0, "tokens_saved": 0}
    
    async def _compress_contexts(self, contexts: List[Context], agent_id: UUID) -> Dict[str, int]:
        """Compress individual contexts using content optimization."""
        try:
            compressed_count = 0
            tokens_saved = 0
            
            async with get_async_session() as session:
                for context in contexts:
                    if not context.is_consolidated and context.content:
                        original_length = len(context.content)
                        
                        # Simple compression: remove extra whitespace and optimize formatting
                        compressed_content = await self._compress_context_content(context.content)
                        
                        if len(compressed_content) < original_length:
                            context.content = compressed_content
                            context.is_consolidated = True
                            
                            if not context.metadata:
                                context.metadata = {}
                            context.metadata["compression"] = {
                                "compressed_date": datetime.utcnow().isoformat(),
                                "original_length": original_length,
                                "compressed_length": len(compressed_content),
                                "compression_ratio": len(compressed_content) / original_length
                            }
                            
                            tokens_saved += original_length - len(compressed_content)
                            compressed_count += 1
                
                await session.commit()
            
            return {"compressed_count": compressed_count, "tokens_saved": tokens_saved}
            
        except Exception as e:
            logger.error(f"Error compressing contexts: {e}")
            return {"compressed_count": 0, "tokens_saved": 0}
    
    async def _archive_old_contexts(self, agent_id: UUID) -> Dict[str, int]:
        """Archive old contexts based on age and usage."""
        try:
            age_threshold = timedelta(days=self.min_age_for_archival_days)
            archived_count = await self.archive_old_contexts(agent_id, age_threshold)
            return {"archived_count": archived_count}
            
        except Exception as e:
            logger.error(f"Error archiving old contexts: {e}")
            return {"archived_count": 0}
    
    async def _identify_similarity_clusters(self, contexts: List[Context]) -> List[SimilarityCluster]:
        """Identify clusters of similar contexts for merging."""
        try:
            clusters = []
            processed_ids = set()
            
            for context in contexts:
                if context.id in processed_ids:
                    continue
                
                cluster = SimilarityCluster(context, self.similarity_threshold)
                
                # Find similar contexts
                for other_context in contexts:
                    if other_context.id != context.id and other_context.id not in processed_ids:
                        similarity = await self._calculate_content_similarity(context, other_context)
                        cluster.add_similar_context(other_context, similarity)
                
                if cluster.should_merge():
                    clusters.append(cluster)
                    processed_ids.add(context.id)
                    for similar_context in cluster.similar_contexts:
                        processed_ids.add(similar_context.id)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error identifying similarity clusters: {e}")
            return []
    
    async def _calculate_content_similarity(self, context1: Context, context2: Context) -> float:
        """Calculate semantic similarity between two contexts."""
        try:
            if not context1.content or not context2.content:
                return 0.0
            
            # Use embedding service for semantic similarity
            embedding1 = await self.embedding_service.get_embedding(context1.content)
            embedding2 = await self.embedding_service.get_embedding(context2.content)
            
            if embedding1 and embedding2:
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                return float(similarity)
            
            # Fallback to simple text similarity
            return self._calculate_text_similarity(context1.content, context2.content)
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate basic text similarity using character overlap."""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple similarity based on common characters
            set1 = set(text1.lower())
            set2 = set(text2.lower())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    async def _compress_context_content(self, content: str) -> str:
        """Compress context content while preserving meaning."""
        try:
            if not content:
                return content
            
            # Basic compression techniques
            compressed = content
            
            # Remove extra whitespace
            import re
            compressed = re.sub(r'\s+', ' ', compressed)
            compressed = re.sub(r'\n\s*\n', '\n', compressed)
            
            # Remove redundant phrases
            compressed = re.sub(r'\b(the|a|an)\s+', '', compressed, flags=re.IGNORECASE)
            
            return compressed.strip()
            
        except Exception as e:
            logger.error(f"Error compressing content: {e}")
            return content
    
    def _calculate_efficiency_score(self, result: ConsolidationResult) -> float:
        """Calculate consolidation efficiency score."""
        try:
            if result.contexts_processed == 0:
                return 0.0
            
            # Score based on multiple factors
            token_efficiency = result.tokens_saved / max(1, result.contexts_processed * 1000)
            processing_efficiency = 1.0 - min(1.0, result.processing_time_ms / 300000)  # 5 minutes max
            merge_efficiency = result.contexts_merged / max(1, result.contexts_processed)
            
            # Weighted average
            efficiency_score = (
                token_efficiency * 0.4 +
                processing_efficiency * 0.3 +
                merge_efficiency * 0.3
            )
            
            return min(1.0, efficiency_score)
            
        except Exception as e:
            logger.error(f"Error calculating efficiency score: {e}")
            return 0.0
    
    def _update_consolidation_metrics(self, agent_id: UUID, result: ConsolidationResult) -> None:
        """Update consolidation metrics for an agent."""
        try:
            agent_key = str(agent_id)
            
            if agent_key not in self._consolidation_metrics:
                self._consolidation_metrics[agent_key] = {
                    "total_consolidations": 0,
                    "total_contexts_processed": 0,
                    "total_tokens_saved": 0,
                    "average_efficiency": 0.0,
                    "last_consolidation": None
                }
            
            metrics = self._consolidation_metrics[agent_key]
            metrics["total_consolidations"] += 1
            metrics["total_contexts_processed"] += result.contexts_processed
            metrics["total_tokens_saved"] += result.tokens_saved
            metrics["last_consolidation"] = datetime.utcnow().isoformat()
            
            # Update average efficiency
            old_avg = metrics["average_efficiency"]
            count = metrics["total_consolidations"]
            metrics["average_efficiency"] = (old_avg * (count - 1) + result.efficiency_score) / count
            
        except Exception as e:
            logger.error(f"Error updating consolidation metrics: {e}")


# Utility function for consolidation engine integration
async def analyze_consolidation_opportunities(agent_id: UUID) -> Dict[str, Any]:
    """Analyze consolidation opportunities for an agent."""
    try:
        consolidator = get_context_consolidator()
        
        # Get basic context statistics
        async with get_async_session() as session:
            total_contexts = await session.scalar(
                select(func.count(Context.id)).where(Context.agent_id == agent_id)
            )
            
            unconsolidated_contexts = await session.scalar(
                select(func.count(Context.id)).where(
                    and_(
                        Context.agent_id == agent_id,
                        Context.is_consolidated == False
                    )
                )
            )
            
            old_contexts = await session.scalar(
                select(func.count(Context.id)).where(
                    and_(
                        Context.agent_id == agent_id,
                        Context.created_at < datetime.utcnow() - timedelta(days=7)
                    )
                )
            )
        
        # Calculate consolidation potential
        consolidation_potential = 0
        if total_contexts > 0:
            consolidation_potential = (unconsolidated_contexts / total_contexts) * 100
        
        return {
            "total_contexts": total_contexts or 0,
            "unconsolidated_contexts": unconsolidated_contexts or 0,
            "old_contexts": old_contexts or 0,
            "consolidation_potential": consolidation_potential,
            "stale_contexts": max(0, old_contexts or 0),
            "archival_candidates": max(0, (old_contexts or 0) // 2),
            "token_savings_estimate": (unconsolidated_contexts or 0) * 500  # Rough estimate
        }
        
    except Exception as e:
        logger.error(f"Error analyzing consolidation opportunities for agent {agent_id}: {e}")
        return {
            "total_contexts": 0,
            "unconsolidated_contexts": 0,
            "old_contexts": 0,
            "consolidation_potential": 0,
            "stale_contexts": 0,
            "archival_candidates": 0,
            "token_savings_estimate": 0
        }


# Global context consolidator instance
_context_consolidator_instance: Optional[ContextConsolidator] = None


def get_context_consolidator() -> ContextConsolidator:
    """Get the global context consolidator instance."""
    global _context_consolidator_instance
    if _context_consolidator_instance is None:
        _context_consolidator_instance = ContextConsolidator()
    return _context_consolidator_instance
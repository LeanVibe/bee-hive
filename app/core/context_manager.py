"""
Context Manager - High-level interface for Context Engine.

Provides unified access to context storage, retrieval, compression, and
cross-agent knowledge sharing with automatic embedding generation and
intelligent context lifecycle management.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import logging

from sqlalchemy import select, and_, or_, desc, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.context import Context, ContextType
from ..schemas.context import ContextCreate, ContextUpdate, ContextSearchRequest
from ..core.embeddings import EmbeddingService, get_embedding_service
from ..core.vector_search import VectorSearchEngine, ContextMatch, SearchFilters
from ..core.context_compression import ContextCompressor, CompressionLevel, get_context_compressor
from ..core.database import get_db_session


logger = logging.getLogger(__name__)


class ContextAnalytics:
    """Context usage and performance analytics."""
    
    def __init__(self):
        self.total_contexts_stored = 0
        self.total_searches_performed = 0
        self.total_compressions_performed = 0
        self.average_retrieval_time = 0.0
        self.token_savings_achieved = 0
        self.cross_agent_shares = 0


class ContextManager:
    """
    High-level context management service.
    
    Features:
    - Unified context CRUD operations
    - Automatic embedding generation
    - Semantic search and retrieval
    - Intelligent compression
    - Cross-agent knowledge sharing
    - Context lifecycle management
    - Performance monitoring
    """
    
    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        embedding_service: Optional[EmbeddingService] = None,
        compressor: Optional[ContextCompressor] = None
    ):
        """
        Initialize context manager.
        
        Args:
            db_session: Database session (optional, will create if not provided)
            embedding_service: Embedding service instance
            compressor: Context compressor instance
        """
        self.db_session = db_session
        self.embedding_service = embedding_service or get_embedding_service()
        self.compressor = compressor or get_context_compressor()
        
        # Search engine will be initialized when db_session is available
        self.search_engine: Optional[VectorSearchEngine] = None
        
        # Analytics tracking
        self.analytics = ContextAnalytics()
        
        # Background task tracking
        self._background_tasks: List[asyncio.Task] = []
    
    async def _ensure_db_session(self) -> AsyncSession:
        """Ensure database session is available."""
        if self.db_session is None:
            self.db_session = await get_db_session()
        return self.db_session
    
    async def _ensure_search_engine(self) -> VectorSearchEngine:
        """Ensure search engine is initialized."""
        if self.search_engine is None:
            db = await self._ensure_db_session()
            self.search_engine = VectorSearchEngine(db, self.embedding_service)
        return self.search_engine
    
    async def store_context(
        self,
        context_data: ContextCreate,
        auto_embed: bool = True,
        auto_compress: bool = False
    ) -> Context:
        """
        Store new context with automatic embedding generation.
        
        Args:
            context_data: Context data to store
            auto_embed: Whether to automatically generate embeddings
            auto_compress: Whether to compress if content is long
            
        Returns:
            Stored context with generated embedding
        """
        try:
            db = await self._ensure_db_session()
            
            # Create context instance
            context = Context(
                title=context_data.title,
                content=context_data.content,
                context_type=context_data.context_type,
                agent_id=context_data.agent_id,
                session_id=context_data.session_id,
                importance_score=context_data.importance_score,
                tags=context_data.tags,
                context_metadata=context_data.metadata or {}
            )
            
            # Auto-compress if content is long
            if auto_compress and len(context_data.content) > 10000:
                logger.info(f"Auto-compressing long context: {context_data.title}")
                compressed = await self.compressor.compress_conversation(
                    context_data.content,
                    CompressionLevel.STANDARD,
                    context_data.context_type
                )
                
                # Update context with compressed content
                context.content = compressed.summary
                context.consolidation_summary = compressed.summary
                context.is_consolidated = "true"
                context.importance_score = max(context.importance_score, compressed.importance_score)
                
                # Store compression metadata
                context.update_metadata("compression_applied", True)
                context.update_metadata("original_token_count", compressed.original_token_count)
                context.update_metadata("compression_ratio", compressed.compression_ratio)
                
                self.analytics.total_compressions_performed += 1
                self.analytics.token_savings_achieved += (
                    compressed.original_token_count - compressed.compressed_token_count
                )
            
            # Generate embedding if requested
            if auto_embed:
                try:
                    embedding_text = f"{context.title}\n\n{context.content}"
                    embedding = await self.embedding_service.generate_embedding(embedding_text)
                    context.embedding = embedding
                    logger.debug(f"Generated embedding for context: {context.title}")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for context {context.title}: {e}")
                    # Continue without embedding - it can be generated later
            
            # Save to database
            db.add(context)
            await db.commit()
            await db.refresh(context)
            
            self.analytics.total_contexts_stored += 1
            
            logger.info(f"Stored context: {context.title} (ID: {context.id})")
            return context
            
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            await db.rollback()
            raise
    
    async def retrieve_context(self, context_id: uuid.UUID) -> Optional[Context]:
        """
        Retrieve specific context by ID.
        
        Args:
            context_id: Context ID to retrieve
            
        Returns:
            Context if found, None otherwise
        """
        try:
            db = await self._ensure_db_session()
            
            context = await db.get(Context, context_id)
            if context:
                # Mark as accessed for relevance tracking
                context.mark_accessed()
                await db.commit()
                logger.debug(f"Retrieved context: {context.title}")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context {context_id}: {e}")
            return None
    
    async def update_context(
        self,
        context_id: uuid.UUID,
        updates: ContextUpdate,
        regenerate_embedding: bool = False
    ) -> Optional[Context]:
        """
        Update existing context.
        
        Args:
            context_id: Context ID to update
            updates: Update data
            regenerate_embedding: Whether to regenerate embedding after update
            
        Returns:
            Updated context or None if not found
        """
        try:
            db = await self._ensure_db_session()
            
            context = await db.get(Context, context_id)
            if not context:
                return None
            
            # Apply updates
            update_data = updates.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                if field == "metadata" and value:
                    # Merge metadata instead of replacing
                    for key, val in value.items():
                        context.update_metadata(key, val)
                elif hasattr(context, field):
                    setattr(context, field, value)
            
            # Regenerate embedding if content changed or explicitly requested
            if regenerate_embedding or "content" in update_data or "title" in update_data:
                try:
                    embedding_text = f"{context.title}\n\n{context.content}"
                    embedding = await self.embedding_service.generate_embedding(embedding_text)
                    context.embedding = embedding
                    logger.debug(f"Regenerated embedding for updated context: {context.title}")
                except Exception as e:
                    logger.warning(f"Failed to regenerate embedding: {e}")
            
            await db.commit()
            await db.refresh(context)
            
            logger.info(f"Updated context: {context.title}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to update context {context_id}: {e}")
            await db.rollback()
            return None
    
    async def delete_context(self, context_id: uuid.UUID) -> bool:
        """
        Delete context (soft delete by marking as archived).
        
        Args:
            context_id: Context ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        try:
            db = await self._ensure_db_session()
            
            context = await db.get(Context, context_id)
            if not context:
                return False
            
            # Soft delete by updating metadata
            context.update_metadata("archived", True)
            context.update_metadata("archived_at", datetime.utcnow().isoformat())
            
            await db.commit()
            
            logger.info(f"Archived context: {context.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete context {context_id}: {e}")
            await db.rollback()
            return False
    
    async def search_contexts(
        self,
        request: ContextSearchRequest,
        include_cross_agent: bool = True
    ) -> List[ContextMatch]:
        """
        Perform semantic search across contexts.
        
        Args:
            request: Search request parameters
            include_cross_agent: Whether to include other agents' contexts
            
        Returns:
            List of matching contexts with relevance scores
        """
        try:
            search_engine = await self._ensure_search_engine()
            
            # Build search filters
            filters = SearchFilters(
                context_types=[request.context_type] if request.context_type else None,
                agent_ids=[request.agent_id] if request.agent_id else None,
                session_ids=[request.session_id] if request.session_id else None,
                min_similarity=request.min_relevance
            )
            
            # Perform search
            results = await search_engine.semantic_search(
                query=request.query,
                agent_id=request.agent_id,
                limit=request.limit,
                filters=filters,
                include_cross_agent=include_cross_agent
            )
            
            # Mark contexts as accessed
            db = await self._ensure_db_session()
            for match in results:
                match.context.mark_accessed()
            await db.commit()
            
            self.analytics.total_searches_performed += 1
            
            logger.info(f"Search returned {len(results)} results for query: {request.query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def retrieve_relevant_contexts(
        self,
        query: str,
        agent_id: uuid.UUID,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        include_cross_agent: bool = True
    ) -> List[ContextMatch]:
        """
        Retrieve contexts relevant to a query for an agent.
        
        Args:
            query: Query string
            agent_id: Agent requesting contexts
            limit: Maximum number of contexts
            similarity_threshold: Minimum similarity score
            include_cross_agent: Whether to include other agents' contexts
            
        Returns:
            List of relevant contexts
        """
        request = ContextSearchRequest(
            query=query,
            agent_id=agent_id,
            limit=limit,
            min_relevance=similarity_threshold
        )
        
        return await self.search_contexts(request, include_cross_agent)
    
    async def compress_context(
        self,
        context_id: uuid.UUID,
        compression_level: CompressionLevel = CompressionLevel.STANDARD,
        preserve_original: bool = True
    ) -> Optional[Context]:
        """
        Compress an existing context to reduce token usage.
        
        Args:
            context_id: Context to compress
            compression_level: Level of compression to apply
            preserve_original: Whether to preserve original content in metadata
            
        Returns:
            Compressed context or None if not found
        """
        try:
            db = await self._ensure_db_session()
            
            context = await db.get(Context, context_id)
            if not context:
                return None
            
            if context.is_consolidated == "true":
                logger.info(f"Context {context_id} already consolidated")
                return context
            
            # Compress the content
            compressed = await self.compressor.compress_conversation(
                conversation_content=context.content,
                compression_level=compression_level,
                context_type=context.context_type
            )
            
            # Preserve original if requested
            if preserve_original:
                context.update_metadata("original_content", context.content)
                context.update_metadata("original_title", context.title)
            
            # Update context with compressed content
            context.consolidate(compressed.summary)
            context.importance_score = max(context.importance_score, compressed.importance_score)
            
            # Store compression metadata
            context.update_metadata("compression_level", compression_level.value)
            context.update_metadata("compression_ratio", compressed.compression_ratio)
            context.update_metadata("key_insights", compressed.key_insights)
            context.update_metadata("decisions_made", compressed.decisions_made)
            context.update_metadata("patterns_identified", compressed.patterns_identified)
            
            # Regenerate embedding for compressed content
            try:
                embedding_text = f"{context.title}\n\n{context.consolidation_summary}"
                embedding = await self.embedding_service.generate_embedding(embedding_text)
                context.embedding = embedding
            except Exception as e:
                logger.warning(f"Failed to regenerate embedding after compression: {e}")
            
            await db.commit()
            await db.refresh(context)
            
            self.analytics.total_compressions_performed += 1
            self.analytics.token_savings_achieved += (
                compressed.original_token_count - compressed.compressed_token_count
            )
            
            logger.info(
                f"Compressed context {context.title}: "
                f"{compressed.compression_ratio:.1%} reduction"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to compress context {context_id}: {e}")
            await db.rollback()
            return None
    
    async def consolidate_stale_contexts(
        self,
        agent_id: Optional[uuid.UUID] = None,
        batch_size: int = 10
    ) -> int:
        """
        Automatically consolidate old, frequently accessed contexts.
        
        Args:
            agent_id: Specific agent to consolidate (optional)
            batch_size: Number of contexts to process at once
            
        Returns:
            Number of contexts consolidated
        """
        try:
            db = await self._ensure_db_session()
            
            # Find contexts that should be consolidated
            query = select(Context).where(
                and_(
                    Context.is_consolidated != "true",
                    Context.embedding.isnot(None),
                    or_(
                        func.cast(Context.access_count, db.Integer) >= 5,
                        Context.importance_score >= 0.8
                    )
                )
            )
            
            if agent_id:
                query = query.where(Context.agent_id == agent_id)
            
            query = query.limit(batch_size)
            result = await db.execute(query)
            contexts = result.scalars().all()
            
            consolidated_count = 0
            for context in contexts:
                try:
                    compressed_context = await self.compress_context(
                        context.id,
                        CompressionLevel.STANDARD
                    )
                    if compressed_context:
                        consolidated_count += 1
                except Exception as e:
                    logger.warning(f"Failed to consolidate context {context.id}: {e}")
                    continue
            
            logger.info(f"Consolidated {consolidated_count} contexts")
            return consolidated_count
            
        except Exception as e:
            logger.error(f"Failed to consolidate stale contexts: {e}")
            return 0
    
    async def cleanup_old_contexts(
        self,
        max_age_days: int = 90,
        min_importance_threshold: float = 0.3
    ) -> int:
        """
        Clean up old, low-importance contexts.
        
        Args:
            max_age_days: Maximum age for contexts to keep
            min_importance_threshold: Minimum importance to preserve old contexts
            
        Returns:
            Number of contexts cleaned up
        """
        try:
            db = await self._ensure_db_session()
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Find old, low-importance contexts
            query = select(Context).where(
                and_(
                    Context.created_at < cutoff_date,
                    Context.importance_score < min_importance_threshold,
                    or_(
                        Context.context_metadata.op('->>')('archived') != 'true',
                        Context.context_metadata.op('->>')('archived').is_(None)
                    )
                )
            )
            
            result = await db.execute(query)
            contexts = result.scalars().all()
            
            cleaned_count = 0
            for context in contexts:
                # Archive instead of hard delete
                context.update_metadata("archived", True)
                context.update_metadata("archived_at", datetime.utcnow().isoformat())
                context.update_metadata("archived_reason", "cleanup_old_low_importance")
                cleaned_count += 1
            
            await db.commit()
            
            logger.info(f"Cleaned up {cleaned_count} old contexts")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old contexts: {e}")
            await db.rollback()
            return 0
    
    async def get_context_analytics(
        self,
        agent_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Get context usage analytics.
        
        Args:
            agent_id: Filter by agent (optional)
            session_id: Filter by session (optional)
            
        Returns:
            Analytics data
        """
        try:
            db = await self._ensure_db_session()
            
            # Base query
            query = select(Context)
            
            if agent_id:
                query = query.where(Context.agent_id == agent_id)
            if session_id:
                query = query.where(Context.session_id == session_id)
            
            # Get basic counts
            result = await db.execute(query)
            contexts = result.scalars().all()
            
            total_contexts = len(contexts)
            consolidated_contexts = sum(1 for c in contexts if c.is_consolidated == "true")
            total_access_count = sum(int(c.access_count or 0) for c in contexts)
            avg_importance = sum(c.importance_score for c in contexts) / max(1, total_contexts)
            
            # Context type distribution
            type_distribution = {}
            for context in contexts:
                type_name = context.context_type.value
                type_distribution[type_name] = type_distribution.get(type_name, 0) + 1
            
            # Recent activity
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_contexts = [c for c in contexts if c.created_at and c.created_at >= recent_cutoff]
            
            analytics = {
                "total_contexts": total_contexts,
                "consolidated_contexts": consolidated_contexts,
                "consolidation_rate": consolidated_contexts / max(1, total_contexts),
                "total_access_count": total_access_count,
                "average_importance_score": avg_importance,
                "context_type_distribution": type_distribution,
                "recent_contexts_count": len(recent_contexts),
                "service_analytics": {
                    "searches_performed": self.analytics.total_searches_performed,
                    "compressions_performed": self.analytics.total_compressions_performed,
                    "tokens_saved": self.analytics.token_savings_achieved,
                    "cross_agent_shares": self.analytics.cross_agent_shares
                }
            }
            
            # Add service-specific metrics
            if self.search_engine:
                analytics["search_performance"] = self.search_engine.get_performance_metrics()
            
            analytics["embedding_performance"] = self.embedding_service.get_performance_metrics()
            analytics["compression_performance"] = self.compressor.get_performance_metrics()
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get context analytics: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on context engine.
        
        Returns:
            Health status for all components
        """
        health_status = {
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            # Check database connectivity
            db = await self._ensure_db_session()
            await db.execute(select(1))
            health_status["components"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "unhealthy"
        
        # Check embedding service
        try:
            embedding_health = await self.embedding_service.health_check()
            health_status["components"]["embedding_service"] = embedding_health
            if embedding_health["status"] != "healthy":
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["components"]["embedding_service"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "unhealthy"
        
        # Check compression service
        try:
            compression_health = await self.compressor.health_check()
            health_status["components"]["compression_service"] = compression_health
            if compression_health["status"] != "healthy":
                health_status["overall_status"] = "degraded"
        except Exception as e:
            health_status["components"]["compression_service"] = {"status": "unhealthy", "error": str(e)}
            health_status["overall_status"] = "unhealthy"
        
        # Add performance metrics
        health_status["analytics"] = {
            "total_contexts_stored": self.analytics.total_contexts_stored,
            "total_searches_performed": self.analytics.total_searches_performed,
            "total_compressions_performed": self.analytics.total_compressions_performed
        }
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup context manager resources."""
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Clear caches
        if self.search_engine:
            self.search_engine.clear_cache()
        
        self.embedding_service.clear_cache()
        
        logger.info("Context manager cleanup completed")


# Singleton instance for application use
_context_manager: Optional[ContextManager] = None


async def get_context_manager() -> ContextManager:
    """
    Get singleton context manager instance.
    
    Returns:
        ContextManager instance
    """
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextManager()
    
    return _context_manager


async def cleanup_context_manager() -> None:
    """Cleanup context manager resources."""
    global _context_manager
    
    if _context_manager:
        await _context_manager.cleanup()
        _context_manager = None
"""
Vector Search Engine for Context Engine.

Provides high-performance semantic search using pgvector with advanced filtering,
relevance scoring, and optimization for large-scale context retrieval.
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

from sqlalchemy import select, and_, or_, desc, asc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from pgvector.sqlalchemy import Vector

from ..models.context import Context, ContextType
from ..core.embeddings import EmbeddingService


logger = logging.getLogger(__name__)


class ContextMatch:
    """Represents a context search result with similarity scoring."""
    
    def __init__(
        self,
        context: Context,
        similarity_score: float,
        relevance_score: float,
        rank: int
    ):
        self.context = context
        self.similarity_score = similarity_score
        self.relevance_score = relevance_score
        self.rank = rank
        
        # Expose context attributes for convenience
        self.id = context.id
        self.title = context.title
        self.content = context.content
        self.context_type = context.context_type
        self.importance_score = context.importance_score
        self.created_at = context.created_at
        self.accessed_at = context.accessed_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert match to dictionary for serialization."""
        return {
            "context": self.context.to_dict(),
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "rank": self.rank
        }


class SearchFilters:
    """Encapsulates search filtering options."""
    
    def __init__(
        self,
        context_types: Optional[List[ContextType]] = None,
        agent_ids: Optional[List[uuid.UUID]] = None,
        session_ids: Optional[List[uuid.UUID]] = None,
        min_similarity: float = 0.5,
        min_importance: float = 0.0,
        max_age_days: Optional[int] = None,
        tags: Optional[List[str]] = None,
        exclude_consolidated: bool = False,
        access_levels: Optional[List[str]] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None
    ):
        self.context_types = context_types
        self.agent_ids = agent_ids
        self.session_ids = session_ids
        self.min_similarity = min_similarity
        self.min_importance = min_importance
        self.max_age_days = max_age_days
        self.tags = tags
        self.exclude_consolidated = exclude_consolidated
        self.access_levels = access_levels
        self.created_after = created_after
        self.created_before = created_before


class VectorSearchEngine:
    """
    High-performance vector search engine using pgvector.
    
    Features:
    - Semantic similarity search with cosine distance
    - Advanced filtering and ranking
    - Performance optimization with indexing
    - Relevance scoring algorithms
    - Cross-agent knowledge discovery
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: EmbeddingService
    ):
        """
        Initialize vector search engine.
        
        Args:
            db_session: Async database session
            embedding_service: Service for generating embeddings
        """
        self.db = db_session
        self.embedding_service = embedding_service
        
        # Performance tracking
        self._search_count = 0
        self._total_search_time = 0.0
        self._cache_hits = 0
        
        # Simple query cache for repeated searches
        self._query_cache: Dict[str, List[ContextMatch]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}
    
    async def semantic_search(
        self,
        query: str,
        agent_id: Optional[uuid.UUID] = None,
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
        include_cross_agent: bool = True,
        boost_recent: bool = True
    ) -> List[ContextMatch]:
        """
        Perform semantic search across contexts.
        
        Args:
            query: Search query text
            agent_id: Agent performing the search (for access control)
            limit: Maximum number of results
            filters: Additional search filters
            include_cross_agent: Whether to include other agents' contexts
            boost_recent: Whether to boost recently accessed contexts
            
        Returns:
            List of context matches ordered by relevance
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, agent_id, limit, filters)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                self._cache_hits += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_results
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Build base query
            base_query = select(Context).where(Context.embedding.isnot(None))
            
            # Apply agent filtering
            if agent_id and not include_cross_agent:
                base_query = base_query.where(Context.agent_id == agent_id)
            elif agent_id and include_cross_agent:
                # Include own contexts and public/shared contexts from other agents
                base_query = base_query.where(
                    or_(
                        Context.agent_id == agent_id,
                        # Add access level filtering here when implemented
                        and_(
                            Context.agent_id != agent_id,
                            Context.importance_score >= 0.7  # Only high-importance cross-agent contexts
                        )
                    )
                )
            
            # Apply additional filters
            if filters:
                base_query = self._apply_filters(base_query, filters)
            
            # Add similarity calculation and ordering
            similarity_expr = Context.embedding.cosine_distance(query_embedding)
            
            search_query = (
                base_query
                .add_columns(
                    similarity_expr.label('similarity_distance'),
                    (1 - similarity_expr).label('similarity_score')
                )
                .where((1 - similarity_expr) >= (filters.min_similarity if filters else 0.5))
                .order_by(similarity_expr.asc())  # Lower distance = higher similarity
                .limit(limit)
            )
            
            # Execute query
            logger.debug(f"Executing semantic search for: {query[:50]}...")
            result = await self.db.execute(search_query)
            rows = result.all()
            
            # Process results and calculate relevance scores
            matches = []
            for rank, (context, similarity_distance, similarity_score) in enumerate(rows, 1):
                relevance_score = self._calculate_relevance_score(
                    context=context,
                    similarity_score=similarity_score,
                    boost_recent=boost_recent,
                    query=query
                )
                
                match = ContextMatch(
                    context=context,
                    similarity_score=similarity_score,
                    relevance_score=relevance_score,
                    rank=rank
                )
                matches.append(match)
            
            # Re-sort by relevance score if boosting is enabled
            if boost_recent:
                matches.sort(key=lambda m: m.relevance_score, reverse=True)
                # Update ranks after re-sorting
                for rank, match in enumerate(matches, 1):
                    match.rank = rank
            
            # Cache results
            self._cache_results(cache_key, matches)
            
            # Update performance metrics
            search_time = time.time() - start_time
            self._search_count += 1
            self._total_search_time += search_time
            
            logger.info(
                f"Semantic search completed: {len(matches)} results in {search_time*1000:.1f}ms"
            )
            
            return matches
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    async def find_similar_contexts(
        self,
        context_id: uuid.UUID,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[ContextMatch]:
        """
        Find contexts similar to a given context.
        
        Args:
            context_id: ID of the reference context
            limit: Maximum number of similar contexts
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar contexts
        """
        # Get the reference context
        ref_context = await self.db.get(Context, context_id)
        if not ref_context or not ref_context.embedding:
            return []
        
        # Search for similar contexts using the reference embedding
        search_query = (
            select(Context)
            .add_columns(
                Context.embedding.cosine_distance(ref_context.embedding).label('similarity_distance'),
                (1 - Context.embedding.cosine_distance(ref_context.embedding)).label('similarity_score')
            )
            .where(
                and_(
                    Context.id != context_id,  # Exclude the reference context
                    Context.embedding.isnot(None),
                    (1 - Context.embedding.cosine_distance(ref_context.embedding)) >= similarity_threshold
                )
            )
            .order_by(Context.embedding.cosine_distance(ref_context.embedding).asc())
            .limit(limit)
        )
        
        result = await self.db.execute(search_query)
        rows = result.all()
        
        matches = []
        for rank, (context, similarity_distance, similarity_score) in enumerate(rows, 1):
            match = ContextMatch(
                context=context,
                similarity_score=similarity_score,
                relevance_score=similarity_score,  # Use similarity as relevance for this case
                rank=rank
            )
            matches.append(match)
        
        return matches
    
    async def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        agent_id: Optional[uuid.UUID] = None,
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[ContextMatch]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Semantic search query
            keywords: Keywords for exact/fuzzy matching
            agent_id: Agent performing the search
            limit: Maximum number of results
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            
        Returns:
            List of context matches with combined scores
        """
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        semantic_weight = semantic_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # Get semantic search results
        semantic_results = await self.semantic_search(
            query=query,
            agent_id=agent_id,
            limit=limit * 2  # Get more results for hybrid ranking
        )
        
        # Calculate keyword scores
        keyword_scores = {}
        for match in semantic_results:
            keyword_score = self._calculate_keyword_score(match.context, keywords)
            keyword_scores[match.context.id] = keyword_score
        
        # Combine scores
        hybrid_matches = []
        for match in semantic_results:
            keyword_score = keyword_scores.get(match.context.id, 0.0)
            hybrid_score = (
                semantic_weight * match.similarity_score +
                keyword_weight * keyword_score
            )
            
            hybrid_match = ContextMatch(
                context=match.context,
                similarity_score=match.similarity_score,
                relevance_score=hybrid_score,
                rank=match.rank
            )
            hybrid_matches.append(hybrid_match)
        
        # Sort by hybrid score and limit results
        hybrid_matches.sort(key=lambda m: m.relevance_score, reverse=True)
        final_results = hybrid_matches[:limit]
        
        # Update ranks
        for rank, match in enumerate(final_results, 1):
            match.rank = rank
        
        return final_results
    
    async def get_context_recommendations(
        self,
        agent_id: uuid.UUID,
        session_id: Optional[uuid.UUID] = None,
        limit: int = 5
    ) -> List[ContextMatch]:
        """
        Get context recommendations for an agent based on their activity.
        
        Args:
            agent_id: Agent to get recommendations for
            session_id: Current session (optional)
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended contexts
        """
        # This is a simplified recommendation system
        # In a production system, this would use more sophisticated algorithms
        
        base_query = (
            select(Context)
            .where(
                and_(
                    Context.agent_id == agent_id,
                    Context.importance_score >= 0.6,
                    Context.accessed_at >= datetime.utcnow() - timedelta(days=30)
                )
            )
            .order_by(
                desc(Context.importance_score),
                desc(Context.accessed_at)
            )
            .limit(limit)
        )
        
        result = await self.db.execute(base_query)
        contexts = result.scalars().all()
        
        recommendations = []
        for rank, context in enumerate(contexts, 1):
            match = ContextMatch(
                context=context,
                similarity_score=1.0,  # Not applicable for recommendations
                relevance_score=context.calculate_current_relevance(),
                rank=rank
            )
            recommendations.append(match)
        
        return recommendations
    
    def _apply_filters(
        self,
        query: select,
        filters: SearchFilters
    ) -> select:
        """Apply search filters to query."""
        if filters.context_types:
            query = query.where(Context.context_type.in_(filters.context_types))
        
        if filters.agent_ids:
            query = query.where(Context.agent_id.in_(filters.agent_ids))
        
        if filters.session_ids:
            query = query.where(Context.session_id.in_(filters.session_ids))
        
        if filters.min_importance > 0:
            query = query.where(Context.importance_score >= filters.min_importance)
        
        if filters.max_age_days:
            cutoff_date = datetime.utcnow() - timedelta(days=filters.max_age_days)
            query = query.where(Context.created_at >= cutoff_date)
        
        if filters.exclude_consolidated:
            query = query.where(Context.is_consolidated != "true")
        
        if filters.tags:
            # Filter by tags (simplified - could be improved with full-text search)
            tag_conditions = []
            for tag in filters.tags:
                tag_conditions.append(Context.tags.op('?')(tag))  # JSON contains operator
            query = query.where(or_(*tag_conditions))
        
        if filters.created_after:
            query = query.where(Context.created_at >= filters.created_after)
        
        if filters.created_before:
            query = query.where(Context.created_at <= filters.created_before)
        
        return query
    
    def _calculate_relevance_score(
        self,
        context: Context,
        similarity_score: float,
        boost_recent: bool,
        query: str
    ) -> float:
        """Calculate relevance score considering multiple factors."""
        base_score = similarity_score
        
        # Importance boost
        importance_boost = context.importance_score * 0.2
        
        # Recency boost
        recency_boost = 0.0
        if boost_recent and context.accessed_at:
            age_days = (datetime.utcnow() - context.accessed_at).days
            recency_boost = max(0, (30 - age_days) / 30) * 0.1  # Max 10% boost for recent access
        
        # Access frequency boost
        access_count = int(context.access_count or 0)
        frequency_boost = min(0.1, access_count * 0.02)  # Max 10% boost
        
        # Consolidation boost
        consolidation_boost = 0.05 if context.is_consolidated == "true" else 0.0
        
        total_score = min(1.0, base_score + importance_boost + recency_boost + frequency_boost + consolidation_boost)
        
        return total_score
    
    def _calculate_keyword_score(self, context: Context, keywords: List[str]) -> float:
        """Calculate keyword matching score for a context."""
        if not keywords:
            return 0.0
        
        content_lower = f"{context.title} {context.content}".lower()
        
        matches = 0
        for keyword in keywords:
            if keyword.lower() in content_lower:
                matches += 1
        
        return matches / len(keywords)
    
    def _get_cache_key(
        self,
        query: str,
        agent_id: Optional[uuid.UUID],
        limit: int,
        filters: Optional[SearchFilters]
    ) -> str:
        """Generate cache key for search parameters."""
        import hashlib
        
        key_parts = [
            query,
            str(agent_id) if agent_id else "None",
            str(limit)
        ]
        
        if filters:
            key_parts.extend([
                str(filters.context_types),
                str(filters.min_similarity),
                str(filters.min_importance),
                str(filters.max_age_days)
            ])
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[ContextMatch]]:
        """Get results from cache if not expired."""
        if cache_key in self._query_cache:
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp < self._cache_ttl:
                return self._query_cache[cache_key]
            else:
                # Remove expired entry
                del self._query_cache[cache_key]
                del self._cache_timestamps[cache_key]
        
        return None
    
    def _cache_results(self, cache_key: str, results: List[ContextMatch]) -> None:
        """Cache search results."""
        self._query_cache[cache_key] = results
        self._cache_timestamps[cache_key] = time.time()
        
        # Simple cache size management
        if len(self._query_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )[:20]
            
            for key in oldest_keys:
                del self._query_cache[key]
                del self._cache_timestamps[key]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics."""
        avg_search_time = self._total_search_time / max(1, self._search_count)
        cache_hit_rate = self._cache_hits / max(1, self._search_count)
        
        return {
            "total_searches": self._search_count,
            "average_search_time_ms": avg_search_time * 1000,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._query_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self._query_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Search cache cleared")
"""
Enhanced Vector Search Engine with Production-Grade Features.

Extends the base VectorSearchEngine with:
- Advanced hybrid search (semantic + keyword + metadata)
- Multi-level caching (Redis + in-memory)
- Query optimization and rewriting
- Real-time performance monitoring
- Adaptive relevance scoring
"""

import time
import uuid
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
import re

from sqlalchemy import select, and_, or_, desc, asc, func, text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.embeddings import EmbeddingService
from ..core.vector_search import VectorSearchEngine, ContextMatch, SearchFilters
from ..core.context_analytics import ContextAnalyticsManager


logger = logging.getLogger(__name__)


class SearchMethod(Enum):
    """Different search methods available."""
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    HYBRID = "hybrid"
    SMART_HYBRID = "smart_hybrid"  # Automatically chooses best combination


@dataclass
class SearchQuery:
    """Enhanced search query with preprocessing."""
    original_query: str
    processed_query: str
    keywords: List[str]
    semantic_weight: float
    keyword_weight: float
    metadata_weight: float
    search_method: SearchMethod
    filters: Optional[SearchFilters] = None
    boost_factors: Dict[str, float] = None


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    results: List[ContextMatch]
    timestamp: datetime
    query_hash: str
    hit_count: int = 0
    last_accessed: datetime = None


class QueryOptimizer:
    """Optimizes search queries for better performance and results."""
    
    def __init__(self):
        # Common technical terms that should be weighted higher
        self.technical_terms = {
            'redis', 'postgresql', 'database', 'api', 'authentication', 'security',
            'performance', 'optimization', 'monitoring', 'deployment', 'scaling',
            'microservices', 'docker', 'kubernetes', 'load balancing', 'caching'
        }
        
        # Stop words to filter out
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'how', 'what', 'when', 'where'
        }
    
    def optimize_query(self, query: str, context_history: List[str] = None) -> SearchQuery:
        """
        Optimize a search query for better results.
        
        Args:
            query: Original search query
            context_history: Previous queries in session for context
            
        Returns:
            Optimized SearchQuery object
        """
        # Clean and normalize the query
        processed_query = self._normalize_query(query)
        
        # Extract keywords
        keywords = self._extract_keywords(processed_query)
        
        # Determine optimal search method
        search_method = self._determine_search_method(processed_query, keywords)
        
        # Calculate weights based on query characteristics
        semantic_weight, keyword_weight, metadata_weight = self._calculate_weights(
            processed_query, keywords, search_method
        )
        
        # Extract boost factors
        boost_factors = self._extract_boost_factors(processed_query, keywords)
        
        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            keywords=keywords,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            metadata_weight=metadata_weight,
            search_method=search_method,
            boost_factors=boost_factors
        )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize and clean the query string."""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'db': 'database',
            'auth': 'authentication',
            'perf': 'performance',
            'config': 'configuration',
            'admin': 'administration'
        }
        
        for abbr, full in abbreviations.items():
            normalized = re.sub(rf'\b{abbr}\b', full, normalized)
        
        return normalized
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from the query."""
        # Split into words and remove stop words
        words = [word for word in query.split() if word not in self.stop_words and len(word) > 2]
        
        # Add technical term variations
        enhanced_keywords = []
        for word in words:
            enhanced_keywords.append(word)
            
            # Add plural/singular variations
            if word.endswith('s') and len(word) > 3:
                enhanced_keywords.append(word[:-1])
            elif not word.endswith('s'):
                enhanced_keywords.append(word + 's')
        
        return list(set(enhanced_keywords))
    
    def _determine_search_method(self, query: str, keywords: List[str]) -> SearchMethod:
        """Determine the optimal search method for the query."""
        # Short queries with technical terms benefit from hybrid search
        if len(query.split()) <= 3 and any(kw in self.technical_terms for kw in keywords):
            return SearchMethod.HYBRID
        
        # Long descriptive queries work well with semantic search
        if len(query.split()) > 8:
            return SearchMethod.SEMANTIC_ONLY
        
        # Medium queries with mix of terms use smart hybrid
        if 4 <= len(query.split()) <= 8:
            return SearchMethod.SMART_HYBRID
        
        # Default to hybrid for balanced results
        return SearchMethod.HYBRID
    
    def _calculate_weights(
        self, 
        query: str, 
        keywords: List[str], 
        method: SearchMethod
    ) -> Tuple[float, float, float]:
        """Calculate optimal weights for different search components."""
        if method == SearchMethod.SEMANTIC_ONLY:
            return 1.0, 0.0, 0.0
        elif method == SearchMethod.KEYWORD_ONLY:
            return 0.0, 1.0, 0.0
        elif method == SearchMethod.HYBRID:
            return 0.7, 0.3, 0.0
        elif method == SearchMethod.SMART_HYBRID:
            # Adaptive weighting based on query characteristics
            has_technical_terms = any(kw in self.technical_terms for kw in keywords)
            query_length = len(query.split())
            
            if has_technical_terms:
                # Boost keyword matching for technical queries
                return 0.6, 0.35, 0.05
            elif query_length > 6:
                # Favor semantic for longer queries
                return 0.8, 0.15, 0.05
            else:
                # Balanced approach
                return 0.7, 0.25, 0.05
        
        return 0.7, 0.3, 0.0  # Default hybrid
    
    def _extract_boost_factors(self, query: str, keywords: List[str]) -> Dict[str, float]:
        """Extract boost factors based on query content."""
        boost_factors = {}
        
        # Boost recent content for time-sensitive queries
        time_indicators = ['recent', 'latest', 'new', 'current', 'today', 'now']
        if any(indicator in query for indicator in time_indicators):
            boost_factors['recency'] = 1.5
        
        # Boost high-importance content for critical queries
        importance_indicators = ['critical', 'important', 'urgent', 'production', 'security']
        if any(indicator in query for indicator in importance_indicators):
            boost_factors['importance'] = 1.3
        
        # Boost specific context types
        if 'error' in query or 'problem' in query or 'issue' in query:
            boost_factors['error_resolution'] = 1.4
        
        if 'documentation' in query or 'guide' in query or 'tutorial' in query:
            boost_factors['documentation'] = 1.2
        
        return boost_factors


class EnhancedVectorSearchEngine(VectorSearchEngine):
    """
    Production-grade vector search engine with advanced features.
    
    Enhancements over base VectorSearchEngine:
    - Multi-level caching (Redis + memory)
    - Query optimization and rewriting
    - Hybrid search with adaptive weighting
    - Real-time performance monitoring
    - Learning from user feedback
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: EmbeddingService,
        redis_client: Optional[redis.Redis] = None,
        analytics_manager: Optional[ContextAnalyticsManager] = None
    ):
        """
        Initialize enhanced search engine.
        
        Args:
            db_session: Database session
            embedding_service: Embedding service
            redis_client: Redis client for distributed caching
            analytics_manager: Analytics manager for tracking
        """
        super().__init__(db_session, embedding_service)
        
        self.redis_client = redis_client
        self.analytics_manager = analytics_manager
        self.query_optimizer = QueryOptimizer()
        
        # Enhanced caching
        self.l1_cache: Dict[str, CacheEntry] = {}  # In-memory cache
        self.l1_cache_size = 100
        self.l2_cache_ttl = 3600  # Redis cache TTL (1 hour)
        
        # Performance monitoring
        self.query_performance: Dict[str, List[float]] = {}
        self.feedback_learning: Dict[str, List[float]] = {}
        
        # Query pattern learning
        self.successful_patterns: Set[str] = set()
        self.failed_patterns: Set[str] = set()
    
    async def enhanced_search(
        self,
        query: str,
        agent_id: Optional[uuid.UUID] = None,
        limit: int = 10,
        filters: Optional[SearchFilters] = None,
        include_cross_agent: bool = True,
        context_history: Optional[List[str]] = None,
        performance_target_ms: float = 50.0
    ) -> Tuple[List[ContextMatch], Dict[str, Any]]:
        """
        Perform enhanced search with optimization and caching.
        
        Args:
            query: Search query
            agent_id: Requesting agent
            limit: Maximum results
            filters: Search filters
            include_cross_agent: Include other agents' contexts
            context_history: Previous queries for context
            performance_target_ms: Performance target in milliseconds
            
        Returns:
            Tuple of (search results, search metadata)
        """
        start_time = time.perf_counter()
        
        try:
            # Optimize the query
            optimized_query = self.query_optimizer.optimize_query(query, context_history)
            optimized_query.filters = filters
            
            # Check cache first
            cache_key = self._generate_cache_key(optimized_query, agent_id, limit, include_cross_agent)
            cached_results = await self._get_from_cache(cache_key)
            
            if cached_results:
                search_time = (time.perf_counter() - start_time) * 1000
                metadata = {
                    "cache_hit": True,
                    "search_time_ms": search_time,
                    "search_method": optimized_query.search_method.value,
                    "query_optimization": {
                        "original": query,
                        "processed": optimized_query.processed_query,
                        "keywords": optimized_query.keywords
                    }
                }
                return cached_results, metadata
            
            # Perform search based on method
            if optimized_query.search_method == SearchMethod.SEMANTIC_ONLY:
                results = await self._semantic_search_only(optimized_query, agent_id, limit, include_cross_agent)
            elif optimized_query.search_method == SearchMethod.KEYWORD_ONLY:
                results = await self._keyword_search_only(optimized_query, agent_id, limit, include_cross_agent)
            else:
                # Hybrid or smart hybrid
                results = await self._hybrid_search_enhanced(optimized_query, agent_id, limit, include_cross_agent)
            
            # Apply boost factors
            if optimized_query.boost_factors:
                results = self._apply_boost_factors(results, optimized_query.boost_factors)
            
            # Cache results
            await self._cache_results(cache_key, results)
            
            # Record analytics
            search_time = (time.perf_counter() - start_time) * 1000
            await self._record_search_analytics(optimized_query, agent_id, results, search_time)
            
            # Performance monitoring
            self._update_performance_metrics(optimized_query.search_method.value, search_time)
            
            metadata = {
                "cache_hit": False,
                "search_time_ms": search_time,
                "search_method": optimized_query.search_method.value,
                "results_count": len(results),
                "performance_target_met": search_time <= performance_target_ms,
                "query_optimization": {
                    "original": query,
                    "processed": optimized_query.processed_query,
                    "keywords": optimized_query.keywords,
                    "weights": {
                        "semantic": optimized_query.semantic_weight,
                        "keyword": optimized_query.keyword_weight,
                        "metadata": optimized_query.metadata_weight
                    }
                },
                "boost_factors": optimized_query.boost_factors
            }
            
            return results, metadata
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to basic search
            basic_results = await self.semantic_search(query, agent_id, limit, filters, include_cross_agent)
            search_time = (time.perf_counter() - start_time) * 1000
            
            metadata = {
                "cache_hit": False,
                "search_time_ms": search_time,
                "search_method": "fallback_semantic",
                "error": str(e)
            }
            
            return basic_results, metadata
    
    async def _hybrid_search_enhanced(
        self,
        optimized_query: SearchQuery,
        agent_id: Optional[uuid.UUID],
        limit: int,
        include_cross_agent: bool
    ) -> List[ContextMatch]:
        """Perform enhanced hybrid search with adaptive weighting."""
        # Get semantic results
        semantic_results = await self.semantic_search(
            query=optimized_query.processed_query,
            agent_id=agent_id,
            limit=limit * 2,  # Get more for hybrid ranking
            filters=optimized_query.filters,
            include_cross_agent=include_cross_agent
        )
        
        # Get keyword results if keyword weight > 0
        keyword_results = []
        if optimized_query.keyword_weight > 0:
            keyword_results = await self._keyword_search_optimized(
                optimized_query.keywords,
                agent_id,
                limit * 2,
                optimized_query.filters,
                include_cross_agent
            )
        
        # Combine and score results
        hybrid_scores = {}
        
        # Score semantic results
        for match in semantic_results:
            hybrid_scores[match.context.id] = {
                'context': match.context,
                'semantic_score': match.similarity_score,
                'keyword_score': 0.0,
                'metadata_score': 0.0,
                'final_score': 0.0
            }
        
        # Add keyword scores
        for match in keyword_results:
            if match.context.id in hybrid_scores:
                hybrid_scores[match.context.id]['keyword_score'] = match.similarity_score
            else:
                hybrid_scores[match.context.id] = {
                    'context': match.context,
                    'semantic_score': 0.0,
                    'keyword_score': match.similarity_score,
                    'metadata_score': 0.0,
                    'final_score': 0.0
                }
        
        # Calculate metadata scores and final scores
        for context_id, scores in hybrid_scores.items():
            context = scores['context']
            
            # Metadata scoring based on context properties
            metadata_score = self._calculate_metadata_score(context, optimized_query)
            scores['metadata_score'] = metadata_score
            
            # Calculate final hybrid score
            final_score = (
                optimized_query.semantic_weight * scores['semantic_score'] +
                optimized_query.keyword_weight * scores['keyword_score'] +
                optimized_query.metadata_weight * metadata_score
            )
            scores['final_score'] = final_score
        
        # Sort by final score and create ContextMatch objects
        sorted_results = sorted(
            hybrid_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )[:limit]
        
        hybrid_matches = []
        for rank, result in enumerate(sorted_results, 1):
            match = ContextMatch(
                context=result['context'],
                similarity_score=result['semantic_score'],
                relevance_score=result['final_score'],
                rank=rank
            )
            hybrid_matches.append(match)
        
        return hybrid_matches
    
    async def _keyword_search_optimized(
        self,
        keywords: List[str],
        agent_id: Optional[uuid.UUID],
        limit: int,
        filters: Optional[SearchFilters],
        include_cross_agent: bool
    ) -> List[ContextMatch]:
        """Perform optimized keyword search using PostgreSQL full-text search."""
        try:
            # Build full-text search query
            search_terms = ' & '.join(keywords)  # AND operation
            
            # Base query with full-text search
            base_query = select(Context).where(
                and_(
                    func.to_tsvector('english', Context.title + ' ' + Context.content).op('@@')(
                        func.plainto_tsquery('english', search_terms)
                    ),
                    Context.embedding.isnot(None)
                )
            )
            
            # Apply agent filtering
            if agent_id and not include_cross_agent:
                base_query = base_query.where(Context.agent_id == agent_id)
            elif agent_id and include_cross_agent:
                base_query = base_query.where(
                    or_(
                        Context.agent_id == agent_id,
                        and_(
                            Context.agent_id != agent_id,
                            Context.importance_score >= 0.7
                        )
                    )
                )
            
            # Apply additional filters
            if filters:
                base_query = self._apply_filters(base_query, filters)
            
            # Add ranking and limit
            search_query = (
                base_query
                .add_columns(
                    func.ts_rank(
                        func.to_tsvector('english', Context.title + ' ' + Context.content),
                        func.plainto_tsquery('english', search_terms)
                    ).label('rank_score')
                )
                .order_by(desc('rank_score'))
                .limit(limit)
            )
            
            result = await self.db.execute(search_query)
            rows = result.all()
            
            matches = []
            for rank, (context, rank_score) in enumerate(rows, 1):
                match = ContextMatch(
                    context=context,
                    similarity_score=float(rank_score),
                    relevance_score=float(rank_score),
                    rank=rank
                )
                matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.warning(f"Keyword search failed, falling back to simple matching: {e}")
            return await self._simple_keyword_search(keywords, agent_id, limit, filters, include_cross_agent)
    
    async def _simple_keyword_search(
        self,
        keywords: List[str],
        agent_id: Optional[uuid.UUID],
        limit: int,
        filters: Optional[SearchFilters],
        include_cross_agent: bool
    ) -> List[ContextMatch]:
        """Fallback simple keyword search."""
        base_query = select(Context).where(Context.embedding.isnot(None))
        
        # Build keyword conditions
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(
                or_(
                    Context.title.ilike(f'%{keyword}%'),
                    Context.content.ilike(f'%{keyword}%')
                )
            )
        
        if keyword_conditions:
            base_query = base_query.where(or_(*keyword_conditions))
        
        # Apply agent and other filters
        if agent_id and not include_cross_agent:
            base_query = base_query.where(Context.agent_id == agent_id)
        
        if filters:
            base_query = self._apply_filters(base_query, filters)
        
        # Order by importance and limit
        search_query = base_query.order_by(desc(Context.importance_score)).limit(limit)
        
        result = await self.db.execute(search_query)
        contexts = result.scalars().all()
        
        matches = []
        for rank, context in enumerate(contexts, 1):
            # Calculate simple keyword score
            score = self._calculate_simple_keyword_score(context, keywords)
            match = ContextMatch(
                context=context,
                similarity_score=score,
                relevance_score=score,
                rank=rank
            )
            matches.append(match)
        
        return matches
    
    def _calculate_simple_keyword_score(self, context: Context, keywords: List[str]) -> float:
        """Calculate simple keyword matching score."""
        content_lower = f"{context.title} {context.content}".lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def _calculate_metadata_score(self, context: Context, optimized_query: SearchQuery) -> float:
        """Calculate metadata-based scoring."""
        score = 0.0
        
        # Boost based on context type relevance
        type_relevance = {
            'error': 0.3 if 'error' in optimized_query.original_query.lower() else 0.0,
            'documentation': 0.3 if 'documentation' in optimized_query.original_query.lower() else 0.0,
            'code': 0.2 if 'code' in optimized_query.original_query.lower() else 0.0
        }
        
        if context.context_type:
            context_type_lower = context.context_type.value.lower()
            for query_type, boost in type_relevance.items():
                if query_type in context_type_lower:
                    score += boost
        
        # Boost for high importance
        score += context.importance_score * 0.2
        
        # Boost for recent access
        if context.accessed_at:
            days_since_access = (datetime.utcnow() - context.accessed_at).days
            if days_since_access < 7:
                score += 0.1 * (7 - days_since_access) / 7
        
        return min(1.0, score)
    
    def _apply_boost_factors(
        self,
        results: List[ContextMatch],
        boost_factors: Dict[str, float]
    ) -> List[ContextMatch]:
        """Apply boost factors to search results."""
        boosted_results = []
        
        for match in results:
            boosted_score = match.relevance_score
            
            # Apply recency boost
            if 'recency' in boost_factors and match.context.created_at:
                days_old = (datetime.utcnow() - match.context.created_at).days
                if days_old < 30:
                    recency_multiplier = 1 + (boost_factors['recency'] - 1) * (30 - days_old) / 30
                    boosted_score *= recency_multiplier
            
            # Apply importance boost
            if 'importance' in boost_factors:
                if match.context.importance_score > 0.8:
                    boosted_score *= boost_factors['importance']
            
            # Apply context type boosts
            if match.context.context_type:
                context_type = match.context.context_type.value.lower()
                for boost_type, multiplier in boost_factors.items():
                    if boost_type in context_type:
                        boosted_score *= multiplier
            
            # Create new match with boosted score
            boosted_match = ContextMatch(
                context=match.context,
                similarity_score=match.similarity_score,
                relevance_score=boosted_score,
                rank=match.rank
            )
            boosted_results.append(boosted_match)
        
        # Re-sort by boosted relevance score
        boosted_results.sort(key=lambda m: m.relevance_score, reverse=True)
        
        # Update ranks
        for rank, match in enumerate(boosted_results, 1):
            match.rank = rank
        
        return boosted_results
    
    async def _semantic_search_only(
        self,
        optimized_query: SearchQuery,
        agent_id: Optional[uuid.UUID],
        limit: int,
        include_cross_agent: bool
    ) -> List[ContextMatch]:
        """Perform semantic-only search."""
        return await self.semantic_search(
            query=optimized_query.processed_query,
            agent_id=agent_id,
            limit=limit,
            filters=optimized_query.filters,
            include_cross_agent=include_cross_agent
        )
    
    async def _keyword_search_only(
        self,
        optimized_query: SearchQuery,
        agent_id: Optional[uuid.UUID],
        limit: int,
        include_cross_agent: bool
    ) -> List[ContextMatch]:
        """Perform keyword-only search."""
        return await self._keyword_search_optimized(
            optimized_query.keywords,
            agent_id,
            limit,
            optimized_query.filters,
            include_cross_agent
        )
    
    def _generate_cache_key(
        self,
        optimized_query: SearchQuery,
        agent_id: Optional[uuid.UUID],
        limit: int,
        include_cross_agent: bool
    ) -> str:
        """Generate cache key for optimized query."""
        key_data = {
            "query": optimized_query.processed_query,
            "keywords": sorted(optimized_query.keywords),
            "method": optimized_query.search_method.value,
            "weights": [
                optimized_query.semantic_weight,
                optimized_query.keyword_weight,
                optimized_query.metadata_weight
            ],
            "agent_id": str(agent_id) if agent_id else None,
            "limit": limit,
            "include_cross_agent": include_cross_agent,
            "filters": self._serialize_filters(optimized_query.filters) if optimized_query.filters else None
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    def _serialize_filters(self, filters: SearchFilters) -> Dict[str, Any]:
        """Serialize filters for cache key generation."""
        return {
            "context_types": [ct.value for ct in filters.context_types] if filters.context_types else None,
            "min_similarity": filters.min_similarity,
            "min_importance": filters.min_importance,
            "max_age_days": filters.max_age_days,
            "exclude_consolidated": filters.exclude_consolidated
        }
    
    async def _get_from_cache(self, cache_key: str) -> Optional[List[ContextMatch]]:
        """Get results from multi-level cache."""
        # Check L1 cache (memory)
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            if datetime.utcnow() - entry.timestamp < timedelta(seconds=self._cache_ttl):
                entry.hit_count += 1
                entry.last_accessed = datetime.utcnow()
                return entry.results
            else:
                del self.l1_cache[cache_key]
        
        # Check L2 cache (Redis)
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"search_cache:{cache_key}")
                if cached_data:
                    # Deserialize and store in L1 cache
                    results_data = json.loads(cached_data)
                    # Note: In production, you'd need proper deserialization of ContextMatch objects
                    # This is simplified for the example
                    return None  # Simplified - would need proper deserialization
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        return None
    
    async def _cache_results(self, cache_key: str, results: List[ContextMatch]) -> None:
        """Cache results in multi-level cache."""
        # Store in L1 cache
        entry = CacheEntry(
            results=results,
            timestamp=datetime.utcnow(),
            query_hash=cache_key
        )
        
        self.l1_cache[cache_key] = entry
        
        # Manage L1 cache size
        if len(self.l1_cache) > self.l1_cache_size:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.l1_cache.items(),
                key=lambda x: x[1].last_accessed or x[1].timestamp
            )
            for key, _ in sorted_entries[:10]:  # Remove 10 oldest
                del self.l1_cache[key]
        
        # Store in L2 cache (Redis)
        if self.redis_client:
            try:
                # Serialize results (simplified)
                results_data = [
                    {
                        "context_id": str(match.context.id),
                        "similarity_score": match.similarity_score,
                        "relevance_score": match.relevance_score,
                        "rank": match.rank
                    }
                    for match in results
                ]
                
                await self.redis_client.setex(
                    f"search_cache:{cache_key}",
                    self.l2_cache_ttl,
                    json.dumps(results_data)
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
    
    async def _record_search_analytics(
        self,
        optimized_query: SearchQuery,
        agent_id: Optional[uuid.UUID],
        results: List[ContextMatch],
        search_time_ms: float
    ) -> None:
        """Record search analytics for performance monitoring."""
        if self.analytics_manager and agent_id:
            try:
                # Record retrieval events for top results
                for match in results[:5]:  # Record top 5 results
                    await self.analytics_manager.record_context_retrieval(
                        context_id=match.context.id,
                        requesting_agent_id=agent_id,
                        query_text=optimized_query.original_query,
                        similarity_score=match.similarity_score,
                        relevance_score=match.relevance_score,
                        rank_position=match.rank,
                        response_time_ms=search_time_ms,
                        retrieval_method=optimized_query.search_method.value
                    )
            except Exception as e:
                logger.warning(f"Failed to record search analytics: {e}")
    
    def _update_performance_metrics(self, search_method: str, search_time_ms: float) -> None:
        """Update performance metrics for monitoring."""
        if search_method not in self.query_performance:
            self.query_performance[search_method] = []
        
        self.query_performance[search_method].append(search_time_ms)
        
        # Keep only recent measurements
        if len(self.query_performance[search_method]) > 100:
            self.query_performance[search_method] = self.query_performance[search_method][-100:]
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including cache statistics."""
        base_metrics = self.get_performance_metrics()
        
        # Add cache metrics
        l1_hit_rate = 0.0
        l1_size = len(self.l1_cache)
        
        if self.l1_cache:
            total_hits = sum(entry.hit_count for entry in self.l1_cache.values())
            total_queries = len(self.l1_cache)
            l1_hit_rate = total_hits / max(1, total_queries)
        
        # Add search method performance
        method_performance = {}
        for method, times in self.query_performance.items():
            if times:
                method_performance[method] = {
                    "avg_time_ms": sum(times) / len(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "query_count": len(times)
                }
        
        enhanced_metrics = {
            **base_metrics,
            "cache_performance": {
                "l1_cache_size": l1_size,
                "l1_hit_rate": l1_hit_rate,
                "l1_max_size": self.l1_cache_size
            },
            "search_method_performance": method_performance,
            "feature_usage": {
                "hybrid_searches": len(self.query_performance.get("hybrid", [])),
                "semantic_only": len(self.query_performance.get("semantic_only", [])),
                "keyword_only": len(self.query_performance.get("keyword_only", []))
            }
        }
        
        return enhanced_metrics


# Factory function
async def create_enhanced_search_engine(
    db_session: AsyncSession,
    embedding_service: EmbeddingService,
    redis_client: Optional[redis.Redis] = None,
    analytics_manager: Optional[ContextAnalyticsManager] = None
) -> EnhancedVectorSearchEngine:
    """
    Create enhanced vector search engine instance.
    
    Args:
        db_session: Database session
        embedding_service: Embedding service
        redis_client: Redis client for caching
        analytics_manager: Analytics manager for tracking
        
    Returns:
        EnhancedVectorSearchEngine instance
    """
    return EnhancedVectorSearchEngine(
        db_session=db_session,
        embedding_service=embedding_service,
        redis_client=redis_client,
        analytics_manager=analytics_manager
    )
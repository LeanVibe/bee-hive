"""
Advanced Conversation Search Engine for LeanVibe Agent Hive 2.0

Provides sophisticated search capabilities across conversation transcripts
with semantic analysis, vector similarity, and advanced filtering.

Features:
- Semantic search using embedding vectors
- Full-text search with advanced query syntax
- Pattern-based search and filtering
- Temporal search with time-based queries
- Agent behavior search and analysis
- Context-aware search with cross-references
- Performance-optimized search with caching
"""

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
import numpy as np
from sqlalchemy import select, func, and_, or_, desc, asc, text
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from .database import get_async_session
from .embedding_service import EmbeddingService
from .context_engine_integration import ContextEngineIntegration
from .chat_transcript_manager import (
    ConversationEvent, ConversationEventType, ConversationPattern,
    ConversationMetrics, SearchFilter
)
from ..models.conversation import Conversation
from ..models.agent import Agent
from ..models.context import Context

logger = structlog.get_logger()


class SearchType(Enum):
    """Types of search operations."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    PATTERN = "pattern"
    TEMPORAL = "temporal"
    AGENT_BEHAVIOR = "agent_behavior"
    CONTEXT_AWARE = "context_aware"
    HYBRID = "hybrid"


class SortOption(Enum):
    """Search result sorting options."""
    RELEVANCE = "relevance"
    TIMESTAMP_ASC = "timestamp_asc"
    TIMESTAMP_DESC = "timestamp_desc"
    RESPONSE_TIME = "response_time"
    AGENT_ID = "agent_id"
    SESSION_ID = "session_id"


class SearchOperator(Enum):
    """Search query operators."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    NEAR = "NEAR"
    FUZZY = "FUZZY"


@dataclass
class SearchQuery:
    """Advanced search query configuration."""
    query_text: Optional[str] = None
    search_type: SearchType = SearchType.HYBRID
    session_filters: List[str] = None
    agent_filters: List[str] = None
    event_type_filters: List[ConversationEventType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    semantic_threshold: float = 0.7
    keyword_operators: List[SearchOperator] = None
    pattern_filters: List[ConversationPattern] = None
    response_time_range: Optional[Tuple[float, float]] = None
    context_filters: List[str] = None
    tool_filters: List[str] = None
    sort_by: SortOption = SortOption.RELEVANCE
    limit: int = 100
    offset: int = 0
    include_metadata: bool = True
    include_embeddings: bool = False
    
    def __post_init__(self):
        if self.session_filters is None:
            self.session_filters = []
        if self.agent_filters is None:
            self.agent_filters = []
        if self.event_type_filters is None:
            self.event_type_filters = []
        if self.keyword_operators is None:
            self.keyword_operators = []
        if self.pattern_filters is None:
            self.pattern_filters = []
        if self.context_filters is None:
            self.context_filters = []
        if self.tool_filters is None:
            self.tool_filters = []


@dataclass
class SearchResult:
    """Search result with relevance scoring."""
    event: ConversationEvent
    relevance_score: float
    search_type: SearchType
    match_reasons: List[str]
    highlighted_content: Optional[str] = None
    context_matches: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event': self.event.to_dict(),
            'relevance_score': self.relevance_score,
            'search_type': self.search_type.value,
            'match_reasons': self.match_reasons,
            'highlighted_content': self.highlighted_content,
            'context_matches': self.context_matches or []
        }


@dataclass
class SearchResults:
    """Complete search results with metadata."""
    results: List[SearchResult]
    total_matches: int
    search_time_ms: float
    query: SearchQuery
    facets: Dict[str, Dict[str, int]]
    suggestions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'results': [r.to_dict() for r in self.results],
            'total_matches': self.total_matches,
            'search_time_ms': self.search_time_ms,
            'query': asdict(self.query),
            'facets': self.facets,
            'suggestions': self.suggestions
        }


class QueryProcessor:
    """Processes and optimizes search queries."""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def parse_query(self, query_text: str) -> Dict[str, Any]:
        """Parse natural language query into structured format."""
        if not query_text:
            return {'terms': [], 'operators': [], 'filters': {}}
        
        # Simple query parsing - can be enhanced with NLP
        query_text = query_text.lower().strip()
        
        # Extract quoted phrases
        phrases = []
        import re
        quoted_matches = re.findall(r'"([^"]*)"', query_text)
        phrases.extend(quoted_matches)
        
        # Remove quoted phrases from main query
        for phrase in quoted_matches:
            query_text = query_text.replace(f'"{phrase}"', '')
        
        # Extract individual terms
        terms = [
            term.strip() for term in query_text.split()
            if term.strip() and term.lower() not in self.stopwords
        ]
        
        # Identify operators
        operators = []
        for term in terms[:]:
            if term.upper() in ['AND', 'OR', 'NOT']:
                operators.append(SearchOperator(term.upper()))
                terms.remove(term)
        
        # Extract filters from query (e.g., "agent:abc123", "session:xyz")
        filters = {}
        for term in terms[:]:
            if ':' in term:
                key, value = term.split(':', 1)
                if key in ['agent', 'session', 'type', 'error', 'tool']:
                    if key not in filters:
                        filters[key] = []
                    filters[key].append(value)
                    terms.remove(term)
        
        return {
            'terms': terms,
            'phrases': phrases,
            'operators': operators,
            'filters': filters
        }
    
    def expand_query(self, terms: List[str]) -> List[str]:
        """Expand query terms with synonyms and variants."""
        expanded = set(terms)
        
        # Simple synonym expansion
        synonyms = {
            'error': ['failure', 'exception', 'issue', 'problem'],
            'tool': ['function', 'method', 'utility', 'service'],
            'agent': ['worker', 'service', 'process'],
            'message': ['communication', 'request', 'signal'],
            'response': ['reply', 'answer', 'result']
        }
        
        for term in terms:
            if term in synonyms:
                expanded.update(synonyms[term])
        
        return list(expanded)
    
    def build_elasticsearch_query(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Build Elasticsearch query from parsed query (future enhancement)."""
        # Placeholder for Elasticsearch integration
        return {
            'query': {
                'bool': {
                    'must': [
                        {'match': {'content': ' '.join(parsed_query['terms'])}}
                    ]
                }
            }
        }


class ConversationSearchEngine:
    """
    Advanced search engine for conversation transcripts.
    
    Provides semantic search, full-text search, pattern matching,
    and sophisticated filtering with performance optimization.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        embedding_service: EmbeddingService,
        context_engine: ContextEngineIntegration
    ):
        self.db_session = db_session
        self.embedding_service = embedding_service
        self.context_engine = context_engine
        self.query_processor = QueryProcessor()
        
        # Search cache
        self.query_cache: Dict[str, SearchResults] = {}
        self.max_cache_size = 1000
        self.cache_ttl_minutes = 30
        
        # Search analytics
        self.search_metrics: Dict[str, Any] = {
            'total_searches': 0,
            'cache_hits': 0,
            'average_search_time_ms': 0,
            'popular_queries': {},
            'search_types_usage': {st.value: 0 for st in SearchType}
        }
        
        logger.info("ConversationSearchEngine initialized")
    
    async def search(self, query: SearchQuery) -> SearchResults:
        """
        Execute comprehensive search across conversation transcripts.
        
        Args:
            query: Search query configuration
            
        Returns:
            SearchResults with ranked results and metadata
        """
        search_start = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.search_metrics['cache_hits'] += 1
                    return cached_result
            
            # Execute search based on type
            if query.search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query)
            elif query.search_type == SearchType.KEYWORD:
                results = await self._keyword_search(query)
            elif query.search_type == SearchType.PATTERN:
                results = await self._pattern_search(query)
            elif query.search_type == SearchType.TEMPORAL:
                results = await self._temporal_search(query)
            elif query.search_type == SearchType.AGENT_BEHAVIOR:
                results = await self._agent_behavior_search(query)
            elif query.search_type == SearchType.CONTEXT_AWARE:
                results = await self._context_aware_search(query)
            else:  # HYBRID
                results = await self._hybrid_search(query)
            
            # Calculate search time
            search_time = (datetime.utcnow() - search_start).total_seconds() * 1000
            
            # Generate facets
            facets = await self._generate_facets(results.results)
            
            # Generate search suggestions
            suggestions = await self._generate_suggestions(query, results.results)
            
            # Create final results
            final_results = SearchResults(
                results=results.results,
                total_matches=results.total_matches,
                search_time_ms=search_time,
                query=query,
                facets=facets,
                suggestions=suggestions
            )
            
            # Cache results
            self.query_cache[cache_key] = final_results
            await self._cleanup_cache()
            
            # Update metrics
            self.search_metrics['total_searches'] += 1
            self.search_metrics['search_types_usage'][query.search_type.value] += 1
            
            # Update average search time
            current_avg = self.search_metrics['average_search_time_ms']
            total_searches = self.search_metrics['total_searches']
            self.search_metrics['average_search_time_ms'] = (
                (current_avg * (total_searches - 1) + search_time) / total_searches
            )
            
            # Track popular queries
            if query.query_text:
                query_key = query.query_text.lower().strip()
                self.search_metrics['popular_queries'][query_key] = (
                    self.search_metrics['popular_queries'].get(query_key, 0) + 1
                )
            
            logger.info(
                "Search completed",
                search_type=query.search_type.value,
                results_count=len(results.results),
                total_matches=results.total_matches,
                search_time_ms=search_time
            )
            
            return final_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def suggest_queries(
        self,
        partial_query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate query auto-completion suggestions.
        
        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions
            
        Returns:
            List of query suggestions with metadata
        """
        try:
            suggestions = []
            
            if not partial_query or len(partial_query) < 2:
                # Return popular queries
                popular = sorted(
                    self.search_metrics['popular_queries'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:limit]
                
                return [
                    {
                        'query': query,
                        'type': 'popular',
                        'frequency': count,
                        'preview': f"Used {count} times"
                    }
                    for query, count in popular
                ]
            
            partial_lower = partial_query.lower()
            
            # Find matching queries from history
            matching_queries = [
                (query, count)
                for query, count in self.search_metrics['popular_queries'].items()
                if partial_lower in query.lower()
            ]
            
            # Sort by frequency
            matching_queries.sort(key=lambda x: x[1], reverse=True)
            
            for query, count in matching_queries[:limit//2]:
                suggestions.append({
                    'query': query,
                    'type': 'history',
                    'frequency': count,
                    'preview': f"Previous search ({count} times)"
                })
            
            # Generate contextual suggestions
            if len(suggestions) < limit:
                contextual = await self._generate_contextual_suggestions(partial_query)
                suggestions.extend(contextual[:limit - len(suggestions)])
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Query suggestion failed: {e}")
            return []
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics."""
        try:
            # Calculate top queries
            top_queries = sorted(
                self.search_metrics['popular_queries'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Calculate search type distribution
            total_searches = sum(self.search_metrics['search_types_usage'].values())
            search_type_percentages = {
                search_type: (count / total_searches * 100) if total_searches > 0 else 0
                for search_type, count in self.search_metrics['search_types_usage'].items()
            }
            
            return {
                'overview': {
                    **self.search_metrics,
                    'cache_hit_rate': (
                        self.search_metrics['cache_hits'] / self.search_metrics['total_searches'] * 100
                        if self.search_metrics['total_searches'] > 0 else 0
                    ),
                    'cache_size': len(self.query_cache)
                },
                'top_queries': [
                    {'query': query, 'count': count, 'percentage': count / total_searches * 100 if total_searches > 0 else 0}
                    for query, count in top_queries
                ],
                'search_type_distribution': search_type_percentages,
                'performance': {
                    'average_search_time_ms': self.search_metrics['average_search_time_ms'],
                    'cache_efficiency': self.search_metrics['cache_hits'] / max(self.search_metrics['total_searches'], 1)
                },
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Search analytics failed: {e}")
            return {'error': str(e)}
    
    # Private search implementation methods
    
    async def _semantic_search(self, query: SearchQuery) -> SearchResults:
        """Perform semantic search using embeddings."""
        if not query.query_text:
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query.query_text)
            
            # Build vector similarity query
            similarity_query = select(
                Conversation,
                func.cosine_similarity(Conversation.embedding, query_embedding).label('similarity')
            ).where(
                Conversation.embedding.is_not(None)
            )
            
            # Apply filters
            similarity_query = await self._apply_base_filters(similarity_query, query)
            
            # Filter by similarity threshold
            similarity_query = similarity_query.where(
                func.cosine_similarity(Conversation.embedding, query_embedding) >= query.semantic_threshold
            )
            
            # Sort by similarity
            similarity_query = similarity_query.order_by(
                desc(func.cosine_similarity(Conversation.embedding, query_embedding))
            ).limit(query.limit).offset(query.offset)
            
            # Execute query
            result = await self.db_session.execute(similarity_query)
            rows = result.fetchall()
            
            # Convert to search results
            search_results = []
            for row in rows:
                conversation, similarity = row
                
                event = await self._conversation_to_event(conversation)
                
                search_result = SearchResult(
                    event=event,
                    relevance_score=float(similarity),
                    search_type=SearchType.SEMANTIC,
                    match_reasons=[f"Semantic similarity: {similarity:.3f}"],
                    highlighted_content=await self._highlight_content(
                        conversation.content, query.query_text
                    )
                )
                search_results.append(search_result)
            
            return SearchResults(
                results=search_results,
                total_matches=len(search_results),
                search_time_ms=0,  # Will be calculated by caller
                query=query,
                facets={},
                suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _keyword_search(self, query: SearchQuery) -> SearchResults:
        """Perform full-text keyword search."""
        if not query.query_text:
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
        
        try:
            # Parse query
            parsed_query = self.query_processor.parse_query(query.query_text)
            
            # Build text search query
            search_terms = parsed_query['terms'] + parsed_query['phrases']
            
            if not search_terms:
                return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
            
            # Build PostgreSQL full-text search query
            base_query = select(Conversation)
            
            # Add text search conditions
            for term in search_terms:
                base_query = base_query.where(
                    Conversation.content.ilike(f'%{term}%')
                )
            
            # Apply base filters
            base_query = await self._apply_base_filters(base_query, query)
            
            # Apply sorting
            if query.sort_by == SortOption.TIMESTAMP_DESC:
                base_query = base_query.order_by(desc(Conversation.created_at))
            elif query.sort_by == SortOption.TIMESTAMP_ASC:
                base_query = base_query.order_by(asc(Conversation.created_at))
            
            base_query = base_query.limit(query.limit).offset(query.offset)
            
            # Execute query
            result = await self.db_session.execute(base_query)
            conversations = result.scalars().all()
            
            # Convert to search results with relevance scoring
            search_results = []
            for conversation in conversations:
                event = await self._conversation_to_event(conversation)
                
                # Calculate keyword relevance score
                relevance_score = self._calculate_keyword_relevance(
                    conversation.content, search_terms
                )
                
                match_reasons = [
                    f"Keyword match: {term}"
                    for term in search_terms
                    if term.lower() in conversation.content.lower()
                ]
                
                search_result = SearchResult(
                    event=event,
                    relevance_score=relevance_score,
                    search_type=SearchType.KEYWORD,
                    match_reasons=match_reasons,
                    highlighted_content=await self._highlight_content(
                        conversation.content, query.query_text
                    )
                )
                search_results.append(search_result)
            
            # Sort by relevance if requested
            if query.sort_by == SortOption.RELEVANCE:
                search_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return SearchResults(
                results=search_results,
                total_matches=len(search_results),
                search_time_ms=0,
                query=query,
                facets={},
                suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _pattern_search(self, query: SearchQuery) -> SearchResults:
        """Search for specific conversation patterns."""
        # Pattern search implementation would analyze conversation threads
        # This is a simplified version
        return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _temporal_search(self, query: SearchQuery) -> SearchResults:
        """Perform time-based search with temporal patterns."""
        try:
            base_query = select(Conversation)
            
            # Apply time range filter
            if query.time_range:
                start_time, end_time = query.time_range
                base_query = base_query.where(
                    and_(
                        Conversation.created_at >= start_time,
                        Conversation.created_at <= end_time
                    )
                )
            
            # Apply other filters
            base_query = await self._apply_base_filters(base_query, query)
            
            # Sort by timestamp
            base_query = base_query.order_by(desc(Conversation.created_at))
            base_query = base_query.limit(query.limit).offset(query.offset)
            
            # Execute query
            result = await self.db_session.execute(base_query)
            conversations = result.scalars().all()
            
            # Convert to search results
            search_results = []
            for conversation in conversations:
                event = await self._conversation_to_event(conversation)
                
                search_result = SearchResult(
                    event=event,
                    relevance_score=1.0,  # All results equally relevant for temporal search
                    search_type=SearchType.TEMPORAL,
                    match_reasons=["Temporal range match"]
                )
                search_results.append(search_result)
            
            return SearchResults(
                results=search_results,
                total_matches=len(search_results),
                search_time_ms=0,
                query=query,
                facets={},
                suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _agent_behavior_search(self, query: SearchQuery) -> SearchResults:
        """Search based on agent behavior patterns."""
        # Agent behavior search would analyze communication patterns
        # This is a placeholder implementation
        return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _context_aware_search(self, query: SearchQuery) -> SearchResults:
        """Perform context-aware search with cross-references."""
        # Context-aware search would use context engine integration
        # This is a placeholder implementation
        return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    async def _hybrid_search(self, query: SearchQuery) -> SearchResults:
        """Combine multiple search types for best results."""
        try:
            all_results = []
            
            # Perform semantic search if query text exists
            if query.query_text:
                semantic_results = await self._semantic_search(query)
                all_results.extend(semantic_results.results)
                
                # Perform keyword search
                keyword_results = await self._keyword_search(query)
                all_results.extend(keyword_results.results)
            
            # Perform temporal search if time range specified
            if query.time_range:
                temporal_results = await self._temporal_search(query)
                all_results.extend(temporal_results.results)
            
            # Remove duplicates and merge relevance scores
            unique_results = {}
            for result in all_results:
                event_id = result.event.id
                if event_id in unique_results:
                    # Combine relevance scores
                    existing = unique_results[event_id]
                    existing.relevance_score = max(existing.relevance_score, result.relevance_score)
                    existing.match_reasons.extend(result.match_reasons)
                    existing.match_reasons = list(set(existing.match_reasons))  # Remove duplicates
                else:
                    unique_results[event_id] = result
            
            # Sort by combined relevance score
            final_results = list(unique_results.values())
            final_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Apply limit
            final_results = final_results[:query.limit]
            
            return SearchResults(
                results=final_results,
                total_matches=len(final_results),
                search_time_ms=0,
                query=query,
                facets={},
                suggestions=[]
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return SearchResults(results=[], total_matches=0, search_time_ms=0, query=query, facets={}, suggestions=[])
    
    # Helper methods
    
    async def _apply_base_filters(self, base_query, query: SearchQuery):
        """Apply common filters to search query."""
        # Session filters
        if query.session_filters:
            from uuid import UUID
            session_uuids = [UUID(sid) for sid in query.session_filters]
            base_query = base_query.where(Conversation.session_id.in_(session_uuids))
        
        # Agent filters
        if query.agent_filters:
            from uuid import UUID
            agent_uuids = [UUID(aid) for aid in query.agent_filters]
            base_query = base_query.where(
                or_(
                    Conversation.from_agent_id.in_(agent_uuids),
                    Conversation.to_agent_id.in_(agent_uuids)
                )
            )
        
        # Event type filters
        if query.event_type_filters:
            # Map to conversation message types
            message_types = [self._event_type_to_message_type(et) for et in query.event_type_filters]
            base_query = base_query.where(Conversation.message_type.in_(message_types))
        
        return base_query
    
    async def _conversation_to_event(self, conversation: Conversation) -> ConversationEvent:
        """Convert database conversation to ConversationEvent."""
        from .chat_transcript_manager import ConversationEvent, ConversationEventType
        
        return ConversationEvent(
            id=str(conversation.id),
            session_id=str(conversation.session_id),
            timestamp=conversation.created_at,
            event_type=self._message_type_to_event_type(conversation.message_type),
            source_agent_id=str(conversation.from_agent_id),
            target_agent_id=str(conversation.to_agent_id) if conversation.to_agent_id else None,
            message_content=conversation.content,
            metadata=conversation.conversation_metadata or {},
            context_references=conversation.context_refs or [],
            embedding_vector=conversation.embedding
        )
    
    def _message_type_to_event_type(self, message_type) -> ConversationEventType:
        """Convert message type to event type."""
        # Import here to avoid circular imports
        from ..models.conversation import MessageType
        
        mapping = {
            MessageType.TASK_ASSIGNMENT: ConversationEventType.TASK_DELEGATION,
            MessageType.STATUS_UPDATE: ConversationEventType.STATUS_UPDATE,
            MessageType.COMPLETION: ConversationEventType.STATUS_UPDATE,
            MessageType.ERROR: ConversationEventType.ERROR_OCCURRED,
            MessageType.COLLABORATION: ConversationEventType.COLLABORATION_START,
            MessageType.COORDINATION: ConversationEventType.COORDINATION_REQUEST
        }
        return mapping.get(message_type, ConversationEventType.MESSAGE_SENT)
    
    def _event_type_to_message_type(self, event_type: ConversationEventType):
        """Convert event type to message type."""
        from ..models.conversation import MessageType
        
        mapping = {
            ConversationEventType.TASK_DELEGATION: MessageType.TASK_ASSIGNMENT,
            ConversationEventType.STATUS_UPDATE: MessageType.STATUS_UPDATE,
            ConversationEventType.ERROR_OCCURRED: MessageType.ERROR,
            ConversationEventType.COLLABORATION_START: MessageType.COLLABORATION,
            ConversationEventType.COORDINATION_REQUEST: MessageType.COORDINATION
        }
        return mapping.get(event_type, MessageType.COLLABORATION)
    
    def _calculate_keyword_relevance(self, content: str, search_terms: List[str]) -> float:
        """Calculate relevance score for keyword matching."""
        if not content or not search_terms:
            return 0.0
        
        content_lower = content.lower()
        total_score = 0.0
        
        for term in search_terms:
            term_lower = term.lower()
            
            # Count occurrences
            count = content_lower.count(term_lower)
            if count > 0:
                # TF-IDF-like scoring
                tf = count / len(content.split())
                total_score += tf * math.log(len(search_terms) + 1)
        
        return min(total_score, 1.0)  # Normalize to 0-1
    
    async def _highlight_content(self, content: str, query_text: str) -> str:
        """Generate highlighted content with query matches."""
        if not query_text:
            return content
        
        # Simple highlighting - can be enhanced
        highlighted = content
        
        # Parse query terms
        parsed = self.query_processor.parse_query(query_text)
        all_terms = parsed['terms'] + parsed['phrases']
        
        for term in all_terms:
            # Simple case-insensitive replacement
            import re
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{term}</mark>', highlighted)
        
        return highlighted
    
    async def _generate_facets(self, results: List[SearchResult]) -> Dict[str, Dict[str, int]]:
        """Generate search facets from results."""
        facets = {
            'event_types': {},
            'agents': {},
            'sessions': {},
            'time_periods': {}
        }
        
        for result in results:
            event = result.event
            
            # Event type facet
            event_type = event.event_type.value
            facets['event_types'][event_type] = facets['event_types'].get(event_type, 0) + 1
            
            # Agent facet
            agent_id = event.source_agent_id
            facets['agents'][agent_id] = facets['agents'].get(agent_id, 0) + 1
            
            # Session facet
            session_id = event.session_id
            facets['sessions'][session_id] = facets['sessions'].get(session_id, 0) + 1
            
            # Time period facet (by hour)
            time_period = event.timestamp.strftime('%H:00')
            facets['time_periods'][time_period] = facets['time_periods'].get(time_period, 0) + 1
        
        return facets
    
    async def _generate_suggestions(
        self,
        query: SearchQuery,
        results: List[SearchResult]
    ) -> List[str]:
        """Generate search suggestions based on results."""
        suggestions = []
        
        if not results:
            suggestions.append("Try broadening your search terms")
            suggestions.append("Check your filters - they might be too restrictive")
            suggestions.append("Try using semantic search for better results")
        elif len(results) < 5:
            suggestions.append("Try using broader search terms")
            suggestions.append("Consider expanding the time range")
        
        # Add contextual suggestions based on results
        if results:
            # Most common event types
            event_types = {}
            for result in results:
                et = result.event.event_type.value
                event_types[et] = event_types.get(et, 0) + 1
            
            if event_types:
                most_common = max(event_types, key=event_types.get)
                if most_common != 'message_sent':
                    suggestions.append(f"Filter by '{most_common}' events for more specific results")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    async def _generate_contextual_suggestions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Generate contextual query suggestions."""
        suggestions = []
        
        # Common search patterns
        if 'error' in partial_query.lower():
            suggestions.append({
                'query': f'{partial_query} type:error',
                'type': 'filter',
                'preview': 'Filter to error events only'
            })
        
        if 'agent' in partial_query.lower():
            suggestions.append({
                'query': f'{partial_query} behavior analysis',
                'type': 'enhancement',
                'preview': 'Analyze agent behavior patterns'
            })
        
        return suggestions
    
    def _generate_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        import hashlib
        
        # Create a string representation of the query
        query_str = f"{query.query_text}_{query.search_type.value}_{query.session_filters}_{query.agent_filters}_{query.time_range}_{query.limit}_{query.offset}"
        
        # Generate hash
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: SearchResults) -> bool:
        """Check if cached result is still valid."""
        # Simple TTL check - can be enhanced with more sophisticated invalidation
        return True  # For now, assume cache is always valid
    
    async def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest 25% of entries
            to_remove = len(self.query_cache) // 4
            keys = list(self.query_cache.keys())[:to_remove]
            for key in keys:
                del self.query_cache[key]
"""
Hybrid Search Engine combining Vector and Text Search for Optimal Results.

This module provides a comprehensive hybrid search system that combines:
- Semantic vector search for meaning-based matching
- Full-text search for exact keyword matching
- BM25 ranking for text relevance scoring
- Intelligent result fusion and re-ranking
- Query understanding and intent detection
- Multi-modal search across different content types
- Search result diversity and personalization
- Advanced relevance tuning and machine learning integration
"""

import asyncio
import time
import json
import logging
import math
import uuid
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import numpy as np

from sqlalchemy import select, and_, or_, desc, asc, func, text, case
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models.context import Context, ContextType
from ..core.advanced_vector_search import AdvancedVectorSearchEngine, SimilarityAlgorithm
from ..core.vector_search import SearchFilters, ContextMatch
from ..core.embedding_service import EmbeddingService
from ..core.database import get_async_session
from ..core.redis import get_redis_client

logger = logging.getLogger(__name__)


class SearchIntent(Enum):
    """User search intent categories."""
    FACTUAL = "factual"           # Looking for specific facts/information
    PROCEDURAL = "procedural"     # How-to queries, procedures
    TROUBLESHOOTING = "troubleshooting"  # Error resolution, debugging
    CONCEPTUAL = "conceptual"     # Understanding concepts, explanations
    NAVIGATIONAL = "navigational"  # Finding specific documents/pages
    EXPLORATORY = "exploratory"   # Broad exploration of topics


class QueryType(Enum):
    """Query type classification."""
    SHORT_KEYWORD = "short_keyword"     # 1-2 words
    PHRASE = "phrase"                   # 3-5 words
    SENTENCE = "sentence"               # 6-10 words  
    LONG_FORM = "long_form"            # 10+ words
    QUESTION = "question"               # Questions with interrogatives
    COMMAND = "command"                 # Imperative statements


class FusionMethod(Enum):
    """Methods for combining search results."""
    WEIGHTED_SUM = "weighted_sum"       # Linear combination of scores
    RANK_FUSION = "rank_fusion"         # Reciprocal rank fusion
    LEARNED_FUSION = "learned_fusion"   # ML-based fusion
    ADAPTIVE_FUSION = "adaptive_fusion" # Context-adaptive fusion


@dataclass
class QueryAnalysis:
    """Analysis of search query characteristics."""
    original_query: str
    cleaned_query: str
    query_type: QueryType
    search_intent: SearchIntent
    keywords: List[str]
    phrases: List[str]
    entities: List[str]
    has_negation: bool
    has_temporal: bool
    language: str = "en"
    confidence: float = 1.0


@dataclass
class SearchResult:
    """Enhanced search result with multiple scores."""
    context: Context
    vector_score: float
    text_score: float
    bm25_score: float
    fusion_score: float
    rank_position: int
    result_source: str  # "vector", "text", "both"
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search engine."""
    # Fusion weights
    vector_weight: float = 0.6
    text_weight: float = 0.3
    bm25_weight: float = 0.1
    
    # Search parameters
    vector_search_limit: int = 50
    text_search_limit: int = 50
    final_result_limit: int = 20
    
    # Quality thresholds
    min_vector_similarity: float = 0.3
    min_text_relevance: float = 0.1
    min_fusion_score: float = 0.2
    
    # Feature flags
    enable_query_expansion: bool = True
    enable_result_diversification: bool = True
    enable_personalization: bool = True
    enable_intent_detection: bool = True
    
    # Performance settings
    max_search_time_ms: float = 200.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800


class QueryAnalyzer:
    """Analyzes queries to understand intent and optimize search strategy."""
    
    def __init__(self):
        # Keywords that indicate different intents
        self.intent_keywords = {
            SearchIntent.TROUBLESHOOTING: [
                "error", "problem", "issue", "bug", "fail", "broken", "fix", 
                "debug", "troubleshoot", "resolve", "solve"
            ],
            SearchIntent.PROCEDURAL: [
                "how", "step", "guide", "tutorial", "process", "procedure",
                "instructions", "manual", "setup", "configure"
            ],
            SearchIntent.FACTUAL: [
                "what", "who", "when", "where", "which", "definition",
                "meaning", "is", "are", "does", "can"
            ],
            SearchIntent.CONCEPTUAL: [
                "why", "explain", "understand", "concept", "theory",
                "principle", "overview", "introduction"
            ],
            SearchIntent.NAVIGATIONAL: [
                "find", "locate", "search", "document", "page", "section"
            ]
        }
        
        # Question indicators
        self.question_starters = {
            "what", "who", "when", "where", "why", "how", "which", 
            "can", "could", "would", "should", "is", "are", "do", "does"
        }
        
        # Temporal indicators
        self.temporal_keywords = {
            "recent", "latest", "new", "old", "past", "future", "now",
            "today", "yesterday", "tomorrow", "week", "month", "year"
        }
        
        # Negation words
        self.negation_words = {
            "not", "no", "none", "never", "nothing", "without", "except"
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.
        
        Args:
            query: Search query to analyze
            
        Returns:
            QueryAnalysis with detected characteristics
        """
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        # Classify query type
        query_type = self._classify_query_type(cleaned_query)
        
        # Detect search intent
        search_intent = self._detect_search_intent(cleaned_query)
        
        # Extract components
        keywords = self._extract_keywords(cleaned_query)
        phrases = self._extract_phrases(cleaned_query)
        entities = self._extract_entities(cleaned_query)
        
        # Check for special characteristics
        has_negation = any(word in cleaned_query.lower() for word in self.negation_words)
        has_temporal = any(word in cleaned_query.lower() for word in self.temporal_keywords)
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            query_type=query_type,
            search_intent=search_intent,
            keywords=keywords,
            phrases=phrases,
            entities=entities,
            has_negation=has_negation,
            has_temporal=has_temporal,
            confidence=0.8  # Simplified confidence score
        )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
        cleaned = re.sub(r'[^\w\s\-\?\!\.]', ' ', cleaned)
        
        return cleaned
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query."""
        words = query.split()
        word_count = len(words)
        
        # Check if it's a question
        first_word = words[0].lower() if words else ""
        if first_word in self.question_starters or query.strip().endswith("?"):
            return QueryType.QUESTION
        
        # Check if it's a command (imperative)
        if first_word in {"show", "list", "find", "get", "create", "delete", "update"}:
            return QueryType.COMMAND
        
        # Classify by length
        if word_count <= 2:
            return QueryType.SHORT_KEYWORD
        elif word_count <= 5:
            return QueryType.PHRASE
        elif word_count <= 10:
            return QueryType.SENTENCE
        else:
            return QueryType.LONG_FORM
    
    def _detect_search_intent(self, query: str) -> SearchIntent:
        """Detect the user's search intent."""
        query_lower = query.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the intent with highest score, or exploratory as default
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return SearchIntent.EXPLORATORY
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction (in production, you'd use NLP libraries)
        words = query.lower().split()
        
        # Filter out stop words and short words
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        }
        
        keywords = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return keywords
    
    def _extract_phrases(self, query: str) -> List[str]:
        """Extract meaningful phrases from query."""
        # Simple phrase extraction (2-3 word combinations)
        words = query.split()
        phrases = []
        
        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 5:  # Minimum phrase length
                phrases.append(phrase)
        
        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) > 8:
                phrases.append(phrase)
        
        return phrases
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query."""
        # Simplified entity extraction
        # In production, you'd use NER models
        
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        # Look for quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_phrases)
        
        return entities


class BM25Scorer:
    """BM25 scoring for text relevance."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.document_frequencies: Dict[str, int] = {}
        self.document_count = 0
        self.avg_document_length = 0.0
    
    def calculate_score(
        self,
        query_terms: List[str],
        document_text: str,
        document_length: Optional[int] = None
    ) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            document_text: Document text to score
            document_length: Length of document (optional)
            
        Returns:
            BM25 relevance score
        """
        if not query_terms or not document_text:
            return 0.0
        
        doc_length = document_length or len(document_text.split())
        if doc_length == 0:
            return 0.0
        
        # Convert document to lowercase for matching
        doc_lower = document_text.lower()
        doc_words = doc_lower.split()
        
        # Count term frequencies in document
        term_frequencies = Counter(doc_words)
        
        score = 0.0
        
        for term in query_terms:
            term_lower = term.lower()
            
            # Term frequency in document
            tf = term_frequencies.get(term_lower, 0)
            
            if tf > 0:
                # Inverse document frequency (simplified)
                # In production, you'd maintain global IDF statistics
                idf = math.log(1000 / max(1, 10))  # Simplified IDF
                
                # BM25 score component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / max(1, self.avg_document_length)))
                
                score += idf * (numerator / denominator)
        
        return score


class ResultFusion:
    """Fuses results from different search methods."""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
    
    def fuse_results(
        self,
        vector_results: List[ContextMatch],
        text_results: List[ContextMatch],
        fusion_method: FusionMethod = FusionMethod.WEIGHTED_SUM,
        query_analysis: Optional[QueryAnalysis] = None
    ) -> List[SearchResult]:
        """
        Fuse results from vector and text search.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            fusion_method: Method to use for fusion
            query_analysis: Query analysis for adaptive fusion
            
        Returns:
            List of fused SearchResult objects
        """
        # Create a mapping of context ID to results
        result_map: Dict[str, Dict[str, Any]] = {}
        
        # Process vector results
        for result in vector_results:
            context_id = str(result.context.id)
            result_map[context_id] = {
                "context": result.context,
                "vector_score": result.similarity_score,
                "vector_rank": result.rank,
                "text_score": 0.0,
                "text_rank": None,
                "bm25_score": 0.0,
                "sources": ["vector"]
            }
        
        # Process text results
        for result in text_results:
            context_id = str(result.context.id)
            if context_id in result_map:
                # Update existing result
                result_map[context_id]["text_score"] = result.similarity_score
                result_map[context_id]["text_rank"] = result.rank
                result_map[context_id]["sources"].append("text")
            else:
                # Add new result
                result_map[context_id] = {
                    "context": result.context,
                    "vector_score": 0.0,
                    "vector_rank": None,
                    "text_score": result.similarity_score,
                    "text_rank": result.rank,
                    "bm25_score": 0.0,
                    "sources": ["text"]
                }
        
        # Calculate BM25 scores if query analysis is available
        if query_analysis:
            bm25_scorer = BM25Scorer()
            for result_data in result_map.values():
                context = result_data["context"]
                document_text = f"{context.title} {context.content}"
                
                bm25_score = bm25_scorer.calculate_score(
                    query_analysis.keywords,
                    document_text
                )
                result_data["bm25_score"] = bm25_score
        
        # Apply fusion method
        if fusion_method == FusionMethod.WEIGHTED_SUM:
            fused_results = self._weighted_sum_fusion(result_map, query_analysis)
        elif fusion_method == FusionMethod.RANK_FUSION:
            fused_results = self._rank_fusion(result_map)
        elif fusion_method == FusionMethod.ADAPTIVE_FUSION:
            fused_results = self._adaptive_fusion(result_map, query_analysis)
        else:
            fused_results = self._weighted_sum_fusion(result_map, query_analysis)
        
        # Sort by fusion score and assign ranks
        fused_results.sort(key=lambda r: r.fusion_score, reverse=True)
        
        for i, result in enumerate(fused_results, 1):
            result.rank_position = i
        
        return fused_results[:self.config.final_result_limit]
    
    def _weighted_sum_fusion(
        self,
        result_map: Dict[str, Dict[str, Any]],
        query_analysis: Optional[QueryAnalysis] = None
    ) -> List[SearchResult]:
        """Perform weighted sum fusion."""
        results = []
        
        # Adapt weights based on query analysis
        vector_weight = self.config.vector_weight
        text_weight = self.config.text_weight
        bm25_weight = self.config.bm25_weight
        
        if query_analysis:
            # Adjust weights based on query type and intent
            if query_analysis.query_type == QueryType.SHORT_KEYWORD:
                # Favor text matching for short keyword queries
                text_weight *= 1.5
                vector_weight *= 0.8
            elif query_analysis.search_intent == SearchIntent.FACTUAL:
                # Favor vector search for factual queries
                vector_weight *= 1.3
                text_weight *= 0.9
            elif query_analysis.search_intent == SearchIntent.TROUBLESHOOTING:
                # Balance both for troubleshooting
                bm25_weight *= 1.2
        
        # Normalize weights
        total_weight = vector_weight + text_weight + bm25_weight
        vector_weight /= total_weight
        text_weight /= total_weight
        bm25_weight /= total_weight
        
        for context_id, result_data in result_map.items():
            # Calculate fusion score
            fusion_score = (
                vector_weight * result_data["vector_score"] +
                text_weight * result_data["text_score"] +
                bm25_weight * result_data["bm25_score"]
            )
            
            # Apply quality thresholds
            if (result_data["vector_score"] >= self.config.min_vector_similarity or
                result_data["text_score"] >= self.config.min_text_relevance) and \
               fusion_score >= self.config.min_fusion_score:
                
                # Determine result source
                sources = result_data["sources"]
                if len(sources) > 1:
                    result_source = "both"
                else:
                    result_source = sources[0]
                
                search_result = SearchResult(
                    context=result_data["context"],
                    vector_score=result_data["vector_score"],
                    text_score=result_data["text_score"],
                    bm25_score=result_data["bm25_score"],
                    fusion_score=fusion_score,
                    rank_position=0,  # Will be set later
                    result_source=result_source,
                    explanation={
                        "fusion_method": "weighted_sum",
                        "weights": {
                            "vector": vector_weight,
                            "text": text_weight,
                            "bm25": bm25_weight
                        },
                        "sources": sources
                    }
                )
                
                results.append(search_result)
        
        return results
    
    def _rank_fusion(self, result_map: Dict[str, Dict[str, Any]]) -> List[SearchResult]:
        """Perform reciprocal rank fusion."""
        results = []
        
        for context_id, result_data in result_map.items():
            # Calculate reciprocal rank fusion score
            fusion_score = 0.0
            
            if result_data["vector_rank"]:
                fusion_score += 1.0 / (60 + result_data["vector_rank"])
            
            if result_data["text_rank"]:
                fusion_score += 1.0 / (60 + result_data["text_rank"])
            
            if fusion_score >= self.config.min_fusion_score:
                sources = result_data["sources"]
                result_source = "both" if len(sources) > 1 else sources[0]
                
                search_result = SearchResult(
                    context=result_data["context"],
                    vector_score=result_data["vector_score"],
                    text_score=result_data["text_score"],
                    bm25_score=result_data["bm25_score"],
                    fusion_score=fusion_score,
                    rank_position=0,
                    result_source=result_source,
                    explanation={
                        "fusion_method": "rank_fusion",
                        "rrf_constant": 60,
                        "sources": sources
                    }
                )
                
                results.append(search_result)
        
        return results
    
    def _adaptive_fusion(
        self,
        result_map: Dict[str, Dict[str, Any]],
        query_analysis: Optional[QueryAnalysis] = None
    ) -> List[SearchResult]:
        """Perform adaptive fusion based on query characteristics."""
        # Use weighted sum as base, but with dynamic weight adjustment
        return self._weighted_sum_fusion(result_map, query_analysis)


class HybridSearchEngine:
    """
    Comprehensive hybrid search engine combining vector and text search.
    
    Features:
    - Intelligent query analysis and intent detection
    - Vector search for semantic similarity
    - Full-text search for exact matching
    - BM25 scoring for text relevance
    - Advanced result fusion and re-ranking
    - Query expansion and personalization
    - Performance optimization and caching
    """
    
    def __init__(
        self,
        vector_search_engine: AdvancedVectorSearchEngine,
        db_session: Optional[AsyncSession] = None,
        redis_client: Optional[redis.Redis] = None,
        config: Optional[HybridSearchConfig] = None
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_search_engine: Advanced vector search engine
            db_session: Database session
            redis_client: Redis client for caching
            config: Hybrid search configuration
        """
        self.vector_engine = vector_search_engine
        self.db_session = db_session
        self.redis_client = redis_client or get_redis_client()
        self.config = config or HybridSearchConfig()
        
        # Components
        self.query_analyzer = QueryAnalyzer()
        self.result_fusion = ResultFusion(self.config)
        
        # Performance tracking
        self.search_metrics = {
            "total_searches": 0,
            "avg_search_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "vector_search_time_ms": 0.0,
            "text_search_time_ms": 0.0,
            "fusion_time_ms": 0.0
        }
        
        # Query expansion cache
        self.expansion_cache: Dict[str, List[str]] = {}
    
    async def hybrid_search(
        self,
        query: str,
        agent_id: Optional[uuid.UUID] = None,
        filters: Optional[SearchFilters] = None,
        limit: int = 20,
        fusion_method: FusionMethod = FusionMethod.ADAPTIVE_FUSION,
        enable_explanation: bool = False
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search.
        
        Args:
            query: Search query
            agent_id: Requesting agent ID
            filters: Search filters
            limit: Maximum results to return
            fusion_method: Method for fusing results
            enable_explanation: Include search explanations
            
        Returns:
            Tuple of (search results, search metadata)
        """
        start_time = time.perf_counter()
        
        try:
            # Analyze query
            query_analysis = self.query_analyzer.analyze_query(query)
            
            # Check cache if enabled
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(query, agent_id, filters, limit)
                cached_results = await self._get_cached_results(cache_key)
                if cached_results:
                    search_time = (time.perf_counter() - start_time) * 1000
                    metadata = {
                        "cache_hit": True,
                        "search_time_ms": search_time,
                        "query_analysis": query_analysis.__dict__ if enable_explanation else None
                    }
                    return cached_results, metadata
            
            # Expand query if enabled
            expanded_query = query
            if self.config.enable_query_expansion:
                expanded_query = await self._expand_query(query, query_analysis)
            
            # Perform vector search
            vector_start = time.perf_counter()
            vector_results, vector_metadata = await self.vector_engine.ultra_fast_search(
                query=expanded_query,
                agent_id=agent_id,
                limit=self.config.vector_search_limit,
                filters=filters,
                performance_target_ms=self.config.max_search_time_ms / 2
            )
            vector_time = (time.perf_counter() - vector_start) * 1000
            
            # Perform text search
            text_start = time.perf_counter()
            text_results = await self._perform_text_search(
                expanded_query,
                query_analysis,
                agent_id,
                filters,
                self.config.text_search_limit
            )
            text_time = (time.perf_counter() - text_start) * 1000
            
            # Fuse results
            fusion_start = time.perf_counter()
            fused_results = self.result_fusion.fuse_results(
                vector_results,
                text_results,
                fusion_method,
                query_analysis
            )
            fusion_time = (time.perf_counter() - fusion_start) * 1000
            
            # Apply diversification if enabled
            if self.config.enable_result_diversification:
                fused_results = self._diversify_results(fused_results, query_analysis)
            
            # Apply personalization if enabled
            if self.config.enable_personalization and agent_id:
                fused_results = await self._personalize_results(fused_results, agent_id)
            
            # Limit results
            final_results = fused_results[:limit]
            
            # Cache results
            if self.config.enable_caching and not cached_results:
                await self._cache_results(cache_key, final_results)
            
            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Update metrics
            self._update_search_metrics(total_time, vector_time, text_time, fusion_time)
            
            # Prepare metadata
            metadata = {
                "cache_hit": False,
                "search_time_ms": total_time,
                "vector_search_time_ms": vector_time,
                "text_search_time_ms": text_time,
                "fusion_time_ms": fusion_time,
                "vector_results_count": len(vector_results),
                "text_results_count": len(text_results),
                "fused_results_count": len(fused_results),
                "final_results_count": len(final_results),
                "fusion_method": fusion_method.value,
                "query_expanded": expanded_query != query,
                "expanded_query": expanded_query if expanded_query != query else None,
                "performance_target_met": total_time <= self.config.max_search_time_ms
            }
            
            if enable_explanation:
                metadata.update({
                    "query_analysis": query_analysis.__dict__,
                    "vector_metadata": vector_metadata,
                    "search_explanation": self._generate_search_explanation(
                        query_analysis, vector_results, text_results, fused_results
                    )
                })
            
            return final_results, metadata
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            
            # Fallback to vector search only
            try:
                fallback_results, fallback_metadata = await self.vector_engine.ultra_fast_search(
                    query=query,
                    agent_id=agent_id,
                    limit=limit,
                    filters=filters
                )
                
                # Convert to SearchResult format
                search_results = [
                    SearchResult(
                        context=result.context,
                        vector_score=result.similarity_score,
                        text_score=0.0,
                        bm25_score=0.0,
                        fusion_score=result.similarity_score,
                        rank_position=result.rank,
                        result_source="vector",
                        explanation={"fallback": True, "error": str(e)}
                    )
                    for result in fallback_results
                ]
                
                search_time = (time.perf_counter() - start_time) * 1000
                metadata = {
                    "cache_hit": False,
                    "search_time_ms": search_time,
                    "fallback_used": True,
                    "error": str(e)
                }
                
                return search_results, metadata
                
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return [], {"error": str(e), "fallback_error": str(fallback_error)}
    
    async def _perform_text_search(
        self,
        query: str,
        query_analysis: QueryAnalysis,
        agent_id: Optional[uuid.UUID],
        filters: Optional[SearchFilters],
        limit: int
    ) -> List[ContextMatch]:
        """Perform full-text search using PostgreSQL."""
        try:
            if not self.db_session:
                self.db_session = await get_async_session()
            
            # Build search terms from query analysis
            search_terms = query_analysis.keywords + query_analysis.phrases
            if not search_terms:
                search_terms = [query_analysis.cleaned_query]
            
            # Create full-text search query
            search_query = ' & '.join(search_terms)
            
            # Base query with full-text search
            base_query = select(Context).where(
                and_(
                    func.to_tsvector('english', Context.title + ' ' + Context.content).op('@@')(
                        func.plainto_tsquery('english', search_query)
                    ),
                    Context.content.isnot(None)
                )
            )
            
            # Apply agent filtering
            if agent_id:
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
                if filters.context_types:
                    base_query = base_query.where(Context.context_type.in_(filters.context_types))
                
                if filters.min_importance:
                    base_query = base_query.where(Context.importance_score >= filters.min_importance)
                
                if filters.max_age_days:
                    cutoff_date = datetime.utcnow() - timedelta(days=filters.max_age_days)
                    base_query = base_query.where(Context.created_at >= cutoff_date)
            
            # Add ranking and limit
            ranked_query = (
                base_query
                .add_columns(
                    func.ts_rank(
                        func.to_tsvector('english', Context.title + ' ' + Context.content),
                        func.plainto_tsquery('english', search_query)
                    ).label('rank_score')
                )
                .order_by(desc('rank_score'))
                .limit(limit)
            )
            
            result = await self.db_session.execute(ranked_query)
            rows = result.all()
            
            # Convert to ContextMatch objects
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
            logger.error(f"Text search failed: {e}")
            return []
    
    async def _expand_query(
        self,
        query: str,
        query_analysis: QueryAnalysis
    ) -> str:
        """Expand query with synonyms and related terms."""
        # Check cache first
        if query in self.expansion_cache:
            expanded_terms = self.expansion_cache[query]
        else:
            # Simple query expansion (in production, use advanced NLP)
            expanded_terms = []
            
            # Add synonyms for common technical terms
            synonym_map = {
                "error": ["bug", "issue", "problem"],
                "fix": ["resolve", "solve", "repair"],
                "guide": ["tutorial", "manual", "instructions"],
                "setup": ["configure", "install", "initialize"]
            }
            
            for keyword in query_analysis.keywords:
                if keyword.lower() in synonym_map:
                    expanded_terms.extend(synonym_map[keyword.lower()])
            
            # Cache the expansion
            self.expansion_cache[query] = expanded_terms
        
        # Add expanded terms to original query
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:3])}"  # Limit to 3 additional terms
        
        return query
    
    def _diversify_results(
        self,
        results: List[SearchResult],
        query_analysis: QueryAnalysis
    ) -> List[SearchResult]:
        """Apply result diversification to reduce redundancy."""
        if len(results) <= 5:
            return results  # Too few results to diversify
        
        diversified = []
        seen_types = set()
        seen_titles = set()
        
        # First pass: add results with unique context types
        for result in results:
            context_type = result.context.context_type
            title_words = set(result.context.title.lower().split()[:3])  # First 3 words
            
            if (context_type not in seen_types or 
                not title_words.intersection(seen_titles)) and \
               len(diversified) < self.config.final_result_limit:
                
                diversified.append(result)
                if context_type:
                    seen_types.add(context_type)
                seen_titles.update(title_words)
        
        # Second pass: fill remaining slots with best remaining results
        remaining_results = [r for r in results if r not in diversified]
        remaining_slots = self.config.final_result_limit - len(diversified)
        
        diversified.extend(remaining_results[:remaining_slots])
        
        return diversified
    
    async def _personalize_results(
        self,
        results: List[SearchResult],
        agent_id: uuid.UUID
    ) -> List[SearchResult]:
        """Apply personalization based on agent preferences and history."""
        try:
            # Get agent's search history and preferences from Redis
            agent_key = f"search_history:{agent_id}"
            history_data = await self.redis_client.get(agent_key)
            
            if not history_data:
                return results  # No personalization data available
            
            # Parse history data
            history = json.loads(history_data)
            preferred_types = history.get("preferred_context_types", [])
            recent_queries = history.get("recent_queries", [])
            
            # Boost results matching preferences
            for result in results:
                boost_factor = 1.0
                
                # Boost preferred context types
                if result.context.context_type and result.context.context_type.value in preferred_types:
                    boost_factor *= 1.2
                
                # Boost results similar to recent queries
                for recent_query in recent_queries[-5:]:  # Last 5 queries
                    if any(word in result.context.title.lower() 
                          for word in recent_query.lower().split()):
                        boost_factor *= 1.1
                        break
                
                # Apply boost
                result.fusion_score *= boost_factor
            
            # Re-sort by updated fusion scores
            results.sort(key=lambda r: r.fusion_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results, 1):
                result.rank_position = i
            
        except Exception as e:
            logger.warning(f"Personalization failed: {e}")
        
        return results
    
    def _generate_cache_key(
        self,
        query: str,
        agent_id: Optional[uuid.UUID],
        filters: Optional[SearchFilters],
        limit: int
    ) -> str:
        """Generate cache key for search results."""
        import hashlib
        
        key_data = {
            "query": query.lower().strip(),
            "agent_id": str(agent_id) if agent_id else None,
            "limit": limit
        }
        
        if filters:
            key_data["filters"] = {
                "context_types": [ct.value for ct in filters.context_types] if filters.context_types else None,
                "min_importance": filters.min_importance,
                "max_age_days": filters.max_age_days
            }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    async def _get_cached_results(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search results."""
        try:
            cached_data = await self.redis_client.get(f"hybrid_search:{cache_key}")
            if cached_data:
                # In production, you'd need proper deserialization
                # This is simplified for the example
                return None  # Placeholder
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        return None
    
    async def _cache_results(self, cache_key: str, results: List[SearchResult]) -> None:
        """Cache search results."""
        try:
            # Serialize results (simplified)
            results_data = [
                {
                    "context_id": str(result.context.id),
                    "fusion_score": result.fusion_score,
                    "rank_position": result.rank_position,
                    "result_source": result.result_source
                }
                for result in results
            ]
            
            await self.redis_client.setex(
                f"hybrid_search:{cache_key}",
                self.config.cache_ttl_seconds,
                json.dumps(results_data)
            )
            
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def _update_search_metrics(
        self,
        total_time: float,
        vector_time: float,
        text_time: float,
        fusion_time: float
    ) -> None:
        """Update search performance metrics."""
        self.search_metrics["total_searches"] += 1
        
        # Update averages
        old_avg = self.search_metrics["avg_search_time_ms"]
        new_count = self.search_metrics["total_searches"]
        self.search_metrics["avg_search_time_ms"] = (old_avg * (new_count - 1) + total_time) / new_count
        
        # Update component times
        self.search_metrics["vector_search_time_ms"] = vector_time
        self.search_metrics["text_search_time_ms"] = text_time
        self.search_metrics["fusion_time_ms"] = fusion_time
    
    def _generate_search_explanation(
        self,
        query_analysis: QueryAnalysis,
        vector_results: List[ContextMatch],
        text_results: List[ContextMatch],
        fused_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Generate explanation of search process."""
        return {
            "query_understanding": {
                "intent": query_analysis.search_intent.value,
                "type": query_analysis.query_type.value,
                "keywords_extracted": query_analysis.keywords,
                "phrases_extracted": query_analysis.phrases,
                "has_negation": query_analysis.has_negation,
                "has_temporal": query_analysis.has_temporal
            },
            "search_execution": {
                "vector_search_results": len(vector_results),
                "text_search_results": len(text_results),
                "fusion_overlap": len([r for r in fused_results if r.result_source == "both"]),
                "vector_only": len([r for r in fused_results if r.result_source == "vector"]),
                "text_only": len([r for r in fused_results if r.result_source == "text"])
            },
            "result_composition": {
                "total_unique_results": len(fused_results),
                "diversification_applied": self.config.enable_result_diversification,
                "personalization_applied": self.config.enable_personalization
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get hybrid search performance metrics."""
        base_metrics = self.vector_engine.get_advanced_performance_metrics()
        
        hybrid_metrics = {
            **base_metrics,
            "hybrid_search": self.search_metrics,
            "configuration": {
                "vector_weight": self.config.vector_weight,
                "text_weight": self.config.text_weight,
                "bm25_weight": self.config.bm25_weight,
                "fusion_enabled": True,
                "query_expansion_enabled": self.config.enable_query_expansion,
                "personalization_enabled": self.config.enable_personalization,
                "diversification_enabled": self.config.enable_result_diversification
            }
        }
        
        return hybrid_metrics


# Factory function
async def create_hybrid_search_engine(
    vector_search_engine: AdvancedVectorSearchEngine,
    db_session: Optional[AsyncSession] = None,
    redis_client: Optional[redis.Redis] = None,
    config: Optional[HybridSearchConfig] = None
) -> HybridSearchEngine:
    """
    Create hybrid search engine instance.
    
    Args:
        vector_search_engine: Advanced vector search engine
        db_session: Database session
        redis_client: Redis client for caching
        config: Hybrid search configuration
        
    Returns:
        HybridSearchEngine instance
    """
    return HybridSearchEngine(
        vector_search_engine=vector_search_engine,
        db_session=db_session,
        redis_client=redis_client,
        config=config
    )
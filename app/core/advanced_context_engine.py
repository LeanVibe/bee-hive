"""
Advanced Context Engine with Semantic Memory for LeanVibe Agent Hive 2.0

This module provides intelligent context compression (60-80%) and cross-agent
knowledge sharing through semantic memory systems and advanced consolidation.
"""

import asyncio
import json
import time
import hashlib
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from enum import Enum
import re
import statistics
import pickle
from pathlib import Path

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from .context_compression import ContextCompressor
from .semantic_memory import SemanticMemoryService
from .config import settings
from .redis import get_redis, get_session_cache

logger = structlog.get_logger()


@dataclass
class ContextSegment:
    """A segment of context with metadata for compression analysis."""
    id: str
    content: str
    timestamp: datetime
    importance_score: float = 0.0
    access_frequency: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    semantic_tags: List[str] = field(default_factory=list)
    compression_ratio: float = 0.0
    original_size: int = 0
    compressed_size: int = 0


@dataclass
class CompressionMetrics:
    """Metrics for context compression operations."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    processing_time: float
    segments_processed: int
    semantic_clusters: int
    knowledge_entities: int
    cross_references: int


@dataclass
class SemanticKnowledge:
    """Semantic knowledge entity for cross-agent sharing."""
    entity_id: str
    entity_type: str
    content: str
    confidence: float
    created_by: str  # agent_id
    created_at: datetime
    accessed_by: Set[str] = field(default_factory=set)
    access_count: int = 0
    semantic_vector: Optional[List[float]] = None
    related_entities: List[str] = field(default_factory=list)


class CompressionStrategy(Enum):
    """Different context compression strategies."""
    SEMANTIC_CLUSTERING = "semantic_clustering"
    IMPORTANCE_RANKING = "importance_ranking"
    FREQUENCY_BASED = "frequency_based"
    TEMPORAL_DECAY = "temporal_decay"
    CROSS_REFERENCE = "cross_reference"
    HYBRID = "hybrid"


class KnowledgeType(Enum):
    """Types of semantic knowledge entities."""
    CONCEPT = "concept"
    PATTERN = "pattern"
    SOLUTION = "solution"
    ERROR_FIX = "error_fix"
    BEST_PRACTICE = "best_practice"
    CODE_SNIPPET = "code_snippet"
    WORKFLOW = "workflow"
    INSIGHT = "insight"


class AdvancedContextEngine:
    """
    Advanced Context Engine with 60-80% compression and semantic memory.
    
    Features:
    - Intelligent context compression using multiple strategies
    - Semantic clustering and importance ranking
    - Cross-agent knowledge sharing
    - Real-time context optimization
    - Memory-efficient storage and retrieval
    - Performance analytics and optimization
    """
    
    def __init__(self):
        self.compression_target = 0.70  # 70% compression target
        self.semantic_threshold = 0.85  # Semantic similarity threshold
        
        # Context storage
        self.context_segments: Dict[str, ContextSegment] = {}
        self.compressed_contexts: Dict[str, bytes] = {}
        self.semantic_knowledge: Dict[str, SemanticKnowledge] = {}
        
        # Compression components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self.pca_reducer = PCA(n_components=100)
        self.clusterer = KMeans(n_clusters=10, random_state=42)
        
        # Performance tracking
        self.compression_history = deque(maxlen=1000)
        self.knowledge_access_patterns = defaultdict(lambda: deque(maxlen=100))
        self.cross_agent_shares = defaultdict(int)
        
        # Caches for optimization
        self.semantic_vectors_cache: Dict[str, np.ndarray] = {}
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Configuration
        self.max_segment_size = 2000  # characters
        self.min_importance_score = 0.3
        self.knowledge_expiry_days = 30
        
        logger.info("🧠 Advanced Context Engine initialized with semantic memory")
    
    async def compress_context(
        self, 
        context: str, 
        agent_id: str,
        strategy: CompressionStrategy = CompressionStrategy.HYBRID
    ) -> Dict[str, Any]:
        """
        Compress context using advanced semantic analysis.
        
        Args:
            context: Raw context to compress
            agent_id: ID of the agent requesting compression
            strategy: Compression strategy to use
        
        Returns:
            Compression result with metrics
        """
        start_time = time.time()
        original_size = len(context.encode('utf-8'))
        
        logger.info(f"🔧 Starting context compression for agent {agent_id} ({original_size:,} bytes)")
        
        # Step 1: Segment context into logical chunks
        segments = await self._segment_context(context, agent_id)
        
        # Step 2: Analyze semantic importance
        segments = await self._analyze_semantic_importance(segments)
        
        # Step 3: Apply compression strategy
        compressed_segments = await self._apply_compression_strategy(segments, strategy)
        
        # Step 4: Generate semantic knowledge
        knowledge_entities = await self._extract_semantic_knowledge(segments, agent_id)
        
        # Step 5: Cross-reference with existing knowledge
        cross_references = await self._create_cross_references(knowledge_entities)
        
        # Step 6: Build compressed context
        compressed_context = await self._build_compressed_context(compressed_segments, cross_references)
        
        # Step 7: Store and cache results
        context_id = await self._store_compressed_context(compressed_context, agent_id)
        
        # Calculate metrics
        compressed_size = len(compressed_context.encode('utf-8'))
        compression_ratio = 1 - (compressed_size / original_size)
        processing_time = time.time() - start_time
        
        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            processing_time=processing_time,
            segments_processed=len(segments),
            semantic_clusters=len(set(s.semantic_tags[0] if s.semantic_tags else 'default' for s in segments)),
            knowledge_entities=len(knowledge_entities),
            cross_references=len(cross_references)
        )
        
        self.compression_history.append(metrics)
        
        logger.info(f"✅ Context compression complete: {compression_ratio:.1%} reduction in {processing_time:.2f}s")
        
        return {
            'context_id': context_id,
            'compressed_context': compressed_context,
            'compression_ratio': compression_ratio,
            'metrics': metrics,
            'knowledge_entities': len(knowledge_entities),
            'cross_references': len(cross_references)
        }
    
    async def _segment_context(self, context: str, agent_id: str) -> List[ContextSegment]:
        """Segment context into logical chunks for analysis."""
        segments = []
        
        # Smart segmentation based on content structure
        chunk_patterns = [
            r'\n\n+',  # Double newlines (paragraphs)
            r'\n(?=\d+\.|\*|\-)',  # Lists and numbered items
            r'\n(?=```|</?\w+>)',  # Code blocks and HTML
            r'(?<=\.)\s+(?=[A-Z])',  # Sentence boundaries
        ]
        
        current_text = context
        segment_id = 0
        
        while current_text and len(segments) < 100:  # Limit segments
            # Find natural break point
            best_split = None
            best_pattern = None
            
            for pattern in chunk_patterns:
                matches = list(re.finditer(pattern, current_text))
                
                for match in matches:
                    split_pos = match.end()
                    if split_pos > 100 and split_pos < self.max_segment_size:
                        if not best_split or abs(split_pos - self.max_segment_size // 2) < abs(best_split - self.max_segment_size // 2):
                            best_split = split_pos
                            best_pattern = pattern
            
            # If no good split found, use max segment size
            if not best_split:
                best_split = min(self.max_segment_size, len(current_text))
            
            # Create segment
            segment_text = current_text[:best_split].strip()
            if segment_text:
                segment = ContextSegment(
                    id=f"{agent_id}_seg_{segment_id}",
                    content=segment_text,
                    timestamp=datetime.utcnow(),
                    original_size=len(segment_text.encode('utf-8'))
                )
                segments.append(segment)
                segment_id += 1
            
            current_text = current_text[best_split:].strip()
            
            if len(current_text) < 50:  # Too small for meaningful segment
                if segments:
                    segments[-1].content += f" {current_text}"
                break
        
        logger.debug(f"Segmented context into {len(segments)} segments")
        return segments
    
    async def _analyze_semantic_importance(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Analyze semantic importance of each segment."""
        if not segments:
            return segments
        
        # Extract text for vectorization
        texts = [segment.content for segment in segments]
        
        try:
            # Generate TF-IDF vectors
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate importance scores
            for i, segment in enumerate(segments):
                # TF-IDF importance (sum of weights)
                tfidf_importance = np.sum(tfidf_matrix[i].toarray())
                
                # Length-based importance (normalized)
                length_importance = min(len(segment.content) / self.max_segment_size, 1.0)
                
                # Keyword-based importance
                keyword_importance = self._calculate_keyword_importance(segment.content)
                
                # Code/technical content importance
                technical_importance = self._calculate_technical_importance(segment.content)
                
                # Combined importance score
                segment.importance_score = (
                    tfidf_importance * 0.3 +
                    length_importance * 0.2 +
                    keyword_importance * 0.3 +
                    technical_importance * 0.2
                )
                
                # Generate semantic tags
                segment.semantic_tags = self._generate_semantic_tags(segment.content)
                
        except Exception as e:
            logger.warning(f"Semantic analysis failed, using fallback: {e}")
            # Fallback: simple length-based importance
            for segment in segments:
                segment.importance_score = min(len(segment.content) / self.max_segment_size, 1.0)
                segment.semantic_tags = ['general']
        
        return segments
    
    def _calculate_keyword_importance(self, text: str) -> float:
        """Calculate importance based on keywords and patterns."""
        importance_keywords = {
            'error': 2.0, 'fix': 2.0, 'solution': 2.0, 'problem': 1.5,
            'important': 1.8, 'critical': 2.0, 'urgent': 1.8,
            'function': 1.2, 'class': 1.2, 'method': 1.2,
            'api': 1.5, 'endpoint': 1.5, 'database': 1.5,
            'performance': 1.6, 'optimization': 1.6, 'memory': 1.4,
            'security': 1.8, 'authentication': 1.6, 'authorization': 1.6,
            'bug': 1.8, 'issue': 1.3, 'feature': 1.2,
            'test': 1.1, 'testing': 1.1, 'deployment': 1.4
        }
        
        text_lower = text.lower()
        score = 0.0
        word_count = len(text.split())
        
        for keyword, weight in importance_keywords.items():
            count = text_lower.count(keyword)
            score += (count / word_count) * weight if word_count > 0 else 0
        
        return min(score, 1.0)
    
    def _calculate_technical_importance(self, text: str) -> float:
        """Calculate importance based on technical content."""
        technical_patterns = [
            r'```[\s\S]*?```',  # Code blocks
            r'`[^`]+`',  # Inline code
            r'https?://\S+',  # URLs
            r'\b\d+\.\d+\.\d+',  # Version numbers
            r'\b[A-Z][a-zA-Z]*Error\b',  # Error types
            r'\bfunction\s+\w+\s*\(',  # Function definitions
            r'\bclass\s+\w+',  # Class definitions
            r'\bimport\s+\w+',  # Import statements
        ]
        
        technical_count = 0
        total_chars = len(text)
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            technical_count += sum(len(match) for match in matches)
        
        return min(technical_count / total_chars, 1.0) if total_chars > 0 else 0
    
    def _generate_semantic_tags(self, text: str) -> List[str]:
        """Generate semantic tags for content categorization."""
        tag_patterns = {
            'code': [r'```', r'function', r'class', r'import', r'def ', r'var ', r'const '],
            'error': [r'error', r'exception', r'failed', r'crash', r'bug'],
            'solution': [r'fix', r'solve', r'resolved', r'solution', r'workaround'],
            'api': [r'endpoint', r'api', r'request', r'response', r'http'],
            'database': [r'database', r'query', r'sql', r'table', r'schema'],
            'performance': [r'performance', r'optimization', r'memory', r'cpu', r'speed'],
            'security': [r'security', r'authentication', r'authorization', r'token'],
            'testing': [r'test', r'testing', r'unit test', r'integration'],
            'deployment': [r'deploy', r'deployment', r'production', r'docker'],
            'documentation': [r'readme', r'documentation', r'guide', r'tutorial']
        }
        
        text_lower = text.lower()
        tags = []
        
        for tag, patterns in tag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    tags.append(tag)
                    break
        
        return tags or ['general']
    
    async def _apply_compression_strategy(
        self, 
        segments: List[ContextSegment], 
        strategy: CompressionStrategy
    ) -> List[ContextSegment]:
        """Apply the selected compression strategy."""
        if strategy == CompressionStrategy.HYBRID:
            return await self._hybrid_compression(segments)
        elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
            return await self._semantic_clustering_compression(segments)
        elif strategy == CompressionStrategy.IMPORTANCE_RANKING:
            return await self._importance_ranking_compression(segments)
        elif strategy == CompressionStrategy.FREQUENCY_BASED:
            return await self._frequency_based_compression(segments)
        elif strategy == CompressionStrategy.TEMPORAL_DECAY:
            return await self._temporal_decay_compression(segments)
        else:
            return segments
    
    async def _hybrid_compression(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Apply hybrid compression using multiple strategies."""
        # Sort by importance score
        segments.sort(key=lambda s: s.importance_score, reverse=True)
        
        # Keep high-importance segments
        high_importance = [s for s in segments if s.importance_score >= 0.7]
        medium_importance = [s for s in segments if 0.4 <= s.importance_score < 0.7]
        low_importance = [s for s in segments if s.importance_score < 0.4]
        
        # Keep all high importance
        result = high_importance.copy()
        
        # Keep 70% of medium importance
        medium_keep_count = int(len(medium_importance) * 0.7)
        result.extend(medium_importance[:medium_keep_count])
        
        # Keep 30% of low importance (most recent or most unique)
        low_importance.sort(key=lambda s: s.timestamp, reverse=True)
        low_keep_count = int(len(low_importance) * 0.3)
        result.extend(low_importance[:low_keep_count])
        
        # Apply semantic clustering to similar content
        result = await self._cluster_similar_segments(result)
        
        return result
    
    async def _cluster_similar_segments(self, segments: List[ContextSegment]) -> List[ContextSegment]:
        """Cluster semantically similar segments and merge them."""
        if len(segments) < 3:
            return segments
        
        try:
            # Generate vectors for clustering
            texts = [segment.content for segment in segments]
            vectors = self.tfidf_vectorizer.transform(texts)
            
            # Perform clustering
            n_clusters = min(max(2, len(segments) // 3), 8)
            clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(vectors)
            
            # Group segments by cluster
            clustered_segments = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                clustered_segments[cluster_id].append(segments[i])
            
            # Merge similar segments within clusters
            result = []
            for cluster_segments in clustered_segments.values():
                if len(cluster_segments) > 2:
                    # Merge segments in this cluster
                    merged_content = self._merge_segment_contents(cluster_segments)
                    merged_segment = ContextSegment(
                        id=f"merged_{cluster_segments[0].id}",
                        content=merged_content,
                        timestamp=max(s.timestamp for s in cluster_segments),
                        importance_score=max(s.importance_score for s in cluster_segments),
                        semantic_tags=list(set().union(*[s.semantic_tags for s in cluster_segments]))
                    )
                    result.append(merged_segment)
                else:
                    result.extend(cluster_segments)
            
            return result
            
        except Exception as e:
            logger.warning(f"Clustering failed, returning original segments: {e}")
            return segments
    
    def _merge_segment_contents(self, segments: List[ContextSegment]) -> str:
        """Merge content from multiple segments intelligently."""
        # Sort by importance
        segments.sort(key=lambda s: s.importance_score, reverse=True)
        
        # Start with highest importance content
        merged = segments[0].content
        
        # Add unique information from other segments
        for segment in segments[1:]:
            # Use simple sentence-based deduplication
            sentences = segment.content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 20 and sentence not in merged:
                    # Check semantic similarity
                    if not self._is_content_similar(sentence, merged):
                        merged += f" {sentence}."
        
        return merged[:self.max_segment_size]  # Respect size limits
    
    def _is_content_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two pieces of content are semantically similar."""
        # Simple word overlap check for performance
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = overlap / union if union > 0 else 0
        return similarity > threshold
    
    async def _extract_semantic_knowledge(
        self, 
        segments: List[ContextSegment], 
        agent_id: str
    ) -> List[SemanticKnowledge]:
        """Extract semantic knowledge entities from segments."""
        knowledge_entities = []
        
        for segment in segments:
            # Extract different types of knowledge
            entities = []
            
            # Code patterns and solutions
            if 'code' in segment.semantic_tags or 'solution' in segment.semantic_tags:
                entities.extend(self._extract_code_knowledge(segment, agent_id))
            
            # Error patterns and fixes
            if 'error' in segment.semantic_tags:
                entities.extend(self._extract_error_knowledge(segment, agent_id))
            
            # API patterns
            if 'api' in segment.semantic_tags:
                entities.extend(self._extract_api_knowledge(segment, agent_id))
            
            # Performance insights
            if 'performance' in segment.semantic_tags:
                entities.extend(self._extract_performance_knowledge(segment, agent_id))
            
            # General concepts
            entities.extend(self._extract_concept_knowledge(segment, agent_id))
            
            knowledge_entities.extend(entities)
        
        # Store knowledge entities
        for entity in knowledge_entities:
            self.semantic_knowledge[entity.entity_id] = entity
        
        return knowledge_entities
    
    def _extract_code_knowledge(self, segment: ContextSegment, agent_id: str) -> List[SemanticKnowledge]:
        """Extract code-related knowledge."""
        entities = []
        
        # Find code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', segment.content)
        for i, code_block in enumerate(code_blocks):
            entity_id = f"{agent_id}_code_{hashlib.md5(code_block.encode()).hexdigest()[:8]}"
            
            entity = SemanticKnowledge(
                entity_id=entity_id,
                entity_type=KnowledgeType.CODE_SNIPPET.value,
                content=code_block,
                confidence=0.8,
                created_by=agent_id,
                created_at=datetime.utcnow()
            )
            entities.append(entity)
        
        # Find function definitions
        functions = re.findall(r'(function\s+\w+\s*\([^)]*\)\s*{[^}]*})', segment.content)
        for func in functions:
            entity_id = f"{agent_id}_func_{hashlib.md5(func.encode()).hexdigest()[:8]}"
            
            entity = SemanticKnowledge(
                entity_id=entity_id,
                entity_type=KnowledgeType.PATTERN.value,
                content=func,
                confidence=0.7,
                created_by=agent_id,
                created_at=datetime.utcnow()
            )
            entities.append(entity)
        
        return entities
    
    def _extract_error_knowledge(self, segment: ContextSegment, agent_id: str) -> List[SemanticKnowledge]:
        """Extract error and solution knowledge."""
        entities = []
        
        # Find error patterns
        error_patterns = re.findall(r'(.*(?:error|exception|failed).*)', segment.content, re.IGNORECASE)
        solution_patterns = re.findall(r'(.*(?:fix|solve|solution|resolved).*)', segment.content, re.IGNORECASE)
        
        for error in error_patterns[:3]:  # Limit to avoid noise
            entity_id = f"{agent_id}_error_{hashlib.md5(error.encode()).hexdigest()[:8]}"
            
            entity = SemanticKnowledge(
                entity_id=entity_id,
                entity_type=KnowledgeType.ERROR_FIX.value,
                content=error.strip(),
                confidence=0.6,
                created_by=agent_id,
                created_at=datetime.utcnow()
            )
            entities.append(entity)
        
        for solution in solution_patterns[:2]:  # Limit solutions
            entity_id = f"{agent_id}_solution_{hashlib.md5(solution.encode()).hexdigest()[:8]}"
            
            entity = SemanticKnowledge(
                entity_id=entity_id,
                entity_type=KnowledgeType.SOLUTION.value,
                content=solution.strip(),
                confidence=0.7,
                created_by=agent_id,
                created_at=datetime.utcnow()
            )
            entities.append(entity)
        
        return entities
    
    def _extract_concept_knowledge(self, segment: ContextSegment, agent_id: str) -> List[SemanticKnowledge]:
        """Extract general conceptual knowledge."""
        entities = []
        
        # Extract key sentences (heuristic-based)
        sentences = segment.content.split('.')
        important_sentences = [
            s.strip() for s in sentences
            if len(s.strip()) > 30 and
            any(keyword in s.lower() for keyword in ['important', 'key', 'note', 'remember', 'crucial'])
        ]
        
        for sentence in important_sentences[:2]:  # Limit to avoid noise
            entity_id = f"{agent_id}_concept_{hashlib.md5(sentence.encode()).hexdigest()[:8]}"
            
            entity = SemanticKnowledge(
                entity_id=entity_id,
                entity_type=KnowledgeType.CONCEPT.value,
                content=sentence,
                confidence=0.5,
                created_by=agent_id,
                created_at=datetime.utcnow()
            )
            entities.append(entity)
        
        return entities
    
    async def _create_cross_references(self, knowledge_entities: List[SemanticKnowledge]) -> Dict[str, List[str]]:
        """Create cross-references between knowledge entities."""
        cross_references = defaultdict(list)
        
        # Simple keyword-based cross-referencing
        for i, entity1 in enumerate(knowledge_entities):
            for j, entity2 in enumerate(knowledge_entities[i+1:], i+1):
                if self._entities_related(entity1, entity2):
                    cross_references[entity1.entity_id].append(entity2.entity_id)
                    cross_references[entity2.entity_id].append(entity1.entity_id)
        
        return dict(cross_references)
    
    def _entities_related(self, entity1: SemanticKnowledge, entity2: SemanticKnowledge) -> bool:
        """Check if two knowledge entities are related."""
        # Simple keyword overlap check
        words1 = set(entity1.content.lower().split())
        words2 = set(entity2.content.lower().split())
        
        # Filter out common words
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        return overlap >= 2  # At least 2 common meaningful words
    
    async def _build_compressed_context(
        self, 
        segments: List[ContextSegment], 
        cross_references: Dict[str, List[str]]
    ) -> str:
        """Build the final compressed context."""
        # Sort segments by importance and timestamp
        segments.sort(key=lambda s: (s.importance_score, s.timestamp), reverse=True)
        
        compressed_parts = []
        
        # Add high-level summary
        compressed_parts.append("=== COMPRESSED CONTEXT SUMMARY ===\n")
        
        # Add semantic categories
        categories = defaultdict(list)
        for segment in segments:
            for tag in segment.semantic_tags:
                categories[tag].append(segment)
        
        for category, cat_segments in categories.items():
            if len(cat_segments) > 1:
                compressed_parts.append(f"\n[{category.upper()}]")
                # Merge similar segments in category
                merged_content = self._merge_segment_contents(cat_segments[:3])  # Top 3
                compressed_parts.append(merged_content)
            else:
                compressed_parts.append(f"\n[{category.upper()}]")
                compressed_parts.append(cat_segments[0].content)
        
        # Add cross-reference summary
        if cross_references:
            compressed_parts.append("\n=== RELATED CONCEPTS ===")
            for entity_id, related_ids in list(cross_references.items())[:5]:  # Top 5
                compressed_parts.append(f"• {entity_id}: {len(related_ids)} related")
        
        compressed_context = '\n'.join(compressed_parts)
        
        # Final compression using text compression
        if len(compressed_context) > self.max_segment_size * 3:
            compressed_context = compressed_context[:self.max_segment_size * 3] + "...[truncated]"
        
        return compressed_context
    
    async def _store_compressed_context(self, compressed_context: str, agent_id: str) -> str:
        """Store compressed context and return context ID."""
        context_id = f"ctx_{agent_id}_{int(time.time())}"
        
        # Store compressed version
        compressed_bytes = zlib.compress(compressed_context.encode('utf-8'))
        self.compressed_contexts[context_id] = compressed_bytes
        
        # Store metadata
        cache = get_session_cache()
        await cache.store_session_state(
            agent_id=agent_id,
            state_key=f"context_meta_{context_id}",
            state_data={
                'context_id': context_id,
                'original_agent': agent_id,
                'created_at': datetime.utcnow().isoformat(),
                'compressed_size': len(compressed_bytes),
                'text_size': len(compressed_context)
            },
            expiry_seconds=86400 * 7  # 7 days
        )
        
        return context_id
    
    async def retrieve_context(self, context_id: str, requesting_agent: str) -> Optional[str]:
        """Retrieve and decompress context."""
        if context_id not in self.compressed_contexts:
            return None
        
        # Decompress context
        compressed_bytes = self.compressed_contexts[context_id]
        context = zlib.decompress(compressed_bytes).decode('utf-8')
        
        # Track access for analytics
        self.knowledge_access_patterns[context_id].append({
            'agent_id': requesting_agent,
            'timestamp': time.time(),
            'access_type': 'retrieval'
        })
        
        return context
    
    async def share_knowledge_across_agents(
        self, 
        source_agent: str, 
        target_agents: List[str],
        knowledge_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Share semantic knowledge between agents."""
        shared_count = 0
        failed_shares = 0
        
        # Filter knowledge based on criteria
        shareable_knowledge = []
        for entity in self.semantic_knowledge.values():
            if entity.created_by == source_agent and entity.confidence >= 0.6:
                # Apply additional filters if provided
                if knowledge_filter:
                    if self._matches_filter(entity, knowledge_filter):
                        shareable_knowledge.append(entity)
                else:
                    shareable_knowledge.append(entity)
        
        # Share with target agents
        for target_agent in target_agents:
            for entity in shareable_knowledge:
                try:
                    # Mark as shared
                    entity.accessed_by.add(target_agent)
                    entity.access_count += 1
                    
                    # Store reference for target agent
                    cache = get_session_cache()
                    await cache.store_session_state(
                        agent_id=target_agent,
                        state_key=f"shared_knowledge_{entity.entity_id}",
                        state_data={
                            'entity_id': entity.entity_id,
                            'entity_type': entity.entity_type,
                            'content': entity.content,
                            'shared_by': source_agent,
                            'shared_at': datetime.utcnow().isoformat(),
                            'confidence': entity.confidence
                        },
                        expiry_seconds=86400 * self.knowledge_expiry_days
                    )
                    
                    shared_count += 1
                    self.cross_agent_shares[f"{source_agent}->{target_agent}"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to share knowledge {entity.entity_id}: {e}")
                    failed_shares += 1
        
        logger.info(f"📤 Shared {shared_count} knowledge entities from {source_agent} to {len(target_agents)} agents")
        
        return {
            'shared_entities': shared_count,
            'failed_shares': failed_shares,
            'target_agents': len(target_agents),
            'shareable_entities': len(shareable_knowledge)
        }
    
    def _matches_filter(self, entity: SemanticKnowledge, knowledge_filter: Dict[str, Any]) -> bool:
        """Check if knowledge entity matches filter criteria."""
        if 'entity_types' in knowledge_filter:
            if entity.entity_type not in knowledge_filter['entity_types']:
                return False
        
        if 'min_confidence' in knowledge_filter:
            if entity.confidence < knowledge_filter['min_confidence']:
                return False
        
        if 'keywords' in knowledge_filter:
            content_lower = entity.content.lower()
            if not any(keyword.lower() in content_lower for keyword in knowledge_filter['keywords']):
                return False
        
        return True
    
    async def get_compression_metrics(self) -> Dict[str, Any]:
        """Get comprehensive compression and knowledge sharing metrics."""
        if not self.compression_history:
            return {'error': 'No compression history available'}
        
        recent_compressions = list(self.compression_history)[-10:]
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'compression_performance': {
                'avg_compression_ratio': statistics.mean([c.compression_ratio for c in recent_compressions]),
                'avg_processing_time': statistics.mean([c.processing_time for c in recent_compressions]),
                'total_compressions': len(self.compression_history),
                'avg_segments_processed': statistics.mean([c.segments_processed for c in recent_compressions])
            },
            'knowledge_base': {
                'total_entities': len(self.semantic_knowledge),
                'entity_types': len(set(e.entity_type for e in self.semantic_knowledge.values())),
                'avg_confidence': statistics.mean([e.confidence for e in self.semantic_knowledge.values()]) if self.semantic_knowledge else 0,
                'cross_references': sum(len(refs) for refs in self.semantic_knowledge.values())
            },
            'sharing_metrics': {
                'total_shares': sum(self.cross_agent_shares.values()),
                'unique_sharing_pairs': len(self.cross_agent_shares),
                'avg_shares_per_pair': statistics.mean(list(self.cross_agent_shares.values())) if self.cross_agent_shares else 0
            },
            'storage_efficiency': {
                'compressed_contexts': len(self.compressed_contexts),
                'total_compressed_size': sum(len(data) for data in self.compressed_contexts.values()),
                'cache_hit_patterns': len(self.knowledge_access_patterns)
            }
        }
        
        return metrics


# Global advanced context engine instance
_advanced_context_engine: Optional[AdvancedContextEngine] = None


async def get_advanced_context_engine() -> AdvancedContextEngine:
    """Get or create the global advanced context engine instance."""
    global _advanced_context_engine
    
    if _advanced_context_engine is None:
        _advanced_context_engine = AdvancedContextEngine()
    
    return _advanced_context_engine


async def shutdown_advanced_context_engine():
    """Shutdown the global advanced context engine."""
    global _advanced_context_engine
    _advanced_context_engine = None
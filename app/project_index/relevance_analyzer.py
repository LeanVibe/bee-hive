"""
AI-Powered Relevance Analyzer for LeanVibe Agent Hive 2.0

Advanced relevance scoring engine that uses multiple algorithms to determine
the relevance of code files for specific development tasks. Provides semantic
similarity analysis, structural importance scoring, and historical relevance.
"""

import asyncio
import json
import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_reading_ease, flesch_kincaid_grade

from .models import FileAnalysisResult, DependencyResult
from .graph import DependencyGraph
from .utils import PathUtils, FileUtils

logger = structlog.get_logger()


class ScoringAlgorithm(Enum):
    """Available relevance scoring algorithms."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURAL_CENTRALITY = "structural_centrality"
    HISTORICAL_PATTERN = "historical_pattern"
    LINGUISTIC_ANALYSIS = "linguistic_analysis"
    GRAPH_METRICS = "graph_metrics"
    COMPLEXITY_ANALYSIS = "complexity_analysis"


@dataclass
class RelevanceFactors:
    """Detailed breakdown of relevance factors."""
    semantic_score: float = 0.0
    structural_score: float = 0.0
    historical_score: float = 0.0
    linguistic_score: float = 0.0
    complexity_score: float = 0.0
    context_score: float = 0.0
    final_score: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "semantic_score": self.semantic_score,
            "structural_score": self.structural_score,
            "historical_score": self.historical_score,
            "linguistic_score": self.linguistic_score,
            "complexity_score": self.complexity_score,
            "context_score": self.context_score,
            "final_score": self.final_score,
            "confidence": self.confidence
        }


@dataclass
class SemanticAnalysisResult:
    """Result of semantic analysis."""
    similarity_score: float
    keyword_matches: List[str]
    concept_matches: List[str]
    topic_similarity: float
    explanation: str


@dataclass
class StructuralAnalysisResult:
    """Result of structural analysis."""
    centrality_score: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank_score: float
    clustering_coefficient: float
    impact_score: float
    explanation: str


@dataclass
class HistoricalAnalysisResult:
    """Result of historical analysis."""
    change_frequency: float
    recent_activity: float
    bug_density: float
    contributor_diversity: float
    stability_score: float
    explanation: str


class RelevanceAnalyzer:
    """
    Advanced AI-powered relevance analyzer for code files.
    
    Provides multiple scoring algorithms:
    - Semantic similarity using NLP techniques
    - Structural importance via graph analysis
    - Historical patterns from version control
    - Linguistic quality assessment
    - Code complexity metrics
    """
    
    def __init__(self):
        """Initialize RelevanceAnalyzer with ML models."""
        # Text analysis models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=100
        )
        
        # Model training state
        self.models_fitted = False
        self.document_corpus = []
        self.vocabulary = set()
        
        # Algorithm weights for different task types
        self.algorithm_weights = {
            "feature": {
                ScoringAlgorithm.SEMANTIC_SIMILARITY: 0.35,
                ScoringAlgorithm.STRUCTURAL_CENTRALITY: 0.25,
                ScoringAlgorithm.HISTORICAL_PATTERN: 0.15,
                ScoringAlgorithm.LINGUISTIC_ANALYSIS: 0.10,
                ScoringAlgorithm.COMPLEXITY_ANALYSIS: 0.15
            },
            "bugfix": {
                ScoringAlgorithm.SEMANTIC_SIMILARITY: 0.20,
                ScoringAlgorithm.STRUCTURAL_CENTRALITY: 0.30,
                ScoringAlgorithm.HISTORICAL_PATTERN: 0.35,
                ScoringAlgorithm.LINGUISTIC_ANALYSIS: 0.05,
                ScoringAlgorithm.COMPLEXITY_ANALYSIS: 0.10
            },
            "refactoring": {
                ScoringAlgorithm.SEMANTIC_SIMILARITY: 0.15,
                ScoringAlgorithm.STRUCTURAL_CENTRALITY: 0.40,
                ScoringAlgorithm.HISTORICAL_PATTERN: 0.20,
                ScoringAlgorithm.LINGUISTIC_ANALYSIS: 0.10,
                ScoringAlgorithm.COMPLEXITY_ANALYSIS: 0.15
            },
            "analysis": {
                ScoringAlgorithm.SEMANTIC_SIMILARITY: 0.30,
                ScoringAlgorithm.STRUCTURAL_CENTRALITY: 0.25,
                ScoringAlgorithm.HISTORICAL_PATTERN: 0.20,
                ScoringAlgorithm.LINGUISTIC_ANALYSIS: 0.15,
                ScoringAlgorithm.COMPLEXITY_ANALYSIS: 0.10
            },
            "default": {
                ScoringAlgorithm.SEMANTIC_SIMILARITY: 0.30,
                ScoringAlgorithm.STRUCTURAL_CENTRALITY: 0.25,
                ScoringAlgorithm.HISTORICAL_PATTERN: 0.20,
                ScoringAlgorithm.LINGUISTIC_ANALYSIS: 0.15,
                ScoringAlgorithm.COMPLEXITY_ANALYSIS: 0.10
            }
        }
        
        # Performance tracking
        self.analysis_stats = {
            "files_analyzed": 0,
            "semantic_analyses": 0,
            "structural_analyses": 0,
            "historical_analyses": 0
        }
    
    async def analyze_relevance(
        self,
        file_result: FileAnalysisResult,
        task_description: str,
        task_type: str,
        dependency_graph: DependencyGraph,
        project_context: Dict[str, Any] = None,
        mentioned_files: List[str] = None
    ) -> RelevanceFactors:
        """
        Perform comprehensive relevance analysis for a file.
        
        Args:
            file_result: File analysis result
            task_description: Description of the development task
            task_type: Type of task (feature, bugfix, etc.)
            dependency_graph: Project dependency graph
            project_context: Additional project context
            mentioned_files: Files explicitly mentioned in task
            
        Returns:
            RelevanceFactors with detailed scoring breakdown
        """
        logger.debug("Analyzing file relevance", 
                    file_path=file_result.file_path,
                    task_type=task_type)
        
        try:
            # Initialize scores
            factors = RelevanceFactors()
            
            # Semantic similarity analysis
            semantic_result = await self._analyze_semantic_similarity(
                file_result, task_description, project_context
            )
            factors.semantic_score = semantic_result.similarity_score
            
            # Structural importance analysis
            structural_result = await self._analyze_structural_importance(
                file_result, dependency_graph, task_type
            )
            factors.structural_score = structural_result.centrality_score
            
            # Historical pattern analysis
            historical_result = await self._analyze_historical_patterns(
                file_result, task_type, project_context
            )
            factors.historical_score = historical_result.stability_score
            
            # Linguistic quality analysis
            factors.linguistic_score = await self._analyze_linguistic_quality(
                file_result, task_description
            )
            
            # Code complexity analysis
            factors.complexity_score = await self._analyze_code_complexity(
                file_result, task_type
            )
            
            # Context bonus (mentioned files, etc.)
            factors.context_score = self._calculate_context_bonus(
                file_result, mentioned_files, task_description
            )
            
            # Calculate weighted final score
            weights = self.algorithm_weights.get(task_type, self.algorithm_weights["default"])
            
            factors.final_score = (
                factors.semantic_score * weights[ScoringAlgorithm.SEMANTIC_SIMILARITY] +
                factors.structural_score * weights[ScoringAlgorithm.STRUCTURAL_CENTRALITY] +
                factors.historical_score * weights[ScoringAlgorithm.HISTORICAL_PATTERN] +
                factors.linguistic_score * weights[ScoringAlgorithm.LINGUISTIC_ANALYSIS] +
                factors.complexity_score * weights[ScoringAlgorithm.COMPLEXITY_ANALYSIS] +
                factors.context_score * 0.1  # Context bonus
            )
            
            # Normalize to 0-1 range
            factors.final_score = max(0.0, min(1.0, factors.final_score))
            
            # Calculate confidence based on score consistency
            factors.confidence = self._calculate_confidence(factors)
            
            # Update statistics
            self.analysis_stats["files_analyzed"] += 1
            
            logger.debug("Relevance analysis completed",
                        file_path=file_result.file_path,
                        final_score=factors.final_score,
                        confidence=factors.confidence)
            
            return factors
            
        except Exception as e:
            logger.error("Relevance analysis failed",
                        file_path=file_result.file_path,
                        error=str(e))
            
            # Return default low relevance
            return RelevanceFactors(
                semantic_score=0.1,
                structural_score=0.1,
                historical_score=0.1,
                linguistic_score=0.1,
                complexity_score=0.1,
                context_score=0.0,
                final_score=0.1,
                confidence=0.1
            )
    
    async def _analyze_semantic_similarity(
        self,
        file_result: FileAnalysisResult,
        task_description: str,
        project_context: Dict[str, Any] = None
    ) -> SemanticAnalysisResult:
        """Analyze semantic similarity between file and task."""
        try:
            self.analysis_stats["semantic_analyses"] += 1
            
            # Extract file content for analysis
            file_content = self._extract_semantic_content(file_result)
            
            if not file_content or not task_description:
                return SemanticAnalysisResult(
                    similarity_score=0.1,
                    keyword_matches=[],
                    concept_matches=[],
                    topic_similarity=0.0,
                    explanation="Insufficient content for semantic analysis"
                )
            
            # Prepare texts for analysis
            task_text = self._preprocess_text(task_description)
            file_text = self._preprocess_text(file_content)
            
            # TF-IDF similarity
            tfidf_similarity = await self._calculate_tfidf_similarity(
                task_text, file_text
            )
            
            # Keyword matching
            keyword_matches = self._find_keyword_matches(task_text, file_text)
            
            # Concept matching (using simple heuristics)
            concept_matches = self._find_concept_matches(
                task_description, file_content, file_result
            )
            
            # Topic similarity (if models are trained)
            topic_similarity = 0.0
            if self.models_fitted:
                topic_similarity = await self._calculate_topic_similarity(
                    task_text, file_text
                )
            
            # Combine scores
            similarity_score = (
                tfidf_similarity * 0.4 +
                len(keyword_matches) * 0.05 +  # Boost for keyword matches
                len(concept_matches) * 0.1 +   # Boost for concept matches
                topic_similarity * 0.3 +
                self._calculate_filename_similarity(file_result.file_path, task_text) * 0.15
            )
            
            similarity_score = min(1.0, similarity_score)
            
            explanation = self._generate_semantic_explanation(
                tfidf_similarity, keyword_matches, concept_matches, topic_similarity
            )
            
            return SemanticAnalysisResult(
                similarity_score=similarity_score,
                keyword_matches=keyword_matches,
                concept_matches=concept_matches,
                topic_similarity=topic_similarity,
                explanation=explanation
            )
            
        except Exception as e:
            logger.warning("Semantic analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            
            return SemanticAnalysisResult(
                similarity_score=0.3,
                keyword_matches=[],
                concept_matches=[],
                topic_similarity=0.0,
                explanation="Semantic analysis failed - using fallback score"
            )
    
    async def _analyze_structural_importance(
        self,
        file_result: FileAnalysisResult,
        dependency_graph: DependencyGraph,
        task_type: str
    ) -> StructuralAnalysisResult:
        """Analyze structural importance of file in dependency graph."""
        try:
            self.analysis_stats["structural_analyses"] += 1
            
            file_path = file_result.file_path
            
            # Basic centrality metrics
            in_degree = len(dependency_graph.get_dependents(file_path))
            out_degree = len(dependency_graph.get_dependencies(file_path))
            total_nodes = len(dependency_graph.get_nodes())
            
            if total_nodes <= 1:
                return StructuralAnalysisResult(
                    centrality_score=0.0,
                    betweenness_centrality=0.0,
                    closeness_centrality=0.0,
                    pagerank_score=0.0,
                    clustering_coefficient=0.0,
                    impact_score=0.0,
                    explanation="Insufficient graph structure for analysis"
                )
            
            # Degree centrality
            max_possible = total_nodes - 1
            degree_centrality = (in_degree + out_degree) / (2 * max_possible)
            
            # Betweenness centrality (simplified)
            betweenness = self._calculate_betweenness_centrality(
                file_path, dependency_graph
            )
            
            # Closeness centrality (simplified)
            closeness = self._calculate_closeness_centrality(
                file_path, dependency_graph
            )
            
            # PageRank-like score
            pagerank = self._calculate_pagerank_score(
                file_path, dependency_graph
            )
            
            # Clustering coefficient
            clustering = self._calculate_clustering_coefficient(
                file_path, dependency_graph
            )
            
            # Impact score (how many files would be affected by changes)
            impact_score = dependency_graph.calculate_impact_score(file_path)
            
            # Task-specific importance adjustments
            task_adjustment = self._get_task_specific_structural_adjustment(
                task_type, in_degree, out_degree, impact_score
            )
            
            # Combined centrality score
            centrality_score = (
                degree_centrality * 0.3 +
                betweenness * 0.2 +
                closeness * 0.15 +
                pagerank * 0.2 +
                impact_score * 0.15 +
                task_adjustment
            )
            
            centrality_score = min(1.0, centrality_score)
            
            explanation = self._generate_structural_explanation(
                degree_centrality, betweenness, closeness, pagerank, impact_score
            )
            
            return StructuralAnalysisResult(
                centrality_score=centrality_score,
                betweenness_centrality=betweenness,
                closeness_centrality=closeness,
                pagerank_score=pagerank,
                clustering_coefficient=clustering,
                impact_score=impact_score,
                explanation=explanation
            )
            
        except Exception as e:
            logger.warning("Structural analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            
            return StructuralAnalysisResult(
                centrality_score=0.3,
                betweenness_centrality=0.0,
                closeness_centrality=0.0,
                pagerank_score=0.0,
                clustering_coefficient=0.0,
                impact_score=0.0,
                explanation="Structural analysis failed - using fallback score"
            )
    
    async def _analyze_historical_patterns(
        self,
        file_result: FileAnalysisResult,
        task_type: str,
        project_context: Dict[str, Any] = None
    ) -> HistoricalAnalysisResult:
        """Analyze historical patterns from version control."""
        try:
            self.analysis_stats["historical_analyses"] += 1
            
            # Simulated historical data (in real implementation, integrate with Git)
            change_frequency = self._simulate_change_frequency(file_result)
            recent_activity = self._simulate_recent_activity(file_result)
            bug_density = self._simulate_bug_density(file_result, task_type)
            contributor_diversity = self._simulate_contributor_diversity(file_result)
            
            # Calculate stability score
            stability_score = self._calculate_stability_score(
                change_frequency, recent_activity, bug_density, task_type
            )
            
            explanation = self._generate_historical_explanation(
                change_frequency, recent_activity, bug_density, contributor_diversity
            )
            
            return HistoricalAnalysisResult(
                change_frequency=change_frequency,
                recent_activity=recent_activity,
                bug_density=bug_density,
                contributor_diversity=contributor_diversity,
                stability_score=stability_score,
                explanation=explanation
            )
            
        except Exception as e:
            logger.warning("Historical analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            
            return HistoricalAnalysisResult(
                change_frequency=0.5,
                recent_activity=0.5,
                bug_density=0.3,
                contributor_diversity=0.5,
                stability_score=0.5,
                explanation="Historical analysis failed - using neutral scores"
            )
    
    async def _analyze_linguistic_quality(
        self,
        file_result: FileAnalysisResult,
        task_description: str
    ) -> float:
        """Analyze linguistic quality of code and comments."""
        try:
            # Extract comments and docstrings
            text_content = self._extract_text_content(file_result)
            
            if not text_content:
                return 0.3  # Neutral score for files without text
            
            # Calculate readability metrics
            readability_score = 0.0
            try:
                reading_ease = flesch_reading_ease(text_content)
                readability_score = reading_ease / 100.0  # Normalize to 0-1
            except:
                readability_score = 0.5  # Neutral if calculation fails
            
            # Calculate documentation density
            doc_density = self._calculate_documentation_density(file_result)
            
            # Calculate naming quality
            naming_quality = self._calculate_naming_quality(file_result)
            
            # Language consistency
            language_consistency = self._calculate_language_consistency(
                text_content, task_description
            )
            
            # Combine scores
            linguistic_score = (
                readability_score * 0.3 +
                doc_density * 0.3 +
                naming_quality * 0.25 +
                language_consistency * 0.15
            )
            
            return min(1.0, linguistic_score)
            
        except Exception as e:
            logger.warning("Linguistic analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            return 0.5
    
    async def _analyze_code_complexity(
        self,
        file_result: FileAnalysisResult,
        task_type: str
    ) -> float:
        """Analyze code complexity and its relevance to the task."""
        try:
            if not file_result.analysis_data:
                return 0.3
            
            # Extract complexity metrics
            line_count = file_result.line_count or 0
            function_count = len(file_result.analysis_data.get('functions', []))
            class_count = len(file_result.analysis_data.get('classes', []))
            
            # Calculate complexity score based on task type
            if task_type == "refactoring":
                # For refactoring, higher complexity = higher relevance
                complexity_score = min(1.0, (line_count / 500) * 0.4 +
                                     (function_count / 20) * 0.3 +
                                     (class_count / 5) * 0.3)
            elif task_type == "bugfix":
                # For bug fixing, moderate complexity is most relevant
                optimal_lines = 200
                line_score = 1.0 - abs(line_count - optimal_lines) / optimal_lines
                complexity_score = max(0.1, line_score * 0.6 +
                                     min(1.0, function_count / 10) * 0.4)
            else:
                # For other tasks, moderate complexity is preferred
                if line_count < 50:
                    complexity_score = 0.3  # Too simple
                elif line_count < 300:
                    complexity_score = 0.8  # Good complexity
                elif line_count < 800:
                    complexity_score = 0.6  # High complexity
                else:
                    complexity_score = 0.4  # Very high complexity
            
            return complexity_score
            
        except Exception as e:
            logger.warning("Complexity analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            return 0.5
    
    def _calculate_context_bonus(
        self,
        file_result: FileAnalysisResult,
        mentioned_files: List[str] = None,
        task_description: str = ""
    ) -> float:
        """Calculate bonus score based on context clues."""
        bonus = 0.0
        
        # Explicitly mentioned files get high bonus
        if mentioned_files and file_result.file_path in mentioned_files:
            bonus += 0.5
        
        # Partial path matches
        if mentioned_files:
            for mentioned in mentioned_files:
                if mentioned in file_result.file_path or file_result.file_path in mentioned:
                    bonus += 0.2
                    break
        
        # Entry point files
        filename = Path(file_result.file_path).name.lower()
        if any(pattern in filename for pattern in ['main', 'index', 'app', 'server']):
            bonus += 0.1
        
        # Configuration files (context-dependent)
        if any(pattern in filename for pattern in ['config', 'settings', 'env']):
            if any(keyword in task_description.lower() for keyword in ['config', 'setting', 'environment']):
                bonus += 0.15
        
        # Test files (context-dependent)
        if 'test' in filename:
            if any(keyword in task_description.lower() for keyword in ['test', 'bug', 'error']):
                bonus += 0.1
        
        return min(0.3, bonus)  # Cap bonus at 0.3
    
    def _calculate_confidence(self, factors: RelevanceFactors) -> float:
        """Calculate confidence in the relevance assessment."""
        scores = [
            factors.semantic_score,
            factors.structural_score,
            factors.historical_score,
            factors.linguistic_score,
            factors.complexity_score
        ]
        
        # Remove zero scores for confidence calculation
        non_zero_scores = [s for s in scores if s > 0.0]
        
        if len(non_zero_scores) < 2:
            return 0.3  # Low confidence with insufficient data
        
        # Calculate coefficient of variation (std/mean)
        mean_score = sum(non_zero_scores) / len(non_zero_scores)
        variance = sum((s - mean_score) ** 2 for s in non_zero_scores) / len(non_zero_scores)
        std_dev = math.sqrt(variance)
        
        if mean_score == 0:
            return 0.1
        
        cv = std_dev / mean_score
        
        # Convert coefficient of variation to confidence (lower CV = higher confidence)
        confidence = max(0.1, 1.0 - cv)
        
        # Boost confidence for consistently high scores
        if mean_score > 0.7 and cv < 0.3:
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    # Helper methods for detailed analysis
    
    def _extract_semantic_content(self, file_result: FileAnalysisResult) -> str:
        """Extract content suitable for semantic analysis."""
        content_parts = []
        
        # Filename and path components
        path_parts = file_result.file_path.replace('/', ' ').replace('_', ' ').replace('-', ' ')
        content_parts.append(path_parts)
        
        # Analysis data
        if file_result.analysis_data:
            # Function and class names
            functions = file_result.analysis_data.get('functions', [])
            for func in functions:
                if isinstance(func, dict):
                    name = func.get('name', '')
                    docstring = func.get('docstring', '')
                    content_parts.append(name.replace('_', ' '))
                    if docstring:
                        content_parts.append(docstring)
                elif isinstance(func, str):
                    content_parts.append(func.replace('_', ' '))
            
            classes = file_result.analysis_data.get('classes', [])
            for cls in classes:
                if isinstance(cls, dict):
                    name = cls.get('name', '')
                    docstring = cls.get('docstring', '')
                    content_parts.append(name.replace('_', ' '))
                    if docstring:
                        content_parts.append(docstring)
                elif isinstance(cls, str):
                    content_parts.append(cls.replace('_', ' '))
            
            # Comments
            comments = file_result.analysis_data.get('comments', [])
            for comment in comments[:10]:  # Limit to first 10 comments
                if isinstance(comment, str) and len(comment) > 10:
                    content_parts.append(comment)
        
        return ' '.join(content_parts)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def _calculate_tfidf_similarity(self, task_text: str, file_text: str) -> float:
        """Calculate TF-IDF similarity between task and file."""
        try:
            texts = [task_text, file_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback to simple word overlap
            task_words = set(task_text.split())
            file_words = set(file_text.split())
            
            if not task_words or not file_words:
                return 0.0
            
            intersection = task_words.intersection(file_words)
            union = task_words.union(file_words)
            
            return len(intersection) / len(union) if union else 0.0
    
    def _find_keyword_matches(self, task_text: str, file_text: str) -> List[str]:
        """Find keyword matches between task and file."""
        task_words = set(task_text.split())
        file_words = set(file_text.split())
        
        # Filter out common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        
        task_words = task_words - common_words
        file_words = file_words - common_words
        
        # Find significant matches (length > 2)
        matches = task_words.intersection(file_words)
        significant_matches = [word for word in matches if len(word) > 2]
        
        return significant_matches
    
    def _find_concept_matches(
        self, 
        task_description: str, 
        file_content: str, 
        file_result: FileAnalysisResult
    ) -> List[str]:
        """Find conceptual matches between task and file."""
        concepts = []
        
        # Domain-specific concept mapping
        concept_keywords = {
            'authentication': ['auth', 'login', 'password', 'token', 'session', 'user'],
            'database': ['db', 'sql', 'query', 'table', 'schema', 'migration'],
            'api': ['api', 'endpoint', 'route', 'handler', 'controller', 'request', 'response'],
            'testing': ['test', 'spec', 'mock', 'assert', 'expect', 'fixture'],
            'configuration': ['config', 'setting', 'env', 'environment', 'option'],
            'security': ['secure', 'encrypt', 'decrypt', 'hash', 'salt', 'certificate'],
            'performance': ['cache', 'optimize', 'fast', 'slow', 'memory', 'cpu'],
            'ui': ['ui', 'interface', 'component', 'render', 'display', 'view']
        }
        
        task_lower = task_description.lower()
        file_lower = file_content.lower()
        
        for concept, keywords in concept_keywords.items():
            task_has_concept = any(keyword in task_lower for keyword in keywords)
            file_has_concept = any(keyword in file_lower for keyword in keywords)
            
            if task_has_concept and file_has_concept:
                concepts.append(concept)
        
        return concepts
    
    async def _calculate_topic_similarity(self, task_text: str, file_text: str) -> float:
        """Calculate topic similarity using LDA."""
        try:
            if not self.models_fitted:
                return 0.0
            
            texts = [task_text, file_text]
            vectors = self.count_vectorizer.transform(texts)
            topics = self.lda_model.transform(vectors)
            
            # Calculate cosine similarity between topic distributions
            similarity = cosine_similarity(topics[0:1], topics[1:2])[0][0]
            return similarity
            
        except:
            return 0.0
    
    def _calculate_filename_similarity(self, file_path: str, task_text: str) -> float:
        """Calculate similarity between filename and task."""
        filename = Path(file_path).stem.lower()
        filename_words = set(filename.replace('_', ' ').replace('-', ' ').split())
        task_words = set(task_text.split())
        
        if not filename_words or not task_words:
            return 0.0
        
        intersection = filename_words.intersection(task_words)
        return len(intersection) / len(filename_words) if filename_words else 0.0
    
    def _calculate_betweenness_centrality(
        self, 
        file_path: str, 
        dependency_graph: DependencyGraph
    ) -> float:
        """Calculate simplified betweenness centrality."""
        # Simplified implementation - in full version, use networkx
        try:
            dependencies = dependency_graph.get_dependencies(file_path)
            dependents = dependency_graph.get_dependents(file_path)
            
            # Files that bridge many connections have higher betweenness
            bridge_score = len(dependencies) * len(dependents)
            max_possible = len(dependency_graph.get_nodes()) ** 2
            
            return min(1.0, bridge_score / max_possible) if max_possible > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_closeness_centrality(
        self, 
        file_path: str, 
        dependency_graph: DependencyGraph
    ) -> float:
        """Calculate simplified closeness centrality."""
        # Simplified implementation
        try:
            all_nodes = dependency_graph.get_nodes()
            if len(all_nodes) <= 1:
                return 0.0
            
            # Simple heuristic: files with more direct connections are "closer"
            direct_connections = (
                len(dependency_graph.get_dependencies(file_path)) +
                len(dependency_graph.get_dependents(file_path))
            )
            
            max_possible = len(all_nodes) - 1
            return direct_connections / max_possible if max_possible > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_pagerank_score(
        self, 
        file_path: str, 
        dependency_graph: DependencyGraph
    ) -> float:
        """Calculate simplified PageRank score."""
        # Simplified implementation
        try:
            dependents = dependency_graph.get_dependents(file_path)
            
            # Weight by the importance of files that depend on this file
            pagerank_score = 0.0
            for dependent in dependents:
                # Each dependent contributes based on its own connections
                dependent_connections = (
                    len(dependency_graph.get_dependencies(dependent)) +
                    len(dependency_graph.get_dependents(dependent))
                )
                pagerank_score += dependent_connections
            
            # Normalize
            total_nodes = len(dependency_graph.get_nodes())
            max_possible = total_nodes * (total_nodes - 1)
            
            return min(1.0, pagerank_score / max_possible) if max_possible > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_clustering_coefficient(
        self, 
        file_path: str, 
        dependency_graph: DependencyGraph
    ) -> float:
        """Calculate clustering coefficient."""
        try:
            neighbors = set()
            neighbors.update(dependency_graph.get_dependencies(file_path))
            neighbors.update(dependency_graph.get_dependents(file_path))
            
            if len(neighbors) < 2:
                return 0.0
            
            # Count connections between neighbors
            connections = 0
            neighbor_list = list(neighbors)
            for i, node1 in enumerate(neighbor_list):
                for node2 in neighbor_list[i+1:]:
                    if (node2 in dependency_graph.get_dependencies(node1) or
                        node1 in dependency_graph.get_dependencies(node2)):
                        connections += 1
            
            max_possible = len(neighbors) * (len(neighbors) - 1) // 2
            return connections / max_possible if max_possible > 0 else 0.0
        except:
            return 0.0
    
    def _get_task_specific_structural_adjustment(
        self, 
        task_type: str, 
        in_degree: int, 
        out_degree: int, 
        impact_score: float
    ) -> float:
        """Get task-specific structural importance adjustment."""
        if task_type == "refactoring":
            # High coupling indicates refactoring need
            return min(0.2, (in_degree + out_degree) * 0.01)
        elif task_type == "bugfix":
            # High impact files more likely to contain bugs
            return impact_score * 0.1
        elif task_type == "feature":
            # Moderate coupling preferred for feature development
            total_degree = in_degree + out_degree
            if 3 <= total_degree <= 10:
                return 0.1
            else:
                return 0.0
        else:
            return 0.0
    
    # Simulation methods (replace with real data integration)
    
    def _simulate_change_frequency(self, file_result: FileAnalysisResult) -> float:
        """Simulate change frequency based on file characteristics."""
        # Heuristic based on file type and size
        base_frequency = 0.5
        
        if file_result.file_type and 'config' in file_result.file_type.value:
            base_frequency = 0.3  # Config files change less frequently
        elif file_result.file_type and 'test' in file_result.file_type.value:
            base_frequency = 0.7  # Test files change more frequently
        
        # Adjust based on file size (larger files might change more)
        if file_result.file_size:
            size_factor = min(1.0, file_result.file_size / 10000)
            base_frequency += size_factor * 0.2
        
        return min(1.0, base_frequency)
    
    def _simulate_recent_activity(self, file_result: FileAnalysisResult) -> float:
        """Simulate recent activity."""
        # Random simulation - in real implementation, use Git data
        import random
        return random.uniform(0.0, 1.0)
    
    def _simulate_bug_density(self, file_result: FileAnalysisResult, task_type: str) -> float:
        """Simulate bug density."""
        base_density = 0.3
        
        # Larger files might have more bugs
        if file_result.file_size and file_result.file_size > 5000:
            base_density += 0.2
        
        # Complex files might have more bugs
        if file_result.analysis_data:
            function_count = len(file_result.analysis_data.get('functions', []))
            if function_count > 20:
                base_density += 0.1
        
        return min(1.0, base_density)
    
    def _simulate_contributor_diversity(self, file_result: FileAnalysisResult) -> float:
        """Simulate contributor diversity."""
        # Heuristic: core files tend to have more contributors
        filename = Path(file_result.file_path).name.lower()
        if any(pattern in filename for pattern in ['main', 'core', 'base', 'index']):
            return 0.8
        elif 'test' in filename:
            return 0.4
        else:
            return 0.6
    
    def _calculate_stability_score(
        self, 
        change_frequency: float, 
        recent_activity: float, 
        bug_density: float, 
        task_type: str
    ) -> float:
        """Calculate stability score based on historical factors."""
        if task_type == "bugfix":
            # For bug fixing, unstable files are more relevant
            instability = change_frequency * 0.4 + bug_density * 0.6
            return instability
        else:
            # For other tasks, stable files are preferred
            stability = (1.0 - change_frequency) * 0.3 + (1.0 - bug_density) * 0.4 + recent_activity * 0.3
            return stability
    
    def _extract_text_content(self, file_result: FileAnalysisResult) -> str:
        """Extract text content (comments, docstrings) from file."""
        text_parts = []
        
        if file_result.analysis_data:
            comments = file_result.analysis_data.get('comments', [])
            for comment in comments:
                if isinstance(comment, str) and len(comment) > 5:
                    text_parts.append(comment)
            
            # Extract docstrings from functions and classes
            functions = file_result.analysis_data.get('functions', [])
            for func in functions:
                if isinstance(func, dict) and 'docstring' in func:
                    docstring = func['docstring']
                    if docstring and len(docstring) > 10:
                        text_parts.append(docstring)
        
        return ' '.join(text_parts)
    
    def _calculate_documentation_density(self, file_result: FileAnalysisResult) -> float:
        """Calculate documentation density score."""
        if not file_result.analysis_data:
            return 0.3
        
        total_elements = (
            len(file_result.analysis_data.get('functions', [])) +
            len(file_result.analysis_data.get('classes', []))
        )
        
        if total_elements == 0:
            return 0.5
        
        documented_elements = 0
        
        # Check function documentation
        for func in file_result.analysis_data.get('functions', []):
            if isinstance(func, dict) and func.get('docstring'):
                documented_elements += 1
        
        # Check class documentation
        for cls in file_result.analysis_data.get('classes', []):
            if isinstance(cls, dict) and cls.get('docstring'):
                documented_elements += 1
        
        return documented_elements / total_elements
    
    def _calculate_naming_quality(self, file_result: FileAnalysisResult) -> float:
        """Calculate naming quality score."""
        if not file_result.analysis_data:
            return 0.5
        
        good_names = 0
        total_names = 0
        
        # Check function names
        for func in file_result.analysis_data.get('functions', []):
            if isinstance(func, dict) and 'name' in func:
                name = func['name']
                total_names += 1
                if self._is_good_name(name):
                    good_names += 1
            elif isinstance(func, str):
                total_names += 1
                if self._is_good_name(func):
                    good_names += 1
        
        # Check class names
        for cls in file_result.analysis_data.get('classes', []):
            if isinstance(cls, dict) and 'name' in cls:
                name = cls['name']
                total_names += 1
                if self._is_good_name(name):
                    good_names += 1
            elif isinstance(cls, str):
                total_names += 1
                if self._is_good_name(cls):
                    good_names += 1
        
        return good_names / total_names if total_names > 0 else 0.5
    
    def _is_good_name(self, name: str) -> bool:
        """Check if a name follows good naming conventions."""
        # Simple heuristics for good naming
        if len(name) < 2:
            return False
        
        # Check for descriptive length
        if len(name) >= 3 and not name.startswith('_'):
            return True
        
        # Single letter variables are generally bad
        if len(name) == 1 and name not in ['i', 'j', 'k', 'x', 'y', 'z']:
            return False
        
        return True
    
    def _calculate_language_consistency(self, text_content: str, task_description: str) -> float:
        """Calculate language consistency between file and task."""
        if not text_content:
            return 0.5
        
        # Simple heuristic: check for shared vocabulary
        task_words = set(task_description.lower().split())
        content_words = set(text_content.lower().split())
        
        if not task_words or not content_words:
            return 0.5
        
        shared_words = task_words.intersection(content_words)
        consistency = len(shared_words) / len(task_words.union(content_words))
        
        return min(1.0, consistency * 2)  # Boost the score
    
    # Explanation generators
    
    def _generate_semantic_explanation(
        self,
        tfidf_similarity: float,
        keyword_matches: List[str],
        concept_matches: List[str],
        topic_similarity: float
    ) -> str:
        """Generate explanation for semantic analysis."""
        explanations = []
        
        if tfidf_similarity > 0.6:
            explanations.append("High textual similarity to task description")
        elif tfidf_similarity > 0.3:
            explanations.append("Moderate textual similarity to task description")
        
        if keyword_matches:
            explanations.append(f"Keyword matches: {', '.join(keyword_matches[:5])}")
        
        if concept_matches:
            explanations.append(f"Conceptual matches: {', '.join(concept_matches)}")
        
        if topic_similarity > 0.5:
            explanations.append("Strong topic similarity")
        
        return '; '.join(explanations) if explanations else "Limited semantic similarity"
    
    def _generate_structural_explanation(
        self,
        degree_centrality: float,
        betweenness: float,
        closeness: float,
        pagerank: float,
        impact_score: float
    ) -> str:
        """Generate explanation for structural analysis."""
        explanations = []
        
        if degree_centrality > 0.6:
            explanations.append("High connectivity in dependency graph")
        elif degree_centrality > 0.3:
            explanations.append("Moderate connectivity in dependency graph")
        
        if betweenness > 0.5:
            explanations.append("Acts as bridge between components")
        
        if impact_score > 0.5:
            explanations.append("High impact on other files")
        
        if pagerank > 0.5:
            explanations.append("Important in dependency hierarchy")
        
        return '; '.join(explanations) if explanations else "Limited structural importance"
    
    def _generate_historical_explanation(
        self,
        change_frequency: float,
        recent_activity: float,
        bug_density: float,
        contributor_diversity: float
    ) -> str:
        """Generate explanation for historical analysis."""
        explanations = []
        
        if change_frequency > 0.7:
            explanations.append("Frequently modified file")
        elif change_frequency < 0.3:
            explanations.append("Stable file with few changes")
        
        if recent_activity > 0.7:
            explanations.append("Recent development activity")
        
        if bug_density > 0.6:
            explanations.append("History of bug fixes")
        elif bug_density < 0.3:
            explanations.append("Low bug density")
        
        if contributor_diversity > 0.7:
            explanations.append("Multiple contributors")
        
        return '; '.join(explanations) if explanations else "Neutral historical pattern"
"""
AI-Powered Context Optimization Engine for LeanVibe Agent Hive 2.0

Advanced context optimization system that uses AI analysis to intelligently select 
the most relevant files, functions, and code sections for specific development tasks.
Provides intelligent context selection, relevance scoring, and smart code context generation.
"""

import asyncio
import json
import math
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from .models import FileAnalysisResult, DependencyResult, AnalysisConfiguration
from .graph import DependencyGraph, GraphNode
from .utils import PathUtils, FileUtils

logger = structlog.get_logger()


class TaskType(Enum):
    """Types of development tasks for context optimization."""
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTORING = "refactoring"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    PERFORMANCE = "performance"
    SECURITY = "security"


class RelevanceAlgorithm(Enum):
    """Algorithms for calculating relevance scores."""
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HISTORICAL = "historical"
    HYBRID = "hybrid"
    ML_ENHANCED = "ml_enhanced"


class AssemblyStrategy(Enum):
    """Strategies for assembling context."""
    HIERARCHICAL = "hierarchical"
    DEPENDENCY_FIRST = "dependency_first"
    TASK_FOCUSED = "task_focused"
    BALANCED = "balanced"
    STREAMING = "streaming"


@dataclass
class ContextRequest:
    """Advanced context optimization request."""
    task_description: str
    task_type: TaskType
    files_mentioned: List[str] = field(default_factory=list)
    context_preferences: Dict[str, Any] = field(default_factory=dict)
    ai_model_info: Dict[str, Any] = field(default_factory=dict)
    historical_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default preferences."""
        default_prefs = {
            "max_files": 15,
            "max_tokens": 32000,
            "include_tests": False,
            "include_docs": True,
            "depth_limit": 3,
            "relevance_threshold": 0.3
        }
        for key, value in default_prefs.items():
            self.context_preferences.setdefault(key, value)


@dataclass
class RelevanceScore:
    """Enhanced relevance score with detailed reasoning."""
    file_path: str
    relevance_score: float
    confidence_score: float
    relevance_reasons: List[str]
    content_summary: str
    key_functions: List[str]
    key_classes: List[str]
    import_relationships: List[str]
    estimated_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextCluster:
    """Intelligent context cluster with enhanced metadata."""
    cluster_id: str
    name: str
    files: List[str]
    central_files: List[str]
    cluster_score: float
    cluster_type: str
    description: str
    cohesion_score: float
    dependency_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizedContext:
    """Complete optimized context result."""
    core_files: List[RelevanceScore]
    supporting_files: List[RelevanceScore]
    dependency_graph: Dict[str, Any]
    context_summary: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    suggestions: Dict[str, List[str]]
    performance_metrics: Dict[str, float]


class ContextOptimizer:
    """
    AI-powered context optimization engine with advanced algorithms.
    
    Features:
    - Semantic similarity analysis using embeddings
    - Structural relevance based on dependency graphs  
    - Historical context from Git analysis
    - ML-enhanced pattern recognition
    - Multi-strategy context assembly
    """
    
    def __init__(self, cache_manager=None, ml_analyzer=None, historical_analyzer=None):
        """Initialize ContextOptimizer with optional dependencies."""
        self.cache_manager = cache_manager
        self.ml_analyzer = ml_analyzer
        self.historical_analyzer = historical_analyzer
        
        # Scoring algorithm weights
        self.algorithm_weights = {
            RelevanceAlgorithm.SEMANTIC: 0.3,
            RelevanceAlgorithm.STRUCTURAL: 0.25,
            RelevanceAlgorithm.HISTORICAL: 0.2,
            RelevanceAlgorithm.ML_ENHANCED: 0.25
        }
        
        # Task-specific weight adjustments
        self.task_weight_adjustments = {
            TaskType.FEATURE: {
                RelevanceAlgorithm.SEMANTIC: 0.4,
                RelevanceAlgorithm.STRUCTURAL: 0.3,
                RelevanceAlgorithm.HISTORICAL: 0.1,
                RelevanceAlgorithm.ML_ENHANCED: 0.2
            },
            TaskType.BUGFIX: {
                RelevanceAlgorithm.SEMANTIC: 0.2,
                RelevanceAlgorithm.STRUCTURAL: 0.3,
                RelevanceAlgorithm.HISTORICAL: 0.4,
                RelevanceAlgorithm.ML_ENHANCED: 0.1
            },
            TaskType.REFACTORING: {
                RelevanceAlgorithm.SEMANTIC: 0.15,
                RelevanceAlgorithm.STRUCTURAL: 0.45,
                RelevanceAlgorithm.HISTORICAL: 0.2,
                RelevanceAlgorithm.ML_ENHANCED: 0.2
            }
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_optimizations": 0,
            "avg_processing_time": 0.0,
            "cache_hit_rate": 0.0,
            "accuracy_score": 0.0
        }
        
        # Initialize vectorizer for semantic analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._vectorizer_fitted = False
    
    async def optimize_context(
        self,
        context_request: ContextRequest,
        file_results: List[FileAnalysisResult],
        dependency_graph: DependencyGraph,
        project_path: str
    ) -> OptimizedContext:
        """
        Perform advanced context optimization for AI agents.
        
        Args:
            context_request: Detailed context request with preferences
            file_results: Analysis results for all project files
            dependency_graph: Project dependency graph
            project_path: Root path of the project
            
        Returns:
            OptimizedContext with intelligent file selection and metadata
        """
        start_time = time.time()
        
        logger.info("Starting AI-powered context optimization",
                   task_type=context_request.task_type.value,
                   file_count=len(file_results),
                   task_description=context_request.task_description[:100])
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context_request, file_results)
            if self.cache_manager:
                cached_result = await self.cache_manager.get_cached_context(cache_key)
                if cached_result:
                    logger.info("Context cache hit", cache_key=cache_key[:16])
                    return cached_result
            
            # Calculate relevance scores using multiple algorithms
            relevance_scores = await self._calculate_enhanced_relevance_scores(
                context_request, file_results, dependency_graph, project_path
            )
            
            # Apply intelligent filtering
            filtered_scores = await self._apply_intelligent_filtering(
                relevance_scores, context_request
            )
            
            # Create dependency graph structure
            dep_graph_data = await self._build_dependency_graph_data(
                filtered_scores, dependency_graph
            )
            
            # Separate core and supporting files
            core_files, supporting_files = await self._separate_core_supporting_files(
                filtered_scores, context_request
            )
            
            # Generate context summary
            context_summary = await self._generate_context_summary(
                core_files, supporting_files, context_request, dependency_graph
            )
            
            # Create optimization metadata
            processing_time = time.time() - start_time
            optimization_metadata = await self._create_optimization_metadata(
                context_request, relevance_scores, processing_time
            )
            
            # Generate suggestions
            suggestions = await self._generate_intelligent_suggestions(
                core_files, supporting_files, context_request, dependency_graph
            )
            
            # Create final result
            optimized_context = OptimizedContext(
                core_files=core_files,
                supporting_files=supporting_files,
                dependency_graph=dep_graph_data,
                context_summary=context_summary,
                optimization_metadata=optimization_metadata,
                suggestions=suggestions,
                performance_metrics=self._get_performance_metrics(processing_time)
            )
            
            # Cache result if cache manager available
            if self.cache_manager:
                await self.cache_manager.cache_context(cache_key, optimized_context)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            logger.info("Context optimization completed",
                       processing_time=processing_time,
                       core_files=len(core_files),
                       supporting_files=len(supporting_files))
            
            return optimized_context
            
        except Exception as e:
            logger.error("Context optimization failed", error=str(e))
            raise
    
    async def _calculate_enhanced_relevance_scores(
        self,
        context_request: ContextRequest,
        file_results: List[FileAnalysisResult],
        dependency_graph: DependencyGraph,
        project_path: str
    ) -> List[RelevanceScore]:
        """Calculate relevance scores using multiple AI algorithms."""
        relevance_scores = []
        
        # Get task-specific algorithm weights
        weights = self.task_weight_adjustments.get(
            context_request.task_type, self.algorithm_weights
        )
        
        for file_result in file_results:
            if not file_result.analysis_successful:
                continue
            
            # Calculate scores using different algorithms
            semantic_score = await self._calculate_semantic_relevance(
                file_result, context_request
            )
            
            structural_score = await self._calculate_structural_relevance(
                file_result, dependency_graph, context_request
            )
            
            historical_score = await self._calculate_historical_relevance(
                file_result, context_request, project_path
            )
            
            ml_score = await self._calculate_ml_enhanced_relevance(
                file_result, context_request
            )
            
            # Calculate weighted final score
            final_score = (
                semantic_score * weights[RelevanceAlgorithm.SEMANTIC] +
                structural_score * weights[RelevanceAlgorithm.STRUCTURAL] +
                historical_score * weights[RelevanceAlgorithm.HISTORICAL] +
                ml_score * weights[RelevanceAlgorithm.ML_ENHANCED]
            )
            
            # Calculate confidence based on score consistency
            scores_list = [semantic_score, structural_score, historical_score, ml_score]
            confidence = self._calculate_score_confidence(scores_list)
            
            # Generate relevance reasons
            reasons = self._generate_relevance_reasons(
                file_result, semantic_score, structural_score, 
                historical_score, ml_score, context_request
            )
            
            # Extract key elements
            key_functions, key_classes = self._extract_key_elements(file_result)
            
            # Calculate estimated tokens
            estimated_tokens = self._estimate_token_count(file_result)
            
            # Create relevance score object
            relevance_score = RelevanceScore(
                file_path=file_result.file_path,
                relevance_score=final_score,
                confidence_score=confidence,
                relevance_reasons=reasons,
                content_summary=self._generate_content_summary(file_result),
                key_functions=key_functions,
                key_classes=key_classes,
                import_relationships=self._extract_import_relationships(file_result),
                estimated_tokens=estimated_tokens,
                metadata={
                    "semantic_score": semantic_score,
                    "structural_score": structural_score,
                    "historical_score": historical_score,
                    "ml_score": ml_score,
                    "file_size": file_result.file_size,
                    "language": file_result.language,
                    "file_type": file_result.file_type.value if file_result.file_type else None
                }
            )
            
            relevance_scores.append(relevance_score)
        
        return relevance_scores
    
    async def _calculate_semantic_relevance(
        self,
        file_result: FileAnalysisResult,
        context_request: ContextRequest
    ) -> float:
        """Calculate semantic relevance using NLP techniques."""
        try:
            # Extract text content from file
            file_content = self._extract_file_text_content(file_result)
            task_text = context_request.task_description
            
            if not file_content or not task_text:
                return 0.1
            
            # Use TF-IDF vectorization for semantic similarity
            if not self._vectorizer_fitted:
                # Fit vectorizer on all available text
                all_texts = [file_content, task_text]
                self.vectorizer.fit(all_texts)
                self._vectorizer_fitted = True
            
            # Calculate cosine similarity
            try:
                vectors = self.vectorizer.transform([task_text, file_content])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                
                # Boost similarity for exact keyword matches
                task_keywords = set(task_text.lower().split())
                file_keywords = set(file_content.lower().split())
                keyword_overlap = len(task_keywords.intersection(file_keywords))
                keyword_boost = min(0.3, keyword_overlap * 0.05)
                
                # Boost for file name matches
                filename_boost = 0.0
                filename = Path(file_result.file_path).name.lower()
                for keyword in task_keywords:
                    if keyword in filename:
                        filename_boost += 0.1
                
                final_score = min(1.0, similarity + keyword_boost + filename_boost)
                return final_score
                
            except Exception:
                # Fallback to simple keyword matching
                return self._simple_keyword_similarity(file_content, task_text)
            
        except Exception as e:
            logger.warning("Semantic relevance calculation failed", 
                         file_path=file_result.file_path, error=str(e))
            return 0.1
    
    async def _calculate_structural_relevance(
        self,
        file_result: FileAnalysisResult,
        dependency_graph: DependencyGraph,
        context_request: ContextRequest
    ) -> float:
        """Calculate structural relevance based on dependency analysis."""
        try:
            file_path = file_result.file_path
            
            # Get dependency metrics
            incoming_deps = len(dependency_graph.get_dependents(file_path))
            outgoing_deps = len(dependency_graph.get_dependencies(file_path))
            
            # Calculate centrality score
            total_nodes = len(dependency_graph.get_nodes())
            if total_nodes <= 1:
                centrality = 0.0
            else:
                max_possible = total_nodes - 1
                centrality = (incoming_deps + outgoing_deps) / (2 * max_possible)
            
            # Calculate impact score
            impact_score = dependency_graph.calculate_impact_score(file_path)
            
            # Boost for files mentioned in context request
            mention_boost = 0.0
            if file_path in context_request.files_mentioned:
                mention_boost = 0.4
            elif any(mentioned in file_path for mentioned in context_request.files_mentioned):
                mention_boost = 0.2
            
            # Task-specific structural importance
            task_boost = 0.0
            if context_request.task_type == TaskType.REFACTORING:
                # High coupling indicates refactoring candidate
                task_boost = centrality * 0.3
            elif context_request.task_type == TaskType.BUGFIX:
                # High impact files more likely to contain bugs
                task_boost = impact_score * 0.2
            
            final_score = min(1.0, centrality * 0.4 + impact_score * 0.4 + 
                            mention_boost + task_boost)
            
            return final_score
            
        except Exception as e:
            logger.warning("Structural relevance calculation failed",
                         file_path=file_result.file_path, error=str(e))
            return 0.1
    
    async def _calculate_historical_relevance(
        self,
        file_result: FileAnalysisResult,
        context_request: ContextRequest,
        project_path: str
    ) -> float:
        """Calculate historical relevance based on Git history analysis."""
        try:
            if not self.historical_analyzer:
                return 0.5  # Neutral score if no historical data
            
            file_path = file_result.file_path
            
            # Get historical metrics
            change_frequency = await self.historical_analyzer.get_change_frequency(
                file_path, project_path
            )
            recent_changes = await self.historical_analyzer.get_recent_changes(
                file_path, project_path, days=30
            )
            bug_history = await self.historical_analyzer.get_bug_history(
                file_path, project_path
            )
            contributor_count = await self.historical_analyzer.get_contributor_count(
                file_path, project_path
            )
            
            # Calculate base historical score
            # More changes = higher relevance for bugs, lower for features
            if context_request.task_type == TaskType.BUGFIX:
                change_score = min(1.0, change_frequency * 0.1)
                bug_score = min(1.0, bug_history * 0.2)
            else:
                change_score = max(0.1, 1.0 - change_frequency * 0.05)
                bug_score = max(0.1, 1.0 - bug_history * 0.1)
            
            # Recent activity boost
            recent_score = min(1.0, recent_changes * 0.1)
            
            # Team knowledge score (more contributors = easier to work with)
            knowledge_score = min(1.0, contributor_count * 0.1)
            
            final_score = (change_score * 0.3 + bug_score * 0.3 + 
                          recent_score * 0.2 + knowledge_score * 0.2)
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning("Historical relevance calculation failed",
                         file_path=file_result.file_path, error=str(e))
            return 0.5
    
    async def _calculate_ml_enhanced_relevance(
        self,
        file_result: FileAnalysisResult,
        context_request: ContextRequest
    ) -> float:
        """Calculate ML-enhanced relevance using embeddings and patterns."""
        try:
            if not self.ml_analyzer:
                return 0.5  # Neutral score if no ML analyzer
            
            # Get code embeddings
            code_embedding = await self.ml_analyzer.get_code_embedding(
                file_result.file_path, file_result.analysis_data
            )
            
            task_embedding = await self.ml_analyzer.get_task_embedding(
                context_request.task_description
            )
            
            if code_embedding is not None and task_embedding is not None:
                # Calculate embedding similarity
                similarity = self.ml_analyzer.calculate_embedding_similarity(
                    code_embedding, task_embedding
                )
            else:
                similarity = 0.5
            
            # Pattern recognition boost
            pattern_score = await self.ml_analyzer.analyze_code_patterns(
                file_result, context_request.task_type
            )
            
            # Anomaly detection (for bug finding)
            anomaly_score = 0.0
            if context_request.task_type == TaskType.BUGFIX:
                anomaly_score = await self.ml_analyzer.detect_anomalies(
                    file_result
                )
            
            final_score = (similarity * 0.6 + pattern_score * 0.3 + 
                          anomaly_score * 0.1)
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning("ML relevance calculation failed",
                         file_path=file_result.file_path, error=str(e))
            return 0.5
    
    def _calculate_score_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score consistency."""
        if len(scores) < 2:
            return 0.5
        
        # Calculate standard deviation
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        # Lower standard deviation = higher confidence
        confidence = max(0.1, 1.0 - std_dev)
        
        # Boost confidence for consistently high scores
        if mean_score > 0.7 and std_dev < 0.2:
            confidence = min(1.0, confidence + 0.2)
        
        return confidence
    
    def _generate_relevance_reasons(
        self,
        file_result: FileAnalysisResult,
        semantic_score: float,
        structural_score: float,
        historical_score: float,
        ml_score: float,
        context_request: ContextRequest
    ) -> List[str]:
        """Generate human-readable reasons for relevance score."""
        reasons = []
        
        # Semantic reasons
        if semantic_score > 0.7:
            reasons.append("High semantic similarity to task description")
        elif semantic_score > 0.5:
            reasons.append("Moderate semantic relevance to task")
        
        # Structural reasons
        if structural_score > 0.7:
            reasons.append("High structural importance in dependency graph")
        elif structural_score > 0.5:
            reasons.append("Moderate dependency centrality")
        
        # Historical reasons
        if historical_score > 0.7:
            reasons.append("Strong historical relevance based on change patterns")
        elif historical_score > 0.5:
            reasons.append("Moderate historical activity")
        
        # ML reasons
        if ml_score > 0.7:
            reasons.append("AI analysis indicates high task relevance")
        elif ml_score > 0.5:
            reasons.append("Pattern recognition suggests relevance")
        
        # File-specific reasons
        if file_result.file_path in context_request.files_mentioned:
            reasons.append("Explicitly mentioned in task description")
        
        filename = Path(file_result.file_path).name.lower()
        if any(keyword in filename for keyword in ['main', 'core', 'base', 'index']):
            reasons.append("Core/entry point file")
        
        if not reasons:
            reasons.append("Basic file type relevance")
        
        return reasons
    
    def _extract_key_elements(self, file_result: FileAnalysisResult) -> Tuple[List[str], List[str]]:
        """Extract key functions and classes from file analysis."""
        functions = []
        classes = []
        
        if file_result.analysis_data:
            # Extract functions
            if 'functions' in file_result.analysis_data:
                for func in file_result.analysis_data['functions']:
                    if isinstance(func, dict) and 'name' in func:
                        functions.append(func['name'])
                    elif isinstance(func, str):
                        functions.append(func)
            
            # Extract classes
            if 'classes' in file_result.analysis_data:
                for cls in file_result.analysis_data['classes']:
                    if isinstance(cls, dict) and 'name' in cls:
                        classes.append(cls['name'])
                    elif isinstance(cls, str):
                        classes.append(cls)
        
        return functions[:10], classes[:10]  # Limit to top 10
    
    def _estimate_token_count(self, file_result: FileAnalysisResult) -> int:
        """Estimate token count for the file content."""
        if not file_result.file_size:
            return 100
        
        # Rough estimation: 1 token per 4 characters on average
        estimated_tokens = file_result.file_size // 4
        
        # Adjust based on file type
        if file_result.language in ['python', 'javascript', 'typescript']:
            # More verbose languages
            estimated_tokens = int(estimated_tokens * 1.2)
        elif file_result.language in ['json', 'yaml']:
            # More compact
            estimated_tokens = int(estimated_tokens * 0.8)
        
        return max(50, min(10000, estimated_tokens))
    
    def _generate_content_summary(self, file_result: FileAnalysisResult) -> str:
        """Generate AI-powered content summary for the file."""
        summary_parts = []
        
        # File type and language
        file_type = file_result.file_type.value if file_result.file_type else "unknown"
        language = file_result.language or "unknown"
        summary_parts.append(f"{file_type.title()} file in {language}")
        
        # Size information
        if file_result.line_count:
            summary_parts.append(f"~{file_result.line_count} lines")
        
        # Key elements
        if file_result.analysis_data:
            functions = file_result.analysis_data.get('functions', [])
            classes = file_result.analysis_data.get('classes', [])
            
            if classes:
                summary_parts.append(f"{len(classes)} class(es)")
            if functions:
                summary_parts.append(f"{len(functions)} function(s)")
        
        # Purpose inference from filename
        filename = Path(file_result.file_path).name.lower()
        if 'test' in filename:
            summary_parts.append("test file")
        elif 'config' in filename:
            summary_parts.append("configuration")
        elif 'util' in filename or 'helper' in filename:
            summary_parts.append("utility functions")
        elif 'main' in filename or 'index' in filename:
            summary_parts.append("entry point")
        
        return ", ".join(summary_parts) if summary_parts else "Code file"
    
    def _extract_import_relationships(self, file_result: FileAnalysisResult) -> List[str]:
        """Extract import relationships from file analysis."""
        imports = []
        
        if file_result.analysis_data:
            if 'imports' in file_result.analysis_data:
                imports_data = file_result.analysis_data['imports']
                if isinstance(imports_data, list):
                    imports = [imp if isinstance(imp, str) else str(imp) for imp in imports_data]
                elif isinstance(imports_data, dict):
                    imports = list(imports_data.keys())
        
        return imports[:20]  # Limit to top 20 imports
    
    def _extract_file_text_content(self, file_result: FileAnalysisResult) -> str:
        """Extract searchable text content from file analysis."""
        content_parts = []
        
        # Filename (without extension)
        filename = Path(file_result.file_path).stem
        content_parts.append(filename.replace('_', ' ').replace('-', ' '))
        
        # Analysis data
        if file_result.analysis_data:
            # Function names
            functions = file_result.analysis_data.get('functions', [])
            for func in functions:
                if isinstance(func, dict) and 'name' in func:
                    content_parts.append(func['name'].replace('_', ' '))
                elif isinstance(func, str):
                    content_parts.append(func.replace('_', ' '))
            
            # Class names
            classes = file_result.analysis_data.get('classes', [])
            for cls in classes:
                if isinstance(cls, dict) and 'name' in cls:
                    content_parts.append(cls['name'].replace('_', ' '))
                elif isinstance(cls, str):
                    content_parts.append(cls.replace('_', ' '))
            
            # Comments/docstrings
            comments = file_result.analysis_data.get('comments', [])
            for comment in comments[:5]:  # Limit to first 5 comments
                if isinstance(comment, str) and len(comment) > 10:
                    content_parts.append(comment[:200])  # Limit comment length
        
        return ' '.join(content_parts)
    
    def _simple_keyword_similarity(self, file_content: str, task_text: str) -> float:
        """Simple keyword-based similarity fallback."""
        task_keywords = set(task_text.lower().split())
        file_keywords = set(file_content.lower().split())
        
        if not task_keywords or not file_keywords:
            return 0.1
        
        intersection = task_keywords.intersection(file_keywords)
        union = task_keywords.union(file_keywords)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        return min(1.0, jaccard_similarity * 3)  # Boost the score
    
    async def _apply_intelligent_filtering(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest
    ) -> List[RelevanceScore]:
        """Apply intelligent filtering based on context preferences."""
        filtered_scores = []
        
        # Apply relevance threshold
        threshold = context_request.context_preferences.get("relevance_threshold", 0.3)
        for score in relevance_scores:
            if score.relevance_score >= threshold:
                filtered_scores.append(score)
        
        # Sort by relevance score
        filtered_scores.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply max files limit
        max_files = context_request.context_preferences.get("max_files", 15)
        filtered_scores = filtered_scores[:max_files]
        
        # Token budget management
        max_tokens = context_request.context_preferences.get("max_tokens", 32000)
        total_tokens = 0
        token_filtered_scores = []
        
        for score in filtered_scores:
            if total_tokens + score.estimated_tokens <= max_tokens:
                token_filtered_scores.append(score)
                total_tokens += score.estimated_tokens
            else:
                break
        
        return token_filtered_scores
    
    async def _separate_core_supporting_files(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest
    ) -> Tuple[List[RelevanceScore], List[RelevanceScore]]:
        """Separate files into core and supporting categories."""
        # Core files: high relevance and high confidence
        core_files = []
        supporting_files = []
        
        for score in relevance_scores:
            if (score.relevance_score >= 0.7 and score.confidence_score >= 0.6):
                core_files.append(score)
            else:
                supporting_files.append(score)
        
        # Ensure at least some core files
        if len(core_files) < 3 and len(relevance_scores) > 0:
            # Promote top scoring files to core
            remaining = 3 - len(core_files)
            promoted = supporting_files[:remaining]
            core_files.extend(promoted)
            supporting_files = supporting_files[remaining:]
        
        return core_files, supporting_files
    
    async def _build_dependency_graph_data(
        self,
        relevance_scores: List[RelevanceScore],
        dependency_graph: DependencyGraph
    ) -> Dict[str, Any]:
        """Build dependency graph data for the optimized context."""
        nodes = []
        edges = []
        
        # Create nodes for selected files
        file_paths = {score.file_path for score in relevance_scores}
        
        for score in relevance_scores:
            node = {
                "id": score.file_path,
                "type": "file",
                "metadata": {
                    "relevance_score": score.relevance_score,
                    "confidence_score": score.confidence_score,
                    "estimated_tokens": score.estimated_tokens,
                    "language": score.metadata.get("language"),
                    "file_type": score.metadata.get("file_type")
                }
            }
            nodes.append(node)
        
        # Create edges for dependencies between selected files
        for score in relevance_scores:
            dependencies = dependency_graph.get_dependencies(score.file_path)
            for dep_path in dependencies:
                if dep_path in file_paths:
                    edge = {
                        "source": score.file_path,
                        "target": dep_path,
                        "type": "dependency",
                        "weight": 1.0
                    }
                    edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    async def _generate_context_summary(
        self,
        core_files: List[RelevanceScore],
        supporting_files: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> Dict[str, Any]:
        """Generate comprehensive context summary."""
        total_files = len(core_files) + len(supporting_files)
        total_tokens = sum(score.estimated_tokens for score in core_files + supporting_files)
        
        # Calculate coverage percentage (simplified)
        coverage_percentage = min(100.0, (total_files / max(1, total_files)) * 100)
        
        # Calculate confidence score
        confidence_scores = [score.confidence_score for score in core_files + supporting_files]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Identify architectural patterns
        architectural_patterns = self._identify_architectural_patterns(core_files + supporting_files)
        
        # Identify potential challenges
        potential_challenges = self._identify_potential_challenges(
            core_files + supporting_files, context_request
        )
        
        # Generate recommended approach
        recommended_approach = self._generate_recommended_approach(
            context_request, core_files, supporting_files
        )
        
        return {
            "total_files": total_files,
            "total_tokens": total_tokens,
            "coverage_percentage": coverage_percentage,
            "confidence_score": avg_confidence,
            "architectural_patterns": architectural_patterns,
            "potential_challenges": potential_challenges,
            "recommended_approach": recommended_approach
        }
    
    def _identify_architectural_patterns(self, relevance_scores: List[RelevanceScore]) -> List[str]:
        """Identify architectural patterns from selected files."""
        patterns = []
        
        # Pattern detection based on file names and structures
        file_paths = [score.file_path for score in relevance_scores]
        
        # MVC pattern
        has_models = any('model' in path.lower() for path in file_paths)
        has_views = any('view' in path.lower() for path in file_paths)
        has_controllers = any('controller' in path.lower() for path in file_paths)
        if has_models and has_views and has_controllers:
            patterns.append("MVC (Model-View-Controller)")
        
        # API pattern
        has_api = any('api' in path.lower() for path in file_paths)
        has_routes = any('route' in path.lower() for path in file_paths)
        if has_api or has_routes:
            patterns.append("RESTful API")
        
        # Microservices pattern
        has_services = len([p for p in file_paths if 'service' in p.lower()]) > 1
        if has_services:
            patterns.append("Service-oriented architecture")
        
        # Test pattern
        has_tests = any('test' in path.lower() for path in file_paths)
        if has_tests:
            patterns.append("Test-driven development")
        
        return patterns if patterns else ["Standard modular architecture"]
    
    def _identify_potential_challenges(
        self,
        relevance_scores: List[RelevanceScore],
        context_request: ContextRequest
    ) -> List[str]:
        """Identify potential challenges for the development task."""
        challenges = []
        
        # Low confidence scores
        low_confidence_count = len([s for s in relevance_scores if s.confidence_score < 0.5])
        if low_confidence_count > len(relevance_scores) * 0.3:
            challenges.append("Some files have low relevance confidence - may need manual review")
        
        # High complexity files
        high_token_files = [s for s in relevance_scores if s.estimated_tokens > 2000]
        if high_token_files:
            challenges.append(f"{len(high_token_files)} large/complex files may require extra attention")
        
        # Task-specific challenges
        if context_request.task_type == TaskType.REFACTORING:
            high_coupling = len([s for s in relevance_scores if len(s.import_relationships) > 10])
            if high_coupling > 0:
                challenges.append("High coupling detected - refactoring may have wide impact")
        
        elif context_request.task_type == TaskType.BUGFIX:
            # Check for files with many dependencies (potential bug sources)
            complex_deps = len([s for s in relevance_scores if len(s.import_relationships) > 15])
            if complex_deps > 0:
                challenges.append("Complex dependency relationships may complicate debugging")
        
        return challenges if challenges else ["No significant challenges identified"]
    
    def _generate_recommended_approach(
        self,
        context_request: ContextRequest,
        core_files: List[RelevanceScore],
        supporting_files: List[RelevanceScore]
    ) -> str:
        """Generate recommended approach for the development task."""
        if context_request.task_type == TaskType.FEATURE:
            return (f"Start with {len(core_files)} core files for feature implementation, "
                   f"then review {len(supporting_files)} supporting files for integration points")
        
        elif context_request.task_type == TaskType.BUGFIX:
            return (f"Begin debugging with the {len(core_files)} most relevant files, "
                   f"focusing on recent changes and error-prone areas")
        
        elif context_request.task_type == TaskType.REFACTORING:
            return (f"Analyze dependencies in {len(core_files)} core files first, "
                   f"then plan refactoring to minimize impact on {len(supporting_files)} dependents")
        
        elif context_request.task_type == TaskType.ANALYSIS:
            return (f"Perform comprehensive analysis starting with {len(core_files)} key files, "
                   f"expanding to {len(supporting_files)} related files as needed")
        
        else:
            return (f"Review {len(core_files)} primary files for context, "
                   f"with {len(supporting_files)} additional files for comprehensive understanding")
    
    async def _create_optimization_metadata(
        self,
        context_request: ContextRequest,
        relevance_scores: List[RelevanceScore],
        processing_time: float
    ) -> Dict[str, Any]:
        """Create optimization metadata for analysis and debugging."""
        algorithm_stats = {
            "semantic_avg": np.mean([s.metadata.get("semantic_score", 0) for s in relevance_scores]),
            "structural_avg": np.mean([s.metadata.get("structural_score", 0) for s in relevance_scores]),
            "historical_avg": np.mean([s.metadata.get("historical_score", 0) for s in relevance_scores]),
            "ml_avg": np.mean([s.metadata.get("ml_score", 0) for s in relevance_scores])
        }
        
        relevance_distribution = {
            "high": len([s for s in relevance_scores if s.relevance_score >= 0.7]),
            "medium": len([s for s in relevance_scores if 0.4 <= s.relevance_score < 0.7]),
            "low": len([s for s in relevance_scores if s.relevance_score < 0.4])
        }
        
        return {
            "algorithm_used": "hybrid_ai_enhanced",
            "processing_time_ms": int(processing_time * 1000),
            "cache_hit_rate": self.performance_metrics.get("cache_hit_rate", 0.0),
            "relevance_distribution": relevance_distribution,
            "algorithm_stats": algorithm_stats,
            "task_type": context_request.task_type.value,
            "total_files_analyzed": len(relevance_scores)
        }
    
    async def _generate_intelligent_suggestions(
        self,
        core_files: List[RelevanceScore],
        supporting_files: List[RelevanceScore],
        context_request: ContextRequest,
        dependency_graph: DependencyGraph
    ) -> Dict[str, List[str]]:
        """Generate intelligent suggestions for context optimization."""
        suggestions = {
            "additional_files": [],
            "alternative_contexts": [],
            "optimization_tips": []
        }
        
        # Additional files suggestions
        all_files = core_files + supporting_files
        file_paths = {score.file_path for score in all_files}
        
        # Find highly connected files not in current selection
        for score in all_files:
            deps = dependency_graph.get_dependencies(score.file_path)
            for dep_path in deps:
                if dep_path not in file_paths:
                    suggestions["additional_files"].append(dep_path)
        
        # Limit additional files suggestions
        suggestions["additional_files"] = suggestions["additional_files"][:5]
        
        # Alternative context strategies
        if len(core_files) < 3:
            suggestions["alternative_contexts"].append(
                "Consider lowering relevance threshold to include more files"
            )
        
        if context_request.context_preferences.get("max_tokens", 32000) > 50000:
            suggestions["alternative_contexts"].append(
                "High token limit - consider focusing on fewer, more relevant files"
            )
        
        # Optimization tips
        low_confidence_files = [s for s in all_files if s.confidence_score < 0.5]
        if low_confidence_files:
            suggestions["optimization_tips"].append(
                f"Review {len(low_confidence_files)} low-confidence files for relevance"
            )
        
        if context_request.task_type == TaskType.FEATURE:
            suggestions["optimization_tips"].append(
                "Include configuration and test files for comprehensive feature development"
            )
        
        suggestions["optimization_tips"].append(
            "Use incremental context expansion if initial selection insufficient"
        )
        
        return suggestions
    
    def _generate_cache_key(
        self, 
        context_request: ContextRequest, 
        file_results: List[FileAnalysisResult]
    ) -> str:
        """Generate cache key for context optimization result."""
        # Create hash from request parameters and file metadata
        import hashlib
        
        key_data = {
            "task_description": context_request.task_description,
            "task_type": context_request.task_type.value,
            "files_mentioned": sorted(context_request.files_mentioned),
            "preferences": context_request.context_preferences,
            "file_count": len(file_results),
            "file_hashes": sorted([
                f"{f.file_path}:{f.file_hash}" for f in file_results 
                if f.file_hash
            ][:20])  # Limit to first 20 for performance
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_performance_metrics(self, processing_time: float) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            "processing_time_seconds": processing_time,
            "files_per_second": self.performance_metrics.get("files_per_second", 0.0),
            "cache_hit_rate": self.performance_metrics.get("cache_hit_rate", 0.0),
            "accuracy_score": self.performance_metrics.get("accuracy_score", 0.0)
        }
    
    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics after optimization."""
        self.performance_metrics["total_optimizations"] += 1
        
        # Update average processing time
        total_ops = self.performance_metrics["total_optimizations"]
        current_avg = self.performance_metrics["avg_processing_time"]
        new_avg = ((current_avg * (total_ops - 1)) + processing_time) / total_ops
        self.performance_metrics["avg_processing_time"] = new_avg
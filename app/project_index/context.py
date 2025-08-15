"""
Context Optimization for LeanVibe Agent Hive 2.0

AI-powered context selection and optimization algorithms for intelligent
code analysis and development assistance. Provides relevance scoring,
file clustering, and context recommendations.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

import structlog

from .models import FileAnalysisResult, DependencyResult
from .graph import DependencyGraph, GraphNode

logger = structlog.get_logger()


class ContextType(Enum):
    """Types of context optimization."""
    DEVELOPMENT = "development"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    FEATURE_IMPLEMENTATION = "feature_implementation"
    BUG_FIX = "bug_fix"
    CODE_REVIEW = "code_review"


@dataclass
class ContextRelevanceScore:
    """Relevance score for a file in a specific context."""
    file_path: str
    relevance_score: float
    confidence: float
    reasoning: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContextCluster:
    """A cluster of related files for a specific context."""
    cluster_id: str
    name: str
    files: List[str]
    central_files: List[str]
    cluster_score: float
    context_type: ContextType
    description: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContextOptimizationResult:
    """Result of context optimization analysis."""
    context_type: ContextType
    total_files: int
    recommended_files: List[ContextRelevanceScore]
    clusters: List[ContextCluster]
    entry_points: List[str]
    high_impact_files: List[str]
    optimization_metrics: Dict[str, float]
    recommendations: List[str]


class ContextOptimizer:
    """
    AI-powered context optimization for intelligent code analysis.
    
    Provides algorithms for file relevance scoring, clustering,
    and context-aware recommendations for development tasks.
    """
    
    def __init__(self):
        """Initialize ContextOptimizer."""
        self._file_scores_cache: Dict[str, Dict[str, float]] = {}
        self._cluster_cache: Dict[str, List[ContextCluster]] = {}
        
        # Scoring weights for different factors
        self.scoring_weights = {
            'dependency_centrality': 0.25,
            'modification_frequency': 0.15,
            'code_complexity': 0.10,
            'test_coverage': 0.10,
            'documentation_quality': 0.05,
            'file_size': 0.05,
            'import_frequency': 0.15,
            'semantic_similarity': 0.15
        }
    
    async def optimize_context(
        self,
        context_type: ContextType,
        file_results: List[FileAnalysisResult],
        dependency_graph: DependencyGraph,
        task_description: Optional[str] = None,
        target_files: Optional[List[str]] = None
    ) -> ContextOptimizationResult:
        """
        Perform context optimization for a specific development task.
        
        Args:
            context_type: Type of context optimization
            file_results: List of file analysis results
            dependency_graph: Project dependency graph
            task_description: Optional description of the task
            target_files: Optional list of files specifically relevant to task
            
        Returns:
            ContextOptimizationResult with recommendations
        """
        logger.info("Starting context optimization", 
                   context_type=context_type.value,
                   file_count=len(file_results))
        
        # Calculate relevance scores for all files
        relevance_scores = await self._calculate_relevance_scores(
            context_type, file_results, dependency_graph, task_description, target_files
        )
        
        # Identify high-impact files
        high_impact_files = self._identify_high_impact_files(
            file_results, dependency_graph
        )
        
        # Find entry points
        entry_points = self._find_entry_points(
            file_results, dependency_graph, context_type
        )
        
        # Create file clusters
        clusters = await self._create_context_clusters(
            context_type, file_results, dependency_graph, relevance_scores
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            context_type, relevance_scores, clusters, high_impact_files, entry_points
        )
        
        # Calculate optimization metrics
        metrics = self._calculate_optimization_metrics(
            relevance_scores, clusters, high_impact_files
        )
        
        # Select top recommended files
        recommended_files = sorted(
            relevance_scores, 
            key=lambda x: x.relevance_score, 
            reverse=True
        )[:50]  # Top 50 most relevant files
        
        result = ContextOptimizationResult(
            context_type=context_type,
            total_files=len(file_results),
            recommended_files=recommended_files,
            clusters=clusters,
            entry_points=entry_points,
            high_impact_files=high_impact_files,
            optimization_metrics=metrics,
            recommendations=recommendations
        )
        
        logger.info("Context optimization completed", 
                   context_type=context_type.value,
                   recommended_files=len(recommended_files),
                   clusters=len(clusters))
        
        return result
    
    async def _calculate_relevance_scores(
        self,
        context_type: ContextType,
        file_results: List[FileAnalysisResult],
        dependency_graph: DependencyGraph,
        task_description: Optional[str] = None,
        target_files: Optional[List[str]] = None
    ) -> List[ContextRelevanceScore]:
        """Calculate relevance scores for all files."""
        scores = []
        
        # Create file lookup
        file_lookup = {result.file_path: result for result in file_results}
        
        for file_result in file_results:
            score_components = {}
            reasoning = []
            
            # Base relevance based on file type and context
            base_score = self._calculate_base_relevance(file_result, context_type)
            score_components['base'] = base_score
            
            if base_score > 0:
                reasoning.append(f"File type {file_result.file_type} relevant for {context_type.value}")
            
            # Dependency centrality score
            centrality_score = self._calculate_dependency_centrality(
                file_result.file_path, dependency_graph
            )
            score_components['centrality'] = centrality_score
            
            if centrality_score > 0.5:
                reasoning.append("High dependency centrality")
            
            # Code complexity score
            complexity_score = self._calculate_complexity_relevance(file_result)
            score_components['complexity'] = complexity_score
            
            # Target file bonus
            target_bonus = 0.0
            if target_files and file_result.file_path in target_files:
                target_bonus = 0.3
                reasoning.append("Explicitly specified as target file")
            
            # Semantic similarity (simplified)
            semantic_score = self._calculate_semantic_similarity(
                file_result, task_description
            )
            score_components['semantic'] = semantic_score
            
            # Language-specific adjustments
            language_bonus = self._get_language_bonus(file_result.language, context_type)
            score_components['language'] = language_bonus
            
            # Calculate weighted final score
            final_score = (
                base_score * self.scoring_weights['dependency_centrality'] +
                centrality_score * self.scoring_weights['dependency_centrality'] +
                complexity_score * self.scoring_weights['code_complexity'] +
                semantic_score * self.scoring_weights['semantic_similarity'] +
                language_bonus * 0.1 +
                target_bonus
            )
            
            # Normalize score to 0-1 range
            final_score = max(0.0, min(1.0, final_score))
            
            # Calculate confidence based on available information
            confidence = self._calculate_confidence(file_result, score_components)
            
            relevance = ContextRelevanceScore(
                file_path=file_result.file_path,
                relevance_score=final_score,
                confidence=confidence,
                reasoning=reasoning,
                metadata=score_components
            )
            
            scores.append(relevance)
        
        return scores
    
    def _calculate_base_relevance(
        self, 
        file_result: FileAnalysisResult, 
        context_type: ContextType
    ) -> float:
        """Calculate base relevance score based on file type and context."""
        if not file_result.analysis_successful:
            return 0.0
        
        # File type relevance mapping
        file_type_scores = {
            ContextType.DEVELOPMENT: {
                'source': 0.9,
                'config': 0.3,
                'test': 0.2,
                'documentation': 0.1,
                'build': 0.2
            },
            ContextType.TESTING: {
                'source': 0.7,
                'test': 0.9,
                'config': 0.4,
                'documentation': 0.1,
                'build': 0.3
            },
            ContextType.DOCUMENTATION: {
                'documentation': 0.9,
                'source': 0.6,
                'test': 0.2,
                'config': 0.3,
                'build': 0.1
            },
            ContextType.DEBUGGING: {
                'source': 0.9,
                'test': 0.8,
                'config': 0.4,
                'documentation': 0.2,
                'build': 0.2
            },
            ContextType.REFACTORING: {
                'source': 0.9,
                'test': 0.7,
                'config': 0.2,
                'documentation': 0.3,
                'build': 0.1
            }
        }
        
        # Default scores for other context types
        default_scores = {
            'source': 0.8,
            'test': 0.5,
            'config': 0.3,
            'documentation': 0.2,
            'build': 0.2
        }
        
        context_scores = file_type_scores.get(context_type, default_scores)
        return context_scores.get(file_result.file_type.value, 0.1)
    
    def _calculate_dependency_centrality(
        self, 
        file_path: str, 
        dependency_graph: DependencyGraph
    ) -> float:
        """Calculate dependency centrality score for a file."""
        node = dependency_graph.get_node(file_path)
        if not node:
            return 0.0
        
        # Count incoming and outgoing dependencies
        incoming = len(dependency_graph.get_dependents(file_path))
        outgoing = len(dependency_graph.get_dependencies(file_path))
        
        # Calculate centrality score
        total_nodes = len(dependency_graph.get_nodes())
        if total_nodes <= 1:
            return 0.0
        
        # Normalize by total possible connections
        max_connections = total_nodes - 1
        centrality = (incoming + outgoing) / (2 * max_connections)
        
        return min(1.0, centrality)
    
    def _calculate_complexity_relevance(self, file_result: FileAnalysisResult) -> float:
        """Calculate relevance based on code complexity."""
        if not file_result.analysis_data:
            return 0.1
        
        # Use line count as a simple complexity metric
        line_count = file_result.line_count or 0
        
        # Optimal complexity range (not too simple, not too complex)
        if line_count < 10:
            return 0.1  # Too simple
        elif line_count < 50:
            return 0.3  # Simple
        elif line_count < 200:
            return 0.7  # Moderate complexity
        elif line_count < 500:
            return 0.9  # High complexity
        else:
            return 0.6  # Very high complexity (might be harder to work with)
    
    def _calculate_semantic_similarity(
        self, 
        file_result: FileAnalysisResult, 
        task_description: Optional[str]
    ) -> float:
        """Calculate semantic similarity between file and task description."""
        if not task_description:
            return 0.5  # Neutral score
        
        # Simple keyword matching (in a full implementation, use embeddings)
        task_keywords = set(task_description.lower().split())
        
        # Extract keywords from file
        file_keywords = set()
        
        # Add filename keywords
        filename_parts = file_result.file_name.lower().replace('_', ' ').replace('-', ' ').split()
        file_keywords.update(filename_parts)
        
        # Add keywords from analysis data
        if file_result.analysis_data:
            # Add function names, class names, etc.
            functions = file_result.analysis_data.get('functions', [])
            for func in functions:
                if isinstance(func, dict) and 'name' in func:
                    func_parts = func['name'].lower().replace('_', ' ').split()
                    file_keywords.update(func_parts)
            
            classes = file_result.analysis_data.get('classes', [])
            for cls in classes:
                if isinstance(cls, dict) and 'name' in cls:
                    cls_parts = cls['name'].lower().replace('_', ' ').split()
                    file_keywords.update(cls_parts)
        
        # Calculate similarity
        if not file_keywords:
            return 0.3
        
        intersection = task_keywords.intersection(file_keywords)
        union = task_keywords.union(file_keywords)
        
        similarity = len(intersection) / len(union) if union else 0.0
        
        return min(1.0, similarity * 2)  # Boost the score
    
    def _get_language_bonus(self, language: Optional[str], context_type: ContextType) -> float:
        """Get language-specific bonus for context type."""
        if not language:
            return 0.0
        
        language_bonuses = {
            ContextType.TESTING: {
                'python': 0.1,  # Good testing frameworks
                'javascript': 0.1,
                'typescript': 0.1
            },
            ContextType.DOCUMENTATION: {
                'markdown': 0.2,
                'restructuredtext': 0.2,
                'text': 0.1
            }
        }
        
        return language_bonuses.get(context_type, {}).get(language, 0.0)
    
    def _calculate_confidence(
        self, 
        file_result: FileAnalysisResult, 
        score_components: Dict[str, float]
    ) -> float:
        """Calculate confidence in the relevance score."""
        confidence_factors = []
        
        # Analysis success
        if file_result.analysis_successful:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.1)
        
        # Available analysis data
        if file_result.analysis_data:
            confidence_factors.append(0.3)
        
        # Language detection
        if file_result.language:
            confidence_factors.append(0.2)
        
        # File size (not too small, not too large)
        if file_result.file_size and 100 < file_result.file_size < 100000:
            confidence_factors.append(0.2)
        
        return sum(confidence_factors)
    
    def _identify_high_impact_files(
        self, 
        file_results: List[FileAnalysisResult], 
        dependency_graph: DependencyGraph
    ) -> List[str]:
        """Identify files with high impact on the project."""
        impact_scores = []
        
        for file_result in file_results:
            impact_score = dependency_graph.calculate_impact_score(file_result.file_path)
            impact_scores.append((file_result.file_path, impact_score))
        
        # Sort by impact score and return top files
        impact_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 20% of files or maximum 20 files
        top_count = max(1, min(20, len(impact_scores) // 5))
        return [file_path for file_path, _ in impact_scores[:top_count]]
    
    def _find_entry_points(
        self, 
        file_results: List[FileAnalysisResult], 
        dependency_graph: DependencyGraph,
        context_type: ContextType
    ) -> List[str]:
        """Find entry point files for the project."""
        entry_points = []
        
        # Common entry point patterns
        entry_patterns = [
            'main.py', 'app.py', '__main__.py', 'index.js', 'index.ts',
            'server.py', 'run.py', 'start.py', 'cli.py'
        ]
        
        # Files with main functions
        main_files = []
        for file_result in file_results:
            filename = file_result.file_name.lower()
            
            # Check filename patterns
            if any(pattern in filename for pattern in entry_patterns):
                entry_points.append(file_result.file_path)
            
            # Check for main functions
            if file_result.analysis_data:
                functions = file_result.analysis_data.get('functions', [])
                for func in functions:
                    if isinstance(func, dict) and func.get('name') == 'main':
                        main_files.append(file_result.file_path)
        
        # Add files with main functions
        entry_points.extend(main_files)
        
        # Files with high outgoing dependencies (potential orchestrators)
        orchestrator_files = []
        for file_result in file_results:
            outgoing_deps = len(dependency_graph.get_dependencies(file_result.file_path))
            if outgoing_deps > 5:  # Arbitrary threshold
                orchestrator_files.append(file_result.file_path)
        
        # Add top orchestrator files
        orchestrator_files.sort(
            key=lambda x: len(dependency_graph.get_dependencies(x)), 
            reverse=True
        )
        entry_points.extend(orchestrator_files[:5])
        
        # Remove duplicates and return
        return list(set(entry_points))
    
    async def _create_context_clusters(
        self,
        context_type: ContextType,
        file_results: List[FileAnalysisResult],
        dependency_graph: DependencyGraph,
        relevance_scores: List[ContextRelevanceScore]
    ) -> List[ContextCluster]:
        """Create clusters of related files for the context."""
        clusters = []
        
        # Group files by directory
        directory_groups = defaultdict(list)
        for file_result in file_results:
            directory = '/'.join(file_result.relative_path.split('/')[:-1])
            directory_groups[directory].append(file_result.file_path)
        
        # Create directory-based clusters
        for directory, files in directory_groups.items():
            if len(files) >= 2:  # Only create clusters with multiple files
                # Calculate cluster score
                relevant_files = [
                    score for score in relevance_scores 
                    if score.file_path in files
                ]
                
                if relevant_files:
                    avg_score = sum(s.relevance_score for s in relevant_files) / len(relevant_files)
                    
                    # Find central files (highest scoring in cluster)
                    central_files = [
                        s.file_path for s in sorted(relevant_files, 
                                                   key=lambda x: x.relevance_score, 
                                                   reverse=True)[:3]
                    ]
                    
                    cluster = ContextCluster(
                        cluster_id=f"dir_{directory.replace('/', '_')}",
                        name=f"Directory: {directory or 'root'}",
                        files=files,
                        central_files=central_files,
                        cluster_score=avg_score,
                        context_type=context_type,
                        description=f"Files in {directory or 'root'} directory",
                        metadata={'directory': directory, 'file_count': len(files)}
                    )
                    
                    clusters.append(cluster)
        
        # Create functionality-based clusters (simplified)
        functionality_clusters = self._create_functionality_clusters(
            file_results, relevance_scores, context_type
        )
        clusters.extend(functionality_clusters)
        
        # Sort clusters by score
        clusters.sort(key=lambda x: x.cluster_score, reverse=True)
        
        return clusters[:10]  # Return top 10 clusters
    
    def _create_functionality_clusters(
        self,
        file_results: List[FileAnalysisResult],
        relevance_scores: List[ContextRelevanceScore],
        context_type: ContextType
    ) -> List[ContextCluster]:
        """Create clusters based on functionality (simplified keyword matching)."""
        clusters = []
        
        # Define functionality keywords
        functionality_keywords = {
            'api': ['api', 'endpoint', 'route', 'controller', 'handler'],
            'database': ['db', 'database', 'model', 'schema', 'migration'],
            'auth': ['auth', 'login', 'password', 'token', 'session'],
            'utils': ['util', 'helper', 'common', 'shared', 'tool'],
            'config': ['config', 'setting', 'env', 'environment'],
            'test': ['test', 'spec', 'mock', 'fixture']
        }
        
        # Group files by functionality
        functionality_groups = defaultdict(list)
        
        for file_result in file_results:
            filename = file_result.file_name.lower()
            filepath = file_result.relative_path.lower()
            
            for functionality, keywords in functionality_keywords.items():
                if any(keyword in filename or keyword in filepath for keyword in keywords):
                    functionality_groups[functionality].append(file_result.file_path)
        
        # Create clusters for each functionality
        for functionality, files in functionality_groups.items():
            if len(files) >= 2:
                relevant_files = [
                    score for score in relevance_scores 
                    if score.file_path in files
                ]
                
                if relevant_files:
                    avg_score = sum(s.relevance_score for s in relevant_files) / len(relevant_files)
                    
                    central_files = [
                        s.file_path for s in sorted(relevant_files, 
                                                   key=lambda x: x.relevance_score, 
                                                   reverse=True)[:3]
                    ]
                    
                    cluster = ContextCluster(
                        cluster_id=f"func_{functionality}",
                        name=f"{functionality.title()} Components",
                        files=files,
                        central_files=central_files,
                        cluster_score=avg_score,
                        context_type=context_type,
                        description=f"Files related to {functionality} functionality",
                        metadata={'functionality': functionality, 'file_count': len(files)}
                    )
                    
                    clusters.append(cluster)
        
        return clusters
    
    def _generate_recommendations(
        self,
        context_type: ContextType,
        relevance_scores: List[ContextRelevanceScore],
        clusters: List[ContextCluster],
        high_impact_files: List[str],
        entry_points: List[str]
    ) -> List[str]:
        """Generate context-specific recommendations."""
        recommendations = []
        
        # Top files recommendation
        top_files = [s for s in relevance_scores if s.relevance_score > 0.7]
        if top_files:
            recommendations.append(
                f"Focus on {len(top_files)} high-relevance files for {context_type.value}"
            )
        
        # Entry points recommendation
        if entry_points:
            recommendations.append(
                f"Start analysis from {len(entry_points)} identified entry points"
            )
        
        # High impact files recommendation
        if high_impact_files:
            recommendations.append(
                f"Pay special attention to {len(high_impact_files)} high-impact files"
            )
        
        # Cluster recommendations
        top_clusters = [c for c in clusters if c.cluster_score > 0.6]
        if top_clusters:
            recommendations.append(
                f"Consider {len(top_clusters)} related file clusters for comprehensive analysis"
            )
        
        # Context-specific recommendations
        context_recommendations = {
            ContextType.DEBUGGING: [
                "Review error handling and logging files first",
                "Check test files for reproduction scenarios"
            ],
            ContextType.TESTING: [
                "Ensure test coverage for high-impact files",
                "Review existing test patterns and utilities"
            ],
            ContextType.REFACTORING: [
                "Identify tightly coupled components for refactoring",
                "Consider impact on dependent modules"
            ],
            ContextType.FEATURE_IMPLEMENTATION: [
                "Review similar existing features for patterns",
                "Check configuration and routing files"
            ]
        }
        
        recommendations.extend(context_recommendations.get(context_type, []))
        
        return recommendations
    
    def _calculate_optimization_metrics(
        self,
        relevance_scores: List[ContextRelevanceScore],
        clusters: List[ContextCluster],
        high_impact_files: List[str]
    ) -> Dict[str, float]:
        """Calculate optimization quality metrics."""
        if not relevance_scores:
            return {}
        
        scores = [s.relevance_score for s in relevance_scores]
        confidences = [s.confidence for s in relevance_scores]
        
        metrics = {
            'average_relevance': sum(scores) / len(scores),
            'max_relevance': max(scores),
            'relevance_std': math.sqrt(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)),
            'average_confidence': sum(confidences) / len(confidences),
            'high_relevance_ratio': len([s for s in scores if s > 0.7]) / len(scores),
            'cluster_count': len(clusters),
            'high_impact_ratio': len(high_impact_files) / len(relevance_scores) if relevance_scores else 0,
            'coverage_efficiency': len([s for s in scores if s > 0.5]) / len(scores)
        }
        
        return metrics
    
    def get_file_recommendations(
        self,
        context_type: ContextType,
        file_results: List[FileAnalysisResult],
        max_files: int = 20
    ) -> List[str]:
        """
        Get quick file recommendations without full optimization.
        
        Args:
            context_type: Type of context
            file_results: List of file analysis results
            max_files: Maximum number of files to recommend
            
        Returns:
            List of recommended file paths
        """
        # Simple scoring based on file type and name patterns
        scored_files = []
        
        for file_result in file_results:
            if not file_result.analysis_successful:
                continue
            
            score = self._calculate_base_relevance(file_result, context_type)
            
            # Boost score for certain patterns
            filename = file_result.file_name.lower()
            if 'main' in filename or 'index' in filename:
                score += 0.2
            if 'core' in filename or 'base' in filename:
                score += 0.1
            
            scored_files.append((file_result.file_path, score))
        
        # Sort and return top files
        scored_files.sort(key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in scored_files[:max_files]]
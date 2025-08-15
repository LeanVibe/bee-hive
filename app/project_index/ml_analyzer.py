"""
Machine Learning Analyzer for LeanVibe Agent Hive 2.0

Advanced ML-powered analysis engine for code embeddings, similarity search,
pattern recognition, and anomaly detection. Provides intelligent code understanding
through neural networks and statistical learning models.
"""

import asyncio
import json
import math
import pickle
import time
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from .models import FileAnalysisResult
from .context_optimizer import TaskType
from .utils import PathUtils, FileUtils

logger = structlog.get_logger()


class EmbeddingType(Enum):
    """Types of code embeddings."""
    TFIDF = "tfidf"
    COUNT = "count"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    HYBRID = "hybrid"


class PatternType(Enum):
    """Types of code patterns."""
    ARCHITECTURAL = "architectural"
    DESIGN = "design"
    ANTIPATTERN = "antipattern"
    IDIOM = "idiom"
    COMPLEXITY = "complexity"


@dataclass
class CodeEmbedding:
    """Code embedding with metadata."""
    file_path: str
    embedding_type: EmbeddingType
    vector: np.ndarray
    dimensions: int
    creation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternMatch:
    """Pattern matching result."""
    pattern_type: PatternType
    pattern_name: str
    confidence: float
    file_path: str
    evidence: List[str]
    similarity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result."""
    file_path: str
    anomaly_score: float
    is_anomaly: bool
    anomaly_reasons: List[str]
    similar_files: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterAnalysisResult:
    """Code clustering analysis result."""
    cluster_id: int
    cluster_label: str
    files: List[str]
    centroid: np.ndarray
    cohesion_score: float
    representative_files: List[str]
    cluster_characteristics: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLAnalyzer:
    """
    Advanced ML analyzer for code understanding and pattern recognition.
    
    Provides:
    - Code embedding generation using multiple techniques
    - Semantic similarity search and clustering
    - Pattern recognition and classification
    - Anomaly detection for unusual code patterns
    - Dimensionality reduction for visualization
    """
    
    def __init__(self, cache_embeddings: bool = True):
        """Initialize MLAnalyzer with ML models."""
        self.cache_embeddings = cache_embeddings
        
        # Embedding models
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.8,
            analyzer='word'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            analyzer='word'
        )
        
        # Dimensionality reduction
        self.pca_model = PCA(n_components=50, random_state=42)
        self.svd_model = TruncatedSVD(n_components=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Clustering models
        self.kmeans_model = KMeans(n_clusters=8, random_state=42, n_init=10)
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=3)
        
        # Anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        
        # Model training state
        self.models_trained = False
        self.embeddings_cache: Dict[str, CodeEmbedding] = {}
        self.pattern_library: Dict[str, Dict[str, Any]] = {}
        
        # Analysis statistics
        self.analysis_stats = {
            "embeddings_generated": 0,
            "patterns_detected": 0,
            "anomalies_found": 0,
            "similarity_searches": 0,
            "cache_hits": 0
        }
        
        # Initialize pattern library
        self._initialize_pattern_library()
    
    async def get_code_embedding(
        self,
        file_path: str,
        analysis_data: Dict[str, Any],
        embedding_type: EmbeddingType = EmbeddingType.HYBRID
    ) -> Optional[np.ndarray]:
        """
        Generate code embedding for a file.
        
        Args:
            file_path: Path to the file
            analysis_data: File analysis data
            embedding_type: Type of embedding to generate
            
        Returns:
            Numpy array representing the code embedding
        """
        try:
            # Check cache first
            cache_key = f"{file_path}_{embedding_type.value}"
            if self.cache_embeddings and cache_key in self.embeddings_cache:
                self.analysis_stats["cache_hits"] += 1
                return self.embeddings_cache[cache_key].vector
            
            # Extract code features
            code_features = self._extract_code_features(file_path, analysis_data)
            
            if not code_features:
                return None
            
            # Generate embedding based on type
            if embedding_type == EmbeddingType.TFIDF:
                embedding = await self._generate_tfidf_embedding(code_features)
            elif embedding_type == EmbeddingType.COUNT:
                embedding = await self._generate_count_embedding(code_features)
            elif embedding_type == EmbeddingType.SEMANTIC:
                embedding = await self._generate_semantic_embedding(code_features)
            elif embedding_type == EmbeddingType.STRUCTURAL:
                embedding = await self._generate_structural_embedding(analysis_data)
            elif embedding_type == EmbeddingType.HYBRID:
                embedding = await self._generate_hybrid_embedding(code_features, analysis_data)
            else:
                embedding = await self._generate_tfidf_embedding(code_features)
            
            if embedding is not None:
                # Cache the embedding
                if self.cache_embeddings:
                    self.embeddings_cache[cache_key] = CodeEmbedding(
                        file_path=file_path,
                        embedding_type=embedding_type,
                        vector=embedding,
                        dimensions=len(embedding),
                        creation_time=time.time(),
                        metadata={"features_count": len(code_features)}
                    )
                
                self.analysis_stats["embeddings_generated"] += 1
            
            return embedding
            
        except Exception as e:
            logger.warning("Code embedding generation failed",
                          file_path=file_path,
                          embedding_type=embedding_type.value,
                          error=str(e))
            return None
    
    async def get_task_embedding(self, task_description: str) -> Optional[np.ndarray]:
        """
        Generate embedding for task description.
        
        Args:
            task_description: Natural language task description
            
        Returns:
            Numpy array representing the task embedding
        """
        try:
            if not task_description.strip():
                return None
            
            # Preprocess task description
            processed_text = self._preprocess_task_text(task_description)
            
            # Generate TF-IDF embedding for task
            if self.models_trained:
                task_vector = self.tfidf_vectorizer.transform([processed_text])
                return task_vector.toarray().flatten()
            else:
                # Simple fallback - use basic word counting
                words = processed_text.split()
                # Create a simple bag of words representation
                word_counts = Counter(words)
                # Convert to vector (simple approach)
                return np.array(list(word_counts.values()), dtype=float)
            
        except Exception as e:
            logger.warning("Task embedding generation failed",
                          task_description=task_description[:100],
                          error=str(e))
            return None
    
    def calculate_embedding_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ('cosine', 'euclidean', 'dot_product')
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Ensure embeddings have the same shape
            if embedding1.shape != embedding2.shape:
                min_dim = min(len(embedding1), len(embedding2))
                embedding1 = embedding1[:min_dim]
                embedding2 = embedding2[:min_dim]
            
            if method == "cosine":
                similarity = cosine_similarity(
                    embedding1.reshape(1, -1),
                    embedding2.reshape(1, -1)
                )[0][0]
                # Convert to 0-1 range (cosine similarity is -1 to 1)
                return (similarity + 1) / 2
                
            elif method == "euclidean":
                distance = euclidean_distances(
                    embedding1.reshape(1, -1),
                    embedding2.reshape(1, -1)
                )[0][0]
                # Convert distance to similarity (inverse relationship)
                max_distance = np.sqrt(len(embedding1))  # Maximum possible distance
                return max(0.0, 1.0 - (distance / max_distance))
                
            elif method == "dot_product":
                # Normalize vectors
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                normalized_dot = np.dot(embedding1, embedding2) / (norm1 * norm2)
                return max(0.0, normalized_dot)
            
            else:
                # Default to cosine similarity
                return self.calculate_embedding_similarity(embedding1, embedding2, "cosine")
            
        except Exception as e:
            logger.warning("Embedding similarity calculation failed",
                          method=method,
                          error=str(e))
            return 0.0
    
    async def analyze_code_patterns(
        self,
        file_result: FileAnalysisResult,
        task_type: TaskType
    ) -> float:
        """
        Analyze code patterns and return relevance score.
        
        Args:
            file_result: File analysis result
            task_type: Type of development task
            
        Returns:
            Pattern-based relevance score
        """
        try:
            pattern_score = 0.0
            
            # Analyze different pattern types
            architectural_score = await self._analyze_architectural_patterns(file_result)
            design_score = await self._analyze_design_patterns(file_result)
            complexity_score = await self._analyze_complexity_patterns(file_result)
            
            # Weight patterns based on task type
            if task_type == TaskType.REFACTORING:
                pattern_score = (
                    architectural_score * 0.4 +
                    design_score * 0.3 +
                    complexity_score * 0.3
                )
            elif task_type == TaskType.FEATURE:
                pattern_score = (
                    architectural_score * 0.3 +
                    design_score * 0.5 +
                    complexity_score * 0.2
                )
            elif task_type == TaskType.ANALYSIS:
                pattern_score = (
                    architectural_score * 0.35 +
                    design_score * 0.35 +
                    complexity_score * 0.3
                )
            else:
                pattern_score = (
                    architectural_score * 0.33 +
                    design_score * 0.33 +
                    complexity_score * 0.34
                )
            
            self.analysis_stats["patterns_detected"] += 1
            
            return min(1.0, pattern_score)
            
        except Exception as e:
            logger.warning("Pattern analysis failed",
                          file_path=file_result.file_path,
                          error=str(e))
            return 0.5
    
    async def detect_anomalies(self, file_result: FileAnalysisResult) -> float:
        """
        Detect anomalies in code structure and patterns.
        
        Args:
            file_result: File analysis result
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        try:
            # Extract numerical features for anomaly detection
            features = self._extract_numerical_features(file_result)
            
            if not features:
                return 0.0
            
            # Convert to numpy array
            feature_vector = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is fitted
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform(feature_vector)
            
            # Use isolation forest for anomaly detection
            if hasattr(self.isolation_forest, 'decision_function'):
                anomaly_score = self.isolation_forest.decision_function(feature_vector)[0]
                # Convert to 0-1 range (lower = more anomalous)
                normalized_score = max(0.0, -anomaly_score / 2.0)
            else:
                # Fallback: use statistical measures
                normalized_score = self._statistical_anomaly_score(features)
            
            self.analysis_stats["anomalies_found"] += 1
            
            return min(1.0, normalized_score)
            
        except Exception as e:
            logger.warning("Anomaly detection failed",
                          file_path=file_result.file_path,
                          error=str(e))
            return 0.0
    
    async def find_similar_files(
        self,
        target_file: str,
        file_results: List[FileAnalysisResult],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find files similar to the target file.
        
        Args:
            target_file: Path to target file
            file_results: List of all file analysis results
            top_k: Number of similar files to return
            
        Returns:
            List of (file_path, similarity_score) tuples
        """
        try:
            self.analysis_stats["similarity_searches"] += 1
            
            # Get target file's analysis data
            target_analysis = None
            for result in file_results:
                if result.file_path == target_file:
                    target_analysis = result
                    break
            
            if not target_analysis:
                return []
            
            # Generate embedding for target file
            target_embedding = await self.get_code_embedding(
                target_file, target_analysis.analysis_data
            )
            
            if target_embedding is None:
                return []
            
            # Calculate similarities with other files
            similarities = []
            
            for result in file_results:
                if result.file_path == target_file:
                    continue
                
                # Generate embedding for comparison file
                comparison_embedding = await self.get_code_embedding(
                    result.file_path, result.analysis_data
                )
                
                if comparison_embedding is not None:
                    similarity = self.calculate_embedding_similarity(
                        target_embedding, comparison_embedding
                    )
                    similarities.append((result.file_path, similarity))
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.warning("Similar files search failed",
                          target_file=target_file,
                          error=str(e))
            return []
    
    async def cluster_files(
        self,
        file_results: List[FileAnalysisResult],
        n_clusters: Optional[int] = None
    ) -> List[ClusterAnalysisResult]:
        """
        Cluster files based on code similarity.
        
        Args:
            file_results: List of file analysis results
            n_clusters: Number of clusters (auto-detected if None)
            
        Returns:
            List of cluster analysis results
        """
        try:
            if len(file_results) < 3:
                return []
            
            # Generate embeddings for all files
            embeddings = []
            file_paths = []
            
            for result in file_results:
                embedding = await self.get_code_embedding(
                    result.file_path, result.analysis_data
                )
                
                if embedding is not None:
                    embeddings.append(embedding)
                    file_paths.append(result.file_path)
            
            if len(embeddings) < 3:
                return []
            
            # Convert to numpy array
            embedding_matrix = np.array(embeddings)
            
            # Scale embeddings
            scaled_embeddings = self.scaler.fit_transform(embedding_matrix)
            
            # Determine number of clusters
            if n_clusters is None:
                n_clusters = min(8, max(2, len(embeddings) // 3))
            
            # Perform clustering
            self.kmeans_model.n_clusters = n_clusters
            cluster_labels = self.kmeans_model.fit_predict(scaled_embeddings)
            
            # Analyze clusters
            clusters = []
            
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                cluster_files = [file_paths[i] for i in cluster_indices]
                cluster_embeddings = scaled_embeddings[cluster_indices]
                
                # Calculate cluster centroid
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate cohesion (average distance to centroid)
                distances = [
                    np.linalg.norm(embedding - centroid)
                    for embedding in cluster_embeddings
                ]
                cohesion = 1.0 / (1.0 + np.mean(distances))  # Higher = more cohesive
                
                # Find representative files (closest to centroid)
                centroid_distances = [
                    (i, np.linalg.norm(cluster_embeddings[i] - centroid))
                    for i in range(len(cluster_embeddings))
                ]
                centroid_distances.sort(key=lambda x: x[1])
                
                representative_indices = [
                    cluster_indices[centroid_distances[i][0]]
                    for i in range(min(3, len(centroid_distances)))
                ]
                representative_files = [file_paths[i] for i in representative_indices]
                
                # Generate cluster characteristics
                characteristics = self._analyze_cluster_characteristics(
                    cluster_files, file_results
                )
                
                cluster_result = ClusterAnalysisResult(
                    cluster_id=cluster_id,
                    cluster_label=f"Cluster {cluster_id + 1}",
                    files=cluster_files,
                    centroid=centroid,
                    cohesion_score=cohesion,
                    representative_files=representative_files,
                    cluster_characteristics=characteristics,
                    metadata={
                        "file_count": len(cluster_files),
                        "avg_distance_to_centroid": np.mean(distances)
                    }
                )
                
                clusters.append(cluster_result)
            
            # Sort clusters by size (descending)
            clusters.sort(key=lambda x: len(x.files), reverse=True)
            
            return clusters
            
        except Exception as e:
            logger.warning("File clustering failed",
                          file_count=len(file_results),
                          error=str(e))
            return []
    
    async def train_models(self, file_results: List[FileAnalysisResult]):
        """
        Train ML models on the provided file analysis results.
        
        Args:
            file_results: List of file analysis results for training
        """
        try:
            logger.info("Training ML models", file_count=len(file_results))
            
            # Extract text features for all files
            text_features = []
            numerical_features = []
            
            for result in file_results:
                # Text features for TF-IDF
                code_text = self._extract_code_features(result.file_path, result.analysis_data)
                if code_text:
                    text_features.append(code_text)
                
                # Numerical features for anomaly detection
                num_features = self._extract_numerical_features(result)
                if num_features:
                    numerical_features.append(num_features)
            
            # Train text vectorizers
            if text_features:
                self.tfidf_vectorizer.fit(text_features)
                self.count_vectorizer.fit(text_features)
                logger.info("Trained text vectorizers", vocabulary_size=len(self.tfidf_vectorizer.vocabulary_))
            
            # Train numerical models
            if numerical_features:
                feature_matrix = np.array(numerical_features)
                
                # Train scaler
                self.scaler.fit(feature_matrix)
                
                # Train dimensionality reduction
                if feature_matrix.shape[1] > 10:
                    self.pca_model.fit(feature_matrix)
                    self.svd_model.fit(feature_matrix)
                
                # Train anomaly detection
                self.isolation_forest.fit(feature_matrix)
                
                logger.info("Trained numerical models", 
                           feature_count=feature_matrix.shape[1],
                           sample_count=feature_matrix.shape[0])
            
            self.models_trained = True
            logger.info("ML model training completed")
            
        except Exception as e:
            logger.error("ML model training failed", error=str(e))
            raise
    
    # Helper methods for feature extraction and analysis
    
    def _extract_code_features(self, file_path: str, analysis_data: Dict[str, Any]) -> str:
        """Extract textual features from code analysis data."""
        features = []
        
        # File name and path components
        path_parts = file_path.replace('/', ' ').replace('_', ' ').replace('-', ' ')
        features.append(path_parts)
        
        if not analysis_data:
            return ' '.join(features)
        
        # Function names
        functions = analysis_data.get('functions', [])
        for func in functions:
            if isinstance(func, dict):
                name = func.get('name', '')
                docstring = func.get('docstring', '')
                features.append(name.replace('_', ' '))
                if docstring:
                    features.append(docstring)
            elif isinstance(func, str):
                features.append(func.replace('_', ' '))
        
        # Class names
        classes = analysis_data.get('classes', [])
        for cls in classes:
            if isinstance(cls, dict):
                name = cls.get('name', '')
                docstring = cls.get('docstring', '')
                features.append(name.replace('_', ' '))
                if docstring:
                    features.append(docstring)
            elif isinstance(cls, str):
                features.append(cls.replace('_', ' '))
        
        # Import statements
        imports = analysis_data.get('imports', [])
        for imp in imports:
            if isinstance(imp, str):
                features.append(imp.replace('.', ' '))
        
        # Comments
        comments = analysis_data.get('comments', [])
        for comment in comments[:5]:  # Limit to first 5 comments
            if isinstance(comment, str) and len(comment) > 10:
                features.append(comment[:200])  # Limit comment length
        
        return ' '.join(features)
    
    def _extract_numerical_features(self, file_result: FileAnalysisResult) -> List[float]:
        """Extract numerical features for ML analysis."""
        features = []
        
        # Basic file metrics
        features.append(float(file_result.file_size or 0))
        features.append(float(file_result.line_count or 0))
        
        if not file_result.analysis_data:
            # Pad with zeros if no analysis data
            return features + [0.0] * 18
        
        # Code structure metrics
        functions = file_result.analysis_data.get('functions', [])
        classes = file_result.analysis_data.get('classes', [])
        imports = file_result.analysis_data.get('imports', [])
        comments = file_result.analysis_data.get('comments', [])
        
        features.extend([
            float(len(functions)),
            float(len(classes)),
            float(len(imports)),
            float(len(comments))
        ])
        
        # Function complexity metrics
        if functions:
            func_names = [
                func.get('name', '') if isinstance(func, dict) else str(func)
                for func in functions
            ]
            avg_func_name_length = sum(len(name) for name in func_names) / len(func_names)
            max_func_name_length = max(len(name) for name in func_names)
            
            features.extend([
                avg_func_name_length,
                float(max_func_name_length)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Class complexity metrics
        if classes:
            class_names = [
                cls.get('name', '') if isinstance(cls, dict) else str(cls)
                for cls in classes
            ]
            avg_class_name_length = sum(len(name) for name in class_names) / len(class_names)
            max_class_name_length = max(len(name) for name in class_names)
            
            features.extend([
                avg_class_name_length,
                float(max_class_name_length)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Import complexity
        if imports:
            import_names = [str(imp) for imp in imports]
            avg_import_length = sum(len(name) for name in import_names) / len(import_names)
            unique_modules = len(set(imp.split('.')[0] for imp in import_names if '.' in str(imp)))
            
            features.extend([
                avg_import_length,
                float(unique_modules)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Code ratios
        total_elements = len(functions) + len(classes)
        if total_elements > 0:
            function_ratio = len(functions) / total_elements
            class_ratio = len(classes) / total_elements
        else:
            function_ratio = 0.0
            class_ratio = 0.0
        
        features.extend([function_ratio, class_ratio])
        
        # Documentation ratio
        if total_elements > 0:
            documented_elements = 0
            
            # Count documented functions
            for func in functions:
                if isinstance(func, dict) and func.get('docstring'):
                    documented_elements += 1
            
            # Count documented classes
            for cls in classes:
                if isinstance(cls, dict) and cls.get('docstring'):
                    documented_elements += 1
            
            doc_ratio = documented_elements / total_elements
        else:
            doc_ratio = 0.0
        
        features.append(doc_ratio)
        
        # Comments ratio
        if file_result.line_count and file_result.line_count > 0:
            comment_ratio = len(comments) / file_result.line_count
        else:
            comment_ratio = 0.0
        
        features.append(comment_ratio)
        
        # Complexity indicators
        avg_elements_per_line = total_elements / file_result.line_count if file_result.line_count else 0
        import_density = len(imports) / file_result.line_count if file_result.line_count else 0
        
        features.extend([avg_elements_per_line, import_density])
        
        return features
    
    def _preprocess_task_text(self, task_description: str) -> str:
        """Preprocess task description for embedding generation."""
        # Convert to lowercase
        text = task_description.lower()
        
        # Remove special characters but keep spaces and alphanumeric
        import re
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    async def _generate_tfidf_embedding(self, code_features: str) -> Optional[np.ndarray]:
        """Generate TF-IDF embedding for code features."""
        try:
            if self.models_trained:
                tfidf_vector = self.tfidf_vectorizer.transform([code_features])
                return tfidf_vector.toarray().flatten()
            else:
                # Simple fallback
                words = code_features.split()
                word_counts = Counter(words)
                return np.array(list(word_counts.values()), dtype=float)
        except:
            return None
    
    async def _generate_count_embedding(self, code_features: str) -> Optional[np.ndarray]:
        """Generate count-based embedding for code features."""
        try:
            if self.models_trained:
                count_vector = self.count_vectorizer.transform([code_features])
                return count_vector.toarray().flatten()
            else:
                # Simple fallback
                words = code_features.split()
                word_counts = Counter(words)
                return np.array(list(word_counts.values()), dtype=float)
        except:
            return None
    
    async def _generate_semantic_embedding(self, code_features: str) -> Optional[np.ndarray]:
        """Generate semantic embedding using advanced NLP techniques."""
        try:
            # This would integrate with actual semantic models like word2vec, BERT, etc.
            # For now, use enhanced TF-IDF with n-grams
            tfidf_embedding = await self._generate_tfidf_embedding(code_features)
            
            if tfidf_embedding is not None and len(tfidf_embedding) > 50:
                # Apply dimensionality reduction for semantic compression
                if hasattr(self.svd_model, 'components_'):
                    semantic_embedding = self.svd_model.transform(tfidf_embedding.reshape(1, -1))
                    return semantic_embedding.flatten()
            
            return tfidf_embedding
        except:
            return None
    
    async def _generate_structural_embedding(self, analysis_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Generate structural embedding based on code structure."""
        try:
            if not analysis_data:
                return None
            
            # Extract structural features
            structural_features = []
            
            # Function and class counts
            functions = analysis_data.get('functions', [])
            classes = analysis_data.get('classes', [])
            imports = analysis_data.get('imports', [])
            
            structural_features.extend([
                len(functions),
                len(classes),
                len(imports)
            ])
            
            # Nesting levels (simplified)
            max_nesting = 0
            for func in functions:
                if isinstance(func, dict) and 'complexity' in func:
                    max_nesting = max(max_nesting, func.get('complexity', 0))
            
            structural_features.append(max_nesting)
            
            # Import categories
            import_categories = defaultdict(int)
            for imp in imports:
                imp_str = str(imp)
                if 'std' in imp_str or 'builtin' in imp_str:
                    import_categories['standard'] += 1
                elif '.' in imp_str:
                    import_categories['external'] += 1
                else:
                    import_categories['local'] += 1
            
            structural_features.extend([
                import_categories['standard'],
                import_categories['external'],
                import_categories['local']
            ])
            
            # Pad to fixed size
            while len(structural_features) < 20:
                structural_features.append(0.0)
            
            return np.array(structural_features[:20], dtype=float)
            
        except:
            return None
    
    async def _generate_hybrid_embedding(
        self,
        code_features: str,
        analysis_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Generate hybrid embedding combining multiple approaches."""
        try:
            embeddings = []
            
            # Get different types of embeddings
            tfidf_emb = await self._generate_tfidf_embedding(code_features)
            structural_emb = await self._generate_structural_embedding(analysis_data)
            
            # Combine embeddings
            if tfidf_emb is not None:
                # Reduce dimensionality of TF-IDF if too large
                if len(tfidf_emb) > 100:
                    tfidf_emb = tfidf_emb[:100]
                embeddings.append(tfidf_emb)
            
            if structural_emb is not None:
                embeddings.append(structural_emb)
            
            if embeddings:
                # Concatenate all embeddings
                hybrid_embedding = np.concatenate(embeddings)
                
                # Normalize
                norm = np.linalg.norm(hybrid_embedding)
                if norm > 0:
                    hybrid_embedding = hybrid_embedding / norm
                
                return hybrid_embedding
            
            return None
            
        except:
            return None
    
    async def _analyze_architectural_patterns(self, file_result: FileAnalysisResult) -> float:
        """Analyze architectural patterns in the file."""
        score = 0.0
        
        # Check for MVC patterns
        file_path_lower = file_result.file_path.lower()
        if any(pattern in file_path_lower for pattern in ['model', 'view', 'controller']):
            score += 0.3
        
        # Check for service patterns
        if any(pattern in file_path_lower for pattern in ['service', 'repository', 'dao']):
            score += 0.2
        
        # Check for factory patterns
        if file_result.analysis_data:
            functions = file_result.analysis_data.get('functions', [])
            for func in functions:
                if isinstance(func, dict):
                    func_name = func.get('name', '').lower()
                    if 'create' in func_name or 'factory' in func_name or 'builder' in func_name:
                        score += 0.1
                        break
        
        return min(1.0, score)
    
    async def _analyze_design_patterns(self, file_result: FileAnalysisResult) -> float:
        """Analyze design patterns in the file."""
        score = 0.0
        
        if not file_result.analysis_data:
            return score
        
        classes = file_result.analysis_data.get('classes', [])
        functions = file_result.analysis_data.get('functions', [])
        
        # Singleton pattern
        for cls in classes:
            if isinstance(cls, dict):
                class_name = cls.get('name', '').lower()
                if 'singleton' in class_name:
                    score += 0.2
                    break
        
        # Observer pattern
        for func in functions:
            if isinstance(func, dict):
                func_name = func.get('name', '').lower()
                if any(pattern in func_name for pattern in ['notify', 'subscribe', 'observe']):
                    score += 0.1
                    break
        
        # Strategy pattern
        if len(classes) > 1:
            # Multiple classes might indicate strategy pattern
            score += 0.1
        
        return min(1.0, score)
    
    async def _analyze_complexity_patterns(self, file_result: FileAnalysisResult) -> float:
        """Analyze complexity patterns in the file."""
        score = 0.0
        
        # File size complexity
        if file_result.file_size:
            if file_result.file_size > 10000:  # Large file
                score += 0.3
            elif file_result.file_size > 5000:  # Medium file
                score += 0.2
        
        # Function count complexity
        if file_result.analysis_data:
            functions = file_result.analysis_data.get('functions', [])
            if len(functions) > 20:
                score += 0.2
            elif len(functions) > 10:
                score += 0.1
        
        # Import complexity
        if file_result.analysis_data:
            imports = file_result.analysis_data.get('imports', [])
            if len(imports) > 15:
                score += 0.2
            elif len(imports) > 10:
                score += 0.1
        
        return min(1.0, score)
    
    def _statistical_anomaly_score(self, features: List[float]) -> float:
        """Calculate anomaly score using statistical methods."""
        if not features:
            return 0.0
        
        # Use z-score based anomaly detection
        features_array = np.array(features)
        
        # Calculate z-scores
        mean_val = np.mean(features_array)
        std_val = np.std(features_array)
        
        if std_val == 0:
            return 0.0
        
        z_scores = np.abs((features_array - mean_val) / std_val)
        
        # Calculate anomaly score based on extreme z-scores
        extreme_z_scores = z_scores[z_scores > 2.0]  # Values more than 2 std devs away
        
        if len(extreme_z_scores) > 0:
            anomaly_score = min(1.0, len(extreme_z_scores) / len(features_array))
        else:
            anomaly_score = 0.0
        
        return anomaly_score
    
    def _analyze_cluster_characteristics(
        self,
        cluster_files: List[str],
        file_results: List[FileAnalysisResult]
    ) -> List[str]:
        """Analyze characteristics of a file cluster."""
        characteristics = []
        
        # File type analysis
        file_types = defaultdict(int)
        for file_path in cluster_files:
            extension = Path(file_path).suffix.lower()
            file_types[extension] += 1
        
        if file_types:
            dominant_type = max(file_types, key=file_types.get)
            characteristics.append(f"Primarily {dominant_type} files")
        
        # Directory analysis
        directories = defaultdict(int)
        for file_path in cluster_files:
            directory = '/'.join(Path(file_path).parts[:-1])
            directories[directory] += 1
        
        if len(directories) == 1:
            characteristics.append("Files from single directory")
        elif len(directories) <= 3:
            characteristics.append("Files from few directories")
        else:
            characteristics.append("Files from multiple directories")
        
        # Size analysis
        file_analysis_map = {result.file_path: result for result in file_results}
        sizes = []
        
        for file_path in cluster_files:
            if file_path in file_analysis_map:
                result = file_analysis_map[file_path]
                if result.file_size:
                    sizes.append(result.file_size)
        
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            if avg_size > 10000:
                characteristics.append("Large files")
            elif avg_size < 1000:
                characteristics.append("Small files")
            else:
                characteristics.append("Medium-sized files")
        
        return characteristics
    
    def _initialize_pattern_library(self):
        """Initialize the pattern library with common code patterns."""
        self.pattern_library = {
            "mvc": {
                "keywords": ["model", "view", "controller", "mvc"],
                "file_patterns": ["*model*", "*view*", "*controller*"],
                "description": "Model-View-Controller architectural pattern"
            },
            "singleton": {
                "keywords": ["singleton", "instance", "getInstance"],
                "class_patterns": ["*Singleton*", "*Instance*"],
                "description": "Singleton design pattern"
            },
            "factory": {
                "keywords": ["factory", "create", "builder", "make"],
                "function_patterns": ["create*", "make*", "*Factory*", "*Builder*"],
                "description": "Factory design pattern"
            },
            "observer": {
                "keywords": ["observer", "notify", "subscribe", "listener"],
                "function_patterns": ["notify*", "subscribe*", "add*Listener*"],
                "description": "Observer design pattern"
            },
            "repository": {
                "keywords": ["repository", "dao", "data", "persistence"],
                "file_patterns": ["*Repository*", "*Dao*", "*Data*"],
                "description": "Repository pattern for data access"
            }
        }
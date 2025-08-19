"""
Advanced Technical Debt Detector for LeanVibe Agent Hive 2.0

ML-powered debt detection using existing infrastructure for sophisticated 
pattern recognition, anomaly detection, and intelligent debt classification.
"""

import asyncio
import ast
import hashlib
import json
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
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

from .debt_analyzer import TechnicalDebtAnalyzer, DebtItem, DebtCategory, DebtSeverity, DebtAnalysisResult
from .ml_analyzer import MLAnalyzer, EmbeddingType, PatternType, PatternMatch, AnomalyDetectionResult
from .historical_analyzer import HistoricalAnalyzer
from .models import AnalysisConfiguration, FileAnalysisResult
from ..models.project_index import FileEntry, ProjectIndex

logger = structlog.get_logger()


class DebtPatternType(Enum):
    """Types of advanced debt patterns."""
    ARCHITECTURAL = "architectural"
    DESIGN_ANTIPATTERN = "design_antipattern"
    PERFORMANCE_HOTSPOT = "performance_hotspot"
    SECURITY_VULNERABILITY = "security_vulnerability"
    COUPLING_VIOLATION = "coupling_violation"
    COHESION_DEGRADATION = "cohesion_degradation"
    ABSTRACTION_LEAK = "abstraction_leak"
    TEMPORAL_COUPLING = "temporal_coupling"


@dataclass
class AdvancedDebtPattern:
    """Advanced debt pattern with ML-powered detection."""
    pattern_type: DebtPatternType
    pattern_name: str
    confidence: float
    files_involved: List[str]
    evidence: Dict[str, Any]
    severity_score: float
    remediation_complexity: int  # 1-10 scale
    architectural_impact: float  # 0-1 scale
    technical_description: str
    business_impact: str
    remediation_strategy: str
    similar_patterns: List[str] = field(default_factory=list)


@dataclass
class DebtCluster:
    """Cluster of related debt items."""
    cluster_id: str
    debt_items: List[DebtItem]
    cluster_center: np.ndarray
    cohesion_score: float
    dominant_categories: List[DebtCategory]
    suggested_refactoring: str
    effort_estimate: int


@dataclass
class DebtTrend:
    """Trend analysis of debt over time."""
    trend_direction: str  # "increasing", "decreasing", "stable", "volatile"
    velocity: float  # rate of change
    projected_debt_in_30_days: float
    risk_level: str
    trend_causes: List[str]
    intervention_recommendations: List[str]


class AdvancedDebtDetector:
    """
    ML-powered advanced debt detection system.
    
    Uses machine learning algorithms to identify complex debt patterns,
    architectural issues, and performance problems that traditional
    static analysis might miss.
    """
    
    def __init__(self, 
                 base_analyzer: TechnicalDebtAnalyzer,
                 ml_analyzer: MLAnalyzer,
                 historical_analyzer: HistoricalAnalyzer):
        """Initialize advanced debt detector with existing analyzers."""
        self.base_analyzer = base_analyzer
        self.ml_analyzer = ml_analyzer
        self.historical_analyzer = historical_analyzer
        
        # ML models for debt detection
        self.debt_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.debt_clusterer = DBSCAN(
            eps=0.3,
            min_samples=2
        )
        
        # Pattern recognition models
        self._trained = False
        self._debt_embeddings = None
        self._pattern_templates = self._initialize_pattern_templates()
    
    def _initialize_pattern_templates(self) -> Dict[DebtPatternType, Dict[str, Any]]:
        """Initialize templates for architectural and design pattern detection."""
        return {
            DebtPatternType.ARCHITECTURAL: {
                'god_class': {
                    'indicators': ['many_responsibilities', 'large_size', 'high_coupling'],
                    'thresholds': {'methods': 20, 'lines': 500, 'dependencies': 15},
                    'description': 'Class that knows or does too much',
                    'remediation': 'Extract responsibilities into separate classes'
                },
                'circular_dependency': {
                    'indicators': ['cyclic_imports', 'mutual_references'],
                    'description': 'Circular dependencies between modules',
                    'remediation': 'Introduce dependency inversion or mediator pattern'
                },
                'spaghetti_code': {
                    'indicators': ['high_complexity', 'poor_structure', 'tangled_control_flow'],
                    'description': 'Code with complex and tangled control structure',
                    'remediation': 'Refactor into smaller, well-structured functions'
                }
            },
            DebtPatternType.DESIGN_ANTIPATTERN: {
                'singleton_abuse': {
                    'indicators': ['multiple_singletons', 'global_state'],
                    'description': 'Overuse of singleton pattern creating global state',
                    'remediation': 'Use dependency injection and reduce global state'
                },
                'anemic_model': {
                    'indicators': ['data_only_classes', 'procedural_logic'],
                    'description': 'Domain objects with no behavior, only data',
                    'remediation': 'Move behavior into domain objects'
                },
                'feature_envy': {
                    'indicators': ['excessive_external_calls', 'method_misplacement'],
                    'description': 'Method that seems more interested in other classes',
                    'remediation': 'Move method to the class it envies'
                }
            },
            DebtPatternType.PERFORMANCE_HOTSPOT: {
                'n_plus_one_query': {
                    'indicators': ['loop_with_queries', 'repeated_db_access'],
                    'description': 'Database query executed in loop causing performance issues',
                    'remediation': 'Use batch queries or eager loading'
                },
                'memory_leak': {
                    'indicators': ['unreleased_resources', 'growing_collections'],
                    'description': 'Memory that is not properly released',
                    'remediation': 'Implement proper resource cleanup and lifecycle management'
                },
                'inefficient_algorithm': {
                    'indicators': ['nested_loops', 'exponential_complexity'],
                    'description': 'Algorithm with poor time complexity',
                    'remediation': 'Replace with more efficient algorithm or data structure'
                }
            },
            DebtPatternType.SECURITY_VULNERABILITY: {
                'injection_risk': {
                    'indicators': ['unsanitized_input', 'dynamic_queries'],
                    'description': 'Potential for SQL injection or code injection',
                    'remediation': 'Use parameterized queries and input validation'
                },
                'hardcoded_secrets': {
                    'indicators': ['password_literals', 'api_key_strings'],
                    'description': 'Sensitive information hardcoded in source',
                    'remediation': 'Use environment variables or secure secret management'
                },
                'insecure_defaults': {
                    'indicators': ['weak_encryption', 'disabled_security'],
                    'description': 'Security features disabled or using weak defaults',
                    'remediation': 'Enable security features with strong defaults'
                }
            }
        }
    
    async def analyze_advanced_debt_patterns(
        self,
        project: ProjectIndex,
        base_analysis: DebtAnalysisResult
    ) -> List[AdvancedDebtPattern]:
        """
        Analyze project for advanced debt patterns using ML techniques.
        
        Args:
            project: Project to analyze
            base_analysis: Results from basic debt analysis
            
        Returns:
            List of advanced debt patterns found
        """
        logger.info("Starting advanced debt pattern analysis", project_id=str(project.id))
        
        patterns = []
        
        # Prepare file embeddings for ML analysis
        file_embeddings = await self._generate_file_embeddings(project.file_entries)
        
        # Detect architectural patterns
        architectural_patterns = await self._detect_architectural_patterns(
            project, base_analysis, file_embeddings
        )
        patterns.extend(architectural_patterns)
        
        # Detect design antipatterns
        antipatterns = await self._detect_design_antipatterns(
            project, base_analysis, file_embeddings
        )
        patterns.extend(antipatterns)
        
        # Detect performance hotspots
        performance_issues = await self._detect_performance_hotspots(
            project, base_analysis
        )
        patterns.extend(performance_issues)
        
        # Detect security vulnerabilities
        security_issues = await self._detect_security_vulnerabilities(
            project, base_analysis
        )
        patterns.extend(security_issues)
        
        # Detect coupling and cohesion issues
        coupling_issues = await self._detect_coupling_cohesion_issues(
            project, file_embeddings
        )
        patterns.extend(coupling_issues)
        
        logger.info(
            "Advanced debt pattern analysis completed",
            project_id=str(project.id),
            patterns_found=len(patterns)
        )
        
        return patterns
    
    async def cluster_debt_items(
        self,
        debt_items: List[DebtItem],
        project: ProjectIndex
    ) -> List[DebtCluster]:
        """
        Cluster related debt items for coordinated remediation.
        
        Args:
            debt_items: Debt items to cluster
            project: Project context
            
        Returns:
            List of debt clusters with remediation suggestions
        """
        if len(debt_items) < 2:
            return []
        
        # Create feature vectors from debt items
        features = await self._create_debt_feature_vectors(debt_items)
        
        if features is None or len(features) == 0:
            return []
        
        # Perform clustering
        cluster_labels = self.debt_clusterer.fit_predict(features)
        
        # Group debt items by cluster
        clusters = defaultdict(list)
        for item, label in zip(debt_items, cluster_labels):
            if label != -1:  # -1 indicates noise in DBSCAN
                clusters[label].append(item)
        
        # Create cluster objects with analysis
        debt_clusters = []
        for cluster_id, cluster_items in clusters.items():
            if len(cluster_items) >= 2:  # Only include meaningful clusters
                cluster = await self._analyze_debt_cluster(
                    str(cluster_id), 
                    cluster_items, 
                    features[cluster_labels == cluster_id]
                )
                debt_clusters.append(cluster)
        
        return debt_clusters
    
    async def analyze_debt_trends(
        self,
        project: ProjectIndex,
        lookback_days: int = 30
    ) -> DebtTrend:
        """
        Analyze debt trends over time using historical data.
        
        Args:
            project: Project to analyze
            lookback_days: Number of days to look back
            
        Returns:
            Debt trend analysis with predictions
        """
        # Get historical debt snapshots
        historical_data = await self.historical_analyzer.analyze_debt_evolution(
            str(project.id)
        )
        
        if len(historical_data) < 2:
            return DebtTrend(
                trend_direction="insufficient_data",
                velocity=0.0,
                projected_debt_in_30_days=0.0,
                risk_level="unknown",
                trend_causes=[],
                intervention_recommendations=["Collect more historical data"]
            )
        
        # Extract debt scores over time
        timestamps = [datetime.fromisoformat(entry['date']) if isinstance(entry['date'], str) else entry['date'] for entry in historical_data]
        debt_scores = [entry['debt_delta'] for entry in historical_data]
        
        # Calculate trend
        trend_direction, velocity = self._calculate_trend(timestamps, debt_scores)
        
        # Project future debt
        current_debt = debt_scores[-1] if debt_scores else 0.0
        projected_debt = current_debt + (velocity * 30)  # 30 days projection
        
        # Assess risk level
        risk_level = self._assess_trend_risk(velocity, projected_debt)
        
        # Identify trend causes
        trend_causes = self._identify_trend_causes(historical_data)
        
        # Generate intervention recommendations
        recommendations = self._generate_intervention_recommendations(
            trend_direction, velocity, risk_level, trend_causes
        )
        
        return DebtTrend(
            trend_direction=trend_direction,
            velocity=velocity,
            projected_debt_in_30_days=projected_debt,
            risk_level=risk_level,
            trend_causes=trend_causes,
            intervention_recommendations=recommendations
        )
    
    async def detect_anomalies(
        self,
        file_entries: List[FileEntry],
        base_analysis: DebtAnalysisResult
    ) -> List[AnomalyDetectionResult]:
        """
        Detect anomalous files that might indicate systemic issues.
        
        Args:
            file_entries: Files to analyze
            base_analysis: Base debt analysis results
            
        Returns:
            List of detected anomalies
        """
        if len(file_entries) < 10:  # Need sufficient data for anomaly detection
            return []
        
        # Create feature matrix for anomaly detection
        features = await self._create_anomaly_feature_matrix(file_entries, base_analysis)
        
        if features is None:
            return []
        
        # Fit anomaly detector
        anomaly_scores = self.anomaly_detector.fit_predict(features)
        decision_scores = self.anomaly_detector.decision_function(features)
        
        # Identify anomalous files
        anomalies = []
        for i, (file_entry, is_anomaly, score) in enumerate(
            zip(file_entries, anomaly_scores, decision_scores)
        ):
            if is_anomaly == -1:  # Anomaly detected
                anomaly = AnomalyDetectionResult(
                    file_path=file_entry.file_path,
                    anomaly_score=abs(score),
                    is_anomaly=True,
                    anomaly_reasons=[
                        f"High debt anomaly score: {score:.3f}",
                        "File shows anomalous debt patterns"
                    ],
                    similar_files=[],  # Could be populated with similar anomalous files
                    metadata={
                        "severity": "high" if abs(score) > 0.5 else "medium",
                        "feature_vector": features[i].tolist(),
                        "file_metrics": await self._extract_file_metrics(file_entry),
                        "recommendations": self._generate_anomaly_recommendations(file_entry, score)
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _generate_file_embeddings(self, file_entries: List[FileEntry]) -> np.ndarray:
        """Generate embeddings for files using existing ML infrastructure."""
        embeddings = []
        
        for file_entry in file_entries:
            if file_entry.is_binary or not file_entry.file_path:
                continue
                
            try:
                # Use existing ML analyzer to generate embeddings
                embedding = await self.ml_analyzer.generate_embeddings(
                    [file_entry], EmbeddingType.STRUCTURAL
                )
                if embedding is not None and len(embedding) > 0:
                    embeddings.append(embedding[0])
            except Exception as e:
                logger.warning(
                    "Failed to generate embedding",
                    file_path=file_entry.file_path,
                    error=str(e)
                )
        
        return np.array(embeddings) if embeddings else np.array([])
    
    async def _detect_architectural_patterns(
        self,
        project: ProjectIndex,
        base_analysis: DebtAnalysisResult,
        file_embeddings: np.ndarray
    ) -> List[AdvancedDebtPattern]:
        """Detect architectural debt patterns."""
        patterns = []
        
        # God class detection using ML clustering
        god_classes = await self._detect_god_classes(project.file_entries, file_embeddings)
        patterns.extend(god_classes)
        
        # Circular dependency detection
        circular_deps = await self._detect_circular_dependencies(project)
        patterns.extend(circular_deps)
        
        # Spaghetti code detection
        spaghetti_patterns = await self._detect_spaghetti_code(
            project.file_entries, base_analysis
        )
        patterns.extend(spaghetti_patterns)
        
        return patterns
    
    async def _detect_design_antipatterns(
        self,
        project: ProjectIndex,
        base_analysis: DebtAnalysisResult,
        file_embeddings: np.ndarray
    ) -> List[AdvancedDebtPattern]:
        """Detect design antipatterns using pattern matching."""
        patterns = []
        
        # Feature envy detection using dependency analysis
        feature_envy = await self._detect_feature_envy(project)
        patterns.extend(feature_envy)
        
        # Anemic model detection
        anemic_models = await self._detect_anemic_models(project.file_entries)
        patterns.extend(anemic_models)
        
        return patterns
    
    async def _detect_performance_hotspots(
        self,
        project: ProjectIndex,
        base_analysis: DebtAnalysisResult
    ) -> List[AdvancedDebtPattern]:
        """Detect performance-related debt patterns."""
        patterns = []
        
        # Detect N+1 query patterns
        n_plus_one = await self._detect_n_plus_one_patterns(project.file_entries)
        patterns.extend(n_plus_one)
        
        # Detect inefficient algorithms
        inefficient_algorithms = await self._detect_inefficient_algorithms(
            project.file_entries
        )
        patterns.extend(inefficient_algorithms)
        
        return patterns
    
    async def _detect_security_vulnerabilities(
        self,
        project: ProjectIndex,
        base_analysis: DebtAnalysisResult
    ) -> List[AdvancedDebtPattern]:
        """Detect security-related debt patterns."""
        patterns = []
        
        # Detect hardcoded secrets
        hardcoded_secrets = await self._detect_hardcoded_secrets(project.file_entries)
        patterns.extend(hardcoded_secrets)
        
        # Detect injection risks
        injection_risks = await self._detect_injection_risks(project.file_entries)
        patterns.extend(injection_risks)
        
        return patterns
    
    async def _detect_coupling_cohesion_issues(
        self,
        project: ProjectIndex,
        file_embeddings: np.ndarray
    ) -> List[AdvancedDebtPattern]:
        """Detect coupling and cohesion issues using ML analysis."""
        patterns = []
        
        if len(file_embeddings) == 0:
            return patterns
        
        # Use similarity analysis to detect tight coupling
        similarity_matrix = cosine_similarity(file_embeddings)
        
        # Find highly coupled file clusters
        high_coupling_threshold = 0.8
        coupled_files = []
        
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > high_coupling_threshold:
                    coupled_files.append((i, j, similarity_matrix[i][j]))
        
        # Create coupling violation patterns
        for file1_idx, file2_idx, coupling_score in coupled_files:
            if file1_idx < len(project.file_entries) and file2_idx < len(project.file_entries):
                file1 = project.file_entries[file1_idx]
                file2 = project.file_entries[file2_idx]
                
                pattern = AdvancedDebtPattern(
                    pattern_type=DebtPatternType.COUPLING_VIOLATION,
                    pattern_name="high_coupling",
                    confidence=coupling_score,
                    files_involved=[file1.file_path, file2.file_path],
                    evidence={
                        "coupling_score": coupling_score,
                        "similarity_analysis": "ML-based structural similarity"
                    },
                    severity_score=coupling_score,
                    remediation_complexity=6,
                    architectural_impact=0.7,
                    technical_description=f"High coupling detected between {file1.file_name} and {file2.file_name}",
                    business_impact="Reduces maintainability and increases change risk",
                    remediation_strategy="Reduce coupling through interface extraction or dependency injection"
                )
                patterns.append(pattern)
        
        return patterns
    
    # Placeholder implementations for specific pattern detection methods
    # These would be implemented with detailed AST analysis and pattern matching
    
    async def _detect_god_classes(self, file_entries: List[FileEntry], embeddings: np.ndarray) -> List[AdvancedDebtPattern]:
        """Detect god classes using size metrics and responsibility analysis."""
        patterns = []
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Simple god class detection based on size and method count
                lines = content.split('\n')
                class_lines = [line for line in lines if line.strip().startswith('class ')]
                
                if len(class_lines) > 0:  # File contains classes
                    method_count = len([line for line in lines if '    def ' in line])
                    file_length = len(lines)
                    
                    # God class heuristics
                    if method_count > 20 and file_length > 500:
                        pattern = AdvancedDebtPattern(
                            pattern_type=DebtPatternType.ARCHITECTURAL,
                            pattern_name="god_class",
                            confidence=min(1.0, (method_count * file_length) / 15000),
                            files_involved=[file_entry.file_path],
                            evidence={
                                "method_count": method_count,
                                "line_count": file_length,
                                "class_names": [line.split()[1].split('(')[0] for line in class_lines]
                            },
                            severity_score=0.8,
                            remediation_complexity=8,
                            architectural_impact=0.9,
                            technical_description=f"God class detected in {file_entry.file_name} with {method_count} methods and {file_length} lines",
                            business_impact="Violates single responsibility principle, hard to maintain and test",
                            remediation_strategy="Extract responsibilities into separate, focused classes"
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning("Error analyzing file for god class", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    async def _detect_circular_dependencies(self, project: ProjectIndex) -> List[AdvancedDebtPattern]:
        """Detect circular dependencies using dependency graph analysis."""
        # This would use the existing dependency relationships
        # For now, returning empty list as placeholder
        return []
    
    async def _detect_spaghetti_code(self, file_entries: List[FileEntry], base_analysis: DebtAnalysisResult) -> List[AdvancedDebtPattern]:
        """Detect spaghetti code patterns using complexity analysis."""
        patterns = []
        
        # Use existing debt items to identify spaghetti code
        high_complexity_items = [
            item for item in base_analysis.debt_items 
            if item.category == DebtCategory.COMPLEXITY and item.severity in [DebtSeverity.HIGH, DebtSeverity.CRITICAL]
        ]
        
        # Group by file to identify files with multiple complexity issues
        complexity_by_file = defaultdict(list)
        for item in high_complexity_items:
            complexity_by_file[item.file_id].append(item)
        
        # Files with multiple complexity issues likely have spaghetti code
        for file_id, items in complexity_by_file.items():
            if len(items) >= 3:  # Multiple complexity issues in same file
                file_entry = next((f for f in file_entries if str(f.id) == file_id), None)
                if file_entry:
                    pattern = AdvancedDebtPattern(
                        pattern_type=DebtPatternType.ARCHITECTURAL,
                        pattern_name="spaghetti_code",
                        confidence=min(1.0, len(items) / 5.0),
                        files_involved=[file_entry.file_path],
                        evidence={
                            "complexity_issues": len(items),
                            "issue_types": [item.debt_type for item in items]
                        },
                        severity_score=0.7,
                        remediation_complexity=7,
                        architectural_impact=0.6,
                        technical_description=f"Spaghetti code pattern detected in {file_entry.file_name} with {len(items)} complexity issues",
                        business_impact="Code is difficult to understand, modify, and maintain",
                        remediation_strategy="Refactor into smaller, well-structured functions with clear responsibilities"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    # Additional placeholder methods for other pattern detection types
    async def _detect_feature_envy(self, project: ProjectIndex) -> List[AdvancedDebtPattern]:
        """Detect feature envy pattern using dependency analysis."""
        patterns = []
        
        # Analyze dependency relationships to find feature envy
        for dependency in project.dependency_relationships:
            # High external call frequency indicates potential feature envy
            if hasattr(dependency, 'call_frequency') and dependency.call_frequency > 10:
                dependent_file = next(
                    (f for f in project.file_entries if str(f.id) == dependency.from_file_id), 
                    None
                )
                target_file = next(
                    (f for f in project.file_entries if str(f.id) == dependency.to_file_id), 
                    None
                )
                
                if dependent_file and target_file:
                    pattern = AdvancedDebtPattern(
                        pattern_type=DebtPatternType.DESIGN_ANTIPATTERN,
                        pattern_name="feature_envy",
                        confidence=min(1.0, dependency.call_frequency / 20.0),
                        files_involved=[dependent_file.file_path, target_file.file_path],
                        evidence={
                            "call_frequency": dependency.call_frequency,
                            "dependent_file": dependent_file.file_name,
                            "target_file": target_file.file_name,
                            "dependency_type": dependency.dependency_type.value if dependency.dependency_type else "unknown"
                        },
                        severity_score=0.6,
                        remediation_complexity=5,
                        architectural_impact=0.5,
                        technical_description=f"Feature envy detected: {dependent_file.file_name} makes excessive calls to {target_file.file_name}",
                        business_impact="Increases coupling and reduces maintainability",
                        remediation_strategy="Move method closer to the data it uses or extract interface"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _detect_anemic_models(self, file_entries: List[FileEntry]) -> List[AdvancedDebtPattern]:
        """Detect anemic model pattern - classes with data but no behavior."""
        patterns = []
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Count methods vs properties/attributes
                        methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]
                        properties = [n for n in node.body if isinstance(n, ast.AnnAssign) or 
                                    (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name))]
                        
                        # Anemic model heuristic: many properties, few methods
                        if len(properties) > 5 and len(methods) <= 2:
                            behavior_ratio = len(methods) / max(len(properties), 1)
                            
                            pattern = AdvancedDebtPattern(
                                pattern_type=DebtPatternType.DESIGN_ANTIPATTERN,
                                pattern_name="anemic_model",
                                confidence=1.0 - behavior_ratio,
                                files_involved=[file_entry.file_path],
                                evidence={
                                    "class_name": node.name,
                                    "property_count": len(properties),
                                    "method_count": len(methods),
                                    "behavior_ratio": behavior_ratio
                                },
                                severity_score=0.7,
                                remediation_complexity=6,
                                architectural_impact=0.6,
                                technical_description=f"Anemic model detected: {node.name} has {len(properties)} properties but only {len(methods)} behavior methods",
                                business_impact="Violates object-oriented principles, logic scattered in other classes",
                                remediation_strategy="Move related behavior into the domain object"
                            )
                            patterns.append(pattern)
                            
            except Exception as e:
                logger.warning("Error analyzing file for anemic models", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    async def _detect_n_plus_one_patterns(self, file_entries: List[FileEntry]) -> List[AdvancedDebtPattern]:
        """Detect N+1 query patterns using code analysis."""
        patterns = []
        
        # Common N+1 patterns to look for
        n_plus_one_indicators = [
            r'for\s+\w+\s+in\s+.*:\s*\n.*(?:query|select|get|fetch)\(',  # Loop with query inside
            r'while\s+.*:\s*\n.*(?:query|select|get|fetch)\(',  # While loop with query
            r'\[.*(?:query|select|get|fetch)\(.*\).*for.*in.*\]',  # List comprehension with query
        ]
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                import re
                for pattern_regex in n_plus_one_indicators:
                    matches = re.finditer(pattern_regex, content, re.MULTILINE | re.IGNORECASE)
                    
                    for match in matches:
                        # Calculate line number
                        line_number = content[:match.start()].count('\n') + 1
                        
                        pattern = AdvancedDebtPattern(
                            pattern_type=DebtPatternType.PERFORMANCE_HOTSPOT,
                            pattern_name="n_plus_one_query",
                            confidence=0.8,
                            files_involved=[file_entry.file_path],
                            evidence={
                                "matched_pattern": match.group(),
                                "line_number": line_number,
                                "pattern_type": "database_query_in_loop"
                            },
                            severity_score=0.8,
                            remediation_complexity=4,
                            architectural_impact=0.7,
                            technical_description=f"Potential N+1 query pattern detected in {file_entry.file_name} at line {line_number}",
                            business_impact="Can cause severe performance degradation with large datasets",
                            remediation_strategy="Use batch queries, eager loading, or query optimization"
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning("Error analyzing file for N+1 patterns", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    async def _detect_inefficient_algorithms(self, file_entries: List[FileEntry]) -> List[AdvancedDebtPattern]:
        """Detect inefficient algorithms using AST analysis."""
        patterns = []
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Detect nested loops (potential O(n²) or worse)
                        nested_loop_depth = self._calculate_nested_loop_depth(node)
                        
                        if nested_loop_depth >= 2:
                            complexity_score = min(1.0, nested_loop_depth / 3.0)
                            
                            pattern = AdvancedDebtPattern(
                                pattern_type=DebtPatternType.PERFORMANCE_HOTSPOT,
                                pattern_name="inefficient_algorithm",
                                confidence=complexity_score,
                                files_involved=[file_entry.file_path],
                                evidence={
                                    "function_name": node.name,
                                    "nested_loop_depth": nested_loop_depth,
                                    "line_number": node.lineno,
                                    "estimated_complexity": f"O(n^{nested_loop_depth})"
                                },
                                severity_score=complexity_score,
                                remediation_complexity=7,
                                architectural_impact=0.6,
                                technical_description=f"Inefficient algorithm in {node.name}: nested loops depth {nested_loop_depth}",
                                business_impact="Poor performance with large datasets, scalability issues",
                                remediation_strategy="Consider more efficient algorithms or data structures"
                            )
                            patterns.append(pattern)
                        
                        # Detect inefficient string operations
                        string_concat_count = self._count_string_concatenations(node)
                        if string_concat_count > 5:
                            pattern = AdvancedDebtPattern(
                                pattern_type=DebtPatternType.PERFORMANCE_HOTSPOT,
                                pattern_name="inefficient_string_operations",
                                confidence=0.7,
                                files_involved=[file_entry.file_path],
                                evidence={
                                    "function_name": node.name,
                                    "string_concatenations": string_concat_count,
                                    "line_number": node.lineno
                                },
                                severity_score=0.5,
                                remediation_complexity=3,
                                architectural_impact=0.3,
                                technical_description=f"Inefficient string operations in {node.name}: {string_concat_count} concatenations",
                                business_impact="Memory waste and performance degradation",
                                remediation_strategy="Use string joining, formatting, or StringBuilder pattern"
                            )
                            patterns.append(pattern)
                            
            except Exception as e:
                logger.warning("Error analyzing file for inefficient algorithms", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    async def _detect_hardcoded_secrets(self, file_entries: List[FileEntry]) -> List[AdvancedDebtPattern]:
        patterns = []
        
        # Simple regex patterns for common secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'hardcoded_api_key'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'hardcoded_secret'),
            (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded_token'),
        ]
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern, secret_type in secret_patterns:
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    
                    if matches:
                        advanced_pattern = AdvancedDebtPattern(
                            pattern_type=DebtPatternType.SECURITY_VULNERABILITY,
                            pattern_name="hardcoded_secrets",
                            confidence=0.9,
                            files_involved=[file_entry.file_path],
                            evidence={
                                "secret_type": secret_type,
                                "match_count": len(matches)
                            },
                            severity_score=0.9,
                            remediation_complexity=3,
                            architectural_impact=0.4,
                            technical_description=f"Hardcoded {secret_type} detected in {file_entry.file_name}",
                            business_impact="Security risk: sensitive information exposed in source code",
                            remediation_strategy="Move secrets to environment variables or secure configuration"
                        )
                        patterns.append(advanced_pattern)
                        break  # One pattern per file to avoid duplicates
                        
            except Exception as e:
                logger.warning("Error scanning for secrets", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    async def _detect_injection_risks(self, file_entries: List[FileEntry]) -> List[AdvancedDebtPattern]:
        """Detect injection vulnerability patterns."""
        patterns = []
        
        # SQL injection patterns
        sql_injection_patterns = [
            r'query\s*=\s*["\'][^"\']*.format\(',  # String formatting in SQL
            r'execute\(.*%.*\)',  # String interpolation in execute
            r'cursor\.execute\(.*\+.*\)',  # String concatenation in SQL
            r'f["\'][^"\']*(SELECT|INSERT|UPDATE|DELETE).*{.*}',  # f-strings in SQL
        ]
        
        # Command injection patterns
        command_injection_patterns = [
            r'os\.system\(.*\+.*\)',  # String concatenation in os.system
            r'subprocess\.(call|run|Popen)\(.*\+.*\)',  # String concat in subprocess
            r'eval\(.*input.*\)',  # eval with user input
            r'exec\(.*input.*\)',  # exec with user input
        ]
        
        for file_entry in file_entries:
            if not file_entry.file_path or file_entry.is_binary:
                continue
                
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                import re
                
                # Check for SQL injection patterns
                for pattern_regex in sql_injection_patterns:
                    matches = re.finditer(pattern_regex, content, re.MULTILINE | re.IGNORECASE)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        
                        pattern = AdvancedDebtPattern(
                            pattern_type=DebtPatternType.SECURITY_VULNERABILITY,
                            pattern_name="sql_injection_risk",
                            confidence=0.85,
                            files_involved=[file_entry.file_path],
                            evidence={
                                "matched_pattern": match.group(),
                                "line_number": line_number,
                                "vulnerability_type": "sql_injection"
                            },
                            severity_score=0.9,
                            remediation_complexity=3,
                            architectural_impact=0.8,
                            technical_description=f"SQL injection risk detected in {file_entry.file_name} at line {line_number}",
                            business_impact="CRITICAL: Potential data breach and unauthorized database access",
                            remediation_strategy="Use parameterized queries and input validation"
                        )
                        patterns.append(pattern)
                
                # Check for command injection patterns
                for pattern_regex in command_injection_patterns:
                    matches = re.finditer(pattern_regex, content, re.MULTILINE | re.IGNORECASE)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        
                        pattern = AdvancedDebtPattern(
                            pattern_type=DebtPatternType.SECURITY_VULNERABILITY,
                            pattern_name="command_injection_risk",
                            confidence=0.9,
                            files_involved=[file_entry.file_path],
                            evidence={
                                "matched_pattern": match.group(),
                                "line_number": line_number,
                                "vulnerability_type": "command_injection"
                            },
                            severity_score=0.95,
                            remediation_complexity=4,
                            architectural_impact=0.9,
                            technical_description=f"Command injection risk detected in {file_entry.file_name} at line {line_number}",
                            business_impact="CRITICAL: Remote code execution and system compromise risk",
                            remediation_strategy="Use subprocess with argument lists, avoid eval/exec, sanitize inputs"
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.warning("Error analyzing file for injection risks", file_path=file_entry.file_path, error=str(e))
        
        return patterns
    
    # Helper methods for clustering and trend analysis
    async def _create_debt_feature_vectors(self, debt_items: List[DebtItem]) -> Optional[np.ndarray]:
        """Create feature vectors for debt clustering."""
        if len(debt_items) == 0:
            return None
        
        features = []
        for item in debt_items:
            # Create feature vector from debt item properties
            feature_vector = [
                item.debt_score,
                item.confidence_score,
                item.estimated_effort_hours or 0,
                1.0 if item.severity == DebtSeverity.CRITICAL else 0.0,
                1.0 if item.severity == DebtSeverity.HIGH else 0.0,
                1.0 if item.category == DebtCategory.COMPLEXITY else 0.0,
                1.0 if item.category == DebtCategory.CODE_DUPLICATION else 0.0,
                1.0 if item.category == DebtCategory.CODE_SMELLS else 0.0,
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _analyze_debt_cluster(
        self,
        cluster_id: str,
        cluster_items: List[DebtItem],
        cluster_features: np.ndarray
    ) -> DebtCluster:
        """Analyze a cluster of debt items."""
        
        # Calculate cluster center
        cluster_center = np.mean(cluster_features, axis=0)
        
        # Calculate cohesion (how similar items in cluster are)
        cohesion_score = 1.0 - np.std(cluster_features)
        
        # Determine dominant categories
        category_counts = Counter(item.category for item in cluster_items)
        dominant_categories = [cat for cat, count in category_counts.most_common(3)]
        
        # Generate refactoring suggestion
        if DebtCategory.COMPLEXITY in dominant_categories:
            suggested_refactoring = "Refactor complex functions into smaller, focused methods"
        elif DebtCategory.CODE_DUPLICATION in dominant_categories:
            suggested_refactoring = "Extract common functionality into shared utilities"
        elif DebtCategory.CODE_SMELLS in dominant_categories:
            suggested_refactoring = "Apply clean code practices and naming conventions"
        else:
            suggested_refactoring = "Address clustered issues systematically"
        
        # Estimate effort (sum of individual estimates with coordination overhead)
        effort_estimate = sum(item.estimated_effort_hours or 1 for item in cluster_items)
        effort_estimate = int(effort_estimate * 1.2)  # Add coordination overhead
        
        return DebtCluster(
            cluster_id=cluster_id,
            debt_items=cluster_items,
            cluster_center=cluster_center,
            cohesion_score=cohesion_score,
            dominant_categories=dominant_categories,
            suggested_refactoring=suggested_refactoring,
            effort_estimate=effort_estimate
        )
    
    def _calculate_trend(
        self, 
        timestamps: List[datetime], 
        debt_scores: List[float]
    ) -> Tuple[str, float]:
        """Calculate debt trend direction and velocity."""
        if len(timestamps) < 2:
            return "insufficient_data", 0.0
        
        # Simple linear regression for trend
        x = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        y = np.array(debt_scores)
        
        if len(x) != len(y) or len(x) == 0:
            return "insufficient_data", 0.0
        
        # Calculate slope (velocity)
        if np.std(x) == 0:  # All timestamps are the same
            return "stable", 0.0
            
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        slope = correlation * (np.std(y) / np.std(x))
        
        # Determine trend direction
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return direction, slope
    
    def _assess_trend_risk(self, velocity: float, projected_debt: float) -> str:
        """Assess risk level based on trend velocity and projection."""
        if velocity > 0.1 or projected_debt > 0.8:
            return "high"
        elif velocity > 0.05 or projected_debt > 0.6:
            return "medium"
        elif velocity < -0.05:
            return "improving"
        else:
            return "low"
    
    def _identify_trend_causes(self, historical_data: List[Dict[str, Any]]) -> List[str]:
        """Identify likely causes of debt trends."""
        causes = []
        
        # Analyze historical data for patterns
        for entry in historical_data[-5:]:  # Last 5 entries
            if 'debt_causes' in entry:
                causes.extend(entry['debt_causes'])
        
        # Return most common causes
        if causes:
            cause_counts = Counter(causes)
            return [cause for cause, count in cause_counts.most_common(3)]
        else:
            return ["Unknown - insufficient historical data"]
    
    def _generate_intervention_recommendations(
        self,
        trend_direction: str,
        velocity: float,
        risk_level: str,
        trend_causes: List[str]
    ) -> List[str]:
        """Generate intervention recommendations based on trend analysis."""
        recommendations = []
        
        if trend_direction == "increasing":
            recommendations.append("Implement debt reduction sprint to address accumulating issues")
            if velocity > 0.1:
                recommendations.append("URGENT: Debt is accumulating rapidly - halt new features until stabilized")
        
        if risk_level == "high":
            recommendations.append("Schedule architectural review and refactoring session")
            recommendations.append("Increase code review standards and add debt gates to CI/CD")
        
        if "complexity" in trend_causes:
            recommendations.append("Focus on simplifying complex functions and classes")
        
        if "duplication" in trend_causes:
            recommendations.append("Prioritize duplicate code elimination and shared utility creation")
        
        if not recommendations:
            recommendations.append("Continue monitoring debt levels and maintain current practices")
        
        return recommendations
    
    async def _create_anomaly_feature_matrix(
        self,
        file_entries: List[FileEntry],
        base_analysis: DebtAnalysisResult
    ) -> Optional[np.ndarray]:
        """Create feature matrix for anomaly detection."""
        features = []
        
        # Create debt lookup by file
        debt_by_file = defaultdict(list)
        for item in base_analysis.debt_items:
            debt_by_file[item.file_id].append(item)
        
        for file_entry in file_entries:
            file_debt_items = debt_by_file.get(str(file_entry.id), [])
            
            # Calculate file-level debt metrics
            total_debt_score = sum(item.debt_score for item in file_debt_items)
            complexity_count = len([item for item in file_debt_items if item.category == DebtCategory.COMPLEXITY])
            duplication_count = len([item for item in file_debt_items if item.category == DebtCategory.CODE_DUPLICATION])
            smell_count = len([item for item in file_debt_items if item.category == DebtCategory.CODE_SMELLS])
            
            feature_vector = [
                total_debt_score,
                len(file_debt_items),
                complexity_count,
                duplication_count,
                smell_count,
                file_entry.line_count or 0,
                file_entry.file_size or 0,
            ]
            features.append(feature_vector)
        
        return np.array(features) if features else None
    
    async def _extract_file_metrics(self, file_entry: FileEntry) -> Dict[str, Any]:
        """Extract metrics for anomaly analysis."""
        return {
            "file_name": file_entry.file_name,
            "file_size": file_entry.file_size,
            "line_count": file_entry.line_count,
            "file_type": file_entry.file_type.value if file_entry.file_type else "unknown",
            "language": file_entry.language
        }
    
    def _generate_anomaly_recommendations(
        self,
        file_entry: FileEntry,
        anomaly_score: float
    ) -> List[str]:
        """Generate recommendations for anomalous files."""
        recommendations = []
        
        if abs(anomaly_score) > 0.7:
            recommendations.append("File shows severe anomalies - prioritize for immediate review")
            recommendations.append("Consider major refactoring or rewriting this file")
        elif abs(anomaly_score) > 0.4:
            recommendations.append("File has moderate anomalies - schedule for refactoring")
            recommendations.append("Break down large functions and classes in this file")
        else:
            recommendations.append("File has minor anomalies - monitor during regular maintenance")
        
        recommendations.append(f"Focus on improving {file_entry.language or 'code'} quality in this file")
        
        return recommendations
    
    def _calculate_nested_loop_depth(self, node: ast.FunctionDef) -> int:
        """Calculate the maximum depth of nested loops in a function."""
        max_depth = 0
        
        def calculate_depth(ast_node: ast.AST, current_depth: int = 0) -> int:
            nonlocal max_depth
            
            if isinstance(ast_node, (ast.For, ast.While, ast.AsyncFor)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                
                # Continue traversing children with increased depth
                for child in ast.iter_child_nodes(ast_node):
                    calculate_depth(child, current_depth)
            else:
                # Continue traversing children without increasing depth
                for child in ast.iter_child_nodes(ast_node):
                    calculate_depth(child, current_depth)
            
            return max_depth
        
        return calculate_depth(node)
    
    def _count_string_concatenations(self, node: ast.FunctionDef) -> int:
        """Count string concatenation operations in a function."""
        concat_count = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.Add):
                # Check if it's likely string concatenation
                # This is a heuristic - in practice you'd want more sophisticated analysis
                concat_count += 1
            elif isinstance(child, ast.Call):
                # Check for .format() or .join() calls which might indicate string building
                if (isinstance(child.func, ast.Attribute) and 
                    child.func.attr in ['format', 'join']):
                    concat_count += 1
        
        return concat_count
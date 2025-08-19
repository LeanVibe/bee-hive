"""
Integrated Technical Debt Analyzer for LeanVibe Agent Hive 2.0

Advanced technical debt detection system integrated with the project index infrastructure.
Provides comprehensive analysis, tracking, and remediation recommendations for code quality issues.
"""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import ast

import structlog
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, text
from sqlalchemy.orm import selectinload

from ..models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, 
    DebtSnapshot, DebtItem, DebtRemediationPlan,
    DebtSeverity as DBDebtSeverity, DebtCategory as DBDebtCategory, DebtStatus as DBDebtStatus
)
from ..core.database import get_session
from .analyzer import CodeAnalyzer
from .ml_analyzer import MLAnalyzer, EmbeddingType, PatternType, PatternMatch
from .historical_analyzer import HistoricalAnalyzer
from .models import AnalysisConfiguration, FileAnalysisResult

logger = structlog.get_logger()


class DebtSeverity(Enum):
    """Technical debt severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DebtCategory(Enum):
    """Categories of technical debt."""
    CODE_DUPLICATION = "code_duplication"
    COMPLEXITY = "complexity"
    CODE_SMELLS = "code_smells"
    ARCHITECTURE = "architecture"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"


class DebtStatus(Enum):
    """Status of debt items."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    IGNORED = "ignored"
    FALSE_POSITIVE = "false_positive"


@dataclass
class DebtItem:
    """Individual technical debt item."""
    id: Optional[str] = None
    project_id: str = ""
    file_id: str = ""
    debt_type: str = ""
    category: DebtCategory = DebtCategory.CODE_SMELLS
    severity: DebtSeverity = DebtSeverity.MEDIUM
    status: DebtStatus = DebtStatus.ACTIVE
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    location: Dict[str, Any] = field(default_factory=dict)
    remediation_suggestion: str = ""
    estimated_effort_hours: int = 1
    debt_score: float = 0.0
    confidence_score: float = 1.0
    first_detected_at: Optional[datetime] = None
    last_detected_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebtSnapshot:
    """Snapshot of project debt at a point in time."""
    id: Optional[str] = None
    project_id: str = ""
    snapshot_date: Optional[datetime] = None
    total_debt_score: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    debt_trend: Dict[str, Any] = field(default_factory=dict)
    file_count_analyzed: int = 0
    lines_of_code_analyzed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebtAnalysisResult:
    """Result of debt analysis operation."""
    project_id: str
    total_debt_score: float
    debt_items: List[DebtItem]
    category_scores: Dict[str, float]
    file_count: int
    lines_of_code: int
    analysis_duration: float
    recommendations: List[str]


class TechnicalDebtAnalyzer:
    """
    Integrated technical debt analyzer using project index infrastructure.
    
    Leverages existing AST analysis, ML pattern detection, and historical analysis
    to provide comprehensive technical debt detection and tracking.
    """
    
    def __init__(self, config: Optional[AnalysisConfiguration] = None):
        """Initialize the technical debt analyzer."""
        self.config = config or AnalysisConfiguration()
        self.code_analyzer = CodeAnalyzer(config)
        self.ml_analyzer = MLAnalyzer(config)
        self.historical_analyzer = HistoricalAnalyzer()
        
        # Debt detection thresholds
        self.complexity_thresholds = {
            'cyclomatic_complexity': 10,
            'cognitive_complexity': 15,
            'function_length': 50,
            'parameter_count': 7,
            'nested_depth': 4,
            'class_length': 500
        }
        
        # Code smell patterns
        self.code_smell_patterns = {
            'long_method': lambda f: len(f.get('body', [])) > 30,
            'long_parameter_list': lambda f: len(f.get('parameters', [])) > 7,
            'large_class': lambda c: c.get('line_count', 0) > 500,
            'duplicate_code': lambda similarity: similarity > 0.8,
            'god_class': lambda c: len(c.get('methods', [])) > 20 and c.get('line_count', 0) > 500
        }
        
        # Initialize similarity analyzer for duplicate detection
        self.similarity_analyzer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 3)
        )
    
    async def analyze_project_debt(
        self, 
        project_id: str, 
        session: Optional[AsyncSession] = None
    ) -> DebtAnalysisResult:
        """
        Analyze technical debt for entire project.
        
        Args:
            project_id: ID of project to analyze
            session: Database session (optional)
            
        Returns:
            DebtAnalysisResult with comprehensive debt analysis
        """
        start_time = time.time()
        
        if session is None:
            async with get_session() as session:
                return await self._analyze_project_debt_with_session(project_id, session, start_time)
        else:
            return await self._analyze_project_debt_with_session(project_id, session, start_time)
    
    async def _analyze_project_debt_with_session(
        self, 
        project_id: str, 
        session: AsyncSession,
        start_time: float
    ) -> DebtAnalysisResult:
        """Internal method to analyze project debt with session."""
        
        logger.info("Starting technical debt analysis", project_id=project_id)
        
        # Get project and file entries
        stmt = select(ProjectIndex).where(ProjectIndex.id == project_id).options(
            selectinload(ProjectIndex.file_entries),
            selectinload(ProjectIndex.dependency_relationships)
        )
        result = await session.execute(stmt)
        project = result.scalar_one_or_none()
        
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Initialize result
        debt_items = []
        category_scores = defaultdict(float)
        total_lines = 0
        file_count = len(project.file_entries)
        
        # Analyze each file for debt
        for file_entry in project.file_entries:
            try:
                file_debt_items = await self._analyze_file_debt(file_entry, session)
                debt_items.extend(file_debt_items)
                total_lines += file_entry.line_count or 0
                
                # Update category scores
                for item in file_debt_items:
                    category_scores[item.category.value] += item.debt_score
                    
            except Exception as e:
                logger.warning(
                    "Failed to analyze file for debt", 
                    file_path=file_entry.file_path, 
                    error=str(e)
                )
        
        # Detect cross-file debt patterns
        cross_file_debt = await self._detect_cross_file_debt(project.file_entries, session)
        debt_items.extend(cross_file_debt)
        
        # Calculate total debt score
        total_debt_score = sum(item.debt_score for item in debt_items) / max(file_count, 1)
        
        # Generate recommendations
        recommendations = self._generate_debt_recommendations(debt_items, category_scores)
        
        analysis_duration = time.time() - start_time
        
        logger.info(
            "Technical debt analysis completed",
            project_id=project_id,
            debt_items_found=len(debt_items),
            total_debt_score=total_debt_score,
            duration=analysis_duration
        )
        
        return DebtAnalysisResult(
            project_id=project_id,
            total_debt_score=total_debt_score,
            debt_items=debt_items,
            category_scores=dict(category_scores),
            file_count=file_count,
            lines_of_code=total_lines,
            analysis_duration=analysis_duration,
            recommendations=recommendations
        )
    
    async def _analyze_file_debt(self, file_entry: FileEntry, session: AsyncSession) -> List[DebtItem]:
        """Analyze technical debt in a single file."""
        debt_items = []
        
        # Skip binary files
        if file_entry.is_binary or not file_entry.file_path:
            return debt_items
        
        try:
            # Read file content
            with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Analyze with code analyzer
            analysis_result = await self.code_analyzer.analyze_file(
                Path(file_entry.file_path), content
            )
            
            if not analysis_result:
                return debt_items
            
            # Check for complexity debt
            complexity_debt = self._detect_complexity_debt(file_entry, analysis_result, content)
            debt_items.extend(complexity_debt)
            
            # Check for code smells
            smell_debt = self._detect_code_smells(file_entry, analysis_result, content)
            debt_items.extend(smell_debt)
            
            # Check for maintainability issues
            maintainability_debt = self._detect_maintainability_debt(file_entry, analysis_result, content)
            debt_items.extend(maintainability_debt)
            
            # Update file entry analysis data with debt information
            if not file_entry.analysis_data:
                file_entry.analysis_data = {}
            
            file_entry.analysis_data['technical_debt'] = {
                'debt_score': sum(item.debt_score for item in debt_items),
                'debt_count': len(debt_items),
                'categories': list(set(item.category.value for item in debt_items)),
                'last_analyzed': datetime.utcnow().isoformat()
            }
            
            await session.commit()
            
        except Exception as e:
            logger.warning(
                "Error analyzing file for debt", 
                file_path=file_entry.file_path, 
                error=str(e)
            )
        
        return debt_items
    
    def _detect_complexity_debt(
        self, 
        file_entry: FileEntry, 
        analysis: FileAnalysisResult, 
        content: str
    ) -> List[DebtItem]:
        """Detect complexity-related technical debt."""
        debt_items = []
        
        # Parse AST for detailed analysis
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return debt_items
        
        # Analyze each function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > self.complexity_thresholds['cyclomatic_complexity']:
                    debt_items.append(DebtItem(
                        project_id=str(file_entry.project_id),
                        file_id=str(file_entry.id),
                        debt_type="high_cyclomatic_complexity",
                        category=DebtCategory.COMPLEXITY,
                        severity=self._get_severity_for_complexity(complexity, 'cyclomatic'),
                        description=f"Function '{node.name}' has high cyclomatic complexity: {complexity}",
                        evidence={
                            "function_name": node.name,
                            "complexity_score": complexity,
                            "threshold": self.complexity_thresholds['cyclomatic_complexity']
                        },
                        location={
                            "line_number": node.lineno,
                            "function_name": node.name
                        },
                        remediation_suggestion="Consider breaking this function into smaller, focused functions",
                        estimated_effort_hours=max(2, complexity // 5),
                        debt_score=min(1.0, complexity / 20.0),
                        confidence_score=0.9
                    ))
                
                # Check function length
                function_length = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if function_length > self.complexity_thresholds['function_length']:
                    debt_items.append(DebtItem(
                        project_id=str(file_entry.project_id),
                        file_id=str(file_entry.id),
                        debt_type="long_function",
                        category=DebtCategory.CODE_SMELLS,
                        severity=self._get_severity_for_length(function_length),
                        description=f"Function '{node.name}' is too long: {function_length} lines",
                        evidence={
                            "function_name": node.name,
                            "length": function_length,
                            "threshold": self.complexity_thresholds['function_length']
                        },
                        location={
                            "line_number": node.lineno,
                            "function_name": node.name
                        },
                        remediation_suggestion="Break this function into smaller, more focused functions",
                        estimated_effort_hours=max(1, function_length // 25),
                        debt_score=min(1.0, function_length / 100.0),
                        confidence_score=0.8
                    ))
                
                # Check parameter count
                param_count = len(node.args.args)
                if param_count > self.complexity_thresholds['parameter_count']:
                    debt_items.append(DebtItem(
                        project_id=str(file_entry.project_id),
                        file_id=str(file_entry.id),
                        debt_type="too_many_parameters",
                        category=DebtCategory.CODE_SMELLS,
                        severity=self._get_severity_for_param_count(param_count),
                        description=f"Function '{node.name}' has too many parameters: {param_count}",
                        evidence={
                            "function_name": node.name,
                            "parameter_count": param_count,
                            "threshold": self.complexity_thresholds['parameter_count']
                        },
                        location={
                            "line_number": node.lineno,
                            "function_name": node.name
                        },
                        remediation_suggestion="Consider using parameter objects or reducing coupling",
                        estimated_effort_hours=2,
                        debt_score=min(1.0, param_count / 15.0),
                        confidence_score=0.9
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Check class size
                class_end = getattr(node, 'end_lineno', node.lineno + 50)
                class_length = class_end - node.lineno
                if class_length > self.complexity_thresholds['class_length']:
                    method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    debt_items.append(DebtItem(
                        project_id=str(file_entry.project_id),
                        file_id=str(file_entry.id),
                        debt_type="large_class",
                        category=DebtCategory.CODE_SMELLS,
                        severity=self._get_severity_for_class_size(class_length, method_count),
                        description=f"Class '{node.name}' is too large: {class_length} lines, {method_count} methods",
                        evidence={
                            "class_name": node.name,
                            "length": class_length,
                            "method_count": method_count,
                            "threshold": self.complexity_thresholds['class_length']
                        },
                        location={
                            "line_number": node.lineno,
                            "class_name": node.name
                        },
                        remediation_suggestion="Consider breaking this class into smaller, focused classes",
                        estimated_effort_hours=max(4, class_length // 100),
                        debt_score=min(1.0, class_length / 1000.0),
                        confidence_score=0.85
                    ))
        
        return debt_items
    
    def _detect_code_smells(
        self, 
        file_entry: FileEntry, 
        analysis: FileAnalysisResult, 
        content: str
    ) -> List[DebtItem]:
        """Detect code smell patterns."""
        debt_items = []
        
        # Detect duplicate code patterns within file
        lines = content.split('\n')
        duplicate_blocks = self._find_duplicate_code_blocks(lines)
        
        for block in duplicate_blocks:
            debt_items.append(DebtItem(
                project_id=str(file_entry.project_id),
                file_id=str(file_entry.id),
                debt_type="duplicate_code_block",
                category=DebtCategory.CODE_DUPLICATION,
                severity=self._get_severity_for_duplication(block['similarity']),
                description=f"Duplicate code block found ({block['lines_count']} lines, {block['similarity']:.1%} similar)",
                evidence=block,
                location={
                    "line_ranges": block['line_ranges']
                },
                remediation_suggestion="Extract common code into a shared function or method",
                estimated_effort_hours=max(1, block['lines_count'] // 10),
                debt_score=block['similarity'] * 0.8,
                confidence_score=0.8
            ))
        
        # Detect naming violations
        naming_issues = self._detect_naming_violations(content)
        for issue in naming_issues:
            debt_items.append(DebtItem(
                project_id=str(file_entry.project_id),
                file_id=str(file_entry.id),
                debt_type="naming_violation",
                category=DebtCategory.CODE_SMELLS,
                severity=DebtSeverity.LOW,
                description=f"Naming violation: {issue['description']}",
                evidence=issue,
                location={
                    "line_number": issue.get('line_number', 0),
                    "name": issue.get('name', '')
                },
                remediation_suggestion=issue.get('suggestion', 'Follow naming conventions'),
                estimated_effort_hours=1,
                debt_score=0.2,
                confidence_score=0.7
            ))
        
        return debt_items
    
    def _detect_maintainability_debt(
        self, 
        file_entry: FileEntry, 
        analysis: FileAnalysisResult, 
        content: str
    ) -> List[DebtItem]:
        """Detect maintainability issues."""
        debt_items = []
        
        # Check for missing docstrings
        if not self._has_module_docstring(content):
            debt_items.append(DebtItem(
                project_id=str(file_entry.project_id),
                file_id=str(file_entry.id),
                debt_type="missing_module_docstring",
                category=DebtCategory.DOCUMENTATION,
                severity=DebtSeverity.LOW,
                description="Module is missing a docstring",
                evidence={
                    "file_name": file_entry.file_name,
                    "has_docstring": False
                },
                location={"line_number": 1},
                remediation_suggestion="Add a comprehensive module docstring explaining purpose and usage",
                estimated_effort_hours=1,
                debt_score=0.1,
                confidence_score=1.0
            ))
        
        # Check comment density
        comment_ratio = self._calculate_comment_ratio(content)
        if comment_ratio < 0.1 and file_entry.line_count and file_entry.line_count > 50:
            debt_items.append(DebtItem(
                project_id=str(file_entry.project_id),
                file_id=str(file_entry.id),
                debt_type="insufficient_comments",
                category=DebtCategory.DOCUMENTATION,
                severity=DebtSeverity.MEDIUM,
                description=f"Low comment density: {comment_ratio:.1%}",
                evidence={
                    "comment_ratio": comment_ratio,
                    "threshold": 0.1,
                    "file_length": file_entry.line_count
                },
                location={},
                remediation_suggestion="Add more explanatory comments for complex logic",
                estimated_effort_hours=2,
                debt_score=0.3,
                confidence_score=0.6
            ))
        
        return debt_items
    
    async def _detect_cross_file_debt(
        self, 
        file_entries: List[FileEntry], 
        session: AsyncSession
    ) -> List[DebtItem]:
        """Detect debt patterns that span multiple files."""
        debt_items = []
        
        # Detect duplicate functionality across files
        if len(file_entries) > 1:
            duplicate_functions = await self._detect_duplicate_functions(file_entries)
            
            for duplicate_group in duplicate_functions:
                if len(duplicate_group) > 1:
                    debt_items.append(DebtItem(
                        project_id=str(file_entries[0].project_id),
                        file_id=str(duplicate_group[0]['file_id']),  # Primary file
                        debt_type="duplicate_functionality",
                        category=DebtCategory.CODE_DUPLICATION,
                        severity=DebtSeverity.HIGH,
                        description=f"Duplicate function found in {len(duplicate_group)} files",
                        evidence={
                            "function_name": duplicate_group[0]['function_name'],
                            "duplicate_files": [d['file_path'] for d in duplicate_group],
                            "similarity_score": duplicate_group[0].get('similarity', 0.9)
                        },
                        location={
                            "function_name": duplicate_group[0]['function_name'],
                            "files": [d['file_path'] for d in duplicate_group]
                        },
                        remediation_suggestion="Consolidate duplicate functions into a shared module",
                        estimated_effort_hours=len(duplicate_group) * 2,
                        debt_score=0.8,
                        confidence_score=0.85
                    ))
        
        return debt_items
    
    async def save_debt_analysis(
        self, 
        result: DebtAnalysisResult, 
        session: Optional[AsyncSession] = None
    ) -> None:
        """Save debt analysis results to database."""
        
        if session is None:
            async with get_session() as session:
                await self._save_debt_analysis_with_session(result, session)
        else:
            await self._save_debt_analysis_with_session(result, session)
    
    async def _save_debt_analysis_with_session(
        self, 
        result: DebtAnalysisResult, 
        session: AsyncSession
    ) -> None:
        """Internal method to save debt analysis with session."""
        
        # Create debt snapshot
        snapshot_data = {
            'project_id': result.project_id,
            'snapshot_date': datetime.utcnow(),
            'total_debt_score': result.total_debt_score,
            'category_scores': result.category_scores,
            'debt_trend': {},  # Will be calculated based on historical data
            'file_count_analyzed': result.file_count,
            'lines_of_code_analyzed': result.lines_of_code,
            'metadata': {
                'analysis_duration': result.analysis_duration,
                'recommendations': result.recommendations
            }
        }
        
        # Create and save debt snapshot using SQLAlchemy model
        snapshot = DebtSnapshot(
            project_id=result.project_id,
            snapshot_date=datetime.utcnow(),
            total_debt_score=result.total_debt_score,
            category_scores=result.category_scores,
            debt_trend={},  # Will be calculated based on historical data
            file_count_analyzed=result.file_count,
            lines_of_code_analyzed=result.lines_of_code,
            meta_data={
                'analysis_duration': result.analysis_duration,
                'recommendations': result.recommendations
            }
        )
        
        session.add(snapshot)
        
        # Save debt items (simplified for Phase 1)
        for item in result.debt_items:
            # Convert enum values for database storage
            db_item = DebtItem(
                project_id=item.project_id,
                file_id=item.file_id,
                debt_type=item.debt_type,
                debt_category=DBDebtCategory(item.category.value),
                severity=DBDebtSeverity(item.severity.value),
                status=DBDebtStatus(item.status.value),
                description=item.description,
                evidence=item.evidence,
                location=item.location,
                remediation_suggestion=item.remediation_suggestion,
                estimated_effort_hours=item.estimated_effort_hours,
                debt_score=item.debt_score,
                confidence_score=item.confidence_score,
                first_detected_at=item.first_detected_at or datetime.utcnow(),
                last_detected_at=item.last_detected_at or datetime.utcnow(),
                meta_data=item.metadata
            )
            
            session.add(db_item)
        
        # Commit all changes
        await session.commit()
        
        logger.info(
            "Debt analysis results saved",
            project_id=result.project_id,
            debt_items=len(result.debt_items),
            total_debt_score=result.total_debt_score
        )
    
    # Helper methods for debt detection
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _get_severity_for_complexity(self, complexity: int, complexity_type: str) -> DebtSeverity:
        """Get severity based on complexity score."""
        if complexity_type == 'cyclomatic':
            if complexity > 20:
                return DebtSeverity.CRITICAL
            elif complexity > 15:
                return DebtSeverity.HIGH
            elif complexity > 10:
                return DebtSeverity.MEDIUM
        return DebtSeverity.LOW
    
    def _get_severity_for_length(self, length: int) -> DebtSeverity:
        """Get severity based on function/class length."""
        if length > 100:
            return DebtSeverity.HIGH
        elif length > 75:
            return DebtSeverity.MEDIUM
        return DebtSeverity.LOW
    
    def _get_severity_for_param_count(self, count: int) -> DebtSeverity:
        """Get severity based on parameter count."""
        if count > 10:
            return DebtSeverity.HIGH
        elif count > 7:
            return DebtSeverity.MEDIUM
        return DebtSeverity.LOW
    
    def _get_severity_for_class_size(self, length: int, method_count: int) -> DebtSeverity:
        """Get severity for class size issues."""
        if length > 1000 or method_count > 30:
            return DebtSeverity.CRITICAL
        elif length > 750 or method_count > 20:
            return DebtSeverity.HIGH
        return DebtSeverity.MEDIUM
    
    def _get_severity_for_duplication(self, similarity: float) -> DebtSeverity:
        """Get severity for code duplication."""
        if similarity > 0.95:
            return DebtSeverity.HIGH
        elif similarity > 0.8:
            return DebtSeverity.MEDIUM
        return DebtSeverity.LOW
    
    def _find_duplicate_code_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find duplicate code blocks within a file."""
        duplicates = []
        
        # Simple implementation - look for repeated line sequences
        min_block_size = 5
        
        for i in range(len(lines) - min_block_size):
            block = lines[i:i + min_block_size]
            block_text = '\n'.join(block).strip()
            
            if not block_text or block_text.startswith('#'):
                continue
            
            # Look for similar blocks later in the file
            for j in range(i + min_block_size, len(lines) - min_block_size):
                other_block = lines[j:j + min_block_size]
                other_text = '\n'.join(other_block).strip()
                
                if not other_text:
                    continue
                
                # Calculate similarity (simple approach)
                similarity = self._calculate_text_similarity(block_text, other_text)
                
                if similarity > 0.8:
                    duplicates.append({
                        'line_ranges': [(i + 1, i + min_block_size), (j + 1, j + min_block_size)],
                        'lines_count': min_block_size,
                        'similarity': similarity,
                        'block_content': block_text[:200]  # First 200 chars
                    })
                    break
        
        return duplicates
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text blocks."""
        if text1 == text2:
            return 1.0
        
        # Simple character-based similarity
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Count common characters
        common = sum(1 for a, b in zip(text1, text2) if a == b)
        max_len = max(len1, len2)
        
        return common / max_len if max_len > 0 else 0.0
    
    def _detect_naming_violations(self, content: str) -> List[Dict[str, Any]]:
        """Detect naming convention violations."""
        violations = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        violations.append({
                            'type': 'function_name',
                            'name': node.name,
                            'line_number': node.lineno,
                            'description': f"Function name '{node.name}' doesn't follow snake_case convention",
                            'suggestion': "Use snake_case for function names"
                        })
                
                elif isinstance(node, ast.ClassDef):
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        violations.append({
                            'type': 'class_name',
                            'name': node.name,
                            'line_number': node.lineno,
                            'description': f"Class name '{node.name}' doesn't follow PascalCase convention",
                            'suggestion': "Use PascalCase for class names"
                        })
        
        except SyntaxError:
            pass
        
        return violations
    
    def _has_module_docstring(self, content: str) -> bool:
        """Check if module has a docstring."""
        try:
            tree = ast.parse(content)
            return (
                tree.body and
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)
            )
        except SyntaxError:
            return False
    
    def _calculate_comment_ratio(self, content: str) -> float:
        """Calculate ratio of comment lines to total lines."""
        lines = content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        total_lines = len(lines)
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    async def _detect_duplicate_functions(self, file_entries: List[FileEntry]) -> List[List[Dict[str, Any]]]:
        """Detect duplicate functions across files."""
        # Simplified implementation - would use ML similarity in production
        function_groups = []
        
        # Extract functions from each file
        functions_by_file = {}
        
        for file_entry in file_entries:
            if file_entry.is_binary or not file_entry.file_path:
                continue
            
            try:
                with open(file_entry.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                functions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Get function signature and basic info
                        functions.append({
                            'file_id': str(file_entry.id),
                            'file_path': file_entry.file_path,
                            'function_name': node.name,
                            'line_number': node.lineno,
                            'arg_count': len(node.args.args),
                            'body_hash': hashlib.md5(
                                ast.dump(node).encode()
                            ).hexdigest()[:8]
                        })
                
                functions_by_file[str(file_entry.id)] = functions
                
            except (SyntaxError, IOError):
                continue
        
        # Find potential duplicates based on name and structure
        all_functions = []
        for functions in functions_by_file.values():
            all_functions.extend(functions)
        
        # Group functions by name and similar characteristics
        name_groups = defaultdict(list)
        for func in all_functions:
            name_groups[func['function_name']].append(func)
        
        # Return groups with more than one function
        for group in name_groups.values():
            if len(group) > 1:
                function_groups.append(group)
        
        return function_groups
    
    def _generate_debt_recommendations(
        self, 
        debt_items: List[DebtItem], 
        category_scores: Dict[str, float]
    ) -> List[str]:
        """Generate intelligent recommendations based on debt analysis."""
        recommendations = []
        
        # Priority recommendations based on debt categories
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        for category, score in sorted_categories[:3]:  # Top 3 categories
            if score > 0:
                if category == 'code_duplication':
                    recommendations.append(
                        f"Address code duplication (score: {score:.2f}) by extracting common functionality into shared modules"
                    )
                elif category == 'complexity':
                    recommendations.append(
                        f"Reduce complexity (score: {score:.2f}) by breaking down large functions and classes"
                    )
                elif category == 'code_smells':
                    recommendations.append(
                        f"Fix code smells (score: {score:.2f}) by improving naming and structure"
                    )
                elif category == 'maintainability':
                    recommendations.append(
                        f"Improve maintainability (score: {score:.2f}) by adding documentation and comments"
                    )
        
        # High-severity items recommendations
        critical_items = [item for item in debt_items if item.severity == DebtSeverity.CRITICAL]
        if critical_items:
            recommendations.append(
                f"Prioritize {len(critical_items)} critical debt items for immediate attention"
            )
        
        # Quick wins
        low_effort_items = [item for item in debt_items if item.estimated_effort_hours <= 2]
        if len(low_effort_items) > 5:
            recommendations.append(
                f"Consider addressing {len(low_effort_items)} low-effort debt items as quick wins"
            )
        
        return recommendations
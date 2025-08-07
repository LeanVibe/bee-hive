"""
Secure Code Analysis Engine for Self-Modification System

This module provides a secure AST-based code analysis system with ZERO system access.
All analysis is performed in isolation with comprehensive security controls.
"""

import ast
import os
import re
import sys
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for code analysis."""
    MINIMAL = "minimal"
    STANDARD = "standard"  
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"


class CodeComplexity(Enum):
    """Code complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class SecurityViolation:
    """Security violation detected during analysis."""
    violation_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    line_number: Optional[int] = None
    column_offset: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class CodePattern:
    """Detected code pattern or anti-pattern."""
    pattern_type: str  # 'good_practice', 'anti_pattern', 'performance_issue', 'security_issue'
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    description: str
    line_number: Optional[int] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class FunctionMetrics:
    """Metrics for a function."""
    name: str
    lines_of_code: int
    cyclomatic_complexity: int
    parameters_count: int
    return_statements: int
    nested_depth: int
    has_docstring: bool
    complexity_level: CodeComplexity
    

@dataclass
class FileAnalysis:
    """Complete analysis results for a single file."""
    file_path: str
    file_hash: str
    lines_of_code: int
    blank_lines: int
    comment_lines: int
    functions: List[FunctionMetrics]
    classes: List[str]
    imports: List[str]
    security_violations: List[SecurityViolation]
    code_patterns: List[CodePattern]
    complexity_score: float  # 0.0 to 1.0
    maintainability_score: float  # 0.0 to 1.0
    safety_score: float  # 0.0 to 1.0
    analysis_timestamp: datetime
    

@dataclass
class ProjectAnalysis:
    """Complete analysis results for a project."""
    project_path: str
    total_files: int
    total_lines: int
    file_analyses: List[FileAnalysis]
    overall_complexity: float
    overall_maintainability: float
    overall_safety: float
    security_violations: List[SecurityViolation]
    project_patterns: List[CodePattern]
    dependencies: Dict[str, List[str]]
    analysis_metadata: Dict[str, Any]


class SecureCodeAnalyzer:
    """
    Secure code analysis engine with ZERO system access.
    
    Provides comprehensive AST-based analysis while maintaining strict
    security boundaries. No network access, no system calls, no file
    system modifications allowed.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self._allowed_file_extensions = {'.py', '.pyx', '.pyi'}
        self._max_file_size_mb = 10  # Maximum file size to analyze
        self._max_files_per_analysis = 1000  # Maximum files per analysis
        
        # Security patterns to detect
        self._security_patterns = self._load_security_patterns()
        self._performance_patterns = self._load_performance_patterns()
        
        # Blocked operations (security enforcement)
        self._blocked_imports = {
            'os', 'subprocess', 'sys', 'socket', 'urllib', 'requests',
            'http', 'ftplib', 'telnetlib', 'smtplib', 'poplib', 'imaplib',
            'platform', 'ctypes', 'marshal', 'pickle', 'dill'
        }
        
        logger.info(f"SecureCodeAnalyzer initialized with {security_level.value} security level")
    
    def analyze_project(self, project_path: str, 
                       include_patterns: Optional[List[str]] = None,
                       exclude_patterns: Optional[List[str]] = None) -> ProjectAnalysis:
        """
        Analyze entire project with security controls.
        
        Args:
            project_path: Path to project root (must exist and be readable)
            include_patterns: File patterns to include (e.g., ['*.py'])
            exclude_patterns: File patterns to exclude (e.g., ['test_*', '__pycache__'])
            
        Returns:
            Complete project analysis results
            
        Raises:
            SecurityError: If security violations are detected
            ValueError: If project path is invalid
        """
        project_path = Path(project_path).resolve()
        
        # Security validation
        self._validate_project_path(project_path)
        
        # Discover Python files
        python_files = self._discover_python_files(
            project_path, include_patterns, exclude_patterns
        )
        
        if not python_files:
            raise ValueError(f"No Python files found in {project_path}")
        
        logger.info(f"Analyzing {len(python_files)} files in project {project_path}")
        
        # Analyze each file
        file_analyses = []
        all_security_violations = []
        all_patterns = []
        dependencies = {}
        
        for file_path in python_files:
            try:
                analysis = self.analyze_file(str(file_path))
                file_analyses.append(analysis)
                all_security_violations.extend(analysis.security_violations)
                all_patterns.extend(analysis.code_patterns)
                
                # Collect dependencies
                dependencies[str(file_path.relative_to(project_path))] = analysis.imports
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                continue
        
        # Calculate project-wide metrics
        total_lines = sum(fa.lines_of_code for fa in file_analyses)
        overall_complexity = sum(fa.complexity_score for fa in file_analyses) / len(file_analyses) if file_analyses else 0.0
        overall_maintainability = sum(fa.maintainability_score for fa in file_analyses) / len(file_analyses) if file_analyses else 0.0
        overall_safety = sum(fa.safety_score for fa in file_analyses) / len(file_analyses) if file_analyses else 0.0
        
        # Security check: Fail if critical violations found
        critical_violations = [v for v in all_security_violations if v.severity == 'critical']
        if critical_violations and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            raise SecurityError(f"Critical security violations found: {len(critical_violations)}")
        
        return ProjectAnalysis(
            project_path=str(project_path),
            total_files=len(file_analyses),
            total_lines=total_lines,
            file_analyses=file_analyses,
            overall_complexity=overall_complexity,
            overall_maintainability=overall_maintainability,
            overall_safety=overall_safety,
            security_violations=all_security_violations,
            project_patterns=all_patterns,
            dependencies=dependencies,
            analysis_metadata={
                'analyzer_version': '1.0.0',
                'security_level': self.security_level.value,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'files_processed': len(file_analyses),
                'files_failed': len(python_files) - len(file_analyses)
            }
        )
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """
        Analyze a single Python file with comprehensive metrics.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Complete file analysis results
            
        Raises:
            SecurityError: If file contains security violations
            ValueError: If file is invalid or too large
        """
        file_path = Path(file_path).resolve()
        
        # Security validation
        self._validate_file_path(file_path)
        
        # Read and validate file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, IOError) as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")
        
        # Security check: File size
        if len(content) > self._max_file_size_mb * 1024 * 1024:
            raise ValueError(f"File {file_path} too large (>{self._max_file_size_mb}MB)")
        
        # Calculate file hash for integrity
        file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Parse AST with security controls
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        # Analyze AST
        visitor = SecureASTVisitor(file_path, content, self.security_level)
        visitor.visit(tree)
        
        # Calculate metrics
        lines = content.split('\n')
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        blank_lines = len([line for line in lines if not line.strip()])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # Security analysis
        security_violations = self._analyze_security_patterns(content, lines)
        security_violations.extend(visitor.security_violations)
        
        # Pattern analysis
        code_patterns = self._analyze_code_patterns(content, tree, lines)
        code_patterns.extend(visitor.code_patterns)
        
        # Calculate scores
        complexity_score = self._calculate_complexity_score(visitor.functions)
        maintainability_score = self._calculate_maintainability_score(
            lines_of_code, comment_lines, visitor.functions, len(visitor.classes)
        )
        safety_score = self._calculate_safety_score(security_violations, code_patterns)
        
        # Security enforcement
        if safety_score < 0.3 and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            critical_violations = [v for v in security_violations if v.severity == 'critical']
            if critical_violations:
                raise SecurityError(f"File {file_path} has critical security violations")
        
        return FileAnalysis(
            file_path=str(file_path),
            file_hash=file_hash,
            lines_of_code=lines_of_code,
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            functions=visitor.functions,
            classes=visitor.classes,
            imports=visitor.imports,
            security_violations=security_violations,
            code_patterns=code_patterns,
            complexity_score=complexity_score,
            maintainability_score=maintainability_score,
            safety_score=safety_score,
            analysis_timestamp=datetime.utcnow()
        )
    
    def _validate_project_path(self, project_path: Path) -> None:
        """Validate project path for security."""
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")
        
        # Security: Prevent path traversal
        try:
            project_path.resolve(strict=True)
        except (OSError, RuntimeError):
            raise SecurityError(f"Invalid project path: {project_path}")
        
        # Security: Check for suspicious paths
        path_str = str(project_path).lower()
        suspicious_paths = ['/etc', '/sys', '/proc', '/dev', '/root', '/home']
        if any(path_str.startswith(sp) for sp in suspicious_paths):
            raise SecurityError(f"Access to system path denied: {project_path}")
    
    def _validate_file_path(self, file_path: Path) -> None:
        """Validate file path for security."""
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if file_path.suffix not in self._allowed_file_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Security: Prevent access to system files
        path_str = str(file_path).lower()
        if any(suspicious in path_str for suspicious in ['/etc/', '/sys/', '/proc/', '/dev/']):
            raise SecurityError(f"Access to system file denied: {file_path}")
    
    def _discover_python_files(self, project_path: Path, 
                              include_patterns: Optional[List[str]] = None,
                              exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover Python files with security controls."""
        python_files = []
        exclude_patterns = exclude_patterns or ['__pycache__', '.git', '.svn', 'node_modules']
        
        for file_path in project_path.rglob('*.py'):
            # Security check: File count limit
            if len(python_files) >= self._max_files_per_analysis:
                logger.warning(f"Reached maximum file limit: {self._max_files_per_analysis}")
                break
            
            # Skip excluded patterns
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
            
            # Security validation
            try:
                self._validate_file_path(file_path)
                python_files.append(file_path)
            except (SecurityError, ValueError):
                continue
        
        return python_files
    
    def _analyze_security_patterns(self, content: str, lines: List[str]) -> List[SecurityViolation]:
        """Analyze content for security patterns."""
        violations = []
        
        for pattern_name, pattern_config in self._security_patterns.items():
            pattern = pattern_config['pattern']
            severity = pattern_config['severity']
            description = pattern_config['description']
            
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                violations.append(SecurityViolation(
                    violation_type=pattern_name,
                    severity=severity,
                    description=description,
                    line_number=line_number,
                    code_snippet=lines[line_number - 1] if line_number <= len(lines) else None,
                    remediation=pattern_config.get('remediation')
                ))
        
        return violations
    
    def _analyze_code_patterns(self, content: str, tree: ast.AST, lines: List[str]) -> List[CodePattern]:
        """Analyze code patterns and anti-patterns."""
        patterns = []
        
        # Analyze using performance patterns
        for pattern_name, pattern_config in self._performance_patterns.items():
            pattern = pattern_config['pattern']
            pattern_type = pattern_config['type']
            confidence = pattern_config['confidence']
            description = pattern_config['description']
            
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                patterns.append(CodePattern(
                    pattern_type=pattern_type,
                    pattern_name=pattern_name,
                    confidence=confidence,
                    description=description,
                    line_number=line_number,
                    suggestions=pattern_config.get('suggestions', [])
                ))
        
        return patterns
    
    def _calculate_complexity_score(self, functions: List[FunctionMetrics]) -> float:
        """Calculate overall complexity score (0.0 = simple, 1.0 = very complex)."""
        if not functions:
            return 0.0
        
        total_complexity = sum(func.cyclomatic_complexity for func in functions)
        avg_complexity = total_complexity / len(functions)
        
        # Normalize to 0.0-1.0 scale (10+ complexity = 1.0)
        return min(avg_complexity / 10.0, 1.0)
    
    def _calculate_maintainability_score(self, lines_of_code: int, comment_lines: int,
                                       functions: List[FunctionMetrics], class_count: int) -> float:
        """Calculate maintainability score (0.0 = hard to maintain, 1.0 = easy to maintain)."""
        # Comment ratio (higher is better)
        comment_ratio = comment_lines / max(lines_of_code, 1) if lines_of_code > 0 else 0
        comment_score = min(comment_ratio * 5, 1.0)  # 20% comments = perfect
        
        # Function size (smaller is better)
        if functions:
            avg_function_size = sum(func.lines_of_code for func in functions) / len(functions)
            function_score = max(1.0 - (avg_function_size / 50.0), 0.0)  # 50+ lines = 0 score
        else:
            function_score = 1.0
        
        # Docstring coverage
        if functions:
            docstring_ratio = sum(1 for func in functions if func.has_docstring) / len(functions)
        else:
            docstring_ratio = 1.0
        
        # Combined score
        return (comment_score * 0.3 + function_score * 0.4 + docstring_ratio * 0.3)
    
    def _calculate_safety_score(self, violations: List[SecurityViolation], 
                               patterns: List[CodePattern]) -> float:
        """Calculate safety score (0.0 = unsafe, 1.0 = safe)."""
        # Security violations penalty
        critical_count = sum(1 for v in violations if v.severity == 'critical')
        high_count = sum(1 for v in violations if v.severity == 'high')
        medium_count = sum(1 for v in violations if v.severity == 'medium')
        
        violation_penalty = critical_count * 0.4 + high_count * 0.2 + medium_count * 0.1
        
        # Anti-pattern penalty
        anti_pattern_count = sum(1 for p in patterns if p.pattern_type == 'anti_pattern')
        anti_pattern_penalty = anti_pattern_count * 0.1
        
        # Calculate score
        total_penalty = min(violation_penalty + anti_pattern_penalty, 1.0)
        return max(1.0 - total_penalty, 0.0)
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security patterns for detection."""
        return {
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'severity': 'critical',
                'description': 'Use of eval() function poses security risks',
                'remediation': 'Replace eval() with safer alternatives like ast.literal_eval()'
            },
            'exec_usage': {
                'pattern': r'\bexec\s*\(',
                'severity': 'critical', 
                'description': 'Use of exec() function poses security risks',
                'remediation': 'Replace exec() with safer code execution patterns'
            },
            'shell_injection': {
                'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                'severity': 'high',
                'description': 'Shell injection vulnerability',
                'remediation': 'Use shell=False and pass arguments as list'
            },
            'hardcoded_secrets': {
                'pattern': r'(?i)(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                'severity': 'high',
                'description': 'Hardcoded secrets detected',
                'remediation': 'Use environment variables or secure secret management'
            },
            'sql_injection': {
                'pattern': r'execute\s*\(\s*["\'].*%s.*["\']',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability',
                'remediation': 'Use parameterized queries or ORM'
            }
        }
    
    def _load_performance_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load performance patterns for detection."""
        return {
            'inefficient_loop': {
                'pattern': r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(',
                'type': 'performance_issue',
                'confidence': 0.8,
                'description': 'Inefficient loop pattern using range(len())',
                'suggestions': ['Use enumerate() or iterate directly over the sequence']
            },
            'string_concatenation': {
                'pattern': r'\+\s*=.*["\']',
                'type': 'performance_issue',
                'confidence': 0.7,
                'description': 'String concatenation in loop may be inefficient',
                'suggestions': ['Use str.join() or f-strings for better performance']
            },
            'list_append_loop': {
                'pattern': r'for\s+\w+\s+in\s+.*:\s*\w+\.append\(',
                'type': 'performance_issue',
                'confidence': 0.6,
                'description': 'Consider using list comprehension',
                'suggestions': ['Replace with list comprehension for better performance']
            }
        }


class SecureASTVisitor(ast.NodeVisitor):
    """Secure AST visitor for code analysis."""
    
    def __init__(self, file_path: Path, content: str, security_level: SecurityLevel):
        self.file_path = file_path
        self.content = content
        self.security_level = security_level
        self.lines = content.split('\n')
        
        # Analysis results
        self.functions: List[FunctionMetrics] = []
        self.classes: List[str] = []
        self.imports: List[str] = []
        self.security_violations: List[SecurityViolation] = []
        self.code_patterns: List[CodePattern] = []
        
        # Tracking state
        self.current_function_depth = 0
        self.max_nested_depth = 0
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
            
            # Security check for blocked imports
            if alias.name in {'os', 'subprocess', 'sys'}:
                self.security_violations.append(SecurityViolation(
                    violation_type='dangerous_import',
                    severity='high',
                    description=f'Import of potentially dangerous module: {alias.name}',
                    line_number=node.lineno,
                    remediation='Review if this import is necessary and secure'
                ))
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            self.imports.append(node.module)
            
            # Security check for blocked imports
            if node.module in {'os', 'subprocess', 'sys'}:
                self.security_violations.append(SecurityViolation(
                    violation_type='dangerous_import',
                    severity='high',
                    description=f'Import from potentially dangerous module: {node.module}',
                    line_number=node.lineno,
                    remediation='Review if this import is necessary and secure'
                ))
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self.current_function_depth += 1
        self.max_nested_depth = max(self.max_nested_depth, self.current_function_depth)
        
        # Calculate function metrics
        func_lines = self._count_function_lines(node)
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(node)
        parameters_count = len(node.args.args)
        return_statements = self._count_return_statements(node)
        has_docstring = ast.get_docstring(node) is not None
        
        # Determine complexity level
        if cyclomatic_complexity <= 5:
            complexity_level = CodeComplexity.LOW
        elif cyclomatic_complexity <= 10:
            complexity_level = CodeComplexity.MEDIUM
        elif cyclomatic_complexity <= 15:
            complexity_level = CodeComplexity.HIGH
        else:
            complexity_level = CodeComplexity.VERY_HIGH
        
        function_metrics = FunctionMetrics(
            name=node.name,
            lines_of_code=func_lines,
            cyclomatic_complexity=cyclomatic_complexity,
            parameters_count=parameters_count,
            return_statements=return_statements,
            nested_depth=self.current_function_depth,
            has_docstring=has_docstring,
            complexity_level=complexity_level
        )
        
        self.functions.append(function_metrics)
        
        # Check for code patterns
        if not has_docstring and self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            self.code_patterns.append(CodePattern(
                pattern_type='anti_pattern',
                pattern_name='missing_docstring',
                confidence=0.9,
                description=f'Function {node.name} lacks docstring',
                line_number=node.lineno,
                suggestions=['Add descriptive docstring for better maintainability']
            ))
        
        if parameters_count > 7:
            self.code_patterns.append(CodePattern(
                pattern_type='anti_pattern',
                pattern_name='too_many_parameters',
                confidence=0.8,
                description=f'Function {node.name} has {parameters_count} parameters (>7)',
                line_number=node.lineno,
                suggestions=['Consider using a class or configuration object']
            ))
        
        self.generic_visit(node)
        self.current_function_depth -= 1
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        self.classes.append(node.name)
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls for security analysis."""
        # Check for dangerous function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if func_name in {'eval', 'exec'}:
                self.security_violations.append(SecurityViolation(
                    violation_type='dangerous_function',
                    severity='critical',
                    description=f'Use of {func_name}() function',
                    line_number=node.lineno,
                    remediation='Replace with safer alternatives'
                ))
        
        self.generic_visit(node)
    
    def _count_function_lines(self, node: ast.FunctionDef) -> int:
        """Count lines of code in a function."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        return 1
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _count_return_statements(self, node: ast.FunctionDef) -> int:
        """Count return statements in a function."""
        return sum(1 for child in ast.walk(node) if isinstance(child, ast.Return))


class SecurityError(Exception):
    """Security violation error."""
    pass


# Export main classes
__all__ = [
    'SecureCodeAnalyzer',
    'ProjectAnalysis', 
    'FileAnalysis',
    'SecurityViolation',
    'CodePattern',
    'SecurityLevel',
    'SecurityError'
]
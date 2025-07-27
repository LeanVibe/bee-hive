"""
Code Analysis Engine

Advanced AST parsing and codebase understanding for the self-modification engine.
Provides comprehensive code analysis, pattern detection, performance bottleneck
identification, and modification opportunity discovery.
"""

import ast
import os
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

import structlog

logger = structlog.get_logger()

# Optional imports for code analysis tools
try:
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit, h_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    logger.warning("Radon not available - complexity metrics will be simplified")

try:
    from bandit.core.manager import BanditManager
    from bandit.core.config import BanditConfig
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False
    logger.warning("Bandit not available - security analysis will be limited")


@dataclass
class CodeMetrics:
    """Comprehensive code metrics for a file or project."""
    
    # Basic metrics
    lines_of_code: int = 0
    lines_of_comments: int = 0
    blank_lines: int = 0
    
    # Complexity metrics
    cyclomatic_complexity: float = 0.0
    maintainability_index: float = 0.0
    halstead_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Code quality metrics
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    todo_count: int = 0
    
    # Security metrics
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance indicators
    potential_bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass  
class CodePattern:
    """Represents a detected code pattern."""
    
    pattern_type: str  # 'anti_pattern', 'performance_issue', 'security_issue', 'code_smell'
    pattern_name: str
    description: str
    file_path: str
    line_number: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0.0 to 1.0
    suggested_fix: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileAnalysis:
    """Complete analysis results for a single file."""
    
    file_path: str
    language: str
    metrics: CodeMetrics
    patterns: List[CodePattern] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    exports: Set[str] = field(default_factory=set)
    ast_tree: Optional[ast.AST] = None
    syntax_errors: List[str] = field(default_factory=list)


@dataclass
class ProjectAnalysis:
    """Complete analysis results for an entire project."""
    
    project_path: str
    files: Dict[str, FileAnalysis] = field(default_factory=dict)
    global_patterns: List[CodePattern] = field(default_factory=list)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    architecture_issues: List[Dict[str, Any]] = field(default_factory=list)
    overall_metrics: CodeMetrics = field(default_factory=CodeMetrics)
    
    @property
    def total_files(self) -> int:
        return len(self.files)
    
    @property
    def total_lines_of_code(self) -> int:
        return sum(analysis.metrics.lines_of_code for analysis in self.files.values())
    
    @property
    def average_complexity(self) -> float:
        complexities = [analysis.metrics.cyclomatic_complexity for analysis in self.files.values()]
        return sum(complexities) / len(complexities) if complexities else 0.0
    
    @property
    def critical_issues_count(self) -> int:
        return sum(
            len([p for p in analysis.patterns if p.severity == 'critical'])
            for analysis in self.files.values()
        )


class PythonASTAnalyzer:
    """Specialized AST analyzer for Python code."""
    
    def __init__(self):
        self.anti_patterns = self._load_anti_patterns()
        self.performance_patterns = self._load_performance_patterns()
    
    def analyze_file(self, file_path: str, content: str) -> FileAnalysis:
        """Analyze a single Python file."""
        try:
            tree = ast.parse(content, filename=file_path)
            
            metrics = self._calculate_metrics(content, tree)
            patterns = self._detect_patterns(tree, file_path, content)
            dependencies = self._extract_dependencies(tree)
            exports = self._extract_exports(tree)
            
            return FileAnalysis(
                file_path=file_path,
                language="python",
                metrics=metrics,
                patterns=patterns,
                dependencies=dependencies,
                exports=exports,
                ast_tree=tree,
                syntax_errors=[]
            )
            
        except SyntaxError as e:
            logger.warning("Syntax error in file", file_path=file_path, error=str(e))
            return FileAnalysis(
                file_path=file_path,
                language="python",
                metrics=CodeMetrics(),
                syntax_errors=[str(e)]
            )
    
    def _calculate_metrics(self, content: str, tree: ast.AST) -> CodeMetrics:
        """Calculate comprehensive code metrics."""
        lines = content.splitlines()
        
        # Basic line counts
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comments = len([line for line in lines if line.strip().startswith('#')])
        blanks = len([line for line in lines if not line.strip()])
        
        # Complexity metrics using radon (if available)
        if RADON_AVAILABLE:
            try:
                complexity_data = cc_visit(content)
                avg_complexity = sum(block.complexity for block in complexity_data) / len(complexity_data) if complexity_data else 0
                
                mi_data = mi_visit(content, multi=False)
                maintainability = mi_data if isinstance(mi_data, (int, float)) else 0
                
                halstead_data = h_visit(content)
                halstead_metrics = {
                    'vocabulary': halstead_data.vocabulary if halstead_data else 0,
                    'length': halstead_data.length if halstead_data else 0,
                    'difficulty': halstead_data.difficulty if halstead_data else 0,
                    'effort': halstead_data.effort if halstead_data else 0,
                }
            except Exception as e:
                logger.warning("Error calculating complexity metrics", error=str(e))
                avg_complexity = 0
                maintainability = 0
                halstead_metrics = {}
        else:
            # Simplified complexity calculation
            avg_complexity = self._calculate_simple_complexity(tree)
            maintainability = 100 - avg_complexity * 10  # Simple heuristic
            halstead_metrics = {}
        
        # AST-based counts
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        import_count = len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])
        
        # TODO count
        todo_count = len(re.findall(r'#\s*TODO', content, re.IGNORECASE))
        
        return CodeMetrics(
            lines_of_code=loc,
            lines_of_comments=comments,
            blank_lines=blanks,
            cyclomatic_complexity=avg_complexity,
            maintainability_index=maintainability,
            halstead_metrics=halstead_metrics,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            todo_count=todo_count
        )
    
    def _detect_patterns(self, tree: ast.AST, file_path: str, content: str) -> List[CodePattern]:
        """Detect code patterns, anti-patterns, and issues."""
        patterns = []
        
        # Anti-pattern detection
        patterns.extend(self._detect_anti_patterns(tree, file_path))
        
        # Performance issue detection
        patterns.extend(self._detect_performance_issues(tree, file_path))
        
        # Code smell detection
        patterns.extend(self._detect_code_smells(tree, file_path, content))
        
        # Security issue detection using Bandit
        patterns.extend(self._detect_security_issues(file_path, content))
        
        return patterns
    
    def _detect_anti_patterns(self, tree: ast.AST, file_path: str) -> List[CodePattern]:
        """Detect common anti-patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            # God class detection (too many methods)
            if isinstance(node, ast.ClassDef):
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                if method_count > 20:
                    patterns.append(CodePattern(
                        pattern_type="anti_pattern",
                        pattern_name="god_class",
                        description=f"Class '{node.name}' has {method_count} methods (threshold: 20)",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="medium",
                        confidence=0.8,
                        suggested_fix="Consider breaking this class into smaller, more focused classes"
                    ))
            
            # Long method detection
            if isinstance(node, ast.FunctionDef):
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    method_length = node.end_lineno - node.lineno
                    if method_length > 50:
                        patterns.append(CodePattern(
                            pattern_type="anti_pattern",
                            pattern_name="long_method",
                            description=f"Method '{node.name}' is {method_length} lines long (threshold: 50)",
                            file_path=file_path,
                            line_number=node.lineno,
                            severity="medium",
                            confidence=0.7,
                            suggested_fix="Consider breaking this method into smaller functions"
                        ))
                
                # Too many parameters
                param_count = len(node.args.args)
                if param_count > 6:
                    patterns.append(CodePattern(
                        pattern_type="anti_pattern",
                        pattern_name="too_many_parameters",
                        description=f"Method '{node.name}' has {param_count} parameters (threshold: 6)",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="low",
                        confidence=0.6,
                        suggested_fix="Consider using a parameter object or breaking the method down"
                    ))
        
        return patterns
    
    def _detect_performance_issues(self, tree: ast.AST, file_path: str) -> List[CodePattern]:
        """Detect potential performance issues."""
        patterns = []
        
        for node in ast.walk(tree):
            # Nested loops detection
            if isinstance(node, (ast.For, ast.While)):
                nested_loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)) and n != node]
                if len(nested_loops) >= 2:
                    patterns.append(CodePattern(
                        pattern_type="performance_issue",
                        pattern_name="nested_loops",
                        description=f"Found {len(nested_loops) + 1} nested loops",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="medium",
                        confidence=0.7,
                        suggested_fix="Consider algorithm optimization or vectorization"
                    ))
            
            # String concatenation in loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        if isinstance(child.target, ast.Name):
                            patterns.append(CodePattern(
                                pattern_type="performance_issue",
                                pattern_name="string_concatenation_in_loop",
                                description="String concatenation in loop detected",
                                file_path=file_path,
                                line_number=child.lineno,
                                severity="medium",
                                confidence=0.8,
                                suggested_fix="Use list comprehension and join() or StringBuilder pattern"
                            ))
            
            # Inefficient list operations
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['append', 'insert'] and isinstance(node.func.value, ast.Name):
                        # Check if this is in a loop
                        for parent in ast.walk(tree):
                            if isinstance(parent, (ast.For, ast.While)):
                                if any(n == node for n in ast.walk(parent)):
                                    patterns.append(CodePattern(
                                        pattern_type="performance_issue",
                                        pattern_name="list_append_in_loop",
                                        description="List append/insert in loop detected",
                                        file_path=file_path,
                                        line_number=node.lineno,
                                        severity="low",
                                        confidence=0.6,
                                        suggested_fix="Consider list comprehension or pre-allocation"
                                    ))
                                    break
        
        return patterns
    
    def _detect_code_smells(self, tree: ast.AST, file_path: str, content: str) -> List[CodePattern]:
        """Detect code smells."""
        patterns = []
        lines = content.splitlines()
        
        # Dead code detection (unreachable code after return)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return) and i < len(node.body) - 1:
                        patterns.append(CodePattern(
                            pattern_type="code_smell",
                            pattern_name="dead_code",
                            description="Unreachable code after return statement",
                            file_path=file_path,
                            line_number=stmt.lineno,
                            severity="low",
                            confidence=0.9,
                            suggested_fix="Remove unreachable code"
                        ))
        
        # Long lines detection
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                patterns.append(CodePattern(
                    pattern_type="code_smell",
                    pattern_name="long_line",
                    description=f"Line {i} is {len(line)} characters (threshold: 120)",
                    file_path=file_path,
                    line_number=i,
                    severity="low",
                    confidence=0.5,
                    suggested_fix="Break line into multiple lines"
                ))
        
        return patterns
    
    def _detect_security_issues(self, file_path: str, content: str) -> List[CodePattern]:
        """Detect security issues using Bandit (if available) or basic patterns."""
        patterns = []
        
        if BANDIT_AVAILABLE:
            try:
                config = BanditConfig()
                manager = BanditManager(config, "file")
                manager.discover_files([file_path])
                manager.run_tests()
                
                for issue in manager.get_issue_list():
                    patterns.append(CodePattern(
                        pattern_type="security_issue",
                        pattern_name=issue.test,
                        description=issue.text,
                        file_path=file_path,
                        line_number=issue.lineno,
                        severity=issue.severity.lower(),
                        confidence=issue.confidence.lower(),
                        suggested_fix=f"Review security implications: {issue.text}",
                        context={
                            'test_id': issue.test,
                            'cwe': getattr(issue, 'cwe', None),
                            'more_info': getattr(issue, 'more_info', None)
                        }
                    ))
            except Exception as e:
                logger.warning("Error running security analysis", file_path=file_path, error=str(e))
        else:
            # Basic security pattern detection
            patterns.extend(self._detect_basic_security_patterns(content, file_path))
        
        return patterns
    
    def _calculate_simple_complexity(self, tree: ast.AST) -> float:
        """Calculate simplified complexity score when radon is not available."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            # Count logical operators
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                complexity += 1
            # Count exception handlers
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
                
        return complexity / max(1, len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]))
    
    def _detect_basic_security_patterns(self, content: str, file_path: str) -> List[CodePattern]:
        """Basic security pattern detection when Bandit is not available."""
        patterns = []
        lines = content.splitlines()
        
        # Common security anti-patterns
        security_patterns = [
            (r'eval\s*\(', 'eval_usage', 'Use of eval() function can be dangerous'),
            (r'exec\s*\(', 'exec_usage', 'Use of exec() function can be dangerous'),
            (r'os\.system\s*\(', 'system_usage', 'Use of os.system() can lead to shell injection'),
            (r'subprocess\.call.*shell=True', 'shell_injection', 'subprocess with shell=True is dangerous'),
            (r'pickle\.loads?\s*\(', 'pickle_usage', 'Pickle deserialization can be unsafe'),
            (r'random\.random\s*\(', 'weak_random', 'Use secrets module for cryptographic randomness'),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, name, description in security_patterns:
                if re.search(pattern, line):
                    patterns.append(CodePattern(
                        pattern_type="security_issue",
                        pattern_name=name,
                        description=description,
                        file_path=file_path,
                        line_number=i,
                        severity="medium",
                        confidence=0.6,
                        suggested_fix=f"Review security implications of {name.replace('_', ' ')}"
                    ))
        
        return patterns
    
    def _extract_dependencies(self, tree: ast.AST) -> Set[str]:
        """Extract import dependencies."""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        return dependencies
    
    def _extract_exports(self, tree: ast.AST) -> Set[str]:
        """Extract public exports (functions, classes, variables)."""
        exports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):
                    exports.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('_'):
                        exports.add(target.id)
        
        return exports
    
    def _load_anti_patterns(self) -> Dict[str, Any]:
        """Load anti-pattern definitions."""
        return {
            'god_class': {'max_methods': 20, 'severity': 'medium'},
            'long_method': {'max_lines': 50, 'severity': 'medium'},
            'too_many_parameters': {'max_params': 6, 'severity': 'low'},
        }
    
    def _load_performance_patterns(self) -> Dict[str, Any]:
        """Load performance pattern definitions."""
        return {
            'nested_loops': {'max_depth': 3, 'severity': 'medium'},
            'string_concatenation': {'in_loop': True, 'severity': 'medium'},
            'inefficient_operations': {'patterns': ['list.append', 'dict.update'], 'severity': 'low'},
        }


class CodeAnalysisEngine:
    """Main code analysis engine for the self-modification system."""
    
    def __init__(self):
        self.python_analyzer = PythonASTAnalyzer()
        self.supported_languages = {'python': self.python_analyzer}
        self.file_extensions = {
            '.py': 'python',
            '.pyx': 'python',
            '.pyi': 'python'
        }
    
    def analyze_project(
        self, 
        project_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> ProjectAnalysis:
        """Analyze an entire project/codebase."""
        logger.info("Starting project analysis", project_path=project_path)
        
        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # Discover files
        files_to_analyze = self._discover_files(
            project_path_obj, include_patterns, exclude_patterns
        )
        
        # Analyze each file
        analysis = ProjectAnalysis(project_path=project_path)
        
        for file_path in files_to_analyze:
            try:
                file_analysis = self._analyze_single_file(file_path)
                if file_analysis:
                    analysis.files[str(file_path)] = file_analysis
                    
            except Exception as e:
                logger.error("Failed to analyze file", file_path=str(file_path), error=str(e))
        
        # Global analysis
        analysis.dependency_graph = self._build_dependency_graph(analysis.files)
        analysis.architecture_issues = self._detect_architecture_issues(analysis)
        analysis.overall_metrics = self._calculate_overall_metrics(analysis.files)
        analysis.global_patterns = self._detect_global_patterns(analysis)
        
        logger.info(
            "Project analysis completed",
            total_files=analysis.total_files,
            total_loc=analysis.total_lines_of_code,
            critical_issues=analysis.critical_issues_count
        )
        
        return analysis
    
    def _discover_files(
        self,
        project_path: Path,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[Path]:
        """Discover files to analyze based on patterns."""
        files = []
        
        # Default exclude patterns
        default_excludes = [
            '*/.*', '*/__pycache__/*', '*/node_modules/*', '*/venv/*', 
            '*/env/*', '*.pyc', '*.pyo', '*/migrations/*', '*/alembic/*'
        ]
        exclude_patterns = (exclude_patterns or []) + default_excludes
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                # Check include patterns
                if include_patterns:
                    if not any(file_path.match(pattern) for pattern in include_patterns):
                        continue
                
                # Check exclude patterns
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue
                
                files.append(file_path)
        
        return files
    
    def _analyze_single_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Analyze a single file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            language = self.file_extensions.get(file_path.suffix)
            
            if language in self.supported_languages:
                analyzer = self.supported_languages[language]
                return analyzer.analyze_file(str(file_path), content)
            
        except Exception as e:
            logger.warning("Failed to read file", file_path=str(file_path), error=str(e))
        
        return None
    
    def _build_dependency_graph(self, files: Dict[str, FileAnalysis]) -> Dict[str, Set[str]]:
        """Build dependency graph between files."""
        graph = defaultdict(set)
        
        for file_path, analysis in files.items():
            for dependency in analysis.dependencies:
                # Try to resolve dependency to actual files in project
                for other_file, other_analysis in files.items():
                    if dependency in other_analysis.exports:
                        graph[file_path].add(other_file)
                        break
        
        return dict(graph)
    
    def _detect_architecture_issues(self, analysis: ProjectAnalysis) -> List[Dict[str, Any]]:
        """Detect architecture-level issues."""
        issues = []
        
        # Circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in analysis.dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for file_path in analysis.files:
            if file_path not in visited:
                if has_cycle(file_path):
                    issues.append({
                        'type': 'circular_dependency',
                        'description': 'Circular dependency detected',
                        'severity': 'high',
                        'files_involved': list(rec_stack)
                    })
        
        # God modules (files with too many dependencies)
        for file_path, dependencies in analysis.dependency_graph.items():
            if len(dependencies) > 20:
                issues.append({
                    'type': 'god_module',
                    'description': f'File {file_path} has {len(dependencies)} dependencies',
                    'severity': 'medium',
                    'file_path': file_path,
                    'dependency_count': len(dependencies)
                })
        
        return issues
    
    def _calculate_overall_metrics(self, files: Dict[str, FileAnalysis]) -> CodeMetrics:
        """Calculate overall project metrics."""
        overall = CodeMetrics()
        
        for analysis in files.values():
            overall.lines_of_code += analysis.metrics.lines_of_code
            overall.lines_of_comments += analysis.metrics.lines_of_comments
            overall.blank_lines += analysis.metrics.blank_lines
            overall.function_count += analysis.metrics.function_count
            overall.class_count += analysis.metrics.class_count
            overall.import_count += analysis.metrics.import_count
            overall.todo_count += analysis.metrics.todo_count
            
            # Collect all security issues
            for pattern in analysis.patterns:
                if pattern.pattern_type == 'security_issue':
                    overall.security_issues.append({
                        'file_path': analysis.file_path,
                        'pattern': pattern.pattern_name,
                        'description': pattern.description,
                        'severity': pattern.severity,
                        'line_number': pattern.line_number
                    })
        
        # Calculate averages
        file_count = len(files)
        if file_count > 0:
            complexities = [f.metrics.cyclomatic_complexity for f in files.values()]
            overall.cyclomatic_complexity = sum(complexities) / len(complexities)
            
            maintainabilities = [f.metrics.maintainability_index for f in files.values()]
            overall.maintainability_index = sum(maintainabilities) / len(maintainabilities)
        
        return overall
    
    def _detect_global_patterns(self, analysis: ProjectAnalysis) -> List[CodePattern]:
        """Detect global patterns across the entire project."""
        patterns = []
        
        # Duplicate code detection (simplified)
        function_signatures = defaultdict(list)
        
        for file_path, file_analysis in analysis.files.items():
            if file_analysis.ast_tree:
                for node in ast.walk(file_analysis.ast_tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a simple signature based on function name and parameter count
                        signature = f"{node.name}_{len(node.args.args)}"
                        function_signatures[signature].append((file_path, node.lineno, node.name))
        
        # Report potential duplicates
        for signature, locations in function_signatures.items():
            if len(locations) > 1:
                patterns.append(CodePattern(
                    pattern_type="code_smell",
                    pattern_name="potential_duplicate_function",
                    description=f"Function signature '{signature}' appears in {len(locations)} files",
                    file_path=locations[0][0],
                    line_number=locations[0][1],
                    severity="low",
                    confidence=0.4,
                    suggested_fix="Review for potential code duplication",
                    context={'all_locations': locations}
                ))
        
        return patterns
    
    def generate_modification_opportunities(self, analysis: ProjectAnalysis) -> List[Dict[str, Any]]:
        """Generate modification opportunities based on analysis."""
        opportunities = []
        
        # Performance opportunities
        for file_path, file_analysis in analysis.files.items():
            for pattern in file_analysis.patterns:
                if pattern.pattern_type == "performance_issue":
                    opportunities.append({
                        'type': 'performance',
                        'priority': self._calculate_priority(pattern),
                        'file_path': file_path,
                        'line_number': pattern.line_number,
                        'description': pattern.description,
                        'suggested_fix': pattern.suggested_fix,
                        'confidence': pattern.confidence,
                        'estimated_impact': self._estimate_performance_impact(pattern)
                    })
        
        # Code quality opportunities
        for file_path, file_analysis in analysis.files.items():
            for pattern in file_analysis.patterns:
                if pattern.pattern_type in ["anti_pattern", "code_smell"]:
                    opportunities.append({
                        'type': 'quality',
                        'priority': self._calculate_priority(pattern),
                        'file_path': file_path,
                        'line_number': pattern.line_number,
                        'description': pattern.description,
                        'suggested_fix': pattern.suggested_fix,
                        'confidence': pattern.confidence,
                        'maintainability_impact': self._estimate_maintainability_impact(pattern)
                    })
        
        # Security opportunities
        for file_path, file_analysis in analysis.files.items():
            for pattern in file_analysis.patterns:
                if pattern.pattern_type == "security_issue":
                    opportunities.append({
                        'type': 'security',
                        'priority': 'high',  # Security issues are always high priority
                        'file_path': file_path,
                        'line_number': pattern.line_number,
                        'description': pattern.description,
                        'suggested_fix': pattern.suggested_fix,
                        'confidence': pattern.confidence,
                        'security_impact': pattern.severity
                    })
        
        # Sort by priority and confidence
        opportunities.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}.get(x['priority'], 0),
            x['confidence']
        ), reverse=True)
        
        return opportunities
    
    def _calculate_priority(self, pattern: CodePattern) -> str:
        """Calculate priority based on pattern severity and confidence."""
        severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        severity_weight = severity_weights.get(pattern.severity, 1)
        
        if pattern.confidence >= 0.8 and severity_weight >= 3:
            return 'high'
        elif pattern.confidence >= 0.6 and severity_weight >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_performance_impact(self, pattern: CodePattern) -> str:
        """Estimate potential performance improvement."""
        impact_map = {
            'nested_loops': 'high',
            'string_concatenation_in_loop': 'medium',
            'list_append_in_loop': 'low',
            'inefficient_algorithm': 'high'
        }
        return impact_map.get(pattern.pattern_name, 'low')
    
    def _estimate_maintainability_impact(self, pattern: CodePattern) -> str:
        """Estimate maintainability improvement."""
        impact_map = {
            'god_class': 'high',
            'long_method': 'medium',
            'too_many_parameters': 'medium',
            'dead_code': 'low',
            'long_line': 'low'
        }
        return impact_map.get(pattern.pattern_name, 'low')


# Export main class
__all__ = ["CodeAnalysisEngine", "ProjectAnalysis", "FileAnalysis", "CodePattern", "CodeMetrics"]
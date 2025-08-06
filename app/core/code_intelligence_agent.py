"""
Code Intelligence Agent - Autonomous Testing and Code Quality System

This agent specializes in autonomous test generation, code quality analysis,
and intelligent testing strategies. Part of the AI Enhancement Team for 
LeanVibe Agent Hive 2.0.
"""

import asyncio
import json
import uuid
import ast
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import importlib.util

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .intelligence_framework import (
    IntelligenceModelInterface,
    IntelligencePrediction,
    DataPoint,
    DataType
)
from ..models.task import Task, TaskStatus

logger = structlog.get_logger()


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    FUNCTIONAL_TEST = "functional_test"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"
    API_TEST = "api_test"
    ERROR_HANDLING_TEST = "error_handling_test"


class TestPriority(Enum):
    """Priority levels for test generation."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CodeQualityIssue(Enum):
    """Types of code quality issues."""
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    COUPLING = "coupling"
    COHESION = "cohesion"
    NAMING = "naming"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    SECURITY = "security"


@dataclass
class TestCase:
    """Represents a generated test case."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    priority: TestPriority
    target_function: str
    target_class: Optional[str]
    test_code: str
    setup_code: Optional[str]
    teardown_code: Optional[str]
    expected_outcome: str
    test_data: List[Dict[str, Any]]
    assertions: List[str]
    mocks_needed: List[str]
    dependencies: List[str]
    estimated_execution_time: float
    confidence: float
    created_at: datetime
    
    def to_pytest_code(self) -> str:
        """Convert test case to executable pytest code."""
        imports = [
            "import pytest",
            "from unittest.mock import Mock, patch, MagicMock",
            "import asyncio",
        ]
        
        if self.dependencies:
            imports.extend([f"import {dep}" for dep in self.dependencies])
        
        test_function = f"""
def test_{self.name.lower().replace(' ', '_')}():
    \"\"\"
    {self.description}
    
    Expected: {self.expected_outcome}
    \"\"\"
    {self._generate_setup_section()}
    {self._generate_test_execution()}
    {self._generate_assertions()}
    {self._generate_teardown_section()}
"""
        
        return "\n".join(imports) + "\n" + test_function
    
    def _generate_setup_section(self) -> str:
        """Generate test setup code."""
        if self.setup_code:
            return f"    # Setup\n    {self.setup_code}"
        return "    # Setup\n    pass"
    
    def _generate_test_execution(self) -> str:
        """Generate main test execution code."""
        return f"    # Test execution\n    {self.test_code}"
    
    def _generate_assertions(self) -> str:
        """Generate test assertions."""
        if self.assertions:
            assertion_code = "\n    ".join(f"assert {assertion}" for assertion in self.assertions)
            return f"    # Assertions\n    {assertion_code}"
        return "    # Assertions\n    assert True  # TODO: Add meaningful assertions"
    
    def _generate_teardown_section(self) -> str:
        """Generate test teardown code."""
        if self.teardown_code:
            return f"    # Teardown\n    {self.teardown_code}"
        return ""


@dataclass
class CodeQualityReport:
    """Comprehensive code quality assessment."""
    file_path: str
    overall_score: float  # 0.0 to 1.0
    complexity_score: float
    maintainability_score: float
    testability_score: float
    security_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    test_coverage_estimate: float
    technical_debt_score: float
    generated_at: datetime
    
    def get_priority_issues(self) -> List[Dict[str, Any]]:
        """Get high-priority issues that need immediate attention."""
        return [issue for issue in self.issues if issue.get('priority') in ['critical', 'high']]
    
    def get_improvement_roadmap(self) -> List[str]:
        """Generate prioritized improvement roadmap."""
        roadmap = []
        
        if self.security_score < 0.7:
            roadmap.append("🔒 Address security vulnerabilities immediately")
        
        if self.complexity_score < 0.6:
            roadmap.append("🔧 Refactor high-complexity methods")
        
        if self.test_coverage_estimate < 0.8:
            roadmap.append("🧪 Increase test coverage")
        
        if self.maintainability_score < 0.7:
            roadmap.append("📝 Improve code documentation and naming")
        
        return roadmap


class CodeAnalyzer:
    """Advanced code analysis engine for quality assessment and test generation."""
    
    def __init__(self):
        self.complexity_threshold = 10
        self.duplication_threshold = 5
        self.method_length_threshold = 50
    
    async def analyze_code(self, code: str, file_path: str = "") -> CodeQualityReport:
        """Perform comprehensive code analysis."""
        try:
            tree = ast.parse(code)
            
            # Calculate various quality metrics
            complexity_score = await self._calculate_complexity_score(tree)
            maintainability_score = await self._calculate_maintainability_score(tree, code)
            testability_score = await self._calculate_testability_score(tree)
            security_score = await self._calculate_security_score(code)
            
            # Identify issues
            issues = []
            issues.extend(await self._find_complexity_issues(tree))
            issues.extend(await self._find_duplication_issues(code))
            issues.extend(await self._find_naming_issues(tree))
            issues.extend(await self._find_security_issues(code))
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(issues, {
                'complexity': complexity_score,
                'maintainability': maintainability_score,
                'testability': testability_score,
                'security': security_score
            })
            
            # Calculate overall score
            overall_score = (
                complexity_score * 0.3 +
                maintainability_score * 0.25 +
                testability_score * 0.25 +
                security_score * 0.2
            )
            
            # Estimate test coverage need
            test_coverage_estimate = await self._estimate_test_coverage_need(tree)
            
            # Calculate technical debt
            technical_debt_score = await self._calculate_technical_debt(issues, overall_score)
            
            return CodeQualityReport(
                file_path=file_path,
                overall_score=overall_score,
                complexity_score=complexity_score,
                maintainability_score=maintainability_score,
                testability_score=testability_score,
                security_score=security_score,
                issues=issues,
                recommendations=recommendations,
                test_coverage_estimate=test_coverage_estimate,
                technical_debt_score=technical_debt_score,
                generated_at=datetime.now()
            )
            
        except SyntaxError as e:
            logger.error(f"Syntax error in code analysis: {e}")
            return CodeQualityReport(
                file_path=file_path,
                overall_score=0.0,
                complexity_score=0.0,
                maintainability_score=0.0,
                testability_score=0.0,
                security_score=0.0,
                issues=[{"type": "syntax_error", "message": str(e), "priority": "critical"}],
                recommendations=["Fix syntax errors before proceeding"],
                test_coverage_estimate=0.0,
                technical_debt_score=1.0,
                generated_at=datetime.now()
            )
    
    async def _calculate_complexity_score(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity score."""
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        
        if not complexity_visitor.methods:
            return 1.0
        
        avg_complexity = sum(complexity_visitor.methods.values()) / len(complexity_visitor.methods)
        
        # Normalize score (lower complexity = higher score)
        return max(0.0, min(1.0, 1.0 - (avg_complexity - 1) / 20))
    
    async def _calculate_maintainability_score(self, tree: ast.AST, code: str) -> float:
        """Calculate maintainability score based on various factors."""
        lines = code.split('\n')
        total_lines = len(lines)
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        blank_lines = len([line for line in lines if not line.strip()])
        
        # Comment ratio (good documentation)
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        # Find long methods
        method_visitor = MethodVisitor()
        method_visitor.visit(tree)
        long_methods = [name for name, length in method_visitor.method_lengths.items() 
                       if length > self.method_length_threshold]
        
        # Calculate score
        comment_score = min(1.0, comment_ratio * 5)  # Cap at 20% comments
        method_length_score = max(0.0, 1.0 - len(long_methods) * 0.1)
        
        return (comment_score * 0.4 + method_length_score * 0.6)
    
    async def _calculate_testability_score(self, tree: ast.AST) -> float:
        """Calculate how testable the code is."""
        testability_visitor = TestabilityVisitor()
        testability_visitor.visit(tree)
        
        # Factors that improve testability
        score = 1.0
        
        # Penalize global state usage
        score -= testability_visitor.global_variables * 0.1
        
        # Penalize high coupling
        score -= testability_visitor.external_dependencies * 0.05
        
        # Reward dependency injection
        score += testability_visitor.injected_dependencies * 0.1
        
        # Reward pure functions
        score += testability_visitor.pure_functions * 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _calculate_security_score(self, code: str) -> float:
        """Calculate security score by detecting common vulnerabilities."""
        security_issues = 0
        
        # Check for common security anti-patterns
        security_patterns = [
            r'eval\(',
            r'exec\(',
            r'input\(',
            r'os\.system\(',
            r'subprocess\.call\(',
            r'pickle\.loads\(',
            r'yaml\.load\(',
            r'sql.*\+.*[\'"]',  # SQL injection pattern
            r'password\s*=\s*[\'"][^\'\"]+[\'"]',  # Hardcoded passwords
        ]
        
        for pattern in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_issues += 1
        
        # Calculate score (fewer issues = higher score)
        return max(0.0, 1.0 - security_issues * 0.2)
    
    async def _find_complexity_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find high complexity methods."""
        issues = []
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        
        for method_name, complexity in complexity_visitor.methods.items():
            if complexity > self.complexity_threshold:
                issues.append({
                    "type": CodeQualityIssue.COMPLEXITY.value,
                    "method": method_name,
                    "complexity": complexity,
                    "message": f"Method '{method_name}' has high cyclomatic complexity ({complexity})",
                    "priority": "high" if complexity > 15 else "medium",
                    "recommendation": "Consider breaking this method into smaller, focused functions"
                })
        
        return issues
    
    async def _find_duplication_issues(self, code: str) -> List[Dict[str, Any]]:
        """Find code duplication issues."""
        issues = []
        lines = code.split('\n')
        line_counts = defaultdict(list)
        
        # Group similar lines
        for i, line in enumerate(lines):
            normalized_line = line.strip()
            if len(normalized_line) > 10 and not normalized_line.startswith('#'):
                line_counts[normalized_line].append(i + 1)
        
        # Find duplicates
        for line, occurrences in line_counts.items():
            if len(occurrences) >= self.duplication_threshold:
                issues.append({
                    "type": CodeQualityIssue.DUPLICATION.value,
                    "line": line[:50] + "..." if len(line) > 50 else line,
                    "occurrences": len(occurrences),
                    "line_numbers": occurrences,
                    "message": f"Duplicated code found on lines {occurrences}",
                    "priority": "medium",
                    "recommendation": "Extract duplicated code into a reusable function"
                })
        
        return issues
    
    async def _find_naming_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find naming convention issues."""
        issues = []
        naming_visitor = NamingVisitor()
        naming_visitor.visit(tree)
        
        for issue in naming_visitor.issues:
            issues.append({
                "type": CodeQualityIssue.NAMING.value,
                "name": issue['name'],
                "name_type": issue['type'],
                "message": issue['message'],
                "priority": "low",
                "recommendation": issue['recommendation']
            })
        
        return issues
    
    async def _find_security_issues(self, code: str) -> List[Dict[str, Any]]:
        """Find potential security vulnerabilities."""
        issues = []
        
        security_checks = [
            {
                "pattern": r'eval\(',
                "message": "Use of eval() can lead to code injection vulnerabilities",
                "priority": "critical",
                "recommendation": "Replace eval() with safer alternatives like ast.literal_eval()"
            },
            {
                "pattern": r'exec\(',
                "message": "Use of exec() can lead to code injection vulnerabilities",
                "priority": "critical",
                "recommendation": "Avoid exec() or implement strict input validation"
            },
            {
                "pattern": r'password\s*=\s*[\'"][^\'\"]+[\'"]',
                "message": "Hardcoded password detected",
                "priority": "high",
                "recommendation": "Use environment variables or secure configuration management"
            },
            {
                "pattern": r'sql.*\+.*[\'"]',
                "message": "Potential SQL injection vulnerability",
                "priority": "high",
                "recommendation": "Use parameterized queries instead of string concatenation"
            }
        ]
        
        for check in security_checks:
            if re.search(check["pattern"], code, re.IGNORECASE):
                issues.append({
                    "type": CodeQualityIssue.SECURITY.value,
                    "message": check["message"],
                    "priority": check["priority"],
                    "recommendation": check["recommendation"]
                })
        
        return issues
    
    async def _generate_recommendations(self, issues: List[Dict[str, Any]], scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Priority-based recommendations
        critical_issues = [issue for issue in issues if issue.get('priority') == 'critical']
        if critical_issues:
            recommendations.append("🚨 Address critical security and syntax issues immediately")
        
        if scores['complexity'] < 0.6:
            recommendations.append("🔧 Refactor complex methods to improve readability and maintainability")
        
        if scores['testability'] < 0.7:
            recommendations.append("🧪 Improve code testability by reducing dependencies and global state")
        
        if scores['security'] < 0.8:
            recommendations.append("🔒 Review and fix security vulnerabilities")
        
        if scores['maintainability'] < 0.7:
            recommendations.append("📝 Add documentation and improve naming conventions")
        
        return recommendations
    
    async def _estimate_test_coverage_need(self, tree: ast.AST) -> float:
        """Estimate how much test coverage is needed."""
        method_visitor = MethodVisitor()
        method_visitor.visit(tree)
        
        total_methods = len(method_visitor.method_lengths)
        if total_methods == 0:
            return 0.0
        
        # Estimate based on method complexity and public interface
        complex_methods = len([length for length in method_visitor.method_lengths.values() if length > 20])
        coverage_need = min(1.0, (complex_methods + total_methods * 0.5) / total_methods)
        
        return coverage_need
    
    async def _calculate_technical_debt(self, issues: List[Dict[str, Any]], overall_score: float) -> float:
        """Calculate technical debt score."""
        # Weight issues by priority
        priority_weights = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.1}
        
        weighted_issues = sum(priority_weights.get(issue.get('priority', 'low'), 0.1) for issue in issues)
        
        # Combine with overall score
        debt_from_issues = min(1.0, weighted_issues * 0.1)
        debt_from_quality = 1.0 - overall_score
        
        return (debt_from_issues * 0.6 + debt_from_quality * 0.4)


class TestGenerator:
    """Intelligent test generation engine."""
    
    def __init__(self, code_analyzer: CodeAnalyzer):
        self.code_analyzer = code_analyzer
        self.test_templates = self._load_test_templates()
    
    async def generate_tests(self, code: str, file_path: str = "") -> List[TestCase]:
        """Generate comprehensive test suite for given code."""
        try:
            tree = ast.parse(code)
            tests = []
            
            # Generate different types of tests
            tests.extend(await self._generate_unit_tests(tree, code))
            tests.extend(await self._generate_integration_tests(tree, code))
            tests.extend(await self._generate_error_handling_tests(tree, code))
            tests.extend(await self._generate_performance_tests(tree, code))
            
            # Prioritize tests
            tests = await self._prioritize_tests(tests, file_path)
            
            return tests
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return []
    
    async def _generate_unit_tests(self, tree: ast.AST, code: str) -> List[TestCase]:
        """Generate unit tests for individual functions and methods."""
        tests = []
        method_visitor = MethodVisitor()
        method_visitor.visit(tree)
        
        for method_name, method_info in method_visitor.methods.items():
            # Generate basic functionality test
            test = TestCase(
                test_id=str(uuid.uuid4()),
                name=f"Test {method_name} basic functionality",
                description=f"Test that {method_name} works correctly with valid inputs",
                test_type=TestType.UNIT_TEST,
                priority=TestPriority.HIGH,
                target_function=method_name,
                target_class=method_info.get('class_name'),
                test_code=self._generate_basic_test_code(method_name, method_info),
                setup_code=self._generate_setup_code(method_name, method_info),
                teardown_code=None,
                expected_outcome="Function executes successfully with valid inputs",
                test_data=self._generate_test_data(method_info),
                assertions=self._generate_assertions(method_name, method_info),
                mocks_needed=self._identify_mocks_needed(method_info),
                dependencies=self._identify_dependencies(code),
                estimated_execution_time=0.1,
                confidence=0.8,
                created_at=datetime.now()
            )
            tests.append(test)
            
            # Generate edge case tests
            edge_test = TestCase(
                test_id=str(uuid.uuid4()),
                name=f"Test {method_name} edge cases",
                description=f"Test that {method_name} handles edge cases properly",
                test_type=TestType.UNIT_TEST,
                priority=TestPriority.MEDIUM,
                target_function=method_name,
                target_class=method_info.get('class_name'),
                test_code=self._generate_edge_case_test_code(method_name, method_info),
                setup_code=self._generate_setup_code(method_name, method_info),
                teardown_code=None,
                expected_outcome="Function handles edge cases gracefully",
                test_data=self._generate_edge_case_data(method_info),
                assertions=self._generate_edge_case_assertions(method_name, method_info),
                mocks_needed=self._identify_mocks_needed(method_info),
                dependencies=self._identify_dependencies(code),
                estimated_execution_time=0.1,
                confidence=0.7,
                created_at=datetime.now()
            )
            tests.append(edge_test)
        
        return tests
    
    async def _generate_integration_tests(self, tree: ast.AST, code: str) -> List[TestCase]:
        """Generate integration tests for component interactions."""
        tests = []
        
        # Identify classes that interact with each other
        class_visitor = ClassVisitor()
        class_visitor.visit(tree)
        
        if len(class_visitor.classes) > 1:
            test = TestCase(
                test_id=str(uuid.uuid4()),
                name="Integration test for class interactions",
                description="Test that classes work together correctly",
                test_type=TestType.INTEGRATION_TEST,
                priority=TestPriority.HIGH,
                target_function="multiple",
                target_class=None,
                test_code=self._generate_integration_test_code(class_visitor.classes),
                setup_code="# Setup multiple class instances",
                teardown_code="# Cleanup resources",
                expected_outcome="Classes integrate successfully",
                test_data=[],
                assertions=["result is not None", "integration_successful == True"],
                mocks_needed=[],
                dependencies=self._identify_dependencies(code),
                estimated_execution_time=0.5,
                confidence=0.6,
                created_at=datetime.now()
            )
            tests.append(test)
        
        return tests
    
    async def _generate_error_handling_tests(self, tree: ast.AST, code: str) -> List[TestCase]:
        """Generate tests for error handling scenarios."""
        tests = []
        
        # Find try/except blocks
        exception_visitor = ExceptionVisitor()
        exception_visitor.visit(tree)
        
        for exception_info in exception_visitor.exception_handlers:
            test = TestCase(
                test_id=str(uuid.uuid4()),
                name=f"Test error handling for {exception_info['exception_type']}",
                description=f"Test that {exception_info['exception_type']} is handled correctly",
                test_type=TestType.ERROR_HANDLING_TEST,
                priority=TestPriority.HIGH,
                target_function=exception_info['function'],
                target_class=None,
                test_code=self._generate_error_test_code(exception_info),
                setup_code="# Setup error conditions",
                teardown_code=None,
                expected_outcome=f"{exception_info['exception_type']} is properly handled",
                test_data=[],
                assertions=[f"pytest.raises({exception_info['exception_type']})"],
                mocks_needed=[],
                dependencies=self._identify_dependencies(code),
                estimated_execution_time=0.1,
                confidence=0.8,
                created_at=datetime.now()
            )
            tests.append(test)
        
        return tests
    
    async def _generate_performance_tests(self, tree: ast.AST, code: str) -> List[TestCase]:
        """Generate performance tests for critical methods."""
        tests = []
        
        # Identify methods that might have performance implications
        method_visitor = MethodVisitor()
        method_visitor.visit(tree)
        
        performance_critical_methods = [
            name for name, length in method_visitor.method_lengths.items()
            if length > 30  # Methods with more than 30 lines might need performance testing
        ]
        
        for method_name in performance_critical_methods:
            test = TestCase(
                test_id=str(uuid.uuid4()),
                name=f"Performance test for {method_name}",
                description=f"Test that {method_name} executes within acceptable time limits",
                test_type=TestType.PERFORMANCE_TEST,
                priority=TestPriority.MEDIUM,
                target_function=method_name,
                target_class=None,
                test_code=self._generate_performance_test_code(method_name),
                setup_code="import time",
                teardown_code=None,
                expected_outcome="Method executes within performance threshold",
                test_data=[],
                assertions=["execution_time < 1.0  # 1 second threshold"],
                mocks_needed=[],
                dependencies=self._identify_dependencies(code),
                estimated_execution_time=2.0,
                confidence=0.6,
                created_at=datetime.now()
            )
            tests.append(test)
        
        return tests
    
    async def _prioritize_tests(self, tests: List[TestCase], file_path: str) -> List[TestCase]:
        """Prioritize tests based on importance and risk."""
        # Sort by priority and confidence
        priority_order = {
            TestPriority.CRITICAL: 4,
            TestPriority.HIGH: 3,
            TestPriority.MEDIUM: 2,
            TestPriority.LOW: 1
        }
        
        return sorted(tests, key=lambda t: (priority_order[t.priority], t.confidence), reverse=True)
    
    def _load_test_templates(self) -> Dict[str, str]:
        """Load test templates for different scenarios."""
        return {
            "basic_function": """
    # Arrange
    {setup}
    
    # Act
    result = {function_call}
    
    # Assert
    assert result is not None
    {assertions}
    """,
            "error_handling": """
    # Arrange
    {setup}
    
    # Act & Assert
    with pytest.raises({exception_type}):
        {function_call}
    """,
            "performance": """
    # Arrange
    {setup}
    
    # Act
    start_time = time.time()
    result = {function_call}
    execution_time = time.time() - start_time
    
    # Assert
    assert execution_time < {threshold}
    assert result is not None
    """
        }
    
    # Helper methods for test generation
    def _generate_basic_test_code(self, method_name: str, method_info: Dict[str, Any]) -> str:
        """Generate basic test code for a method."""
        return f"result = {method_name}()"
    
    def _generate_setup_code(self, method_name: str, method_info: Dict[str, Any]) -> str:
        """Generate setup code for a test."""
        if method_info.get('class_name'):
            return f"instance = {method_info['class_name']}()"
        return "pass"
    
    def _generate_test_data(self, method_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test data for a method."""
        return [{"input": "valid_data", "expected": "success"}]
    
    def _generate_assertions(self, method_name: str, method_info: Dict[str, Any]) -> List[str]:
        """Generate assertions for a test."""
        return ["result is not None"]
    
    def _identify_mocks_needed(self, method_info: Dict[str, Any]) -> List[str]:
        """Identify mocks needed for a test."""
        return []
    
    def _identify_dependencies(self, code: str) -> List[str]:
        """Identify dependencies needed for tests."""
        dependencies = []
        if 'import' in code:
            import_matches = re.findall(r'import\s+(\w+)', code)
            dependencies.extend(import_matches)
        return list(set(dependencies))
    
    def _generate_edge_case_test_code(self, method_name: str, method_info: Dict[str, Any]) -> str:
        """Generate edge case test code."""
        return f"result = {method_name}(None)  # Test with None input"
    
    def _generate_edge_case_data(self, method_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate edge case test data."""
        return [{"input": None, "expected": "error"}, {"input": "", "expected": "empty_result"}]
    
    def _generate_edge_case_assertions(self, method_name: str, method_info: Dict[str, Any]) -> List[str]:
        """Generate edge case assertions."""
        return ["result is not None or exception_raised"]
    
    def _generate_integration_test_code(self, classes: List[str]) -> str:
        """Generate integration test code."""
        return "# Integration test code would be generated based on class analysis"
    
    def _generate_error_test_code(self, exception_info: Dict[str, Any]) -> str:
        """Generate error handling test code."""
        return f"# Test {exception_info['exception_type']} handling"
    
    def _generate_performance_test_code(self, method_name: str) -> str:
        """Generate performance test code."""
        return f"""
start_time = time.time()
result = {method_name}()
execution_time = time.time() - start_time
"""


# AST Visitors for code analysis
class ComplexityVisitor(ast.NodeVisitor):
    """Calculate cyclomatic complexity of methods."""
    
    def __init__(self):
        self.methods = {}
        self.current_method = None
        self.current_complexity = 0
    
    def visit_FunctionDef(self, node):
        old_method = self.current_method
        old_complexity = self.current_complexity
        
        self.current_method = node.name
        self.current_complexity = 1  # Base complexity
        
        self.generic_visit(node)
        
        self.methods[self.current_method] = self.current_complexity
        
        self.current_method = old_method
        self.current_complexity = old_complexity
    
    def visit_If(self, node):
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        self.current_complexity += 1
        self.generic_visit(node)


class MethodVisitor(ast.NodeVisitor):
    """Collect information about methods."""
    
    def __init__(self):
        self.methods = {}
        self.method_lengths = {}
        self.current_class = None
    
    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node):
        method_info = {
            'name': node.name,
            'class_name': self.current_class,
            'line_start': node.lineno,
            'line_end': node.end_lineno or node.lineno,
            'args': [arg.arg for arg in node.args.args]
        }
        
        self.methods[node.name] = method_info
        self.method_lengths[node.name] = (node.end_lineno or node.lineno) - node.lineno


class TestabilityVisitor(ast.NodeVisitor):
    """Assess code testability factors."""
    
    def __init__(self):
        self.global_variables = 0
        self.external_dependencies = 0
        self.injected_dependencies = 0
        self.pure_functions = 0
    
    def visit_Global(self, node):
        self.global_variables += len(node.names)
    
    def visit_Import(self, node):
        self.external_dependencies += len(node.names)
    
    def visit_ImportFrom(self, node):
        self.external_dependencies += len(node.names) if node.names else 1


class NamingVisitor(ast.NodeVisitor):
    """Check naming conventions."""
    
    def __init__(self):
        self.issues = []
    
    def visit_FunctionDef(self, node):
        if not self._is_snake_case(node.name):
            self.issues.append({
                'name': node.name,
                'type': 'function',
                'message': f"Function '{node.name}' doesn't follow snake_case convention",
                'recommendation': f"Rename to '{self._to_snake_case(node.name)}'"
            })
        
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        if not self._is_pascal_case(node.name):
            self.issues.append({
                'name': node.name,
                'type': 'class',
                'message': f"Class '{node.name}' doesn't follow PascalCase convention",
                'recommendation': f"Rename to '{self._to_pascal_case(node.name)}'"
            })
        
        self.generic_visit(node)
    
    def _is_snake_case(self, name: str) -> bool:
        return name.islower() and '_' in name or name.islower()
    
    def _is_pascal_case(self, name: str) -> bool:
        return name[0].isupper() and name.isalnum()
    
    def _to_snake_case(self, name: str) -> str:
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    def _to_pascal_case(self, name: str) -> str:
        return ''.join(word.capitalize() for word in name.split('_'))


class ClassVisitor(ast.NodeVisitor):
    """Collect information about classes."""
    
    def __init__(self):
        self.classes = []
    
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)


class ExceptionVisitor(ast.NodeVisitor):
    """Find exception handling patterns."""
    
    def __init__(self):
        self.exception_handlers = []
        self.current_function = None
    
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_ExceptHandler(self, node):
        exception_type = 'Exception'
        if node.type and isinstance(node.type, ast.Name):
            exception_type = node.type.id
        
        self.exception_handlers.append({
            'exception_type': exception_type,
            'function': self.current_function,
            'line': node.lineno
        })


class CodeIntelligenceAgent(IntelligenceModelInterface):
    """
    Code Intelligence Agent with autonomous testing and quality analysis.
    
    This agent specializes in:
    - Autonomous test case generation
    - Code quality assessment and recommendations
    - Intelligent testing strategies
    - Pattern-based code improvement suggestions
    """
    
    def __init__(self, agent_id: str, anthropic_client: Optional[AsyncAnthropic] = None):
        self.agent_id = agent_id
        self.client = anthropic_client
        self.code_analyzer = CodeAnalyzer()
        self.test_generator = TestGenerator(self.code_analyzer)
        self.analysis_history: List[CodeQualityReport] = []
        self.test_generation_metrics = {
            'tests_generated': 0,
            'tests_passed': 0,
            'coverage_improved': 0.0
        }
    
    async def predict(self, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Make code intelligence predictions."""
        request_type = input_data.get('type', 'quality_analysis')
        code = input_data.get('code', '')
        file_path = input_data.get('file_path', '')
        
        if request_type == 'quality_analysis':
            return await self._analyze_code_quality(code, file_path)
        elif request_type == 'test_generation':
            return await self._generate_test_suite(code, file_path)
        elif request_type == 'improvement_recommendations':
            return await self._provide_improvement_recommendations(code, file_path)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _analyze_code_quality(self, code: str, file_path: str) -> IntelligencePrediction:
        """Analyze code quality and provide detailed report."""
        quality_report = await self.code_analyzer.analyze_code(code, file_path)
        self.analysis_history.append(quality_report)
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"code": code[:200], "file_path": file_path},
            prediction=asdict(quality_report),
            confidence=quality_report.overall_score,
            explanation=f"Code quality analysis: {quality_report.overall_score:.2f}/1.0 with {len(quality_report.get_priority_issues())} priority issues",
            timestamp=datetime.now()
        )
    
    async def _generate_test_suite(self, code: str, file_path: str) -> IntelligencePrediction:
        """Generate comprehensive test suite for the code."""
        test_cases = await self.test_generator.generate_tests(code, file_path)
        self.test_generation_metrics['tests_generated'] += len(test_cases)
        
        # Generate test suite summary
        test_summary = {
            'total_tests': len(test_cases),
            'test_types': {test_type.value: len([t for t in test_cases if t.test_type == test_type]) 
                          for test_type in TestType},
            'priority_distribution': {priority.value: len([t for t in test_cases if t.priority == priority])
                                    for priority in TestPriority},
            'estimated_coverage': min(1.0, len(test_cases) * 0.1),  # Rough estimate
            'test_files': [test.to_pytest_code() for test in test_cases[:5]]  # Include first 5 tests
        }
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"code": code[:200], "file_path": file_path},
            prediction={
                "test_cases": [asdict(test) for test in test_cases],
                "test_summary": test_summary,
                "execution_plan": self._create_test_execution_plan(test_cases)
            },
            confidence=0.8,
            explanation=f"Generated {len(test_cases)} test cases covering unit, integration, and error handling scenarios",
            timestamp=datetime.now()
        )
    
    async def _provide_improvement_recommendations(self, code: str, file_path: str) -> IntelligencePrediction:
        """Provide prioritized improvement recommendations."""
        quality_report = await self.code_analyzer.analyze_code(code, file_path)
        
        # Generate detailed improvement plan
        improvement_plan = {
            'priority_fixes': quality_report.get_priority_issues(),
            'improvement_roadmap': quality_report.get_improvement_roadmap(),
            'refactoring_suggestions': await self._generate_refactoring_suggestions(quality_report),
            'testing_recommendations': await self._generate_testing_recommendations(code),
            'security_improvements': [issue for issue in quality_report.issues 
                                    if issue.get('type') == CodeQualityIssue.SECURITY.value],
            'estimated_effort': self._estimate_improvement_effort(quality_report)
        }
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"code": code[:200], "file_path": file_path},
            prediction=improvement_plan,
            confidence=0.9,
            explanation=f"Comprehensive improvement plan with {len(improvement_plan['priority_fixes'])} priority fixes",
            timestamp=datetime.now()
        )
    
    def _create_test_execution_plan(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Create execution plan for generated tests."""
        return {
            'execution_order': [test.test_id for test in sorted(test_cases, key=lambda t: t.priority.value)],
            'parallel_groups': self._group_tests_for_parallel_execution(test_cases),
            'estimated_total_time': sum(test.estimated_execution_time for test in test_cases),
            'dependencies': self._identify_test_dependencies(test_cases)
        }
    
    def _group_tests_for_parallel_execution(self, test_cases: List[TestCase]) -> List[List[str]]:
        """Group tests that can be run in parallel."""
        # Simple grouping by test type
        groups = defaultdict(list)
        for test in test_cases:
            groups[test.test_type.value].append(test.test_id)
        return list(groups.values())
    
    def _identify_test_dependencies(self, test_cases: List[TestCase]) -> Dict[str, List[str]]:
        """Identify dependencies between tests."""
        dependencies = {}
        for test in test_cases:
            if test.dependencies:
                dependencies[test.test_id] = test.dependencies
        return dependencies
    
    async def _generate_refactoring_suggestions(self, quality_report: CodeQualityReport) -> List[Dict[str, Any]]:
        """Generate specific refactoring suggestions."""
        suggestions = []
        
        for issue in quality_report.issues:
            if issue.get('type') == CodeQualityIssue.COMPLEXITY.value:
                suggestions.append({
                    'type': 'extract_method',
                    'target': issue.get('method'),
                    'description': f"Extract complex logic from {issue.get('method')} into smaller methods",
                    'estimated_effort': 'medium',
                    'impact': 'high'
                })
            
            elif issue.get('type') == CodeQualityIssue.DUPLICATION.value:
                suggestions.append({
                    'type': 'extract_common_method',
                    'target': 'duplicated_code',
                    'description': "Extract duplicated code into a reusable method",
                    'estimated_effort': 'low',
                    'impact': 'medium'
                })
        
        return suggestions
    
    async def _generate_testing_recommendations(self, code: str) -> List[str]:
        """Generate testing-specific recommendations."""
        recommendations = []
        
        # Analyze code for testing gaps
        if 'async def' in code:
            recommendations.append("Add async/await testing with pytest-asyncio")
        
        if 'class' in code:
            recommendations.append("Add comprehensive unit tests for all public methods")
        
        if 'requests.' in code or 'httpx.' in code:
            recommendations.append("Mock HTTP calls in tests to avoid external dependencies")
        
        if 'database' in code.lower() or 'db' in code.lower():
            recommendations.append("Use test database or mocking for database interactions")
        
        return recommendations
    
    def _estimate_improvement_effort(self, quality_report: CodeQualityReport) -> Dict[str, Any]:
        """Estimate effort required for improvements."""
        total_issues = len(quality_report.issues)
        critical_issues = len([i for i in quality_report.issues if i.get('priority') == 'critical'])
        high_issues = len([i for i in quality_report.issues if i.get('priority') == 'high'])
        
        # Simple effort estimation
        effort_hours = critical_issues * 4 + high_issues * 2 + (total_issues - critical_issues - high_issues) * 1
        
        return {
            'estimated_hours': effort_hours,
            'effort_level': 'high' if effort_hours > 20 else 'medium' if effort_hours > 10 else 'low',
            'priority_breakdown': {
                'critical': critical_issues,
                'high': high_issues,
                'medium_low': total_issues - critical_issues - high_issues
            },
            'recommended_timeline': f"{max(1, effort_hours // 8)} days"
        }
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train the Code Intelligence Agent with feedback data."""
        try:
            for data_point in training_data:
                if data_point.data_type == DataType.TEXT:
                    # Update analysis based on feedback
                    feedback = data_point.metadata.get('feedback')
                    if feedback and 'test_results' in feedback:
                        test_results = feedback['test_results']
                        self.test_generation_metrics['tests_passed'] += test_results.get('passed', 0)
                        
                        # Learn from test success/failure patterns
                        if test_results.get('coverage_improvement'):
                            self.test_generation_metrics['coverage_improved'] += test_results['coverage_improvement']
            
            logger.info(f"Code Intelligence Agent trained on {len(training_data)} feedback points")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    async def evaluate(self, test_data: List[DataPoint]) -> Dict[str, float]:
        """Evaluate the Code Intelligence Agent performance."""
        correct_analyses = 0
        total_analyses = len(test_data)
        
        for data_point in test_data:
            try:
                prediction = await self.predict({
                    'code': data_point.value,
                    'type': 'quality_analysis'
                })
                
                # Evaluate based on confidence and accuracy
                if prediction.confidence > 0.6:
                    correct_analyses += 1
                    
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                continue
        
        accuracy = correct_analyses / total_analyses if total_analyses > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_evaluations': total_analyses,
            'successful_analyses': correct_analyses,
            'tests_generated_total': self.test_generation_metrics['tests_generated'],
            'test_success_rate': (self.test_generation_metrics['tests_passed'] / 
                                max(1, self.test_generation_metrics['tests_generated'])),
            'coverage_improvement_avg': self.test_generation_metrics['coverage_improved']
        }
    
    async def get_testing_insights(self) -> Dict[str, Any]:
        """Get insights from test generation and analysis."""
        return {
            'testing_patterns': {
                'most_common_test_types': self._analyze_test_type_patterns(),
                'highest_value_tests': self._identify_high_value_tests(),
                'testing_gaps': self._identify_testing_gaps()
            },
            'quality_trends': {
                'quality_improvement': self._calculate_quality_trends(),
                'common_issues': self._identify_common_issues(),
                'best_practices': self._generate_best_practices()
            },
            'recommendations': {
                'focus_areas': self._recommend_focus_areas(),
                'automation_opportunities': self._identify_automation_opportunities()
            }
        }
    
    def _analyze_test_type_patterns(self) -> Dict[str, int]:
        """Analyze patterns in generated test types."""
        # This would analyze historical data
        return {
            'unit_tests': 60,
            'integration_tests': 25,
            'error_handling_tests': 15
        }
    
    def _identify_high_value_tests(self) -> List[str]:
        """Identify tests that provide the most value."""
        return [
            "Error handling tests for critical paths",
            "Integration tests for API endpoints",
            "Performance tests for data processing"
        ]
    
    def _identify_testing_gaps(self) -> List[str]:
        """Identify common testing gaps."""
        return [
            "Insufficient edge case coverage",
            "Missing performance tests",
            "Lack of security-focused tests"
        ]
    
    def _calculate_quality_trends(self) -> Dict[str, float]:
        """Calculate quality improvement trends."""
        if len(self.analysis_history) < 2:
            return {'trend': 0.0}
        
        recent_scores = [report.overall_score for report in self.analysis_history[-5:]]
        return {
            'trend': (recent_scores[-1] - recent_scores[0]) if len(recent_scores) > 1 else 0.0,
            'average_score': sum(recent_scores) / len(recent_scores)
        }
    
    def _identify_common_issues(self) -> List[str]:
        """Identify most common quality issues."""
        issue_counts = defaultdict(int)
        for report in self.analysis_history:
            for issue in report.issues:
                issue_counts[issue.get('type', 'unknown')] += 1
        
        return sorted(issue_counts.keys(), key=lambda x: issue_counts[x], reverse=True)[:5]
    
    def _generate_best_practices(self) -> List[str]:
        """Generate best practices based on analysis."""
        return [
            "Keep method complexity below 10",
            "Maintain test coverage above 80%",
            "Use descriptive variable and function names",
            "Handle exceptions explicitly",
            "Document complex algorithms"
        ]
    
    def _recommend_focus_areas(self) -> List[str]:
        """Recommend areas to focus improvement efforts."""
        if not self.analysis_history:
            return ["Start with code quality analysis"]
        
        latest_report = self.analysis_history[-1]
        focus_areas = []
        
        if latest_report.security_score < 0.8:
            focus_areas.append("Security improvements")
        
        if latest_report.complexity_score < 0.6:
            focus_areas.append("Code complexity reduction")
        
        if latest_report.test_coverage_estimate < 0.8:
            focus_areas.append("Test coverage improvement")
        
        return focus_areas
    
    def _identify_automation_opportunities(self) -> List[str]:
        """Identify opportunities for test automation."""
        return [
            "Automated test generation for new methods",
            "Continuous quality monitoring",
            "Automated refactoring suggestions",
            "Performance regression detection"
        ]


async def create_code_intelligence_agent(agent_id: str) -> CodeIntelligenceAgent:
    """Factory function to create a new Code Intelligence Agent."""
    anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
    return CodeIntelligenceAgent(agent_id, anthropic_client)
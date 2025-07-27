"""
Safety Validator

Comprehensive security scanning and safety validation for code modifications.
Ensures all modifications meet security standards and safety requirements
before application to prevent introduction of vulnerabilities or instabilities.
"""

import ast
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

import structlog

logger = structlog.get_logger()


class SecurityLevel(Enum):
    """Security levels for validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class SecurityIssue:
    """Represents a security issue found during validation."""
    
    issue_type: str
    severity: SecurityLevel
    description: str
    file_path: str
    line_number: int
    confidence: float  # 0.0 to 1.0
    cwe_id: Optional[str] = None
    recommendation: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyValidationReport:
    """Comprehensive safety validation report."""
    
    validation_result: ValidationResult
    overall_score: float  # 0.0 to 1.0
    security_issues: List[SecurityIssue] = field(default_factory=list)
    performance_impact: Optional[float] = None
    complexity_score: float = 0.0
    
    # Validation categories
    syntax_validation: ValidationResult = ValidationResult.PASS
    security_validation: ValidationResult = ValidationResult.PASS
    performance_validation: ValidationResult = ValidationResult.PASS
    compatibility_validation: ValidationResult = ValidationResult.PASS
    
    # Detailed results
    validation_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_safe_to_apply(self) -> bool:
        """Check if modification is safe to apply."""
        return (
            self.validation_result in [ValidationResult.PASS, ValidationResult.WARNING] and
            self.overall_score >= 0.6 and
            not any(issue.severity == SecurityLevel.CRITICAL for issue in self.security_issues)
        )
    
    @property
    def critical_issues_count(self) -> int:
        """Count of critical security issues."""
        return len([issue for issue in self.security_issues if issue.severity == SecurityLevel.CRITICAL])
    
    @property
    def high_issues_count(self) -> int:
        """Count of high severity issues."""
        return len([issue for issue in self.security_issues if issue.severity == SecurityLevel.HIGH])


class SafetyValidator:
    """Validates code modifications for security and safety."""
    
    def __init__(self):
        self.security_patterns = self._load_security_patterns()
        self.dangerous_functions = self._load_dangerous_functions()
        self.syntax_checkers = {
            "python": self._validate_python_syntax,
            "javascript": self._validate_javascript_syntax,
            "typescript": self._validate_typescript_syntax
        }
        
    def validate_modification(
        self,
        original_content: str,
        modified_content: str,
        file_path: str,
        language: str = "python",
        safety_level: str = "conservative"
    ) -> SafetyValidationReport:
        """Perform comprehensive safety validation of a code modification."""
        
        logger.info(
            "Starting safety validation",
            file_path=file_path,
            language=language,
            safety_level=safety_level
        )
        
        report = SafetyValidationReport(validation_result=ValidationResult.PASS, overall_score=1.0)
        
        try:
            # 1. Syntax validation
            report.syntax_validation = self._validate_syntax(modified_content, language, report)
            
            # 2. Security validation
            report.security_validation = self._validate_security(
                original_content, modified_content, file_path, report
            )
            
            # 3. Performance validation
            report.performance_validation = self._validate_performance(
                original_content, modified_content, language, report
            )
            
            # 4. Compatibility validation
            report.compatibility_validation = self._validate_compatibility(
                original_content, modified_content, language, report
            )
            
            # 5. Calculate overall scores
            self._calculate_overall_score(report, safety_level)
            
            # 6. Generate recommendations
            self._generate_recommendations(report)
            
            logger.info(
                "Safety validation completed",
                file_path=file_path,
                result=report.validation_result.value,
                score=report.overall_score,
                issues_count=len(report.security_issues)
            )
            
            return report
            
        except Exception as e:
            logger.error("Safety validation failed", file_path=file_path, error=str(e))
            report.validation_result = ValidationResult.ERROR
            report.overall_score = 0.0
            report.validation_details["error"] = str(e)
            return report
    
    def validate_multiple_modifications(
        self,
        modifications: Dict[str, Tuple[str, str]],  # file_path -> (original, modified)
        language: str = "python",
        safety_level: str = "conservative"
    ) -> Dict[str, SafetyValidationReport]:
        """Validate multiple modifications together."""
        
        reports = {}
        
        # Individual validation
        for file_path, (original, modified) in modifications.items():
            reports[file_path] = self.validate_modification(
                original, modified, file_path, language, safety_level
            )
        
        # Cross-file validation
        self._validate_cross_file_impacts(modifications, reports)
        
        return reports
    
    def _validate_syntax(
        self,
        content: str,
        language: str,
        report: SafetyValidationReport
    ) -> ValidationResult:
        """Validate syntax for the given language."""
        
        validator = self.syntax_checkers.get(language)
        if not validator:
            report.validation_details["syntax"] = f"No syntax validator for {language}"
            return ValidationResult.WARNING
        
        try:
            errors = validator(content)
            if errors:
                report.validation_details["syntax_errors"] = errors
                
                # Add syntax errors as security issues
                for error in errors:
                    report.security_issues.append(SecurityIssue(
                        issue_type="syntax_error",
                        severity=SecurityLevel.HIGH,
                        description=f"Syntax error: {error}",
                        file_path="<content>",
                        line_number=0,
                        confidence=1.0,
                        recommendation="Fix syntax errors before applying modification"
                    ))
                
                return ValidationResult.FAIL
            
            return ValidationResult.PASS
            
        except Exception as e:
            report.validation_details["syntax_validation_error"] = str(e)
            return ValidationResult.ERROR
    
    def _validate_python_syntax(self, content: str) -> List[str]:
        """Validate Python syntax."""
        
        errors = []
        
        try:
            # Parse with AST
            ast.parse(content)
            
            # Additional Python-specific checks
            lines = content.splitlines()
            for i, line in enumerate(lines, 1):
                # Check for common syntax issues
                if re.search(r'^\s*print\s+', line):  # Python 2 style print
                    errors.append(f"Line {i}: Use print() function instead of print statement")
                
                if re.search(r'except\s+\w+\s*,\s*\w+:', line):  # Python 2 style except
                    errors.append(f"Line {i}: Use 'except Exception as e:' syntax")
                
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"Syntax validation error: {str(e)}")
        
        return errors
    
    def _validate_javascript_syntax(self, content: str) -> List[str]:
        """Validate JavaScript syntax (simplified)."""
        
        errors = []
        
        # Basic JavaScript syntax checks
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Check for common issues
            if re.search(r'==\s*(?:null|undefined)', line):
                errors.append(f"Line {i}: Consider using === for null/undefined checks")
            
            if re.search(r'var\s+\w+', line):
                errors.append(f"Line {i}: Consider using 'let' or 'const' instead of 'var'")
        
        return errors
    
    def _validate_typescript_syntax(self, content: str) -> List[str]:
        """Validate TypeScript syntax (simplified)."""
        
        # For now, use JavaScript validation as base
        return self._validate_javascript_syntax(content)
    
    def _validate_security(
        self,
        original_content: str,
        modified_content: str,
        file_path: str,
        report: SafetyValidationReport
    ) -> ValidationResult:
        """Validate security aspects of the modification."""
        
        issues = []
        
        # 1. Check for dangerous patterns
        issues.extend(self._check_dangerous_patterns(modified_content, file_path))
        
        # 2. Check for security regressions
        issues.extend(self._check_security_regressions(original_content, modified_content, file_path))
        
        # 3. Check for injection vulnerabilities
        issues.extend(self._check_injection_vulnerabilities(modified_content, file_path))
        
        # 4. Check for cryptographic issues
        issues.extend(self._check_cryptographic_issues(modified_content, file_path))
        
        # 5. Check for authentication/authorization issues
        issues.extend(self._check_auth_issues(modified_content, file_path))
        
        report.security_issues.extend(issues)
        
        # Determine validation result
        if any(issue.severity == SecurityLevel.CRITICAL for issue in issues):
            return ValidationResult.FAIL
        elif any(issue.severity == SecurityLevel.HIGH for issue in issues):
            return ValidationResult.WARNING
        elif issues:
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASS
    
    def _check_dangerous_patterns(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Check for dangerous coding patterns."""
        
        issues = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_info in self.security_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_type=issue_info["type"],
                        severity=SecurityLevel(issue_info["severity"]),
                        description=issue_info["description"],
                        file_path=file_path,
                        line_number=i,
                        confidence=issue_info["confidence"],
                        cwe_id=issue_info.get("cwe_id"),
                        recommendation=issue_info.get("recommendation"),
                        context={"pattern": pattern, "line": line.strip()}
                    ))
        
        return issues
    
    def _check_security_regressions(
        self,
        original: str,
        modified: str,
        file_path: str
    ) -> List[SecurityIssue]:
        """Check if modifications introduce security regressions."""
        
        issues = []
        
        # Check if security features were removed
        security_keywords = [
            "authenticate", "authorize", "validate", "sanitize", "escape",
            "csrf", "xss", "sql_injection", "permission", "access_control"
        ]
        
        for keyword in security_keywords:
            if keyword in original.lower() and keyword not in modified.lower():
                issues.append(SecurityIssue(
                    issue_type="security_regression",
                    severity=SecurityLevel.HIGH,
                    description=f"Potential removal of security feature: {keyword}",
                    file_path=file_path,
                    line_number=0,
                    confidence=0.7,
                    recommendation=f"Ensure {keyword} functionality is preserved"
                ))
        
        return issues
    
    def _check_injection_vulnerabilities(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Check for injection vulnerability patterns."""
        
        issues = []
        lines = content.splitlines()
        
        injection_patterns = [
            (r'execute\s*\([^)]*\+[^)]*\)', "sql_injection", "Potential SQL injection"),
            (r'system\s*\([^)]*\+[^)]*\)', "command_injection", "Potential command injection"),
            (r'eval\s*\([^)]*\+[^)]*\)', "code_injection", "Potential code injection"),
            (r'innerHTML\s*=\s*[^;]*\+', "xss", "Potential XSS vulnerability"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, vuln_type, description in injection_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_type=vuln_type,
                        severity=SecurityLevel.HIGH,
                        description=description,
                        file_path=file_path,
                        line_number=i,
                        confidence=0.8,
                        recommendation="Use parameterized queries or proper escaping",
                        context={"line": line.strip()}
                    ))
        
        return issues
    
    def _check_cryptographic_issues(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Check for cryptographic security issues."""
        
        issues = []
        lines = content.splitlines()
        
        crypto_patterns = [
            (r'md5\s*\(', "weak_crypto", "MD5 is cryptographically weak"),
            (r'sha1\s*\(', "weak_crypto", "SHA1 is cryptographically weak"),
            (r'random\.random\s*\(', "weak_random", "Use secrets module for cryptographic randomness"),
            (r'DES|3DES', "weak_encryption", "DES/3DES encryption is weak"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_type, description in crypto_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_type=issue_type,
                        severity=SecurityLevel.MEDIUM,
                        description=description,
                        file_path=file_path,
                        line_number=i,
                        confidence=0.9,
                        recommendation="Use stronger cryptographic algorithms",
                        context={"line": line.strip()}
                    ))
        
        return issues
    
    def _check_auth_issues(self, content: str, file_path: str) -> List[SecurityIssue]:
        """Check for authentication and authorization issues."""
        
        issues = []
        lines = content.splitlines()
        
        auth_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "hardcoded_password", "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "hardcoded_api_key", "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "hardcoded_secret", "Hardcoded secret detected"),
            (r'admin\s*=\s*True', "privilege_escalation", "Potential privilege escalation"),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, issue_type, description in auth_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_type=issue_type,
                        severity=SecurityLevel.HIGH,
                        description=description,
                        file_path=file_path,
                        line_number=i,
                        confidence=0.8,
                        recommendation="Use environment variables or secure configuration",
                        context={"line": line.strip()}
                    ))
        
        return issues
    
    def _validate_performance(
        self,
        original: str,
        modified: str,
        language: str,
        report: SafetyValidationReport
    ) -> ValidationResult:
        """Validate performance impact of modifications."""
        
        # Simple performance heuristics
        performance_score = 1.0
        issues = []
        
        # Check for performance anti-patterns
        modified_lines = modified.splitlines()
        
        for i, line in enumerate(modified_lines, 1):
            # Nested loops
            if re.search(r'for\s+.*:\s*\n\s*for\s+.*:', modified):
                performance_score -= 0.2
                issues.append(f"Line {i}: Nested loops detected - consider optimization")
            
            # String concatenation in loops
            if 'for ' in line and '+=' in line and 'str' in line:
                performance_score -= 0.1
                issues.append(f"Line {i}: String concatenation in loop - use join() instead")
            
            # Inefficient operations
            if re.search(r'\.append\s*\([^)]*\)\s*in\s+.*for.*in', line):
                performance_score -= 0.1
                issues.append(f"Line {i}: Consider list comprehension for better performance")
        
        report.performance_impact = 1.0 - performance_score
        report.validation_details["performance_issues"] = issues
        
        if performance_score < 0.5:
            return ValidationResult.WARNING
        else:
            return ValidationResult.PASS
    
    def _validate_compatibility(
        self,
        original: str,
        modified: str,
        language: str,
        report: SafetyValidationReport
    ) -> ValidationResult:
        """Validate compatibility and breaking changes."""
        
        issues = []
        
        # Check for breaking changes (simplified)
        if language == "python":
            # Check if public interface changed
            original_functions = self._extract_public_functions(original)
            modified_functions = self._extract_public_functions(modified)
            
            removed_functions = original_functions - modified_functions
            if removed_functions:
                issues.append(f"Public functions removed: {', '.join(removed_functions)}")
        
        report.validation_details["compatibility_issues"] = issues
        
        return ValidationResult.WARNING if issues else ValidationResult.PASS
    
    def _extract_public_functions(self, content: str) -> Set[str]:
        """Extract public function names from Python code."""
        
        functions = set()
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    functions.add(node.name)
        except:
            pass
        
        return functions
    
    def _validate_cross_file_impacts(
        self,
        modifications: Dict[str, Tuple[str, str]],
        reports: Dict[str, SafetyValidationReport]
    ) -> None:
        """Validate impacts across multiple files."""
        
        # Check for circular dependencies
        imports_by_file = {}
        
        for file_path, (_, modified) in modifications.items():
            imports_by_file[file_path] = self._extract_imports(modified)
        
        # Simple circular dependency check
        for file_path, imports in imports_by_file.items():
            for imported in imports:
                if imported in imports_by_file and file_path in imports_by_file[imported]:
                    # Potential circular dependency
                    issue = SecurityIssue(
                        issue_type="circular_dependency",
                        severity=SecurityLevel.MEDIUM,
                        description=f"Potential circular dependency between {file_path} and {imported}",
                        file_path=file_path,
                        line_number=0,
                        confidence=0.6,
                        recommendation="Review import structure to avoid circular dependencies"
                    )
                    reports[file_path].security_issues.append(issue)
    
    def _extract_imports(self, content: str) -> Set[str]:
        """Extract import statements from code."""
        
        imports = set()
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
        except:
            pass
        
        return imports
    
    def _calculate_overall_score(
        self,
        report: SafetyValidationReport,
        safety_level: str
    ) -> None:
        """Calculate overall safety score."""
        
        base_score = 1.0
        
        # Deduct points for validation failures
        validation_scores = {
            ValidationResult.PASS: 1.0,
            ValidationResult.WARNING: 0.8,
            ValidationResult.FAIL: 0.0,
            ValidationResult.ERROR: 0.0
        }
        
        component_weights = {
            "syntax": 0.3,
            "security": 0.4,
            "performance": 0.2,
            "compatibility": 0.1
        }
        
        weighted_score = (
            validation_scores[report.syntax_validation] * component_weights["syntax"] +
            validation_scores[report.security_validation] * component_weights["security"] +
            validation_scores[report.performance_validation] * component_weights["performance"] +
            validation_scores[report.compatibility_validation] * component_weights["compatibility"]
        )
        
        # Adjust for security issues
        for issue in report.security_issues:
            severity_penalties = {
                SecurityLevel.LOW: 0.05,
                SecurityLevel.MEDIUM: 0.1,
                SecurityLevel.HIGH: 0.2,
                SecurityLevel.CRITICAL: 0.5
            }
            weighted_score -= severity_penalties[issue.severity] * issue.confidence
        
        # Adjust for safety level requirements
        safety_level_adjustments = {
            "conservative": 0.0,  # No adjustment - strict requirements
            "moderate": 0.1,      # Slightly more lenient
            "aggressive": 0.2     # More lenient for higher risk tolerance
        }
        
        weighted_score += safety_level_adjustments.get(safety_level, 0.0)
        
        report.overall_score = max(0.0, min(1.0, weighted_score))
        
        # Determine overall validation result
        if report.overall_score >= 0.8:
            report.validation_result = ValidationResult.PASS
        elif report.overall_score >= 0.6:
            report.validation_result = ValidationResult.WARNING
        else:
            report.validation_result = ValidationResult.FAIL
    
    def _generate_recommendations(self, report: SafetyValidationReport) -> None:
        """Generate actionable recommendations."""
        
        recommendations = []
        
        # Syntax recommendations
        if report.syntax_validation != ValidationResult.PASS:
            recommendations.append("Fix all syntax errors before applying modification")
        
        # Security recommendations
        critical_issues = [i for i in report.security_issues if i.severity == SecurityLevel.CRITICAL]
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical security issues")
        
        high_issues = [i for i in report.security_issues if i.severity == SecurityLevel.HIGH]
        if high_issues:
            recommendations.append(f"Review {len(high_issues)} high-severity security issues")
        
        # Performance recommendations
        if report.performance_impact and report.performance_impact > 0.2:
            recommendations.append("Consider performance optimization - significant impact detected")
        
        # General recommendations
        if report.overall_score < 0.8:
            recommendations.append("Consider reviewing and improving code quality before application")
        
        report.recommendations = recommendations
    
    def _load_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load security pattern definitions."""
        
        return {
            r'eval\s*\(': {
                "type": "dangerous_function",
                "severity": "high",
                "description": "Use of eval() function",
                "confidence": 0.9,
                "cwe_id": "CWE-94",
                "recommendation": "Avoid eval() - use safer alternatives"
            },
            r'exec\s*\(': {
                "type": "dangerous_function", 
                "severity": "high",
                "description": "Use of exec() function",
                "confidence": 0.9,
                "cwe_id": "CWE-94",
                "recommendation": "Avoid exec() - use safer alternatives"
            },
            r'os\.system\s*\(': {
                "type": "command_injection",
                "severity": "high",
                "description": "Use of os.system() - potential command injection",
                "confidence": 0.8,
                "cwe_id": "CWE-78",
                "recommendation": "Use subprocess with proper input validation"
            },
            r'subprocess\..*shell=True': {
                "type": "command_injection",
                "severity": "medium",
                "description": "Subprocess with shell=True",
                "confidence": 0.7,
                "cwe_id": "CWE-78",
                "recommendation": "Avoid shell=True or validate input carefully"
            },
            r'pickle\.loads?\s*\(': {
                "type": "deserialization",
                "severity": "high",
                "description": "Unsafe pickle deserialization",
                "confidence": 0.8,
                "cwe_id": "CWE-502",
                "recommendation": "Use safe serialization formats like JSON"
            }
        }
    
    def _load_dangerous_functions(self) -> Set[str]:
        """Load set of dangerous functions to monitor."""
        
        return {
            "eval", "exec", "compile", "__import__",
            "open", "file", "input", "raw_input",
            "execfile", "reload", "apply"
        }


# Export main classes
__all__ = [
    "SafetyValidator",
    "SafetyValidationReport",
    "SecurityIssue",
    "SecurityLevel",
    "ValidationResult"
]
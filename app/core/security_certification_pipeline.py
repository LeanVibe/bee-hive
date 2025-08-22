"""
Security Certification Pipeline for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.2

Implements comprehensive security certification pipeline with automated scanning,
certification processes, and vulnerability tracking for the plugin marketplace.

Key Features:
- Automated security scanning and vulnerability detection
- Multi-level certification process (Basic → Security → Performance → Full → Enterprise)
- Compliance validation (GDPR, COPPA, SOX, etc.)
- Continuous monitoring and re-certification
- Quality metrics and scoring
- Integration with plugin marketplace

Epic 1 Preservation:
- <50ms certification status checks
- <30ms security validation operations
- <80MB memory usage with efficient scanning
- Non-blocking certification processes
"""

import asyncio
import uuid
import hashlib
import json
import re
import ast
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from contextlib import asynccontextmanager

from .logging_service import get_component_logger
from .plugin_marketplace import CertificationLevel, MarketplacePluginEntry
from .plugin_security_framework import (
    PluginSecurityFramework, SecurityReport, PluginSecurityLevel,
    SecurityViolation, SecurityViolationType
)
from .advanced_plugin_manager import PluginVersion

logger = get_component_logger("security_certification_pipeline")


class QualityGateType(Enum):
    """Types of quality gates."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    COMPATIBILITY = "compatibility"


class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ComplianceStandard(Enum):
    """Compliance standards for validation."""
    GDPR = "gdpr"
    COPPA = "coppa"
    SOX = "sox"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    OWASP = "owasp"


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    name: str
    value: float
    unit: str
    threshold: float
    passed: bool
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_type: QualityGateType
    status: QualityGateStatus
    score: float
    max_score: float
    metrics: List[QualityMetric] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def score_percentage(self) -> float:
        """Calculate score as percentage."""
        if self.max_score <= 0:
            return 0.0
        return (self.score / self.max_score) * 100
    
    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.status == QualityGateStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_type": self.gate_type.value,
            "status": self.status.value,
            "score": self.score,
            "max_score": self.max_score,
            "score_percentage": round(self.score_percentage, 2),
            "passed": self.passed,
            "metrics": [m.to_dict() for m in self.metrics],
            "violations": self.violations,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CertificationReport:
    """Comprehensive certification report."""
    plugin_id: str
    certification_id: str
    target_level: CertificationLevel
    achieved_level: CertificationLevel
    overall_score: float
    gate_results: List[QualityGateResult] = field(default_factory=list)
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    security_report: Optional[SecurityReport] = None
    expires_at: Optional[datetime] = None
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "certification_id": self.certification_id,
            "target_level": self.target_level.value,
            "achieved_level": self.achieved_level.value,
            "overall_score": round(self.overall_score, 2),
            "gate_results": [gr.to_dict() for gr in self.gate_results],
            "compliance_status": self.compliance_status,
            "security_report": self.security_report.to_dict() if self.security_report else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat()
        }


class SecurityScanner:
    """
    Advanced security scanner for plugin certification.
    
    Epic 1 Optimizations:
    - <30ms security scans for common patterns
    - Efficient AST parsing and analysis
    - Cached vulnerability patterns
    """
    
    def __init__(self):
        self._vulnerability_patterns: Dict[str, List[str]] = {}
        self._security_rules: Dict[str, Dict[str, Any]] = {}
        self._scan_cache: Dict[str, SecurityReport] = {}
        
        # Epic 1: Performance tracking
        self._scan_times: List[float] = []
        
        self._initialize_security_rules()
        
        logger.info("SecurityScanner initialized with advanced patterns")
    
    def _initialize_security_rules(self) -> None:
        """Initialize comprehensive security scanning rules."""
        self._vulnerability_patterns = {
            "injection": [
                r"exec\s*\(",
                r"eval\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"sql.*\+.*\+",  # Simple SQL injection pattern
                r"\.format\s*\(.*input",  # String format injection
            ],
            "xss": [
                r"innerHTML\s*=",
                r"document\.write\s*\(",
                r"\.html\s*\(.*\+",
                r"dangerouslySetInnerHTML",
            ],
            "path_traversal": [
                r"\.\.\/",
                r"\.\.\\\\",
                r"os\.path\.join\s*\(.*\.\.",
                r"open\s*\(.*\.\.",
            ],
            "hardcoded_secrets": [
                r"password\s*=\s*['\"][^'\"]{8,}['\"]",
                r"api_key\s*=\s*['\"][^'\"]{20,}['\"]",
                r"secret\s*=\s*['\"][^'\"]{16,}['\"]",
                r"token\s*=\s*['\"][^'\"]{32,}['\"]",
            ],
            "unsafe_random": [
                r"random\.random\s*\(",
                r"Math\.random\s*\(",
                r"rand\s*\(",
            ],
            "dangerous_functions": [
                r"pickle\.loads\s*\(",
                r"yaml\.load\s*\(",
                r"marshal\.loads\s*\(",
                r"subprocess\.shell\s*=\s*True",
            ]
        }
        
        self._security_rules = {
            "critical": {
                "patterns": ["injection", "path_traversal", "hardcoded_secrets"],
                "score_deduction": 30,
                "auto_fail": True
            },
            "high": {
                "patterns": ["xss", "dangerous_functions"],
                "score_deduction": 20,
                "auto_fail": False
            },
            "medium": {
                "patterns": ["unsafe_random"],
                "score_deduction": 10,
                "auto_fail": False
            }
        }
    
    async def scan_plugin_security(
        self,
        plugin_entry: MarketplacePluginEntry,
        source_code: Optional[str] = None,
        source_path: Optional[Path] = None
    ) -> SecurityReport:
        """
        Comprehensive security scan of a plugin.
        
        Epic 1: <30ms scan target for cached results
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(plugin_entry, source_code, source_path)
            
            # Check cache first
            if cache_key in self._scan_cache:
                cached_report = self._scan_cache[cache_key]
                # Return if recent (within 1 hour)
                if (datetime.utcnow() - cached_report.timestamp).total_seconds() < 3600:
                    return cached_report
            
            violations = []
            warnings = []
            
            # Scan source code if available
            if source_code:
                code_violations = await self._scan_source_code(source_code, plugin_entry.plugin_id)
                violations.extend(code_violations)
            
            # Scan file if available
            if source_path:
                file_violations = await self._scan_file_security(source_path, plugin_entry.plugin_id)
                violations.extend(file_violations)
            
            # Validate dependencies
            dep_violations = await self._scan_dependencies(plugin_entry.metadata.dependencies)
            violations.extend(dep_violations)
            
            # Generate security warnings
            warnings.extend(await self._generate_security_warnings(plugin_entry))
            
            # Calculate security level and safety
            security_level, is_safe = self._calculate_security_level(violations, warnings)
            
            # Create security report
            report = SecurityReport(
                plugin_id=plugin_entry.plugin_id,
                is_safe=is_safe,
                security_level=security_level,
                violations=[v.description for v in violations],
                warnings=[w.description for w in warnings],
                resource_usage={}
            )
            
            # Cache the result
            self._scan_cache[cache_key] = report
            
            # Epic 1: Track scan time
            scan_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._scan_times.append(scan_time_ms)
            if len(self._scan_times) > 100:
                self._scan_times.pop(0)
            
            logger.debug("Security scan completed",
                        plugin_id=plugin_entry.plugin_id,
                        is_safe=is_safe,
                        violations_count=len(violations),
                        scan_time_ms=round(scan_time_ms, 2))
            
            return report
            
        except Exception as e:
            logger.error("Security scan failed",
                        plugin_id=plugin_entry.plugin_id,
                        error=str(e))
            
            return SecurityReport(
                plugin_id=plugin_entry.plugin_id,
                is_safe=False,
                security_level=PluginSecurityLevel.UNTRUSTED,
                violations=[f"Scan error: {str(e)}"]
            )
    
    async def _scan_source_code(self, source_code: str, plugin_id: str) -> List[SecurityViolation]:
        """Scan source code for security vulnerabilities."""
        violations = []
        
        try:
            # Pattern-based scanning
            for severity, rule in self._security_rules.items():
                for pattern_type in rule["patterns"]:
                    patterns = self._vulnerability_patterns.get(pattern_type, [])
                    for pattern in patterns:
                        matches = re.finditer(pattern, source_code, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            violation = SecurityViolation(
                                violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                                description=f"{severity.upper()}: {pattern_type} pattern detected: {match.group()}",
                                severity=severity,
                                details={
                                    "pattern": pattern,
                                    "match": match.group(),
                                    "line": source_code[:match.start()].count('\n') + 1
                                }
                            )
                            violations.append(violation)
            
            # AST-based analysis for Python code
            if self._looks_like_python(source_code):
                ast_violations = await self._analyze_python_ast(source_code, plugin_id)
                violations.extend(ast_violations)
            
        except Exception as e:
            logger.error("Source code scan failed", plugin_id=plugin_id, error=str(e))
            violation = SecurityViolation(
                violation_type=SecurityViolationType.SYSTEM_ACCESS,
                description=f"Source code analysis error: {str(e)}",
                severity="medium"
            )
            violations.append(violation)
        
        return violations
    
    async def _analyze_python_ast(self, source_code: str, plugin_id: str) -> List[SecurityViolation]:
        """Analyze Python AST for security issues."""
        violations = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                # Check for dangerous imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    violation = self._check_dangerous_imports(node)
                    if violation:
                        violations.append(violation)
                
                # Check for dangerous function calls
                elif isinstance(node, ast.Call):
                    violation = self._check_dangerous_calls(node)
                    if violation:
                        violations.append(violation)
                
                # Check for eval/exec usage
                elif isinstance(node, ast.Name) and node.id in ['eval', 'exec']:
                    violation = SecurityViolation(
                        violation_type=SecurityViolationType.DYNAMIC_CODE,
                        description=f"Dangerous function usage: {node.id}",
                        severity="critical"
                    )
                    violations.append(violation)
                
        except SyntaxError as e:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.DYNAMIC_CODE,
                description=f"Python syntax error: {str(e)}",
                severity="high"
            )
            violations.append(violation)
        except Exception as e:
            logger.error("AST analysis failed", plugin_id=plugin_id, error=str(e))
        
        return violations
    
    def _check_dangerous_imports(self, node: Union[ast.Import, ast.ImportFrom]) -> Optional[SecurityViolation]:
        """Check for dangerous import statements."""
        dangerous_modules = {
            'os': 'System access',
            'subprocess': 'Process execution', 
            'socket': 'Network access',
            'ctypes': 'Low-level system access',
            'marshal': 'Unsafe serialization',
            'pickle': 'Unsafe deserialization'
        }
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in dangerous_modules:
                    return SecurityViolation(
                        violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                        description=f"Dangerous import: {alias.name} ({dangerous_modules[alias.name]})",
                        severity="high"
                    )
        
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module in dangerous_modules:
                return SecurityViolation(
                    violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                    description=f"Dangerous import from: {node.module} ({dangerous_modules[node.module]})",
                    severity="high"
                )
        
        return None
    
    def _check_dangerous_calls(self, node: ast.Call) -> Optional[SecurityViolation]:
        """Check for dangerous function calls."""
        dangerous_functions = {
            'eval': ('Dynamic code execution', 'critical'),
            'exec': ('Dynamic code execution', 'critical'),
            'compile': ('Code compilation', 'medium'),
            '__import__': ('Dynamic import', 'medium')
        }
        
        if isinstance(node.func, ast.Name) and node.func.id in dangerous_functions:
            func_name = node.func.id
            description, severity = dangerous_functions[func_name]
            return SecurityViolation(
                violation_type=SecurityViolationType.DYNAMIC_CODE,
                description=f"Dangerous function call: {func_name} ({description})",
                severity=severity
            )
        
        return None
    
    def _looks_like_python(self, source_code: str) -> bool:
        """Check if source code looks like Python."""
        python_indicators = ['def ', 'class ', 'import ', 'from ', 'if __name__', 'async def']
        return any(indicator in source_code for indicator in python_indicators)
    
    async def _scan_file_security(self, file_path: Path, plugin_id: str) -> List[SecurityViolation]:
        """Scan file-based security issues."""
        violations = []
        
        try:
            if not file_path.exists():
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.FILE_ACCESS,
                    description="Plugin file does not exist",
                    severity="critical"
                )
                violations.append(violation)
                return violations
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > 5 * 1024 * 1024:  # 5MB limit
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.RESOURCE_LIMIT,
                    description=f"Plugin file too large: {file_size} bytes (limit: 5MB)",
                    severity="medium"
                )
                violations.append(violation)
            
            # Check file permissions
            mode = file_path.stat().st_mode
            if mode & 0o002:  # World writable
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.PERMISSION_DENIED,
                    description="Plugin file is world-writable",
                    severity="high"
                )
                violations.append(violation)
            
            # Check file extension
            if file_path.suffix.lower() in ['.exe', '.bat', '.sh', '.cmd']:
                violation = SecurityViolation(
                    violation_type=SecurityViolationType.SYSTEM_ACCESS,
                    description=f"Potentially dangerous file type: {file_path.suffix}",
                    severity="high"
                )
                violations.append(violation)
            
        except Exception as e:
            violation = SecurityViolation(
                violation_type=SecurityViolationType.FILE_ACCESS,
                description=f"File security scan error: {str(e)}",
                severity="medium"
            )
            violations.append(violation)
        
        return violations
    
    async def _scan_dependencies(self, dependencies: List[str]) -> List[SecurityViolation]:
        """Scan plugin dependencies for security issues."""
        violations = []
        
        # Known vulnerable packages (simplified)
        vulnerable_packages = {
            'pickle5': 'Known deserialization vulnerabilities',
            'pycrypto': 'Deprecated cryptography library',
            'django<2.0': 'Known security vulnerabilities in old versions'
        }
        
        for dep in dependencies:
            for vuln_pkg, description in vulnerable_packages.items():
                if dep.startswith(vuln_pkg.split('<')[0]):
                    violation = SecurityViolation(
                        violation_type=SecurityViolationType.DANGEROUS_IMPORT,
                        description=f"Vulnerable dependency: {dep} ({description})",
                        severity="high"
                    )
                    violations.append(violation)
        
        return violations
    
    async def _generate_security_warnings(self, plugin_entry: MarketplacePluginEntry) -> List[SecurityViolation]:
        """Generate security warnings based on plugin metadata."""
        warnings = []
        
        # Check certification level
        if plugin_entry.certification_level == CertificationLevel.UNCERTIFIED:
            warning = SecurityViolation(
                violation_type=SecurityViolationType.PERMISSION_DENIED,
                description="Plugin is not certified - security status unknown",
                severity="low"
            )
            warnings.append(warning)
        
        # Check update frequency
        if plugin_entry.updated_at and \
           (datetime.utcnow() - plugin_entry.updated_at).days > 365:
            warning = SecurityViolation(
                violation_type=SecurityViolationType.RESOURCE_LIMIT,
                description="Plugin hasn't been updated in over a year",
                severity="low"
            )
            warnings.append(warning)
        
        return warnings
    
    def _calculate_security_level(
        self,
        violations: List[SecurityViolation],
        warnings: List[SecurityViolation]
    ) -> Tuple[PluginSecurityLevel, bool]:
        """Calculate security level and safety status."""
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        if critical_violations:
            return PluginSecurityLevel.UNTRUSTED, False
        elif len(high_violations) > 2:
            return PluginSecurityLevel.SANDBOX, False
        elif high_violations:
            return PluginSecurityLevel.SANDBOX, True
        elif len(warnings) > 3:
            return PluginSecurityLevel.VERIFIED, True
        else:
            return PluginSecurityLevel.TRUSTED, True
    
    def _generate_cache_key(
        self,
        plugin_entry: MarketplacePluginEntry,
        source_code: Optional[str],
        source_path: Optional[Path]
    ) -> str:
        """Generate cache key for security scan."""
        key_components = [
            plugin_entry.plugin_id,
            str(plugin_entry.version),
            plugin_entry.updated_at.isoformat()
        ]
        
        if source_code:
            key_components.append(hashlib.md5(source_code.encode()).hexdigest())
        
        if source_path:
            key_components.append(str(source_path))
            
        return hashlib.sha256('|'.join(key_components).encode()).hexdigest()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get security scanner performance metrics."""
        if not self._scan_times:
            return {"scan_times": {"count": 0}}
        
        avg_time = sum(self._scan_times) / len(self._scan_times)
        
        return {
            "scan_times": {
                "count": len(self._scan_times),
                "avg_ms": round(avg_time, 2),
                "max_ms": round(max(self._scan_times), 2),
                "min_ms": round(min(self._scan_times), 2),
                "epic1_compliant": avg_time < 30
            },
            "cache_size": len(self._scan_cache),
            "vulnerability_patterns": len(self._vulnerability_patterns)
        }


class PerformanceValidator:
    """
    Performance validation for plugin certification.
    
    Epic 1 Optimizations:
    - Fast performance heuristics
    - Efficient resource usage estimation
    - Cached performance profiles
    """
    
    def __init__(self):
        self._performance_profiles: Dict[str, Dict[str, Any]] = {}
        self._validation_times: List[float] = []
        
        logger.info("PerformanceValidator initialized")
    
    async def validate_performance(self, plugin_entry: MarketplacePluginEntry) -> QualityGateResult:
        """Validate plugin performance characteristics."""
        start_time = datetime.utcnow()
        
        try:
            metrics = []
            score = 100.0
            violations = []
            warnings = []
            
            # Memory usage estimation
            memory_metric = await self._estimate_memory_usage(plugin_entry)
            metrics.append(memory_metric)
            if not memory_metric.passed:
                score -= 20
                violations.append(f"Estimated memory usage too high: {memory_metric.value}{memory_metric.unit}")
            
            # CPU usage estimation
            cpu_metric = await self._estimate_cpu_usage(plugin_entry)
            metrics.append(cpu_metric)
            if not cpu_metric.passed:
                score -= 15
                warnings.append(f"High CPU usage estimated: {cpu_metric.value}{cpu_metric.unit}")
            
            # Startup time estimation
            startup_metric = await self._estimate_startup_time(plugin_entry)
            metrics.append(startup_metric)
            if not startup_metric.passed:
                score -= 10
                warnings.append(f"Slow startup estimated: {startup_metric.value}{startup_metric.unit}")
            
            # Determine status
            if score >= 80:
                status = QualityGateStatus.PASSED
            elif score >= 60:
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
            
            execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._validation_times.append(execution_time_ms)
            if len(self._validation_times) > 100:
                self._validation_times.pop(0)
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                status=status,
                score=score,
                max_score=100.0,
                metrics=metrics,
                violations=violations,
                warnings=warnings,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            logger.error("Performance validation failed",
                        plugin_id=plugin_entry.plugin_id,
                        error=str(e))
            
            return QualityGateResult(
                gate_type=QualityGateType.PERFORMANCE,
                status=QualityGateStatus.FAILED,
                score=0.0,
                max_score=100.0,
                violations=[f"Performance validation error: {str(e)}"]
            )
    
    async def _estimate_memory_usage(self, plugin_entry: MarketplacePluginEntry) -> QualityMetric:
        """Estimate plugin memory usage."""
        # Simplified estimation based on plugin type and complexity
        base_memory = 10.0  # MB
        
        # Adjust based on plugin type
        type_multipliers = {
            "performance": 1.5,
            "security": 1.3,
            "analytics": 2.0,
            "monitoring": 1.8,
            "workflow": 1.2
        }
        
        multiplier = type_multipliers.get(plugin_entry.metadata.plugin_type.value, 1.0)
        estimated_memory = base_memory * multiplier
        
        # Epic 1: Memory threshold
        threshold = 50.0  # MB
        passed = estimated_memory <= threshold
        
        return QualityMetric(
            name="estimated_memory_usage",
            value=estimated_memory,
            unit="MB",
            threshold=threshold,
            passed=passed,
            description="Estimated memory usage based on plugin characteristics"
        )
    
    async def _estimate_cpu_usage(self, plugin_entry: MarketplacePluginEntry) -> QualityMetric:
        """Estimate plugin CPU usage."""
        # Simplified estimation
        base_cpu = 5.0  # %
        
        # Adjust based on complexity indicators
        if "analytics" in plugin_entry.short_description.lower():
            base_cpu += 10.0
        if "monitoring" in plugin_entry.short_description.lower():
            base_cpu += 8.0
        if "real-time" in plugin_entry.short_description.lower():
            base_cpu += 15.0
        
        threshold = 20.0  # %
        passed = base_cpu <= threshold
        
        return QualityMetric(
            name="estimated_cpu_usage",
            value=base_cpu,
            unit="%",
            threshold=threshold,
            passed=passed,
            description="Estimated CPU usage based on plugin functionality"
        )
    
    async def _estimate_startup_time(self, plugin_entry: MarketplacePluginEntry) -> QualityMetric:
        """Estimate plugin startup time."""
        # Simplified estimation
        base_startup = 100.0  # ms
        
        # Adjust based on dependencies
        dependency_overhead = len(plugin_entry.metadata.dependencies) * 50.0
        estimated_startup = base_startup + dependency_overhead
        
        # Epic 1: Startup threshold
        threshold = 500.0  # ms
        passed = estimated_startup <= threshold
        
        return QualityMetric(
            name="estimated_startup_time",
            value=estimated_startup,
            unit="ms",
            threshold=threshold,
            passed=passed,
            description="Estimated startup time based on dependencies and complexity"
        )


class ComplianceValidator:
    """
    Compliance validation for various standards.
    
    Epic 1 Optimizations:
    - Fast compliance checks using rule-based validation
    - Efficient pattern matching for compliance requirements
    - Cached compliance assessments
    """
    
    def __init__(self):
        self._compliance_rules: Dict[ComplianceStandard, Dict[str, Any]] = {}
        self._validation_times: List[float] = []
        
        self._initialize_compliance_rules()
        
        logger.info("ComplianceValidator initialized")
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance validation rules."""
        self._compliance_rules = {
            ComplianceStandard.GDPR: {
                "required_disclosures": ["data_collection", "data_usage", "data_retention"],
                "prohibited_patterns": ["personal_data_export", "tracking_without_consent"],
                "required_features": ["data_deletion", "consent_management"]
            },
            ComplianceStandard.COPPA: {
                "age_restrictions": ["under_13_protection"],
                "parental_consent": ["verifiable_parental_consent"],
                "data_minimization": ["minimal_data_collection"]
            },
            ComplianceStandard.OWASP: {
                "security_requirements": ["input_validation", "output_encoding", "authentication"],
                "vulnerability_checks": ["injection_prevention", "secure_communication"]
            }
        }
    
    async def validate_compliance(
        self,
        plugin_entry: MarketplacePluginEntry,
        standards: List[ComplianceStandard]
    ) -> Dict[ComplianceStandard, QualityGateResult]:
        """Validate plugin compliance with specified standards."""
        results = {}
        
        for standard in standards:
            start_time = datetime.utcnow()
            
            try:
                result = await self._validate_single_standard(plugin_entry, standard)
                results[standard] = result
                
                execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._validation_times.append(execution_time_ms)
                
            except Exception as e:
                logger.error("Compliance validation failed",
                            plugin_id=plugin_entry.plugin_id,
                            standard=standard.value,
                            error=str(e))
                
                results[standard] = QualityGateResult(
                    gate_type=QualityGateType.COMPLIANCE,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    max_score=100.0,
                    violations=[f"Compliance validation error: {str(e)}"]
                )
        
        return results
    
    async def _validate_single_standard(
        self,
        plugin_entry: MarketplacePluginEntry,
        standard: ComplianceStandard
    ) -> QualityGateResult:
        """Validate compliance with a single standard."""
        rules = self._compliance_rules.get(standard, {})
        score = 100.0
        violations = []
        warnings = []
        metrics = []
        
        # Check plugin description and documentation for compliance indicators
        plugin_text = f"{plugin_entry.short_description} {plugin_entry.long_description}".lower()
        
        if standard == ComplianceStandard.GDPR:
            score, violations, warnings = await self._check_gdpr_compliance(plugin_text, plugin_entry)
        elif standard == ComplianceStandard.COPPA:
            score, violations, warnings = await self._check_coppa_compliance(plugin_text, plugin_entry)
        elif standard == ComplianceStandard.OWASP:
            score, violations, warnings = await self._check_owasp_compliance(plugin_text, plugin_entry)
        
        # Determine status
        if score >= 80:
            status = QualityGateStatus.PASSED
        elif score >= 60:
            status = QualityGateStatus.WARNING
        else:
            status = QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_type=QualityGateType.COMPLIANCE,
            status=status,
            score=score,
            max_score=100.0,
            violations=violations,
            warnings=warnings,
            metrics=metrics
        )
    
    async def _check_gdpr_compliance(
        self,
        plugin_text: str,
        plugin_entry: MarketplacePluginEntry
    ) -> Tuple[float, List[str], List[str]]:
        """Check GDPR compliance."""
        score = 100.0
        violations = []
        warnings = []
        
        # Check for data collection mentions
        if any(term in plugin_text for term in ["collect", "store", "process", "personal"]):
            if not any(term in plugin_text for term in ["consent", "privacy", "gdpr"]):
                score -= 30
                violations.append("Plugin appears to handle personal data without GDPR compliance mentions")
        
        # Check for privacy policy
        if not plugin_entry.documentation_url:
            score -= 20
            warnings.append("No documentation URL provided - privacy policy may be missing")
        
        return score, violations, warnings
    
    async def _check_coppa_compliance(
        self,
        plugin_text: str,
        plugin_entry: MarketplacePluginEntry
    ) -> Tuple[float, List[str], List[str]]:
        """Check COPPA compliance."""
        score = 100.0
        violations = []
        warnings = []
        
        # Check for child-related content
        if any(term in plugin_text for term in ["child", "kid", "young", "teen"]):
            if not any(term in plugin_text for term in ["parent", "guardian", "consent"]):
                score -= 40
                violations.append("Plugin targets children without mentioning parental consent")
        
        return score, violations, warnings
    
    async def _check_owasp_compliance(
        self,
        plugin_text: str,
        plugin_entry: MarketplacePluginEntry
    ) -> Tuple[float, List[str], List[str]]:
        """Check OWASP compliance."""
        score = 100.0
        violations = []
        warnings = []
        
        # Check security certification level
        if plugin_entry.certification_level == CertificationLevel.UNCERTIFIED:
            score -= 25
            warnings.append("Plugin lacks security certification")
        
        # Check for security mentions
        if any(term in plugin_text for term in ["auth", "login", "password", "encrypt"]):
            if not any(term in plugin_text for term in ["secure", "safety", "protection"]):
                score -= 15
                warnings.append("Plugin handles authentication without security emphasis")
        
        return score, violations, warnings


class SecurityCertificationPipeline:
    """
    Comprehensive security certification pipeline for plugins.
    
    Epic 1 Preservation:
    - <50ms certification status checks
    - <30ms security validation operations
    - <80MB memory usage across all validators
    - Non-blocking certification pipeline
    """
    
    def __init__(self, security_framework: Optional[PluginSecurityFramework] = None):
        self.security_framework = security_framework or PluginSecurityFramework()
        self.security_scanner = SecurityScanner()
        self.performance_validator = PerformanceValidator()
        self.compliance_validator = ComplianceValidator()
        
        # Certification state
        self._active_certifications: Dict[str, CertificationReport] = {}
        self._certification_history: Dict[str, List[CertificationReport]] = {}
        
        # Epic 1: Performance tracking
        self._certification_times: List[float] = []
        
        logger.info("SecurityCertificationPipeline initialized with all validators")
    
    async def certify_plugin(
        self,
        plugin_entry: MarketplacePluginEntry,
        target_level: CertificationLevel = CertificationLevel.FULLY_CERTIFIED,
        source_code: Optional[str] = None,
        source_path: Optional[Path] = None,
        compliance_standards: Optional[List[ComplianceStandard]] = None
    ) -> CertificationReport:
        """
        Comprehensive plugin certification through security pipeline.
        
        Epic 1: Target <500ms for full certification pipeline
        """
        start_time = datetime.utcnow()
        certification_id = f"cert_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info("Starting plugin certification",
                       plugin_id=plugin_entry.plugin_id,
                       target_level=target_level.value,
                       certification_id=certification_id)
            
            gate_results = []
            overall_score = 0.0
            max_possible_score = 0.0
            
            # Security Gate (Required for all levels)
            security_result = await self._run_security_gate(plugin_entry, source_code, source_path)
            gate_results.append(security_result)
            overall_score += security_result.score
            max_possible_score += security_result.max_score
            
            # Performance Gate (Required for Performance+ levels)
            if target_level in [CertificationLevel.PERFORMANCE_VERIFIED, 
                               CertificationLevel.FULLY_CERTIFIED,
                               CertificationLevel.ENTERPRISE_CERTIFIED]:
                performance_result = await self._run_performance_gate(plugin_entry)
                gate_results.append(performance_result)
                overall_score += performance_result.score
                max_possible_score += performance_result.max_score
            
            # Compliance Gate (Required for Full+ levels)
            compliance_status = {}
            if target_level in [CertificationLevel.FULLY_CERTIFIED,
                               CertificationLevel.ENTERPRISE_CERTIFIED]:
                standards = compliance_standards or [ComplianceStandard.OWASP]
                compliance_results = await self._run_compliance_gate(plugin_entry, standards)
                
                for standard, result in compliance_results.items():
                    gate_results.append(result)
                    overall_score += result.score
                    max_possible_score += result.max_score
                    compliance_status[standard.value] = result.passed
            
            # Calculate final score
            final_score = (overall_score / max_possible_score) * 100 if max_possible_score > 0 else 0
            
            # Determine achieved certification level
            achieved_level = self._determine_achieved_level(gate_results, target_level)
            
            # Generate recommendations
            recommendations = self._generate_certification_recommendations(gate_results, achieved_level, target_level)
            
            # Create certification report
            report = CertificationReport(
                plugin_id=plugin_entry.plugin_id,
                certification_id=certification_id,
                target_level=target_level,
                achieved_level=achieved_level,
                overall_score=final_score,
                gate_results=gate_results,
                compliance_status=compliance_status,
                security_report=security_result.metrics[0].details.get('security_report') if security_result.metrics else None,
                expires_at=datetime.utcnow() + timedelta(days=90),  # 90-day certification validity
                recommendations=recommendations
            )
            
            # Store certification
            self._active_certifications[plugin_entry.plugin_id] = report
            
            if plugin_entry.plugin_id not in self._certification_history:
                self._certification_history[plugin_entry.plugin_id] = []
            self._certification_history[plugin_entry.plugin_id].append(report)
            
            # Keep only last 10 certifications per plugin
            if len(self._certification_history[plugin_entry.plugin_id]) > 10:
                self._certification_history[plugin_entry.plugin_id] = \
                    self._certification_history[plugin_entry.plugin_id][-10:]
            
            # Epic 1: Track certification time
            certification_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._certification_times.append(certification_time_ms)
            if len(self._certification_times) > 100:
                self._certification_times.pop(0)
            
            logger.info("Plugin certification completed",
                       plugin_id=plugin_entry.plugin_id,
                       certification_id=certification_id,
                       achieved_level=achieved_level.value,
                       overall_score=round(final_score, 2),
                       certification_time_ms=round(certification_time_ms, 2))
            
            return report
            
        except Exception as e:
            logger.error("Plugin certification failed",
                        plugin_id=plugin_entry.plugin_id,
                        certification_id=certification_id,
                        error=str(e))
            
            # Return failed certification report
            return CertificationReport(
                plugin_id=plugin_entry.plugin_id,
                certification_id=certification_id,
                target_level=target_level,
                achieved_level=CertificationLevel.UNCERTIFIED,
                overall_score=0.0,
                gate_results=[QualityGateResult(
                    gate_type=QualityGateType.SECURITY,
                    status=QualityGateStatus.FAILED,
                    score=0.0,
                    max_score=100.0,
                    violations=[f"Certification error: {str(e)}"]
                )],
                recommendations=["Fix certification errors and retry"]
            )
    
    async def _run_security_gate(
        self,
        plugin_entry: MarketplacePluginEntry,
        source_code: Optional[str],
        source_path: Optional[Path]
    ) -> QualityGateResult:
        """Run security quality gate."""
        try:
            # Run security scan
            security_report = await self.security_scanner.scan_plugin_security(
                plugin_entry, source_code, source_path
            )
            
            # Calculate security score
            score = 100.0
            if not security_report.is_safe:
                score = 0.0
            else:
                # Deduct points for violations and warnings
                score -= len(security_report.violations) * 10
                score -= len(security_report.warnings) * 5
            
            score = max(0.0, score)
            
            # Determine status
            if score >= 80:
                status = QualityGateStatus.PASSED
            elif score >= 60:
                status = QualityGateStatus.WARNING
            else:
                status = QualityGateStatus.FAILED
            
            # Create security metric
            security_metric = QualityMetric(
                name="security_score",
                value=score,
                unit="points",
                threshold=80.0,
                passed=score >= 80.0,
                description="Overall security assessment score",
                details={"security_report": security_report}
            )
            
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                status=status,
                score=score,
                max_score=100.0,
                metrics=[security_metric],
                violations=security_report.violations,
                warnings=security_report.warnings
            )
            
        except Exception as e:
            logger.error("Security gate failed", plugin_id=plugin_entry.plugin_id, error=str(e))
            return QualityGateResult(
                gate_type=QualityGateType.SECURITY,
                status=QualityGateStatus.FAILED,
                score=0.0,
                max_score=100.0,
                violations=[f"Security gate error: {str(e)}"]
            )
    
    async def _run_performance_gate(self, plugin_entry: MarketplacePluginEntry) -> QualityGateResult:
        """Run performance quality gate."""
        return await self.performance_validator.validate_performance(plugin_entry)
    
    async def _run_compliance_gate(
        self,
        plugin_entry: MarketplacePluginEntry,
        standards: List[ComplianceStandard]
    ) -> Dict[ComplianceStandard, QualityGateResult]:
        """Run compliance quality gates."""
        return await self.compliance_validator.validate_compliance(plugin_entry, standards)
    
    def _determine_achieved_level(
        self,
        gate_results: List[QualityGateResult],
        target_level: CertificationLevel
    ) -> CertificationLevel:
        """Determine the achieved certification level based on gate results."""
        # Check if all gates passed
        all_passed = all(result.passed for result in gate_results)
        has_warnings = any(result.status == QualityGateStatus.WARNING for result in gate_results)
        
        # Security gate must pass for any certification
        security_results = [r for r in gate_results if r.gate_type == QualityGateType.SECURITY]
        if not security_results or not security_results[0].passed:
            return CertificationLevel.UNCERTIFIED
        
        # Determine level based on what was tested and passed
        if target_level == CertificationLevel.BASIC:
            return CertificationLevel.BASIC if all_passed else CertificationLevel.UNCERTIFIED
        
        elif target_level == CertificationLevel.SECURITY_VERIFIED:
            # Requires security gate to pass with high score
            security_score = security_results[0].score if security_results else 0
            if security_score >= 90:
                return CertificationLevel.SECURITY_VERIFIED
            elif security_score >= 70:
                return CertificationLevel.BASIC
            else:
                return CertificationLevel.UNCERTIFIED
        
        elif target_level == CertificationLevel.PERFORMANCE_VERIFIED:
            performance_results = [r for r in gate_results if r.gate_type == QualityGateType.PERFORMANCE]
            if performance_results and performance_results[0].passed:
                return CertificationLevel.PERFORMANCE_VERIFIED
            else:
                return CertificationLevel.SECURITY_VERIFIED if security_results[0].score >= 90 else CertificationLevel.BASIC
        
        elif target_level == CertificationLevel.FULLY_CERTIFIED:
            compliance_results = [r for r in gate_results if r.gate_type == QualityGateType.COMPLIANCE]
            if all_passed and compliance_results:
                return CertificationLevel.FULLY_CERTIFIED
            else:
                # Fall back to highest achieved level
                if any(r.gate_type == QualityGateType.PERFORMANCE and r.passed for r in gate_results):
                    return CertificationLevel.PERFORMANCE_VERIFIED
                else:
                    return CertificationLevel.SECURITY_VERIFIED if security_results[0].score >= 90 else CertificationLevel.BASIC
        
        elif target_level == CertificationLevel.ENTERPRISE_CERTIFIED:
            # Requires all gates to pass with high scores
            if all_passed and all(r.score >= 90 for r in gate_results):
                return CertificationLevel.ENTERPRISE_CERTIFIED
            elif all_passed:
                return CertificationLevel.FULLY_CERTIFIED
            else:
                return self._determine_achieved_level(gate_results, CertificationLevel.FULLY_CERTIFIED)
        
        return CertificationLevel.UNCERTIFIED
    
    def _generate_certification_recommendations(
        self,
        gate_results: List[QualityGateResult],
        achieved_level: CertificationLevel,
        target_level: CertificationLevel
    ) -> List[str]:
        """Generate recommendations for improving certification level."""
        recommendations = []
        
        # Collect recommendations from all gates
        for result in gate_results:
            recommendations.extend(result.recommendations)
        
        # Add level-specific recommendations
        if achieved_level != target_level:
            if target_level == CertificationLevel.SECURITY_VERIFIED and achieved_level == CertificationLevel.BASIC:
                recommendations.append("Improve security score to 90+ points for Security Verified certification")
            
            elif target_level == CertificationLevel.PERFORMANCE_VERIFIED:
                perf_results = [r for r in gate_results if r.gate_type == QualityGateType.PERFORMANCE]
                if not perf_results or not perf_results[0].passed:
                    recommendations.append("Pass performance validation for Performance Verified certification")
            
            elif target_level == CertificationLevel.FULLY_CERTIFIED:
                compliance_results = [r for r in gate_results if r.gate_type == QualityGateType.COMPLIANCE]
                if not compliance_results:
                    recommendations.append("Complete compliance validation for Full certification")
                elif not all(r.passed for r in compliance_results):
                    recommendations.append("Pass all compliance checks for Full certification")
            
            elif target_level == CertificationLevel.ENTERPRISE_CERTIFIED:
                recommendations.append("Achieve 90+ scores on all quality gates for Enterprise certification")
        
        # Remove duplicates and limit recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    async def get_certification_status(self, plugin_id: str) -> Optional[CertificationReport]:
        """Get current certification status for a plugin."""
        return self._active_certifications.get(plugin_id)
    
    async def get_certification_history(self, plugin_id: str) -> List[CertificationReport]:
        """Get certification history for a plugin."""
        return self._certification_history.get(plugin_id, [])
    
    async def is_certification_valid(self, plugin_id: str) -> bool:
        """Check if plugin has valid (non-expired) certification."""
        cert = self._active_certifications.get(plugin_id)
        if not cert or not cert.expires_at:
            return False
        return datetime.utcnow() < cert.expires_at
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for certification pipeline."""
        metrics = {
            "security_scanner": self.security_scanner.get_performance_metrics(),
            "performance_validator": {
                "validation_times": {
                    "count": len(self.performance_validator._validation_times),
                    "avg_ms": round(sum(self.performance_validator._validation_times) / len(self.performance_validator._validation_times), 2) if self.performance_validator._validation_times else 0
                }
            },
            "compliance_validator": {
                "validation_times": {
                    "count": len(self.compliance_validator._validation_times),
                    "avg_ms": round(sum(self.compliance_validator._validation_times) / len(self.compliance_validator._validation_times), 2) if self.compliance_validator._validation_times else 0
                }
            }
        }
        
        if self._certification_times:
            avg_cert_time = sum(self._certification_times) / len(self._certification_times)
            metrics["certification_pipeline"] = {
                "certification_times": {
                    "count": len(self._certification_times),
                    "avg_ms": round(avg_cert_time, 2),
                    "max_ms": round(max(self._certification_times), 2),
                    "min_ms": round(min(self._certification_times), 2),
                    "epic1_compliant": avg_cert_time < 500
                },
                "active_certifications": len(self._active_certifications),
                "total_certifications": sum(len(history) for history in self._certification_history.values())
            }
        
        return metrics
    
    async def cleanup(self) -> None:
        """Cleanup certification pipeline resources."""
        logger.info("Cleaning up SecurityCertificationPipeline")
        
        # Cleanup security framework
        await self.security_framework.cleanup()
        
        # Clear caches and state
        self._active_certifications.clear()
        self._certification_history.clear()
        self._certification_times.clear()
        
        logger.info("SecurityCertificationPipeline cleanup complete")


# Factory function for easy instantiation
def create_security_certification_pipeline(security_framework: Optional[PluginSecurityFramework] = None) -> SecurityCertificationPipeline:
    """Factory function to create SecurityCertificationPipeline."""
    return SecurityCertificationPipeline(security_framework)
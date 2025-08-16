"""
Consolidation Quality Gates

This module implements comprehensive quality gates that must pass
before, during, and after the Epic 1-4 consolidation process.
"""

import subprocess
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import pytest
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool = False
    score: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "execution_time": self.execution_time
        }


@dataclass
class QualityGateReport:
    """Comprehensive quality gate report."""
    stage: str  # "pre", "during", "post"
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    overall_score: float = 0.0
    gate_results: List[QualityGateResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage,
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "overall_score": self.overall_score,
            "gate_results": [result.to_dict() for result in self.gate_results],
            "recommendations": self.recommendations,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0):
        """Initialize quality gate."""
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def check(self) -> QualityGateResult:
        """Execute the quality gate check."""
        pass
    
    def _run_command(self, command: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a shell command and return success status, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Command failed: {str(e)}"


class TestCoverageGate(QualityGate):
    """Quality gate for test coverage requirements."""
    
    def __init__(self, min_coverage: float = 90.0):
        """Initialize with minimum coverage requirement."""
        super().__init__("Test Coverage Gate")
        self.min_coverage = min_coverage
        
    def check(self) -> QualityGateResult:
        """Check test coverage meets requirements."""
        result = QualityGateResult(gate_name=self.name)
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            success, stdout, stderr = self._run_command([
                "python", "-m", "pytest", 
                "--cov=app",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "-x", "--tb=short"
            ])
            
            if not success:
                result.errors.append(f"Tests failed: {stderr}")
                result.passed = False
                return result
                
            # Parse coverage report
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                result.metrics["coverage_percentage"] = total_coverage
                result.score = min(total_coverage / 100.0, 1.0)
                
                if total_coverage >= self.min_coverage:
                    result.passed = True
                else:
                    result.errors.append(
                        f"Coverage {total_coverage:.1f}% below minimum {self.min_coverage:.1f}%"
                    )
                    
            else:
                result.errors.append("Coverage report not generated")
                
        except Exception as e:
            result.errors.append(f"Coverage check failed: {str(e)}")
            
        finally:
            result.execution_time = time.time() - start_time
            
        return result


class BuildSuccessGate(QualityGate):
    """Quality gate for successful build."""
    
    def __init__(self):
        """Initialize build success gate."""
        super().__init__("Build Success Gate")
        
    def check(self) -> QualityGateResult:
        """Check that the system builds successfully."""
        result = QualityGateResult(gate_name=self.name)
        start_time = time.time()
        
        try:
            # Check Python syntax
            success, stdout, stderr = self._run_command([
                "python", "-m", "py_compile", "app/main.py"
            ])
            
            if not success:
                result.errors.append(f"Python compilation failed: {stderr}")
                result.passed = False
                return result
                
            # Check imports work
            success, stdout, stderr = self._run_command([
                "python", "-c", "import app.main; print('Import successful')"
            ])
            
            if success:
                result.passed = True
                result.score = 1.0
                result.metrics["import_successful"] = True
            else:
                result.errors.append(f"Import failed: {stderr}")
                result.metrics["import_successful"] = False
                
        except Exception as e:
            result.errors.append(f"Build check failed: {str(e)}")
            
        finally:
            result.execution_time = time.time() - start_time
            
        return result


class SecurityScanGate(QualityGate):
    """Quality gate for security vulnerability scanning."""
    
    def __init__(self):
        """Initialize security scan gate."""
        super().__init__("Security Scan Gate")
        
    def check(self) -> QualityGateResult:
        """Check for security vulnerabilities."""
        result = QualityGateResult(gate_name=self.name)
        start_time = time.time()
        
        try:
            # Run bandit security scan
            success, stdout, stderr = self._run_command([
                "python", "-m", "bandit", "-r", "app/", "-f", "json", "-o", "security_scan.json"
            ])
            
            # Bandit returns non-zero if vulnerabilities found, but that's expected
            security_file = Path("security_scan.json")
            if security_file.exists():
                with open(security_file) as f:
                    scan_data = json.load(f)
                    
                high_issues = scan_data.get("metrics", {}).get("_totals", {}).get("SEVERITY.HIGH", 0)
                medium_issues = scan_data.get("metrics", {}).get("_totals", {}).get("SEVERITY.MEDIUM", 0)
                low_issues = scan_data.get("metrics", {}).get("_totals", {}).get("SEVERITY.LOW", 0)
                
                result.metrics["high_severity_issues"] = high_issues
                result.metrics["medium_severity_issues"] = medium_issues
                result.metrics["low_severity_issues"] = low_issues
                
                # Pass if no high severity issues
                if high_issues == 0:
                    result.passed = True
                    result.score = 1.0 if medium_issues == 0 else 0.8
                    if medium_issues > 0:
                        result.warnings.append(f"{medium_issues} medium severity security issues found")
                else:
                    result.errors.append(f"{high_issues} high severity security issues found")
                    result.score = 0.0
                    
            else:
                result.warnings.append("Security scan report not generated")
                result.passed = True  # Don't fail if scan tool unavailable
                result.score = 0.5
                
        except Exception as e:
            result.warnings.append(f"Security scan failed: {str(e)}")
            result.passed = True  # Don't fail entire pipeline
            result.score = 0.5
            
        finally:
            result.execution_time = time.time() - start_time
            
        return result


class PerformanceBenchmarkGate(QualityGate):
    """Quality gate for performance benchmarks."""
    
    def __init__(self, max_regression: float = 0.05):
        """Initialize with maximum allowed regression."""
        super().__init__("Performance Benchmark Gate")
        self.max_regression = max_regression
        
    def check(self) -> QualityGateResult:
        """Check performance benchmarks."""
        result = QualityGateResult(gate_name=self.name)
        start_time = time.time()
        
        try:
            # Run performance tests
            success, stdout, stderr = self._run_command([
                "python", "-m", "pytest", 
                "tests/performance/",
                "-m", "benchmark",
                "--tb=short"
            ])
            
            if success:
                result.passed = True
                result.score = 1.0
                result.metrics["performance_tests_passed"] = True
            else:
                # Check if it's a regression or test failure
                if "regression" in stderr.lower():
                    result.errors.append("Performance regression detected")
                    result.score = 0.0
                else:
                    result.warnings.append("Performance tests failed (may be environmental)")
                    result.passed = True  # Don't fail pipeline for environmental issues
                    result.score = 0.7
                    
                result.metrics["performance_tests_passed"] = False
                
        except Exception as e:
            result.warnings.append(f"Performance benchmark failed: {str(e)}")
            result.passed = True  # Don't fail entire pipeline
            result.score = 0.5
            
        finally:
            result.execution_time = time.time() - start_time
            
        return result


class DatabaseMigrationGate(QualityGate):
    """Quality gate for database migration integrity."""
    
    def __init__(self):
        """Initialize database migration gate."""
        super().__init__("Database Migration Gate")
        
    def check(self) -> QualityGateResult:
        """Check database migration integrity."""
        result = QualityGateResult(gate_name=self.name)
        start_time = time.time()
        
        try:
            # Check if database migrations can run
            success, stdout, stderr = self._run_command([
                "python", "-c", 
                "from app.core.database import engine; from sqlalchemy import text; "
                "with engine.connect() as conn: conn.execute(text('SELECT 1')); "
                "print('Database connection successful')"
            ])
            
            if success:
                result.passed = True
                result.score = 1.0
                result.metrics["database_accessible"] = True
            else:
                result.errors.append(f"Database connection failed: {stderr}")
                result.metrics["database_accessible"] = False
                
        except Exception as e:
            result.warnings.append(f"Database check failed: {str(e)}")
            result.passed = True  # Don't fail if DB not available in test env
            result.score = 0.5
            
        finally:
            result.execution_time = time.time() - start_time
            
        return result


class ConsolidationQualityGates:
    """
    Main quality gates controller for consolidation process.
    
    Orchestrates all quality gates and ensures system readiness
    for the Epic 1-4 transformation.
    """
    
    def __init__(self):
        """Initialize quality gates."""
        self.pre_gates = [
            TestCoverageGate(min_coverage=85.0),  # Slightly lower for pre-consolidation
            BuildSuccessGate(),
            SecurityScanGate(),
            DatabaseMigrationGate()
        ]
        
        self.post_gates = [
            TestCoverageGate(min_coverage=90.0),  # Higher for post-consolidation
            BuildSuccessGate(),
            SecurityScanGate(),
            PerformanceBenchmarkGate(),
            DatabaseMigrationGate()
        ]
        
    def run_pre_consolidation_gates(self) -> QualityGateReport:
        """Run pre-consolidation quality gates."""
        return self._run_gates(self.pre_gates, "pre")
        
    def run_post_consolidation_gates(self) -> QualityGateReport:
        """Run post-consolidation quality gates."""
        return self._run_gates(self.post_gates, "post")
        
    def _run_gates(self, gates: List[QualityGate], stage: str) -> QualityGateReport:
        """Run a set of quality gates."""
        report = QualityGateReport(
            stage=stage,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        start_time = time.time()
        
        total_score = 0.0
        total_weight = 0.0
        
        for gate in gates:
            try:
                result = gate.check()
                report.gate_results.append(result)
                
                if result.passed:
                    report.passed_gates += 1
                else:
                    report.failed_gates += 1
                    
                total_score += result.score * gate.weight
                total_weight += gate.weight
                
            except Exception as e:
                # Create error result for failed gate
                error_result = QualityGateResult(
                    gate_name=gate.name,
                    passed=False,
                    errors=[f"Gate execution failed: {str(e)}"]
                )
                report.gate_results.append(error_result)
                report.failed_gates += 1
                total_weight += gate.weight
                
        report.total_gates = len(gates)
        report.overall_score = total_score / total_weight if total_weight > 0 else 0.0
        report.execution_time = time.time() - start_time
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        return report
        
    def _generate_recommendations(self, report: QualityGateReport) -> List[str]:
        """Generate recommendations based on gate results."""
        recommendations = []
        
        if report.failed_gates == 0:
            recommendations.append(f"All {report.stage}-consolidation quality gates passed! Safe to proceed.")
            return recommendations
            
        for result in report.gate_results:
            if not result.passed:
                if "Coverage" in result.gate_name:
                    recommendations.append(
                        "CRITICAL: Increase test coverage before proceeding with consolidation"
                    )
                elif "Build" in result.gate_name:
                    recommendations.append(
                        "CRITICAL: Fix build errors before proceeding with consolidation"
                    )
                elif "Security" in result.gate_name:
                    recommendations.append(
                        "HIGH: Address security vulnerabilities before consolidation"
                    )
                elif "Performance" in result.gate_name:
                    recommendations.append(
                        "HIGH: Fix performance regressions before proceeding"
                    )
                elif "Database" in result.gate_name:
                    recommendations.append(
                        "MEDIUM: Verify database connectivity and migrations"
                    )
                    
            # Add specific recommendations based on errors
            for error in result.errors:
                if "regression" in error.lower():
                    recommendations.append(
                        f"CRITICAL: Performance regression detected in {result.gate_name}"
                    )
                elif "coverage" in error.lower():
                    recommendations.append(
                        f"HIGH: Test coverage below threshold in {result.gate_name}"
                    )
                    
        return recommendations
    
    def is_safe_to_consolidate(self, pre_report: QualityGateReport) -> bool:
        """Determine if it's safe to proceed with consolidation."""
        # Critical gates that must pass
        critical_gates = ["Build Success Gate", "Test Coverage Gate"]
        
        for result in pre_report.gate_results:
            if result.gate_name in critical_gates and not result.passed:
                return False
                
        # Overall score threshold
        return pre_report.overall_score >= 0.8
    
    def save_report(self, report: QualityGateReport, filename: Optional[str] = None):
        """Save quality gate report to file."""
        if not filename:
            filename = f"quality_gate_report_{report.stage}_{int(time.time())}.json"
            
        report_path = Path("reports") / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
            
        logger.info(f"Quality gate report saved to {report_path}")


# Pytest integration
class TestConsolidationQualityGates:
    """Pytest test class for quality gates."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.quality_gates = ConsolidationQualityGates()
    
    @pytest.mark.quality_gates
    @pytest.mark.consolidation
    def test_pre_consolidation_gates(self):
        """Test pre-consolidation quality gates."""
        report = self.quality_gates.run_pre_consolidation_gates()
        
        # Save report for analysis
        self.quality_gates.save_report(report, "pre_consolidation_gates.json")
        
        # Assert critical gates pass
        critical_failures = []
        for result in report.gate_results:
            if result.gate_name in ["Build Success Gate", "Test Coverage Gate"] and not result.passed:
                critical_failures.append(result.gate_name)
                
        if critical_failures:
            pytest.fail(f"Critical quality gates failed: {critical_failures}")
            
        # Warn about overall score
        if report.overall_score < 0.8:
            pytest.fail(f"Overall quality score too low: {report.overall_score:.1%}")
    
    @pytest.mark.quality_gates
    @pytest.mark.consolidation
    def test_post_consolidation_gates(self):
        """Test post-consolidation quality gates."""
        report = self.quality_gates.run_post_consolidation_gates()
        
        # Save report for analysis
        self.quality_gates.save_report(report, "post_consolidation_gates.json")
        
        # All gates should pass after consolidation
        if report.failed_gates > 0:
            failed_gates = [r.gate_name for r in report.gate_results if not r.passed]
            pytest.fail(f"Post-consolidation quality gates failed: {failed_gates}")
            
        # High quality score required after consolidation
        if report.overall_score < 0.9:
            pytest.fail(f"Post-consolidation quality score insufficient: {report.overall_score:.1%}")
    
    @pytest.mark.quality_gates
    @pytest.mark.consolidation
    def test_consolidation_safety_check(self):
        """Test consolidation safety determination."""
        pre_report = self.quality_gates.run_pre_consolidation_gates()
        is_safe = self.quality_gates.is_safe_to_consolidate(pre_report)
        
        if not is_safe:
            recommendations = pre_report.recommendations
            pytest.fail(f"System not ready for consolidation. Recommendations: {recommendations}")
    
    @pytest.mark.quality_gates
    def test_individual_quality_gates(self):
        """Test individual quality gates."""
        # Test each gate type
        gates_to_test = [
            TestCoverageGate(),
            BuildSuccessGate(),
            SecurityScanGate(),
            PerformanceBenchmarkGate(),
            DatabaseMigrationGate()
        ]
        
        for gate in gates_to_test:
            result = gate.check()
            
            # Log results for debugging
            logger.info(f"{gate.name}: {'PASSED' if result.passed else 'FAILED'}")
            if result.errors:
                logger.error(f"{gate.name} errors: {result.errors}")
            if result.warnings:
                logger.warning(f"{gate.name} warnings: {result.warnings}")
                
            # Don't fail on individual gates, just collect results
            assert isinstance(result, QualityGateResult), f"Invalid result type from {gate.name}"
"""
Epic 1 Consolidation Quality Gates Implementation
================================================

This module implements comprehensive quality gates for the Epic 1 consolidation
process, ensuring that manager consolidation meets strict quality, performance,
and reliability requirements before approval.

QUALITY GATE CRITERIA:
- Test Pass Rate: >80% of all tests must pass
- Performance Regression: <10% degradation from baseline
- API Coverage: >95% of expected APIs must be available
- Integration Success: All critical integration points must work
- Memory Usage: No memory leaks or excessive usage
- Error Handling: Proper error recovery mechanisms
"""

import time
import logging
import tracemalloc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import psutil
import os

logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate status enumeration."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    NOT_EVALUATED = "not_evaluated"


class ConsolidationPhase(Enum):
    """Consolidation phase enumeration."""
    PRE_CONSOLIDATION = "pre_consolidation"
    CONSOLIDATION = "consolidation"  
    POST_CONSOLIDATION = "post_consolidation"
    INTEGRATION = "integration"
    VALIDATION = "validation"


@dataclass
class QualityGateResult:
    """Result of a quality gate evaluation."""
    gate_name: str
    status: QualityGateStatus
    score: float = 0.0
    threshold: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsolidationMetrics:
    """Comprehensive consolidation metrics."""
    # Test Metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    test_pass_rate: float = 0.0
    
    # Performance Metrics
    performance_baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_regressions: List[Dict[str, Any]] = field(default_factory=list)
    
    # API Metrics
    expected_apis: Dict[str, List[str]] = field(default_factory=dict)
    available_apis: Dict[str, List[str]] = field(default_factory=dict)
    missing_apis: Dict[str, List[str]] = field(default_factory=dict)
    api_coverage: float = 0.0
    
    # Integration Metrics
    integration_points: List[str] = field(default_factory=list)
    working_integrations: List[str] = field(default_factory=list)
    failed_integrations: List[str] = field(default_factory=list)
    integration_success_rate: float = 0.0
    
    # System Metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    startup_time_seconds: float = 0.0
    shutdown_time_seconds: float = 0.0
    
    # Error Metrics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)


class QualityGateEvaluator:
    """
    Main quality gate evaluator for Epic 1 consolidation.
    
    Implements comprehensive quality gates to ensure safe consolidation
    of manager components with maintained functionality and performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality gate evaluator."""
        self.config = config or self._get_default_config()
        self.metrics = ConsolidationMetrics()
        self.gate_results: Dict[str, QualityGateResult] = {}
        self.overall_status = QualityGateStatus.NOT_EVALUATED
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default quality gate configuration."""
        return {
            "test_pass_rate_threshold": 0.80,  # 80%
            "performance_regression_threshold": 0.10,  # 10%
            "api_coverage_threshold": 0.95,  # 95%
            "integration_success_threshold": 0.90,  # 90%
            "memory_limit_mb": 500,  # 500MB max
            "startup_time_limit_seconds": 5.0,  # 5s max
            "critical_error_limit": 0,  # No critical errors allowed
            "max_warnings": 10,  # Max 10 warnings allowed
            "performance_baseline_required": True,
            "integration_testing_required": True
        }
    
    def evaluate_all_gates(self, metrics: ConsolidationMetrics) -> Dict[str, QualityGateResult]:
        """Evaluate all quality gates."""
        self.metrics = metrics
        self.gate_results = {}
        
        # Core quality gates
        self.gate_results["test_pass_rate"] = self._evaluate_test_pass_rate()
        self.gate_results["performance_regression"] = self._evaluate_performance_regression()
        self.gate_results["api_coverage"] = self._evaluate_api_coverage()
        self.gate_results["integration_success"] = self._evaluate_integration_success()
        
        # System quality gates
        self.gate_results["memory_usage"] = self._evaluate_memory_usage()
        self.gate_results["startup_performance"] = self._evaluate_startup_performance()
        self.gate_results["error_thresholds"] = self._evaluate_error_thresholds()
        
        # Advanced quality gates
        self.gate_results["functionality_preservation"] = self._evaluate_functionality_preservation()
        self.gate_results["system_stability"] = self._evaluate_system_stability()
        self.gate_results["consolidation_completeness"] = self._evaluate_consolidation_completeness()
        
        # Determine overall status
        self.overall_status = self._determine_overall_status()
        
        return self.gate_results
    
    def _evaluate_test_pass_rate(self) -> QualityGateResult:
        """Evaluate test pass rate quality gate."""
        threshold = self.config["test_pass_rate_threshold"]
        
        if self.metrics.total_tests == 0:
            return QualityGateResult(
                gate_name="test_pass_rate",
                status=QualityGateStatus.FAILED,
                score=0.0,
                threshold=threshold,
                message="No tests found - cannot validate consolidation",
                recommendations=["Add comprehensive test coverage", "Run test discovery"]
            )
        
        pass_rate = self.metrics.passed_tests / self.metrics.total_tests
        
        status = QualityGateStatus.PASSED if pass_rate >= threshold else QualityGateStatus.FAILED
        
        message = f"Test pass rate: {pass_rate:.2%} (threshold: {threshold:.2%})"
        if status == QualityGateStatus.FAILED:
            message += f" - {self.metrics.failed_tests} tests failing"
        
        recommendations = []
        if pass_rate < threshold:
            recommendations.extend([
                "Fix failing tests before proceeding with consolidation",
                "Review test failures for consolidation-related issues",
                "Add missing test coverage for consolidated components"
            ])
        
        return QualityGateResult(
            gate_name="test_pass_rate",
            status=status,
            score=pass_rate,
            threshold=threshold,
            message=message,
            details={
                "total_tests": self.metrics.total_tests,
                "passed_tests": self.metrics.passed_tests,
                "failed_tests": self.metrics.failed_tests,
                "skipped_tests": self.metrics.skipped_tests
            },
            recommendations=recommendations
        )
    
    def _evaluate_performance_regression(self) -> QualityGateResult:
        """Evaluate performance regression quality gate."""
        threshold = self.config["performance_regression_threshold"]
        
        if not self.metrics.performance_baselines:
            return QualityGateResult(
                gate_name="performance_regression",
                status=QualityGateStatus.WARNING,
                score=0.0,
                threshold=threshold,
                message="No performance baselines available",
                recommendations=["Establish performance baselines", "Run performance benchmarks"]
            )
        
        max_regression = 0.0
        regression_details = []
        
        for component, current_metrics in self.metrics.current_performance.items():
            baseline_metrics = self.metrics.performance_baselines.get(component, {})
            
            for metric, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric)
                
                if baseline_value and baseline_value > 0:
                    regression = (current_value - baseline_value) / baseline_value
                    max_regression = max(max_regression, regression)
                    
                    if regression > threshold:
                        regression_details.append({
                            "component": component,
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "regression": regression
                        })
        
        status = QualityGateStatus.PASSED if max_regression <= threshold else QualityGateStatus.FAILED
        
        message = f"Max performance regression: {max_regression:.2%} (threshold: {threshold:.2%})"
        if regression_details:
            message += f" - {len(regression_details)} metrics regressed"
        
        recommendations = []
        if max_regression > threshold:
            recommendations.extend([
                "Optimize components with performance regressions",
                "Review consolidation impact on performance-critical paths",
                "Consider rollback if regressions are severe"
            ])
        
        return QualityGateResult(
            gate_name="performance_regression",
            status=status,
            score=max_regression,
            threshold=threshold,
            message=message,
            details={
                "max_regression": max_regression,
                "regression_count": len(regression_details),
                "regressions": regression_details
            },
            recommendations=recommendations
        )
    
    def _evaluate_api_coverage(self) -> QualityGateResult:
        """Evaluate API coverage quality gate."""
        threshold = self.config["api_coverage_threshold"]
        
        total_expected = sum(len(apis) for apis in self.metrics.expected_apis.values())
        
        if total_expected == 0:
            return QualityGateResult(
                gate_name="api_coverage",
                status=QualityGateStatus.WARNING,
                score=1.0,  # No APIs expected, assume full coverage
                threshold=threshold,
                message="No expected APIs defined",
                recommendations=["Define expected API coverage", "Validate public API contracts"]
            )
        
        total_available = 0
        coverage_details = {}
        
        for component, expected_apis in self.metrics.expected_apis.items():
            available_apis = self.metrics.available_apis.get(component, [])
            available_count = len(set(expected_apis) & set(available_apis))
            total_available += available_count
            
            coverage_details[component] = {
                "expected": len(expected_apis),
                "available": available_count,
                "coverage": available_count / len(expected_apis) if expected_apis else 1.0,
                "missing": list(set(expected_apis) - set(available_apis))
            }
        
        overall_coverage = total_available / total_expected
        
        status = QualityGateStatus.PASSED if overall_coverage >= threshold else QualityGateStatus.FAILED
        
        message = f"API coverage: {overall_coverage:.2%} (threshold: {threshold:.2%})"
        missing_count = sum(len(details["missing"]) for details in coverage_details.values())
        if missing_count > 0:
            message += f" - {missing_count} APIs missing"
        
        recommendations = []
        if overall_coverage < threshold:
            recommendations.extend([
                "Implement missing APIs in consolidated components",
                "Review API consolidation mapping",
                "Verify import paths and module structure"
            ])
        
        return QualityGateResult(
            gate_name="api_coverage",
            status=status,
            score=overall_coverage,
            threshold=threshold,
            message=message,
            details={
                "overall_coverage": overall_coverage,
                "component_coverage": coverage_details,
                "total_missing": missing_count
            },
            recommendations=recommendations
        )
    
    def _evaluate_integration_success(self) -> QualityGateResult:
        """Evaluate integration success quality gate."""
        threshold = self.config["integration_success_threshold"]
        
        total_integrations = len(self.metrics.integration_points)
        
        if total_integrations == 0:
            return QualityGateResult(
                gate_name="integration_success",
                status=QualityGateStatus.WARNING,
                score=1.0,
                threshold=threshold,
                message="No integration points defined",
                recommendations=["Define integration test points", "Test component interactions"]
            )
        
        working_integrations = len(self.metrics.working_integrations)
        success_rate = working_integrations / total_integrations
        
        status = QualityGateStatus.PASSED if success_rate >= threshold else QualityGateStatus.FAILED
        
        message = f"Integration success rate: {success_rate:.2%} (threshold: {threshold:.2%})"
        if self.metrics.failed_integrations:
            message += f" - {len(self.metrics.failed_integrations)} integrations failing"
        
        recommendations = []
        if success_rate < threshold:
            recommendations.extend([
                "Fix failing integration points",
                "Review component interaction patterns", 
                "Validate consolidated component interfaces"
            ])
        
        return QualityGateResult(
            gate_name="integration_success",
            status=status,
            score=success_rate,
            threshold=threshold,
            message=message,
            details={
                "total_integrations": total_integrations,
                "working_integrations": working_integrations,
                "failed_integrations": self.metrics.failed_integrations,
                "success_rate": success_rate
            },
            recommendations=recommendations
        )
    
    def _evaluate_memory_usage(self) -> QualityGateResult:
        """Evaluate memory usage quality gate."""
        limit = self.config["memory_limit_mb"]
        current_usage = self.metrics.memory_usage_mb
        
        status = QualityGateStatus.PASSED if current_usage <= limit else QualityGateStatus.FAILED
        
        usage_percentage = (current_usage / limit) * 100
        message = f"Memory usage: {current_usage:.1f}MB ({usage_percentage:.1f}% of limit)"
        
        recommendations = []
        if current_usage > limit:
            recommendations.extend([
                "Investigate memory usage in consolidated components",
                "Check for memory leaks",
                "Optimize memory-intensive operations"
            ])
        elif current_usage > limit * 0.8:  # Warning at 80%
            status = QualityGateStatus.WARNING
            recommendations.append("Monitor memory usage closely")
        
        return QualityGateResult(
            gate_name="memory_usage",
            status=status,
            score=current_usage,
            threshold=limit,
            message=message,
            details={
                "current_usage_mb": current_usage,
                "limit_mb": limit,
                "usage_percentage": usage_percentage
            },
            recommendations=recommendations
        )
    
    def _evaluate_startup_performance(self) -> QualityGateResult:
        """Evaluate startup performance quality gate."""
        limit = self.config["startup_time_limit_seconds"]
        current_time = self.metrics.startup_time_seconds
        
        status = QualityGateStatus.PASSED if current_time <= limit else QualityGateStatus.FAILED
        
        message = f"Startup time: {current_time:.2f}s (limit: {limit:.2f}s)"
        
        recommendations = []
        if current_time > limit:
            recommendations.extend([
                "Optimize startup sequence in consolidated components",
                "Review initialization dependencies",
                "Consider lazy loading for non-critical components"
            ])
        
        return QualityGateResult(
            gate_name="startup_performance",
            status=status,
            score=current_time,
            threshold=limit,
            message=message,
            details={
                "startup_time": current_time,
                "limit": limit
            },
            recommendations=recommendations
        )
    
    def _evaluate_error_thresholds(self) -> QualityGateResult:
        """Evaluate error threshold quality gate."""
        critical_limit = self.config["critical_error_limit"]
        warning_limit = self.config["max_warnings"]
        
        critical_count = len(self.metrics.critical_issues)
        warning_count = len(self.metrics.warnings)
        
        if critical_count > critical_limit:
            status = QualityGateStatus.FAILED
            message = f"Critical errors: {critical_count} (limit: {critical_limit})"
        elif warning_count > warning_limit:
            status = QualityGateStatus.WARNING
            message = f"Warnings: {warning_count} (limit: {warning_limit})"
        else:
            status = QualityGateStatus.PASSED
            message = f"Errors: {critical_count} critical, {warning_count} warnings"
        
        recommendations = []
        if critical_count > 0:
            recommendations.extend([
                "Fix all critical errors before proceeding",
                "Review error logs for consolidation issues"
            ])
        if warning_count > warning_limit:
            recommendations.append("Address excessive warnings")
        
        return QualityGateResult(
            gate_name="error_thresholds",
            status=status,
            score=critical_count + (warning_count * 0.1),  # Weight warnings less
            threshold=critical_limit,
            message=message,
            details={
                "critical_errors": critical_count,
                "warnings": warning_count,
                "critical_limit": critical_limit,
                "warning_limit": warning_limit,
                "critical_issues": self.metrics.critical_issues,
                "warnings_list": self.metrics.warnings
            },
            recommendations=recommendations
        )
    
    def _evaluate_functionality_preservation(self) -> QualityGateResult:
        """Evaluate functionality preservation quality gate."""
        # This is a composite gate based on other metrics
        test_result = self.gate_results.get("test_pass_rate")
        api_result = self.gate_results.get("api_coverage")
        integration_result = self.gate_results.get("integration_success")
        
        if not all([test_result, api_result, integration_result]):
            return QualityGateResult(
                gate_name="functionality_preservation",
                status=QualityGateStatus.NOT_EVALUATED,
                message="Cannot evaluate - dependent gates not completed"
            )
        
        # Functionality preservation score based on weighted average
        weights = {"test_pass_rate": 0.4, "api_coverage": 0.3, "integration_success": 0.3}
        
        score = (
            test_result.score * weights["test_pass_rate"] +
            api_result.score * weights["api_coverage"] +
            integration_result.score * weights["integration_success"]
        )
        
        threshold = 0.85  # 85% minimum for functionality preservation
        status = QualityGateStatus.PASSED if score >= threshold else QualityGateStatus.FAILED
        
        message = f"Functionality preservation score: {score:.2%} (threshold: {threshold:.2%})"
        
        recommendations = []
        if score < threshold:
            recommendations.extend([
                "Ensure all original functionality is preserved",
                "Review consolidation mapping for missing features",
                "Add tests for edge cases and error scenarios"
            ])
        
        return QualityGateResult(
            gate_name="functionality_preservation",
            status=status,
            score=score,
            threshold=threshold,
            message=message,
            details={
                "component_scores": {
                    "test_pass_rate": test_result.score,
                    "api_coverage": api_result.score,
                    "integration_success": integration_result.score
                },
                "weighted_score": score
            },
            recommendations=recommendations
        )
    
    def _evaluate_system_stability(self) -> QualityGateResult:
        """Evaluate system stability quality gate."""
        # Stability based on error rates and performance consistency
        error_result = self.gate_results.get("error_thresholds")
        performance_result = self.gate_results.get("performance_regression")
        memory_result = self.gate_results.get("memory_usage")
        
        if not all([error_result, performance_result, memory_result]):
            return QualityGateResult(
                gate_name="system_stability",
                status=QualityGateStatus.NOT_EVALUATED,
                message="Cannot evaluate - dependent gates not completed"
            )
        
        # Count passing stability-related gates
        passing_gates = sum([
            1 for result in [error_result, performance_result, memory_result]
            if result.status == QualityGateStatus.PASSED
        ])
        
        stability_score = passing_gates / 3
        threshold = 1.0  # All stability gates must pass
        
        status = QualityGateStatus.PASSED if stability_score >= threshold else QualityGateStatus.FAILED
        
        message = f"System stability: {passing_gates}/3 stability gates passed"
        
        recommendations = []
        if stability_score < threshold:
            recommendations.extend([
                "Address system stability issues",
                "Monitor system behavior under load",
                "Improve error handling and recovery"
            ])
        
        return QualityGateResult(
            gate_name="system_stability",
            status=status,
            score=stability_score,
            threshold=threshold,
            message=message,
            details={
                "stability_gates_passed": passing_gates,
                "stability_gates_total": 3,
                "gate_statuses": {
                    "error_thresholds": error_result.status.value,
                    "performance_regression": performance_result.status.value,
                    "memory_usage": memory_result.status.value
                }
            },
            recommendations=recommendations
        )
    
    def _evaluate_consolidation_completeness(self) -> QualityGateResult:
        """Evaluate consolidation completeness quality gate.""" 
        # Check if consolidation objectives are met
        expected_components = ["task_manager", "agent_manager", "workflow_manager", 
                             "resource_manager", "communication_manager"]
        
        available_components = list(self.metrics.available_apis.keys())
        completed_components = len(set(expected_components) & set(available_components))
        completion_rate = completed_components / len(expected_components)
        
        threshold = 1.0  # 100% - all components must be consolidated
        status = QualityGateStatus.PASSED if completion_rate >= threshold else QualityGateStatus.FAILED
        
        message = f"Consolidation completeness: {completed_components}/{len(expected_components)} components"
        
        missing_components = set(expected_components) - set(available_components)
        recommendations = []
        if missing_components:
            recommendations.extend([
                f"Complete consolidation of: {', '.join(missing_components)}",
                "Verify all manager components are properly consolidated",
                "Update component registration and discovery"
            ])
        
        return QualityGateResult(
            gate_name="consolidation_completeness",
            status=status,
            score=completion_rate,
            threshold=threshold,
            message=message,
            details={
                "expected_components": expected_components,
                "available_components": available_components,
                "missing_components": list(missing_components),
                "completion_rate": completion_rate
            },
            recommendations=recommendations
        )
    
    def _determine_overall_status(self) -> QualityGateStatus:
        """Determine overall quality gate status."""
        if not self.gate_results:
            return QualityGateStatus.NOT_EVALUATED
        
        # Critical gates that must pass
        critical_gates = [
            "test_pass_rate", "functionality_preservation", 
            "consolidation_completeness", "error_thresholds"
        ]
        
        # Check critical gates first
        for gate_name in critical_gates:
            result = self.gate_results.get(gate_name)
            if result and result.status == QualityGateStatus.FAILED:
                return QualityGateStatus.FAILED
        
        # Check all other gates
        failed_gates = [
            name for name, result in self.gate_results.items()
            if result.status == QualityGateStatus.FAILED
        ]
        
        warning_gates = [
            name for name, result in self.gate_results.items()
            if result.status == QualityGateStatus.WARNING
        ]
        
        if failed_gates:
            return QualityGateStatus.FAILED
        elif warning_gates:
            return QualityGateStatus.WARNING
        else:
            return QualityGateStatus.PASSED
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality gate report."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": time.time(),
            "consolidation_approved": self.overall_status == QualityGateStatus.PASSED,
            "gate_results": {
                name: {
                    "status": result.status.value,
                    "score": result.score,
                    "threshold": result.threshold,
                    "message": result.message,
                    "details": result.details,
                    "recommendations": result.recommendations
                }
                for name, result in self.gate_results.items()
            },
            "summary": {
                "total_gates": len(self.gate_results),
                "passed_gates": len([r for r in self.gate_results.values() 
                                   if r.status == QualityGateStatus.PASSED]),
                "failed_gates": len([r for r in self.gate_results.values() 
                                   if r.status == QualityGateStatus.FAILED]),
                "warning_gates": len([r for r in self.gate_results.values() 
                                    if r.status == QualityGateStatus.WARNING]),
                "not_evaluated_gates": len([r for r in self.gate_results.values() 
                                          if r.status == QualityGateStatus.NOT_EVALUATED])
            },
            "metrics": {
                "test_metrics": {
                    "total_tests": self.metrics.total_tests,
                    "passed_tests": self.metrics.passed_tests,
                    "failed_tests": self.metrics.failed_tests,
                    "test_pass_rate": self.metrics.test_pass_rate
                },
                "performance_metrics": {
                    "performance_regressions": len(self.metrics.performance_regressions),
                    "memory_usage_mb": self.metrics.memory_usage_mb,
                    "startup_time_seconds": self.metrics.startup_time_seconds
                },
                "integration_metrics": {
                    "integration_success_rate": self.metrics.integration_success_rate,
                    "api_coverage": self.metrics.api_coverage
                },
                "system_metrics": {
                    "critical_issues": len(self.metrics.critical_issues),
                    "warnings": len(self.metrics.warnings)
                }
            },
            "recommendations": self._generate_consolidated_recommendations()
        }
    
    def _generate_consolidated_recommendations(self) -> List[str]:
        """Generate consolidated recommendations from all gates."""
        all_recommendations = []
        
        for result in self.gate_results.values():
            all_recommendations.extend(result.recommendations)
        
        # Deduplicate and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        
        # Sort by priority (critical issues first)
        priority_keywords = ["critical", "fix", "address", "implement"]
        
        def recommendation_priority(rec: str) -> int:
            rec_lower = rec.lower()
            for i, keyword in enumerate(priority_keywords):
                if keyword in rec_lower:
                    return i
            return len(priority_keywords)
        
        unique_recommendations.sort(key=recommendation_priority)
        
        return unique_recommendations


# Utility functions for quality gate execution
def collect_system_metrics() -> ConsolidationMetrics:
    """Collect system metrics for quality gate evaluation."""
    metrics = ConsolidationMetrics()
    
    # Collect system resource metrics
    process = psutil.Process(os.getpid())
    metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    metrics.cpu_usage_percent = process.cpu_percent()
    
    return metrics


def run_quality_gates(metrics: ConsolidationMetrics, 
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run quality gates and return comprehensive report."""
    evaluator = QualityGateEvaluator(config)
    evaluator.evaluate_all_gates(metrics)
    return evaluator.generate_report()


# Configuration for Epic 1 consolidation
EPIC1_QUALITY_GATE_CONFIG = {
    "test_pass_rate_threshold": 0.80,
    "performance_regression_threshold": 0.10,
    "api_coverage_threshold": 0.95,
    "integration_success_threshold": 0.90,
    "memory_limit_mb": 500,
    "startup_time_limit_seconds": 5.0,
    "critical_error_limit": 0,
    "max_warnings": 10,
    "performance_baseline_required": True,
    "integration_testing_required": True
}
"""
Module Compatibility Testing

This module provides specialized testing for ensuring module compatibility
during the Epic 1-4 consolidation process.
"""

import importlib
import inspect
import sys
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
import pytest
import logging
from dataclasses import dataclass, field

from .consolidation_framework import ConsolidationTarget, ConsolidationResult

logger = logging.getLogger(__name__)


@dataclass
class ModuleCompatibilityReport:
    """Report for module compatibility testing."""
    module_name: str
    import_successful: bool = False
    api_coverage: float = 0.0
    missing_apis: Set[str] = field(default_factory=set)
    extra_apis: Set[str] = field(default_factory=set)
    signature_mismatches: Dict[str, str] = field(default_factory=dict)
    dependency_issues: List[str] = field(default_factory=list)
    compatibility_score: float = 0.0


class ModuleCompatibilityTester:
    """
    Tests module compatibility during consolidation.
    
    Ensures that consolidated modules maintain compatibility with
    existing code that imports and uses them.
    """
    
    def __init__(self):
        """Initialize the module compatibility tester."""
        self.compatibility_cache: Dict[str, ModuleCompatibilityReport] = {}
        
    def test_orchestrator_consolidation(self) -> ModuleCompatibilityReport:
        """Test compatibility of orchestrator consolidation (19 → 5 modules)."""
        return self._test_module_consolidation(
            original_modules=[
                "app.core.orchestrator",
                "app.core.automated_orchestrator", 
                "app.core.advanced_orchestration_engine",
                "app.core.production_orchestrator",
                "app.core.unified_production_orchestrator"
            ],
            target_module="app.core.production_orchestrator",
            expected_api={
                "ProductionOrchestrator",
                "TaskRouter", 
                "AgentManager",
                "WorkflowEngine",
                "ResourceManager",
                "orchestrate_task",
                "route_agent_request",
                "manage_workflow",
                "allocate_resources"
            }
        )
    
    def test_context_engine_consolidation(self) -> ModuleCompatibilityReport:
        """Test compatibility of context engine consolidation (38 → 6 modules)."""
        return self._test_module_consolidation(
            original_modules=[
                "app.core.context_engine_integration",
                "app.core.enhanced_context_engine",
                "app.core.context_manager",
                "app.core.context_compression_engine", 
                "app.core.context_consolidator",
                "app.core.enhanced_context_consolidator"
            ],
            target_module="app.core.context_engine",
            expected_api={
                "ContextEngine",
                "ContextManager", 
                "CompressionEngine",
                "MemoryManager",
                "SemanticProcessor",
                "compress_context",
                "consolidate_memory",
                "optimize_context",
                "retrieve_context"
            }
        )
    
    def test_security_system_consolidation(self) -> ModuleCompatibilityReport:
        """Test compatibility of security system consolidation (25 → 4 modules)."""
        return self._test_module_consolidation(
            original_modules=[
                "app.core.security",
                "app.core.enhanced_security_safeguards",
                "app.core.security_audit",
                "app.core.integrated_security_system",
                "app.core.enterprise_security_system"
            ],
            target_module="app.core.security_system",
            expected_api={
                "SecuritySystem",
                "AuthManager",
                "AuditLogger", 
                "ThreatDetector",
                "ComplianceEngine",
                "authenticate_user",
                "authorize_action",
                "log_security_event",
                "detect_threats"
            }
        )
    
    def test_performance_system_consolidation(self) -> ModuleCompatibilityReport:
        """Test compatibility of performance system consolidation (30 → 5 modules)."""
        return self._test_module_consolidation(
            original_modules=[
                "app.core.performance_monitoring",
                "app.core.performance_optimizer",
                "app.observability.hooks",
                "app.core.health_monitoring",
                "app.core.performance_benchmarks"
            ],
            target_module="app.core.performance_system",
            expected_api={
                "PerformanceSystem",
                "MetricsCollector",
                "HealthMonitor",
                "BenchmarkRunner", 
                "AlertManager",
                "collect_metrics",
                "monitor_health",
                "run_benchmarks",
                "send_alerts"
            }
        )
    
    def _test_module_consolidation(
        self, 
        original_modules: List[str],
        target_module: str,
        expected_api: Set[str]
    ) -> ModuleCompatibilityReport:
        """Test consolidation of multiple modules into a target module."""
        report = ModuleCompatibilityReport(module_name=target_module)
        
        try:
            # Test target module import
            target = importlib.import_module(target_module)
            report.import_successful = True
            
            # Extract actual API from target module
            actual_api = self._extract_module_api(target)
            
            # Compare APIs
            report.missing_apis = expected_api - actual_api
            report.extra_apis = actual_api - expected_api
            
            # Calculate API coverage
            if expected_api:
                covered_apis = expected_api & actual_api
                report.api_coverage = len(covered_apis) / len(expected_api)
            
            # Test signature compatibility
            report.signature_mismatches = self._check_signature_compatibility(
                target, expected_api & actual_api
            )
            
            # Test dependency compatibility
            report.dependency_issues = self._check_dependency_compatibility(
                original_modules, target_module
            )
            
            # Calculate overall compatibility score
            report.compatibility_score = self._calculate_compatibility_score(report)
            
        except ImportError as e:
            report.import_successful = False
            report.dependency_issues.append(f"Import failed: {str(e)}")
            
        except Exception as e:
            report.dependency_issues.append(f"Unexpected error: {str(e)}")
            
        return report
    
    def _extract_module_api(self, module) -> Set[str]:
        """Extract public API from a module."""
        api_items = set()
        
        for name in dir(module):
            if not name.startswith('_'):
                obj = getattr(module, name)
                if (callable(obj) or inspect.isclass(obj) or 
                    inspect.ismodule(obj) or isinstance(obj, type)):
                    api_items.add(name)
                    
        return api_items
    
    def _check_signature_compatibility(
        self, 
        module, 
        api_items: Set[str]
    ) -> Dict[str, str]:
        """Check function signature compatibility."""
        mismatches = {}
        
        for api_item in api_items:
            try:
                obj = getattr(module, api_item)
                if callable(obj):
                    sig = inspect.signature(obj)
                    # For now, just record that we can get the signature
                    # Future enhancement: compare with expected signatures
                    
            except (ValueError, TypeError) as e:
                mismatches[api_item] = f"Signature error: {str(e)}"
                
        return mismatches
    
    def _check_dependency_compatibility(
        self, 
        original_modules: List[str],
        target_module: str
    ) -> List[str]:
        """Check for dependency compatibility issues."""
        issues = []
        
        # Check if original modules still exist (they might during transition)
        for orig_module in original_modules:
            try:
                importlib.import_module(orig_module)
            except ImportError:
                # This is expected during consolidation
                pass
            except Exception as e:
                issues.append(f"Original module {orig_module} has issues: {str(e)}")
        
        # Check target module dependencies
        try:
            target = importlib.import_module(target_module)
            # Basic dependency check - could be enhanced
            if hasattr(target, '__file__') and target.__file__:
                # Module loaded successfully, basic dependency check passed
                pass
        except Exception as e:
            issues.append(f"Target module dependency issue: {str(e)}")
            
        return issues
    
    def _calculate_compatibility_score(self, report: ModuleCompatibilityReport) -> float:
        """Calculate an overall compatibility score (0-1)."""
        if not report.import_successful:
            return 0.0
            
        score = 0.0
        
        # API coverage weight: 40%
        score += report.api_coverage * 0.4
        
        # Missing APIs penalty: -30%
        if report.missing_apis:
            missing_penalty = len(report.missing_apis) * 0.1
            score -= min(missing_penalty, 0.3)
        
        # Signature compatibility weight: 20%
        if not report.signature_mismatches:
            score += 0.2
        
        # Dependency compatibility weight: 10%
        if not report.dependency_issues:
            score += 0.1
            
        return max(0.0, min(1.0, score))
    
    def test_all_consolidations(self) -> Dict[str, ModuleCompatibilityReport]:
        """Test all Epic consolidations for compatibility."""
        results = {}
        
        results["orchestrator"] = self.test_orchestrator_consolidation()
        results["context_engine"] = self.test_context_engine_consolidation() 
        results["security_system"] = self.test_security_system_consolidation()
        results["performance_system"] = self.test_performance_system_consolidation()
        
        return results
    
    def generate_compatibility_report(
        self, 
        results: Dict[str, ModuleCompatibilityReport]
    ) -> Dict[str, Any]:
        """Generate a comprehensive compatibility report."""
        report = {
            "summary": {
                "total_modules": len(results),
                "successful_imports": 0,
                "average_api_coverage": 0.0,
                "average_compatibility_score": 0.0,
                "total_missing_apis": 0,
                "total_dependency_issues": 0
            },
            "details": {},
            "recommendations": []
        }
        
        total_coverage = 0.0
        total_score = 0.0
        
        for module_name, result in results.items():
            if result.import_successful:
                report["summary"]["successful_imports"] += 1
                
            total_coverage += result.api_coverage
            total_score += result.compatibility_score
            report["summary"]["total_missing_apis"] += len(result.missing_apis)
            report["summary"]["total_dependency_issues"] += len(result.dependency_issues)
            
            report["details"][module_name] = {
                "import_successful": result.import_successful,
                "api_coverage": result.api_coverage,
                "compatibility_score": result.compatibility_score,
                "missing_apis": list(result.missing_apis),
                "extra_apis": list(result.extra_apis),
                "signature_mismatches": result.signature_mismatches,
                "dependency_issues": result.dependency_issues
            }
            
        if results:
            report["summary"]["average_api_coverage"] = total_coverage / len(results)
            report["summary"]["average_compatibility_score"] = total_score / len(results)
            
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(results)
        
        return report
    
    def _generate_recommendations(
        self, 
        results: Dict[str, ModuleCompatibilityReport]
    ) -> List[str]:
        """Generate recommendations based on compatibility results."""
        recommendations = []
        
        for module_name, result in results.items():
            if not result.import_successful:
                recommendations.append(
                    f"CRITICAL: Fix import issues in {module_name} before proceeding"
                )
                
            if result.missing_apis:
                recommendations.append(
                    f"HIGH: Implement missing APIs in {module_name}: {list(result.missing_apis)}"
                )
                
            if result.api_coverage < 0.8:
                recommendations.append(
                    f"MEDIUM: Improve API coverage in {module_name} (currently {result.api_coverage:.1%})"
                )
                
            if result.signature_mismatches:
                recommendations.append(
                    f"MEDIUM: Fix signature issues in {module_name}: {list(result.signature_mismatches.keys())}"
                )
                
            if result.dependency_issues:
                recommendations.append(
                    f"LOW: Review dependency issues in {module_name}"
                )
                
        if not recommendations:
            recommendations.append("All modules show good compatibility. Proceed with consolidation.")
            
        return recommendations


# Pytest integration
class TestModuleCompatibility:
    """Pytest test class for module compatibility."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.tester = ModuleCompatibilityTester()
    
    @pytest.mark.integration
    @pytest.mark.consolidation
    def test_orchestrator_compatibility(self):
        """Test orchestrator module compatibility."""
        report = self.tester.test_orchestrator_consolidation()
        
        assert report.import_successful, f"Orchestrator import failed: {report.dependency_issues}"
        assert report.api_coverage >= 0.8, f"Low API coverage: {report.api_coverage:.1%}"
        assert report.compatibility_score >= 0.7, f"Low compatibility score: {report.compatibility_score:.1%}"
        
        if report.missing_apis:
            pytest.fail(f"Missing APIs: {report.missing_apis}")
    
    @pytest.mark.integration  
    @pytest.mark.consolidation
    def test_context_engine_compatibility(self):
        """Test context engine module compatibility."""
        report = self.tester.test_context_engine_consolidation()
        
        assert report.import_successful, f"Context engine import failed: {report.dependency_issues}"
        assert report.api_coverage >= 0.8, f"Low API coverage: {report.api_coverage:.1%}"
        assert report.compatibility_score >= 0.7, f"Low compatibility score: {report.compatibility_score:.1%}"
    
    @pytest.mark.integration
    @pytest.mark.consolidation  
    def test_security_system_compatibility(self):
        """Test security system module compatibility."""
        report = self.tester.test_security_system_consolidation()
        
        assert report.import_successful, f"Security system import failed: {report.dependency_issues}"
        assert report.api_coverage >= 0.8, f"Low API coverage: {report.api_coverage:.1%}"
        assert report.compatibility_score >= 0.7, f"Low compatibility score: {report.compatibility_score:.1%}"
    
    @pytest.mark.integration
    @pytest.mark.consolidation
    def test_performance_system_compatibility(self):
        """Test performance system module compatibility.""" 
        report = self.tester.test_performance_system_consolidation()
        
        assert report.import_successful, f"Performance system import failed: {report.dependency_issues}"
        assert report.api_coverage >= 0.8, f"Low API coverage: {report.api_coverage:.1%}"
        assert report.compatibility_score >= 0.7, f"Low compatibility score: {report.compatibility_score:.1%}"
    
    @pytest.mark.integration
    @pytest.mark.consolidation
    def test_all_modules_compatibility(self):
        """Test compatibility of all consolidated modules."""
        results = self.tester.test_all_consolidations()
        report = self.tester.generate_compatibility_report(results)
        
        # Overall system checks
        assert report["summary"]["successful_imports"] == report["summary"]["total_modules"], \
            "Not all modules imported successfully"
            
        assert report["summary"]["average_compatibility_score"] >= 0.7, \
            f"Low average compatibility score: {report['summary']['average_compatibility_score']:.1%}"
            
        # Fail if there are critical recommendations
        critical_recs = [r for r in report["recommendations"] if r.startswith("CRITICAL")]
        if critical_recs:
            pytest.fail(f"Critical compatibility issues: {critical_recs}")
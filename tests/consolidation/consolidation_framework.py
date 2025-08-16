"""
Core Consolidation Testing Framework

This framework provides the foundational testing capabilities for validating
file consolidations during the Epic 1-4 transformation from 313 files to 50 modules.
"""

import inspect
import importlib
import sys
import time
import tracemalloc
from typing import List, Dict, Any, Optional, Set, Callable
from pathlib import Path
import pytest
import ast
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationTarget:
    """Represents a consolidation target mapping original files to target module."""
    original_files: List[str] = field(default_factory=list)
    target_module: str = ""
    target_path: str = ""
    expected_public_api: Set[str] = field(default_factory=set)
    performance_baseline: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)


@dataclass 
class ConsolidationResult:
    """Results of a consolidation validation."""
    target: ConsolidationTarget
    functionality_preserved: bool = False
    api_compatible: bool = False
    performance_acceptable: bool = False
    integration_intact: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ConsolidationValidator(ABC):
    """Abstract base class for consolidation validators."""
    
    @abstractmethod
    def validate(self, target: ConsolidationTarget) -> ConsolidationResult:
        """Validate a consolidation target."""
        pass


class FunctionalityPreservationValidator(ConsolidationValidator):
    """Validates that all functionality from original files is preserved."""
    
    def validate(self, target: ConsolidationTarget) -> ConsolidationResult:
        """Validate functionality preservation during consolidation."""
        result = ConsolidationResult(target=target)
        
        try:
            # Extract public API from original files
            original_api = self._extract_original_api(target.original_files)
            
            # Extract public API from target module
            target_api = self._extract_target_api(target.target_module)
            
            # Check if all original functionality is present
            missing_functions = original_api - target_api
            if missing_functions:
                result.errors.append(f"Missing functions in target: {missing_functions}")
                result.functionality_preserved = False
            else:
                result.functionality_preserved = True
                
            result.metrics["original_api_count"] = len(original_api)
            result.metrics["target_api_count"] = len(target_api)
            result.metrics["missing_count"] = len(missing_functions)
            
        except Exception as e:
            result.errors.append(f"Functionality validation failed: {str(e)}")
            result.functionality_preserved = False
            
        return result
    
    def _extract_original_api(self, file_paths: List[str]) -> Set[str]:
        """Extract public API from original files."""
        api_items = set()
        
        for file_path in file_paths:
            try:
                if not Path(file_path).exists():
                    continue
                    
                with open(file_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not node.name.startswith('_'):  # Public items only
                            api_items.add(node.name)
                            
            except Exception as e:
                logger.warning(f"Could not parse {file_path}: {e}")
                
        return api_items
    
    def _extract_target_api(self, module_path: str) -> Set[str]:
        """Extract public API from target module."""
        api_items = set()
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get all public attributes
            for name in dir(module):
                if not name.startswith('_'):
                    obj = getattr(module, name)
                    if callable(obj) or inspect.isclass(obj):
                        api_items.add(name)
                        
        except Exception as e:
            logger.warning(f"Could not import {module_path}: {e}")
            
        return api_items


class APICompatibilityValidator(ConsolidationValidator):
    """Validates API compatibility after consolidation."""
    
    def validate(self, target: ConsolidationTarget) -> ConsolidationResult:
        """Validate API compatibility during consolidation."""
        result = ConsolidationResult(target=target)
        
        try:
            # Check import compatibility
            import_compatibility = self._check_import_compatibility(target)
            
            # Check function signature compatibility  
            signature_compatibility = self._check_signature_compatibility(target)
            
            result.api_compatible = import_compatibility and signature_compatibility
            result.metrics["import_compatible"] = import_compatibility
            result.metrics["signature_compatible"] = signature_compatibility
            
        except Exception as e:
            result.errors.append(f"API compatibility validation failed: {str(e)}")
            result.api_compatible = False
            
        return result
    
    def _check_import_compatibility(self, target: ConsolidationTarget) -> bool:
        """Check if imports still work after consolidation."""
        try:
            # Try importing the target module
            module = importlib.import_module(target.target_module)
            
            # Check if expected public API is available
            for api_item in target.expected_public_api:
                if not hasattr(module, api_item):
                    return False
                    
            return True
            
        except ImportError:
            return False
    
    def _check_signature_compatibility(self, target: ConsolidationTarget) -> bool:
        """Check if function signatures remain compatible."""
        try:
            module = importlib.import_module(target.target_module)
            
            for api_item in target.expected_public_api:
                if hasattr(module, api_item):
                    obj = getattr(module, api_item)
                    if callable(obj):
                        # Basic signature check - could be enhanced
                        try:
                            inspect.signature(obj)
                        except (ValueError, TypeError):
                            return False
                            
            return True
            
        except Exception:
            return False


class PerformanceRegressionValidator(ConsolidationValidator):
    """Validates performance is maintained during consolidation."""
    
    def __init__(self, regression_threshold: float = 0.05):
        """Initialize with regression threshold (5% by default)."""
        self.regression_threshold = regression_threshold
    
    def validate(self, target: ConsolidationTarget) -> ConsolidationResult:
        """Validate performance during consolidation."""
        result = ConsolidationResult(target=target)
        
        try:
            # Measure current performance
            current_metrics = self._measure_performance(target.target_module)
            
            # Compare with baseline
            performance_acceptable = self._compare_with_baseline(
                current_metrics, target.performance_baseline
            )
            
            result.performance_acceptable = performance_acceptable
            result.metrics.update(current_metrics)
            result.metrics["baseline"] = target.performance_baseline
            
        except Exception as e:
            result.errors.append(f"Performance validation failed: {str(e)}")
            result.performance_acceptable = False
            
        return result
    
    def _measure_performance(self, module_path: str) -> Dict[str, float]:
        """Measure performance metrics for a module."""
        metrics = {}
        
        try:
            # Measure import time
            start_time = time.perf_counter()
            tracemalloc.start()
            
            module = importlib.import_module(module_path)
            
            import_time = time.perf_counter() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            metrics["import_time"] = import_time
            metrics["memory_current"] = current
            metrics["memory_peak"] = peak
            
            # Measure module size
            if hasattr(module, '__file__') and module.__file__:
                file_size = Path(module.__file__).stat().st_size
                metrics["file_size"] = file_size
                
        except Exception as e:
            logger.warning(f"Could not measure performance for {module_path}: {e}")
            
        return metrics
    
    def _compare_with_baseline(self, current: Dict[str, float], baseline: Dict[str, float]) -> bool:
        """Compare current metrics with baseline."""
        if not baseline:
            return True  # No baseline to compare against
            
        for metric, baseline_value in baseline.items():
            if metric in current:
                current_value = current[metric]
                if baseline_value > 0:
                    regression = (current_value - baseline_value) / baseline_value
                    if regression > self.regression_threshold:
                        return False
                        
        return True


class ConsolidationTestFramework:
    """
    Main framework for testing file consolidations during Epic transformation.
    
    This framework orchestrates multiple validators to ensure safe consolidation
    of 313 files into 50 modules.
    """
    
    def __init__(self):
        """Initialize the consolidation test framework."""
        self.validators: List[ConsolidationValidator] = [
            FunctionalityPreservationValidator(),
            APICompatibilityValidator(),
            PerformanceRegressionValidator()
        ]
        
        self.consolidation_targets: List[ConsolidationTarget] = []
        
    def add_consolidation_target(self, target: ConsolidationTarget):
        """Add a consolidation target for testing."""
        self.consolidation_targets.append(target)
        
    def validate_consolidation(self, target: ConsolidationTarget) -> ConsolidationResult:
        """Validate a single consolidation target."""
        overall_result = ConsolidationResult(target=target)
        
        for validator in self.validators:
            result = validator.validate(target)
            
            # Aggregate results
            if not result.functionality_preserved:
                overall_result.functionality_preserved = False
            if not result.api_compatible:
                overall_result.api_compatible = False
            if not result.performance_acceptable:
                overall_result.performance_acceptable = False
                
            overall_result.errors.extend(result.errors)
            overall_result.warnings.extend(result.warnings)
            overall_result.metrics.update(result.metrics)
            
        # Set overall integration status
        overall_result.integration_intact = (
            overall_result.functionality_preserved and
            overall_result.api_compatible and
            overall_result.performance_acceptable
        )
        
        return overall_result
    
    def validate_all_consolidations(self) -> List[ConsolidationResult]:
        """Validate all registered consolidation targets."""
        results = []
        
        for target in self.consolidation_targets:
            result = self.validate_consolidation(target)
            results.append(result)
            
        return results
    
    def generate_report(self, results: List[ConsolidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive consolidation report."""
        report = {
            "total_consolidations": len(results),
            "successful_consolidations": 0,
            "failed_consolidations": 0,
            "warnings_count": 0,
            "errors_count": 0,
            "performance_regressions": 0,
            "api_breaks": 0,
            "functionality_losses": 0,
            "details": []
        }
        
        for result in results:
            detail = {
                "target_module": result.target.target_module,
                "original_files_count": len(result.target.original_files),
                "success": result.integration_intact,
                "errors": result.errors,
                "warnings": result.warnings,
                "metrics": result.metrics
            }
            
            report["details"].append(detail)
            
            if result.integration_intact:
                report["successful_consolidations"] += 1
            else:
                report["failed_consolidations"] += 1
                
            if not result.functionality_preserved:
                report["functionality_losses"] += 1
            if not result.api_compatible:
                report["api_breaks"] += 1
            if not result.performance_acceptable:
                report["performance_regressions"] += 1
                
            report["warnings_count"] += len(result.warnings)
            report["errors_count"] += len(result.errors)
            
        return report


# Predefined consolidation targets for Epic 1-4 transformation
EPIC_CONSOLIDATION_TARGETS = [
    # Epic 1: Orchestrator Consolidation (19 → 5 modules)
    ConsolidationTarget(
        original_files=[
            "app/core/orchestrator.py",
            "app/core/automated_orchestrator.py", 
            "app/core/advanced_orchestration_engine.py",
            "app/core/production_orchestrator.py",
            "app/core/unified_production_orchestrator.py"
            # ... more orchestrator files
        ],
        target_module="app.core.production_orchestrator",
        target_path="app/core/production_orchestrator.py",
        expected_public_api={
            "ProductionOrchestrator", "TaskRouter", "AgentManager", 
            "WorkflowEngine", "ResourceManager"
        }
    ),
    
    # Epic 2: Context Engine Consolidation (38 → 6 modules)  
    ConsolidationTarget(
        original_files=[
            "app/core/context_engine_integration.py",
            "app/core/enhanced_context_engine.py",
            "app/core/context_manager.py",
            "app/core/context_compression_engine.py",
            "app/core/context_consolidator.py"
            # ... more context files
        ],
        target_module="app.core.context_engine",
        target_path="app/core/context_engine.py", 
        expected_public_api={
            "ContextEngine", "ContextManager", "CompressionEngine",
            "MemoryManager", "SemanticProcessor"
        }
    ),
    
    # Epic 3: Security System Consolidation (25 → 4 modules)
    ConsolidationTarget(
        original_files=[
            "app/core/security.py",
            "app/core/enhanced_security_safeguards.py",
            "app/core/security_audit.py", 
            "app/core/integrated_security_system.py",
            "app/core/enterprise_security_system.py"
            # ... more security files
        ],
        target_module="app.core.security_system",
        target_path="app/core/security_system.py",
        expected_public_api={
            "SecuritySystem", "AuthManager", "AuditLogger",
            "ThreatDetector", "ComplianceEngine"
        }
    ),
    
    # Epic 4: Performance & Monitoring Consolidation (30 → 5 modules)
    ConsolidationTarget(
        original_files=[
            "app/core/performance_monitoring.py",
            "app/core/performance_optimizer.py",
            "app/observability/hooks.py",
            "app/core/health_monitoring.py",
            "app/core/performance_benchmarks.py"
            # ... more performance files
        ],
        target_module="app.core.performance_system", 
        target_path="app/core/performance_system.py",
        expected_public_api={
            "PerformanceSystem", "MetricsCollector", "HealthMonitor",
            "BenchmarkRunner", "AlertManager"
        }
    )
]
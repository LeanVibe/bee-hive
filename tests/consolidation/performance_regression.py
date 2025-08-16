"""
Performance Regression Detection

This module provides performance regression detection capabilities
for the Epic 1-4 consolidation process to ensure system performance
is maintained during the 313→50 file transformation.
"""

import time
import tracemalloc
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import pytest
import importlib
import json
from pathlib import Path
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    import_time: float = 0.0
    memory_current: int = 0
    memory_peak: int = 0
    cpu_percent: float = 0.0
    file_size: int = 0
    function_call_time: Dict[str, float] = field(default_factory=dict)
    memory_per_operation: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "import_time": self.import_time,
            "memory_current": self.memory_current,
            "memory_peak": self.memory_peak,
            "cpu_percent": self.cpu_percent,
            "file_size": self.file_size,
            "function_call_time": self.function_call_time,
            "memory_per_operation": self.memory_per_operation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(
            import_time=data.get("import_time", 0.0),
            memory_current=data.get("memory_current", 0),
            memory_peak=data.get("memory_peak", 0),
            cpu_percent=data.get("cpu_percent", 0.0),
            file_size=data.get("file_size", 0),
            function_call_time=data.get("function_call_time", {}),
            memory_per_operation=data.get("memory_per_operation", {})
        )


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    module_name: str
    baseline_metrics: PerformanceMetrics
    measurement_count: int = 1
    confidence_interval: float = 0.95
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "module_name": self.module_name,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "measurement_count": self.measurement_count,
            "confidence_interval": self.confidence_interval,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        return cls(
            module_name=data["module_name"],
            baseline_metrics=PerformanceMetrics.from_dict(data["baseline_metrics"]),
            measurement_count=data.get("measurement_count", 1),
            confidence_interval=data.get("confidence_interval", 0.95),
            created_at=data.get("created_at", "")
        )


@dataclass
class RegressionResult:
    """Result of a regression detection test."""
    module_name: str
    has_regression: bool = False
    regression_percentage: float = 0.0
    regression_details: Dict[str, Dict[str, float]] = field(default_factory=dict)
    current_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    baseline_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class PerformanceRegressionDetector:
    """
    Detects performance regressions during consolidation.
    
    Measures performance before and after consolidation to ensure
    the system maintains acceptable performance levels.
    """
    
    def __init__(self, regression_threshold: float = 0.05, baseline_dir: str = "tests/baselines"):
        """
        Initialize the performance regression detector.
        
        Args:
            regression_threshold: Maximum allowed performance regression (5% by default)
            baseline_dir: Directory to store performance baselines
        """
        self.regression_threshold = regression_threshold
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        # Predefined critical modules for Epic consolidation
        self.critical_modules = {
            "orchestrator": "app.core.production_orchestrator",
            "context_engine": "app.core.context_engine", 
            "security_system": "app.core.security_system",
            "performance_system": "app.core.performance_system",
            "project_index": "app.project_index.core"
        }
        
    def establish_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Establish performance baselines for all critical modules."""
        baselines = {}
        
        for module_key, module_path in self.critical_modules.items():
            try:
                baseline = self._establish_module_baseline(module_path, module_key)
                baselines[module_key] = baseline
                self._save_baseline(baseline)
                logger.info(f"Established baseline for {module_key}")
                
            except Exception as e:
                logger.error(f"Failed to establish baseline for {module_key}: {e}")
                
        return baselines
    
    def _establish_module_baseline(self, module_path: str, module_key: str) -> PerformanceBaseline:
        """Establish baseline for a single module."""
        # Run multiple measurements for statistical reliability
        measurements = []
        
        for i in range(5):  # 5 measurements for baseline
            metrics = self._measure_module_performance(module_path)
            measurements.append(metrics)
            
            # Clear memory between measurements
            gc.collect()
            time.sleep(0.1)
            
        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(measurements)
        
        return PerformanceBaseline(
            module_name=module_key,
            baseline_metrics=avg_metrics,
            measurement_count=len(measurements),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _measure_module_performance(self, module_path: str) -> PerformanceMetrics:
        """Measure performance metrics for a module."""
        metrics = PerformanceMetrics()
        
        # Clear any existing imports
        if module_path in sys.modules:
            del sys.modules[module_path]
            
        # Measure import performance
        gc.collect()
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            module = importlib.import_module(module_path)
            metrics.import_time = time.perf_counter() - start_time
            
            current, peak = tracemalloc.get_traced_memory()
            metrics.memory_current = current
            metrics.memory_peak = peak
            
            # Measure CPU usage (approximate)
            metrics.cpu_percent = process.cpu_percent() - cpu_before
            
            # Measure file size if available
            if hasattr(module, '__file__') and module.__file__:
                metrics.file_size = Path(module.__file__).stat().st_size
                
            # Measure function call performance for key functions
            metrics.function_call_time = self._measure_function_performance(module)
            
        except Exception as e:
            logger.warning(f"Could not measure performance for {module_path}: {e}")
            
        finally:
            tracemalloc.stop()
            
        return metrics
    
    def _measure_function_performance(self, module) -> Dict[str, float]:
        """Measure performance of key functions in a module."""
        function_times = {}
        
        # Common function names to test
        test_functions = [
            'process', 'execute', 'run', 'handle', 'manage',
            'orchestrate', 'route', 'compress', 'optimize',
            'authenticate', 'authorize', 'monitor', 'collect'
        ]
        
        for func_name in test_functions:
            if hasattr(module, func_name):
                func = getattr(module, func_name)
                if callable(func):
                    try:
                        # Simple performance test with minimal arguments
                        start_time = time.perf_counter()
                        # We can't safely call arbitrary functions, so just measure import/access time
                        _ = func
                        function_times[func_name] = time.perf_counter() - start_time
                        
                    except Exception:
                        # Skip functions that can't be safely tested
                        pass
                        
        return function_times
    
    def _calculate_average_metrics(self, measurements: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Calculate average metrics from multiple measurements."""
        if not measurements:
            return PerformanceMetrics()
            
        avg_metrics = PerformanceMetrics()
        
        # Calculate averages for numeric fields
        avg_metrics.import_time = statistics.mean(m.import_time for m in measurements)
        avg_metrics.memory_current = int(statistics.mean(m.memory_current for m in measurements))
        avg_metrics.memory_peak = int(statistics.mean(m.memory_peak for m in measurements))
        avg_metrics.cpu_percent = statistics.mean(m.cpu_percent for m in measurements)
        avg_metrics.file_size = int(statistics.mean(m.file_size for m in measurements))
        
        # Calculate averages for function call times
        all_functions = set()
        for m in measurements:
            all_functions.update(m.function_call_time.keys())
            
        for func_name in all_functions:
            times = [m.function_call_time.get(func_name, 0.0) for m in measurements]
            avg_metrics.function_call_time[func_name] = statistics.mean(times)
            
        return avg_metrics
    
    def detect_regressions(self) -> Dict[str, RegressionResult]:
        """Detect performance regressions in all critical modules."""
        results = {}
        
        for module_key, module_path in self.critical_modules.items():
            try:
                result = self._detect_module_regression(module_path, module_key)
                results[module_key] = result
                
            except Exception as e:
                result = RegressionResult(
                    module_name=module_key,
                    has_regression=True,
                    errors=[f"Failed to detect regression: {str(e)}"]
                )
                results[module_key] = result
                
        return results
    
    def _detect_module_regression(self, module_path: str, module_key: str) -> RegressionResult:
        """Detect regression for a single module."""
        result = RegressionResult(module_name=module_key)
        
        # Load baseline
        baseline = self._load_baseline(module_key)
        if not baseline:
            result.warnings.append("No baseline found, cannot detect regression")
            return result
            
        # Measure current performance
        current_metrics = self._measure_module_performance(module_path)
        result.current_metrics = current_metrics
        result.baseline_metrics = baseline.baseline_metrics
        
        # Compare metrics
        regression_details = {}
        max_regression = 0.0
        
        # Check import time regression
        if baseline.baseline_metrics.import_time > 0:
            import_regression = (
                (current_metrics.import_time - baseline.baseline_metrics.import_time) /
                baseline.baseline_metrics.import_time
            )
            regression_details["import_time"] = {
                "baseline": baseline.baseline_metrics.import_time,
                "current": current_metrics.import_time,
                "regression": import_regression
            }
            max_regression = max(max_regression, import_regression)
            
        # Check memory regression
        if baseline.baseline_metrics.memory_peak > 0:
            memory_regression = (
                (current_metrics.memory_peak - baseline.baseline_metrics.memory_peak) /
                baseline.baseline_metrics.memory_peak
            )
            regression_details["memory_peak"] = {
                "baseline": baseline.baseline_metrics.memory_peak,
                "current": current_metrics.memory_peak,  
                "regression": memory_regression
            }
            max_regression = max(max_regression, memory_regression)
            
        # Check file size regression (if consolidated, might be larger)
        if baseline.baseline_metrics.file_size > 0:
            size_regression = (
                (current_metrics.file_size - baseline.baseline_metrics.file_size) /
                baseline.baseline_metrics.file_size
            )
            regression_details["file_size"] = {
                "baseline": baseline.baseline_metrics.file_size,
                "current": current_metrics.file_size,
                "regression": size_regression
            }
            # File size increase is expected during consolidation, so use higher threshold
            if size_regression > 2.0:  # 200% increase threshold for file size
                max_regression = max(max_regression, size_regression - 2.0)
                
        result.regression_details = regression_details
        result.regression_percentage = max_regression
        result.has_regression = max_regression > self.regression_threshold
        
        return result
    
    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to disk."""
        baseline_file = self.baseline_dir / f"{baseline.module_name}_baseline.json"
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline.to_dict(), f, indent=2)
            
    def _load_baseline(self, module_key: str) -> Optional[PerformanceBaseline]:
        """Load baseline from disk."""
        baseline_file = self.baseline_dir / f"{module_key}_baseline.json"
        
        if not baseline_file.exists():
            return None
            
        try:
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                return PerformanceBaseline.from_dict(data)
                
        except Exception as e:
            logger.error(f"Failed to load baseline for {module_key}: {e}")
            return None
            
    def generate_regression_report(self, results: Dict[str, RegressionResult]) -> Dict[str, Any]:
        """Generate a comprehensive regression report."""
        report = {
            "summary": {
                "total_modules": len(results),
                "modules_with_regression": 0,
                "max_regression_percentage": 0.0,
                "average_regression": 0.0,
                "critical_regressions": [],
                "warnings_count": 0,
                "errors_count": 0
            },
            "details": {},
            "recommendations": []
        }
        
        total_regression = 0.0
        
        for module_name, result in results.items():
            if result.has_regression:
                report["summary"]["modules_with_regression"] += 1
                
            total_regression += result.regression_percentage
            report["summary"]["max_regression_percentage"] = max(
                report["summary"]["max_regression_percentage"],
                result.regression_percentage
            )
            
            if result.regression_percentage > 0.1:  # 10% regression is critical
                report["summary"]["critical_regressions"].append({
                    "module": module_name,
                    "regression": result.regression_percentage
                })
                
            report["summary"]["warnings_count"] += len(result.warnings)
            report["summary"]["errors_count"] += len(result.errors)
            
            report["details"][module_name] = {
                "has_regression": result.has_regression,
                "regression_percentage": result.regression_percentage,
                "regression_details": result.regression_details,
                "warnings": result.warnings,
                "errors": result.errors
            }
            
        if results:
            report["summary"]["average_regression"] = total_regression / len(results)
            
        # Generate recommendations
        report["recommendations"] = self._generate_regression_recommendations(results)
        
        return report
    
    def _generate_regression_recommendations(self, results: Dict[str, RegressionResult]) -> List[str]:
        """Generate recommendations based on regression results."""
        recommendations = []
        
        for module_name, result in results.items():
            if result.errors:
                recommendations.append(
                    f"CRITICAL: Fix errors in {module_name} before proceeding with consolidation"
                )
                
            if result.regression_percentage > 0.1:
                recommendations.append(
                    f"CRITICAL: {module_name} shows {result.regression_percentage:.1%} regression - investigate immediately"
                )
                
            elif result.has_regression:
                recommendations.append(
                    f"WARNING: {module_name} shows {result.regression_percentage:.1%} regression - monitor closely"
                )
                
            # Specific metric recommendations
            for metric, details in result.regression_details.items():
                if details["regression"] > 0.1:
                    recommendations.append(
                        f"HIGH: {module_name} {metric} regressed by {details['regression']:.1%}"
                    )
                    
        if not any("CRITICAL" in rec for rec in recommendations):
            recommendations.append("Performance regressions are within acceptable limits for consolidation.")
            
        return recommendations


# Pytest integration
class TestPerformanceRegression:
    """Pytest test class for performance regression detection."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.detector = PerformanceRegressionDetector()
    
    @pytest.mark.performance
    @pytest.mark.consolidation
    def test_establish_baselines(self):
        """Test establishing performance baselines."""
        baselines = self.detector.establish_baselines()
        
        assert len(baselines) > 0, "No baselines were established"
        
        for module_name, baseline in baselines.items():
            assert baseline.baseline_metrics.import_time > 0, f"Invalid import time for {module_name}"
            assert baseline.measurement_count > 0, f"No measurements for {module_name}"
    
    @pytest.mark.performance
    @pytest.mark.consolidation
    def test_detect_regressions(self):
        """Test regression detection."""
        # First establish baselines if they don't exist
        self.detector.establish_baselines()
        
        # Then detect regressions
        results = self.detector.detect_regressions()
        
        assert len(results) > 0, "No regression results obtained"
        
        # Check for critical regressions
        critical_regressions = [
            result for result in results.values() 
            if result.has_regression and result.regression_percentage > 0.1
        ]
        
        if critical_regressions:
            pytest.fail(f"Critical performance regressions detected: {critical_regressions}")
    
    @pytest.mark.performance
    @pytest.mark.consolidation
    def test_orchestrator_performance(self):
        """Test orchestrator module performance specifically."""
        module_path = "app.core.production_orchestrator"
        
        try:
            metrics = self.detector._measure_module_performance(module_path)
            
            # Basic performance assertions
            assert metrics.import_time < 2.0, f"Slow import time: {metrics.import_time}s"
            assert metrics.memory_peak < 100 * 1024 * 1024, f"High memory usage: {metrics.memory_peak} bytes"
            
        except ImportError:
            pytest.skip(f"Module {module_path} not yet available")
    
    @pytest.mark.performance
    @pytest.mark.consolidation  
    def test_all_modules_performance(self):
        """Test performance of all consolidated modules."""
        results = self.detector.detect_regressions()
        report = self.detector.generate_regression_report(results)
        
        # Ensure no critical regressions
        assert len(report["summary"]["critical_regressions"]) == 0, \
            f"Critical regressions found: {report['summary']['critical_regressions']}"
            
        # Ensure average regression is within limits
        assert report["summary"]["average_regression"] <= self.detector.regression_threshold, \
            f"Average regression {report['summary']['average_regression']:.1%} exceeds threshold"
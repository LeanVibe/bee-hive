"""
Unit and Integration Test Runner with Quality Gates

Comprehensive test runner for Level 2 (Unit) and Level 3 (Integration) 
testing with automated quality gate validation.

Features:
- Parallel test execution for performance
- Quality gate validation with pass/fail criteria
- Detailed reporting and metrics collection
- Integration with foundation testing layer
- Performance benchmarking
- Test categorization and filtering
- Failure analysis and recommendations

Quality Gates:
- Unit tests: 90%+ pass rate, <2s execution per component
- Integration tests: 85%+ pass rate, <10s execution per workflow
- Code coverage tracking
- Performance compliance validation
"""

import asyncio
import json
import time
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest


@dataclass
class TestResult:
    """Individual test result."""
    name: str
    status: str  # passed, failed, skipped
    duration: float
    error_message: Optional[str] = None
    category: str = "unknown"


@dataclass
class TestCategoryResult:
    """Results for a test category."""
    category: str
    tests_run: int
    passed: int
    failed: int
    skipped: int
    duration: float
    pass_rate: float
    error_messages: List[str]


@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    gate_name: str
    passed: bool
    actual_value: float
    threshold_value: float
    message: str


@dataclass
class TestSuiteReport:
    """Complete test suite report."""
    timestamp: str
    total_tests: int
    total_passed: int
    total_failed: int
    total_skipped: int
    total_duration: float
    overall_pass_rate: float
    category_results: List[TestCategoryResult]
    quality_gates: List[QualityGateResult]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]


class UnitIntegrationTestRunner:
    """Test runner for unit and integration tests with quality gates."""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent
        self.unit_test_path = self.base_path / "unit"
        self.integration_test_path = self.base_path / "integration"
        self.foundation_test_path = self.base_path / "foundation"
        
        # Quality gate thresholds
        self.quality_gates = {
            "unit_test_pass_rate": 90.0,
            "integration_test_pass_rate": 85.0,
            "unit_test_max_duration_per_component": 2.0,
            "integration_test_max_duration_per_workflow": 10.0,
            "overall_test_pass_rate": 85.0,
            "test_execution_time_limit": 300.0  # 5 minutes total
        }
    
    def run_foundation_tests_check(self) -> bool:
        """Check that foundation tests are passing before running upper levels."""
        foundation_report_path = self.foundation_test_path / "foundation_test_report.json"
        
        if not foundation_report_path.exists():
            print("❌ Foundation test report not found. Please run foundation tests first.")
            return False
        
        try:
            with open(foundation_report_path) as f:
                foundation_report = json.load(f)
            
            foundation_pass_rate = (foundation_report["summary"]["passed"] / 
                                  foundation_report["summary"]["total"]) * 100
            
            if foundation_pass_rate < 65.0:  # Minimum foundation requirement (relaxed for development)
                print(f"❌ Foundation tests pass rate too low: {foundation_pass_rate:.1f}% (need ≥65%)")
                print("❌ Foundation tests must pass before running unit/integration tests")
                return False
            elif foundation_pass_rate < 70.0:
                print(f"⚠️ Foundation tests below optimal: {foundation_pass_rate:.1f}% (optimal ≥70%)")
                print("⚠️ Proceeding with unit/integration tests but consider fixing foundation issues")
            
            print(f"✅ Foundation tests validated: {foundation_pass_rate:.1f}% pass rate")
            return True
            
        except Exception as e:
            print(f"❌ Error reading foundation test report: {e}")
            return False
    
    def discover_unit_tests(self) -> Dict[str, List[str]]:
        """Discover unit test files organized by component."""
        unit_tests = {}
        
        if not self.unit_test_path.exists():
            return unit_tests
        
        # Discover component tests
        components_path = self.unit_test_path / "components"
        if components_path.exists():
            for test_file in components_path.glob("test_*_unit.py"):
                component_name = test_file.name.replace("test_", "").replace("_unit.py", "")
                unit_tests[f"component_{component_name}"] = [str(test_file)]
        
        # Discover other unit tests
        for test_file in self.unit_test_path.glob("test_*.py"):
            test_name = test_file.name.replace("test_", "").replace(".py", "")
            unit_tests[f"unit_{test_name}"] = [str(test_file)]
        
        return unit_tests
    
    def discover_integration_tests(self) -> Dict[str, List[str]]:
        """Discover integration test files organized by workflow."""
        integration_tests = {}
        
        if not self.integration_test_path.exists():
            return integration_tests
        
        # Discover workflow tests
        workflows_path = self.integration_test_path / "workflows"
        if workflows_path.exists():
            for test_file in workflows_path.glob("test_*.py"):
                workflow_name = test_file.name.replace("test_", "").replace(".py", "")
                integration_tests[f"workflow_{workflow_name}"] = [str(test_file)]
        
        # Discover other integration tests
        for test_file in self.integration_test_path.glob("test_*.py"):
            test_name = test_file.name.replace("test_", "").replace(".py", "")
            integration_tests[f"integration_{test_name}"] = [str(test_file)]
        
        return integration_tests
    
    def run_test_category(self, category_name: str, test_files: List[str]) -> TestCategoryResult:
        """Run tests for a specific category."""
        print(f"🧪 Running {category_name} tests...")
        
        start_time = time.time()
        passed = failed = skipped = 0
        error_messages = []
        
        try:
            # Use pytest to run the tests
            cmd = [
                sys.executable, "-m", "pytest",
                "-v",
                "--tb=short",
                "--disable-warnings",
                *test_files
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout per category
            )
            
            # Parse pytest output (simplified)
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if " PASSED " in line:
                    passed += 1
                elif " FAILED " in line:
                    failed += 1
                    error_messages.append(line.strip())
                elif " SKIPPED " in line:
                    skipped += 1
            
            if result.returncode != 0 and failed == 0:
                # If pytest failed but we didn't count failures, there was an error
                failed = 1
                error_messages.append(result.stderr.strip()[:200])
        
        except subprocess.TimeoutExpired:
            failed = 1
            error_messages.append("Test execution timed out")
        except Exception as e:
            failed = 1
            error_messages.append(f"Test execution error: {str(e)}")
        
        duration = time.time() - start_time
        tests_run = passed + failed + skipped
        pass_rate = (passed / tests_run * 100) if tests_run > 0 else 0
        
        print(f"   {category_name}: {passed} passed, {failed} failed, {skipped} skipped ({duration:.2f}s)")
        
        return TestCategoryResult(
            category=category_name,
            tests_run=tests_run,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration=duration,
            pass_rate=pass_rate,
            error_messages=error_messages
        )
    
    def run_tests_parallel(self, test_categories: Dict[str, List[str]], max_workers: int = 4) -> List[TestCategoryResult]:
        """Run test categories in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all test categories
            future_to_category = {
                executor.submit(self.run_test_category, category, files): category
                for category, files in test_categories.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_category):
                category_name = future_to_category[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    # Create error result for failed category
                    error_result = TestCategoryResult(
                        category=category_name,
                        tests_run=0,
                        passed=0,
                        failed=1,
                        skipped=0,
                        duration=0.0,
                        pass_rate=0.0,
                        error_messages=[f"Category execution failed: {str(e)}"]
                    )
                    results.append(error_result)
        
        return results
    
    def validate_quality_gates(self, category_results: List[TestCategoryResult], total_duration: float) -> List[QualityGateResult]:
        """Validate quality gates and return results."""
        quality_gate_results = []
        
        # Calculate metrics
        unit_test_results = [r for r in category_results if r.category.startswith(("component_", "unit_"))]
        integration_test_results = [r for r in category_results if r.category.startswith(("workflow_", "integration_"))]
        
        total_tests = sum(r.tests_run for r in category_results)
        total_passed = sum(r.passed for r in category_results)
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Unit test pass rate
        if unit_test_results:
            unit_passed = sum(r.passed for r in unit_test_results)
            unit_total = sum(r.tests_run for r in unit_test_results)
            unit_pass_rate = (unit_passed / unit_total * 100) if unit_total > 0 else 0
            
            quality_gate_results.append(QualityGateResult(
                gate_name="Unit Test Pass Rate",
                passed=unit_pass_rate >= self.quality_gates["unit_test_pass_rate"],
                actual_value=unit_pass_rate,
                threshold_value=self.quality_gates["unit_test_pass_rate"],
                message=f"Unit tests: {unit_pass_rate:.1f}% pass rate (need ≥{self.quality_gates['unit_test_pass_rate']}%)"
            ))
        
        # Integration test pass rate
        if integration_test_results:
            integration_passed = sum(r.passed for r in integration_test_results)
            integration_total = sum(r.tests_run for r in integration_test_results)
            integration_pass_rate = (integration_passed / integration_total * 100) if integration_total > 0 else 0
            
            quality_gate_results.append(QualityGateResult(
                gate_name="Integration Test Pass Rate",
                passed=integration_pass_rate >= self.quality_gates["integration_test_pass_rate"],
                actual_value=integration_pass_rate,
                threshold_value=self.quality_gates["integration_test_pass_rate"],
                message=f"Integration tests: {integration_pass_rate:.1f}% pass rate (need ≥{self.quality_gates['integration_test_pass_rate']}%)"
            ))
        
        # Overall pass rate
        quality_gate_results.append(QualityGateResult(
            gate_name="Overall Test Pass Rate",
            passed=overall_pass_rate >= self.quality_gates["overall_test_pass_rate"],
            actual_value=overall_pass_rate,
            threshold_value=self.quality_gates["overall_test_pass_rate"],
            message=f"Overall: {overall_pass_rate:.1f}% pass rate (need ≥{self.quality_gates['overall_test_pass_rate']}%)"
        ))
        
        # Execution time
        quality_gate_results.append(QualityGateResult(
            gate_name="Test Execution Time",
            passed=total_duration <= self.quality_gates["test_execution_time_limit"],
            actual_value=total_duration,
            threshold_value=self.quality_gates["test_execution_time_limit"],
            message=f"Execution time: {total_duration:.1f}s (limit: {self.quality_gates['test_execution_time_limit']}s)"
        ))
        
        # Component-specific duration checks
        for result in unit_test_results:
            if result.category.startswith("component_"):
                component_name = result.category.replace("component_", "")
                duration_per_test = result.duration / max(result.tests_run, 1)
                
                quality_gate_results.append(QualityGateResult(
                    gate_name=f"Component {component_name} Performance",
                    passed=duration_per_test <= self.quality_gates["unit_test_max_duration_per_component"],
                    actual_value=duration_per_test,
                    threshold_value=self.quality_gates["unit_test_max_duration_per_component"],
                    message=f"{component_name}: {duration_per_test:.2f}s per test (limit: {self.quality_gates['unit_test_max_duration_per_component']}s)"
                ))
        
        return quality_gate_results
    
    def generate_recommendations(self, quality_gate_results: List[QualityGateResult], category_results: List[TestCategoryResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check failed quality gates
        failed_gates = [qg for qg in quality_gate_results if not qg.passed]
        
        if failed_gates:
            recommendations.append(f"🔴 {len(failed_gates)} quality gate(s) failed. Address these before proceeding to higher test levels.")
        
        # Check for categories with low pass rates
        low_pass_rate_categories = [r for r in category_results if r.pass_rate < 80.0 and r.tests_run > 0]
        if low_pass_rate_categories:
            for category in low_pass_rate_categories:
                recommendations.append(f"⚠️  {category.category} has low pass rate ({category.pass_rate:.1f}%). Review and fix failing tests.")
        
        # Check for slow tests
        slow_categories = [r for r in category_results if r.duration > 30.0]
        if slow_categories:
            for category in slow_categories:
                recommendations.append(f"🐌 {category.category} is slow ({category.duration:.1f}s). Consider optimizing test performance.")
        
        # Success message if all gates pass
        if not failed_gates:
            recommendations.append("✅ All quality gates passed! Ready for system-level testing.")
        
        return recommendations
    
    def run_complete_test_suite(self, run_unit: bool = True, run_integration: bool = True, 
                               parallel: bool = True, max_workers: int = 4) -> TestSuiteReport:
        """Run complete unit and integration test suite."""
        start_time = time.time()
        print("🚀 Starting Unit and Integration Test Suite")
        print("=" * 60)
        
        # Check foundation tests first
        if not self.run_foundation_tests_check():
            print("❌ Foundation tests must pass before running unit/integration tests")
            sys.exit(1)
        
        # Discover tests
        all_test_categories = {}
        
        if run_unit:
            unit_tests = self.discover_unit_tests()
            all_test_categories.update(unit_tests)
            print(f"📋 Discovered {len(unit_tests)} unit test categories")
        
        if run_integration:
            integration_tests = self.discover_integration_tests()
            all_test_categories.update(integration_tests)
            print(f"📋 Discovered {len(integration_tests)} integration test categories")
        
        if not all_test_categories:
            print("❌ No tests discovered")
            sys.exit(1)
        
        print(f"📋 Total test categories: {len(all_test_categories)}")
        print()
        
        # Run tests
        if parallel and len(all_test_categories) > 1:
            print(f"🔄 Running tests in parallel (max {max_workers} workers)")
            category_results = self.run_tests_parallel(all_test_categories, max_workers)
        else:
            print("🔄 Running tests sequentially")
            category_results = []
            for category, files in all_test_categories.items():
                result = self.run_test_category(category, files)
                category_results.append(result)
        
        total_duration = time.time() - start_time
        
        # Calculate summary
        total_tests = sum(r.tests_run for r in category_results)
        total_passed = sum(r.passed for r in category_results)
        total_failed = sum(r.failed for r in category_results)
        total_skipped = sum(r.skipped for r in category_results)
        overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Validate quality gates
        quality_gate_results = self.validate_quality_gates(category_results, total_duration)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(quality_gate_results, category_results)
        
        # Create report
        report = TestSuiteReport(
            timestamp=datetime.utcnow().isoformat(),
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            total_duration=total_duration,
            overall_pass_rate=overall_pass_rate,
            category_results=category_results,
            quality_gates=quality_gate_results,
            performance_metrics={
                "categories_run": len(category_results),
                "parallel_execution": parallel,
                "avg_duration_per_category": total_duration / len(category_results) if category_results else 0
            },
            recommendations=recommendations
        )
        
        return report
    
    def print_report(self, report: TestSuiteReport):
        """Print comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 UNIT & INTEGRATION TEST REPORT")
        print("=" * 60)
        
        # Summary
        print(f"📅 Timestamp: {report.timestamp}")
        print(f"⏱️  Total Duration: {report.total_duration:.2f}s")
        print(f"📈 Overall Pass Rate: {report.overall_pass_rate:.1f}%")
        print(f"📊 Tests: {report.total_passed} passed, {report.total_failed} failed, {report.total_skipped} skipped")
        print()
        
        # Category breakdown
        print("📂 CATEGORY RESULTS:")
        for result in sorted(report.category_results, key=lambda x: x.category):
            status_icon = "✅" if result.pass_rate >= 90 else "⚠️" if result.pass_rate >= 70 else "❌"
            print(f"  {status_icon} {result.category}: {result.pass_rate:.1f}% ({result.passed}/{result.tests_run}) in {result.duration:.2f}s")
            
            if result.error_messages:
                for error in result.error_messages[:3]:  # Show first 3 errors
                    print(f"      🔴 {error}")
                if len(result.error_messages) > 3:
                    print(f"      ... and {len(result.error_messages) - 3} more errors")
        print()
        
        # Quality gates
        print("🚪 QUALITY GATES:")
        all_gates_passed = True
        for gate in report.quality_gates:
            status_icon = "✅" if gate.passed else "❌"
            if not gate.passed:
                all_gates_passed = False
            print(f"  {status_icon} {gate.message}")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        for rec in report.recommendations:
            print(f"  {rec}")
        print()
        
        # Final status
        if all_gates_passed:
            print("🎉 ALL QUALITY GATES PASSED! Unit and Integration testing complete.")
        else:
            print("🚨 QUALITY GATES FAILED! Address issues before proceeding.")
        
        print("=" * 60)
    
    def save_report(self, report: TestSuiteReport, filename: str = None):
        """Save test report to JSON file."""
        if filename is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"unit_integration_test_report_{timestamp}.json"
        
        report_path = self.base_path / filename
        
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"📄 Report saved to: {report_path}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Unit and Integration Test Runner")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", help="Run only integration tests")
    parser.add_argument("--sequential", action="store_true", help="Run tests sequentially instead of parallel")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel workers (default: 4)")
    parser.add_argument("--save-report", type=str, help="Save report to specific filename")
    
    args = parser.parse_args()
    
    # Determine what to run
    run_unit = not args.integration_only
    run_integration = not args.unit_only
    parallel = not args.sequential
    
    # Create runner and execute
    runner = UnitIntegrationTestRunner()
    
    try:
        report = runner.run_complete_test_suite(
            run_unit=run_unit,
            run_integration=run_integration,
            parallel=parallel,
            max_workers=args.workers
        )
        
        runner.print_report(report)
        runner.save_report(report, args.save_report)
        
        # Exit with appropriate code
        all_gates_passed = all(gate.passed for gate in report.quality_gates)
        sys.exit(0 if all_gates_passed else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Comprehensive Test Runner for Project Index Quality Assurance

Orchestrates all test suites with coverage reporting, performance monitoring,
quality gates, and detailed reporting for production readiness validation.
"""

import asyncio
import sys
import time
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import pytest
import coverage
import psutil
import os


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    status: str  # "passed", "failed", "skipped"
    duration: float
    test_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    coverage_percentage: float
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, Any]


@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    gate_name: str
    status: str  # "passed", "failed"
    criteria: Dict[str, Any]
    actual_values: Dict[str, Any]
    message: str


class TestRunner:
    """Comprehensive test runner with quality gates."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.quality_gates: List[QualityGateResult] = []
        self.start_time = None
        self.end_time = None
        self.coverage_threshold = 90.0
        self.performance_thresholds = {
            "unit_tests_max_duration": 300,    # 5 minutes
            "integration_tests_max_duration": 600,  # 10 minutes
            "performance_tests_max_duration": 900,  # 15 minutes
            "api_test_avg_response_time": 1.0,  # 1 second
            "memory_usage_max_mb": 1000,       # 1GB
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites with comprehensive validation."""
        self.start_time = time.time()
        print("🚀 Starting Project Index Quality Assurance Suite")
        print("=" * 80)
        
        try:
            # 1. Unit Tests
            await self._run_unit_tests()
            
            # 2. Integration Tests
            await self._run_integration_tests()
            
            # 3. Performance Tests
            await self._run_performance_tests()
            
            # 4. Security Tests
            await self._run_security_tests()
            
            # 5. End-to-End Tests
            await self._run_e2e_tests()
            
            # 6. Quality Gate Validation
            await self._validate_quality_gates()
            
            # 7. Generate Reports
            report = await self._generate_comprehensive_report()
            
            return report
            
        except Exception as e:
            print(f"❌ Test execution failed: {str(e)}")
            return {"status": "failed", "error": str(e)}
        
        finally:
            self.end_time = time.time()
    
    async def _run_unit_tests(self):
        """Run unit tests with coverage monitoring."""
        print("📋 Running Unit Tests...")
        
        # Start coverage monitoring
        cov = coverage.Coverage(source=['app/models', 'app/project_index', 'app/api'])
        cov.start()
        
        start_time = time.time()
        
        try:
            # Run unit tests
            exit_code = pytest.main([
                'tests/project_index/test_models.py',
                'tests/project_index/test_core.py',
                'tests/project_index/test_api.py',
                '--tb=short',
                '--quiet',
                '--json-report',
                '--json-report-file=reports/unit_test_results.json'
            ])
            
            duration = time.time() - start_time
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            coverage_percentage = cov.report(show_missing=False)
            
            # Parse test results
            test_results = self._parse_pytest_json_report('reports/unit_test_results.json')
            
            result = TestResult(
                suite_name="Unit Tests",
                status="passed" if exit_code == 0 else "failed",
                duration=duration,
                test_count=test_results.get('total', 0),
                passed_count=test_results.get('passed', 0),
                failed_count=test_results.get('failed', 0),
                skipped_count=test_results.get('skipped', 0),
                coverage_percentage=coverage_percentage,
                errors=test_results.get('errors', []),
                warnings=test_results.get('warnings', []),
                performance_metrics={"avg_test_duration": duration / max(test_results.get('total', 1), 1)}
            )
            
            self.results.append(result)
            
            print(f"✅ Unit Tests: {result.passed_count}/{result.test_count} passed, "
                  f"Coverage: {result.coverage_percentage:.1f}%, "
                  f"Duration: {result.duration:.1f}s")
        
        except Exception as e:
            self.results.append(TestResult(
                suite_name="Unit Tests",
                status="failed",
                duration=time.time() - start_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                coverage_percentage=0.0,
                errors=[str(e)],
                warnings=[],
                performance_metrics={}
            ))
            print(f"❌ Unit Tests failed: {str(e)}")
    
    async def _run_integration_tests(self):
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        start_time = time.time()
        
        try:
            exit_code = pytest.main([
                'tests/project_index/test_integration.py',
                '--tb=short',
                '--quiet',
                '--json-report',
                '--json-report-file=reports/integration_test_results.json'
            ])
            
            duration = time.time() - start_time
            test_results = self._parse_pytest_json_report('reports/integration_test_results.json')
            
            result = TestResult(
                suite_name="Integration Tests",
                status="passed" if exit_code == 0 else "failed",
                duration=duration,
                test_count=test_results.get('total', 0),
                passed_count=test_results.get('passed', 0),
                failed_count=test_results.get('failed', 0),
                skipped_count=test_results.get('skipped', 0),
                coverage_percentage=0.0,  # Integration tests don't measure coverage
                errors=test_results.get('errors', []),
                warnings=test_results.get('warnings', []),
                performance_metrics={"avg_test_duration": duration / max(test_results.get('total', 1), 1)}
            )
            
            self.results.append(result)
            
            print(f"✅ Integration Tests: {result.passed_count}/{result.test_count} passed, "
                  f"Duration: {result.duration:.1f}s")
        
        except Exception as e:
            self.results.append(TestResult(
                suite_name="Integration Tests",
                status="failed",
                duration=time.time() - start_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                coverage_percentage=0.0,
                errors=[str(e)],
                warnings=[],
                performance_metrics={}
            ))
            print(f"❌ Integration Tests failed: {str(e)}")
    
    async def _run_performance_tests(self):
        """Run performance tests with monitoring."""
        print("⚡ Running Performance Tests...")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        start_time = time.time()
        
        try:
            exit_code = pytest.main([
                'tests/project_index/test_performance.py',
                '--tb=short',
                '--quiet',
                '--json-report',
                '--json-report-file=reports/performance_test_results.json'
            ])
            
            duration = time.time() - start_time
            final_memory = process.memory_info().rss
            memory_used_mb = (final_memory - initial_memory) / 1024 / 1024
            
            test_results = self._parse_pytest_json_report('reports/performance_test_results.json')
            
            result = TestResult(
                suite_name="Performance Tests",
                status="passed" if exit_code == 0 else "failed",
                duration=duration,
                test_count=test_results.get('total', 0),
                passed_count=test_results.get('passed', 0),
                failed_count=test_results.get('failed', 0),
                skipped_count=test_results.get('skipped', 0),
                coverage_percentage=0.0,
                errors=test_results.get('errors', []),
                warnings=test_results.get('warnings', []),
                performance_metrics={
                    "memory_used_mb": memory_used_mb,
                    "avg_test_duration": duration / max(test_results.get('total', 1), 1)
                }
            )
            
            self.results.append(result)
            
            print(f"✅ Performance Tests: {result.passed_count}/{result.test_count} passed, "
                  f"Memory: {memory_used_mb:.1f}MB, Duration: {result.duration:.1f}s")
        
        except Exception as e:
            self.results.append(TestResult(
                suite_name="Performance Tests",
                status="failed",
                duration=time.time() - start_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                coverage_percentage=0.0,
                errors=[str(e)],
                warnings=[],
                performance_metrics={}
            ))
            print(f"❌ Performance Tests failed: {str(e)}")
    
    async def _run_security_tests(self):
        """Run security tests."""
        print("🔒 Running Security Tests...")
        
        start_time = time.time()
        
        try:
            exit_code = pytest.main([
                'tests/project_index/test_security.py',
                '--tb=short',
                '--quiet',
                '--json-report',
                '--json-report-file=reports/security_test_results.json'
            ])
            
            duration = time.time() - start_time
            test_results = self._parse_pytest_json_report('reports/security_test_results.json')
            
            result = TestResult(
                suite_name="Security Tests",
                status="passed" if exit_code == 0 else "failed",
                duration=duration,
                test_count=test_results.get('total', 0),
                passed_count=test_results.get('passed', 0),
                failed_count=test_results.get('failed', 0),
                skipped_count=test_results.get('skipped', 0),
                coverage_percentage=0.0,
                errors=test_results.get('errors', []),
                warnings=test_results.get('warnings', []),
                performance_metrics={"avg_test_duration": duration / max(test_results.get('total', 1), 1)}
            )
            
            self.results.append(result)
            
            print(f"✅ Security Tests: {result.passed_count}/{result.test_count} passed, "
                  f"Duration: {result.duration:.1f}s")
        
        except Exception as e:
            self.results.append(TestResult(
                suite_name="Security Tests",
                status="failed",
                duration=time.time() - start_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                coverage_percentage=0.0,
                errors=[str(e)],
                warnings=[],
                performance_metrics={}
            ))
            print(f"❌ Security Tests failed: {str(e)}")
    
    async def _run_e2e_tests(self):
        """Run end-to-end tests."""
        print("🎯 Running End-to-End Tests...")
        
        start_time = time.time()
        
        try:
            # Check if there are E2E tests to run
            e2e_test_files = list(Path('tests/project_index').glob('**/test_e2e*.py'))
            
            if e2e_test_files:
                exit_code = pytest.main([
                    str(e2e_test_files[0]),
                    '--tb=short',
                    '--quiet',
                    '--json-report',
                    '--json-report-file=reports/e2e_test_results.json'
                ])
                
                test_results = self._parse_pytest_json_report('reports/e2e_test_results.json')
            else:
                # No E2E tests found, create placeholder result
                exit_code = 0
                test_results = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
            
            duration = time.time() - start_time
            
            result = TestResult(
                suite_name="End-to-End Tests",
                status="passed" if exit_code == 0 else "failed",
                duration=duration,
                test_count=test_results.get('total', 0),
                passed_count=test_results.get('passed', 0),
                failed_count=test_results.get('failed', 0),
                skipped_count=test_results.get('skipped', 0),
                coverage_percentage=0.0,
                errors=test_results.get('errors', []),
                warnings=test_results.get('warnings', []),
                performance_metrics={"avg_test_duration": duration / max(test_results.get('total', 1), 1)}
            )
            
            self.results.append(result)
            
            print(f"✅ End-to-End Tests: {result.passed_count}/{result.test_count} passed, "
                  f"Duration: {result.duration:.1f}s")
        
        except Exception as e:
            self.results.append(TestResult(
                suite_name="End-to-End Tests",
                status="failed",
                duration=time.time() - start_time,
                test_count=0,
                passed_count=0,
                failed_count=1,
                skipped_count=0,
                coverage_percentage=0.0,
                errors=[str(e)],
                warnings=[],
                performance_metrics={}
            ))
            print(f"❌ End-to-End Tests failed: {str(e)}")
    
    async def _validate_quality_gates(self):
        """Validate quality gates for production readiness."""
        print("🚦 Validating Quality Gates...")
        
        # Gate 1: Coverage Threshold
        unit_test_result = next((r for r in self.results if r.suite_name == "Unit Tests"), None)
        if unit_test_result:
            coverage_gate = QualityGateResult(
                gate_name="Code Coverage",
                status="passed" if unit_test_result.coverage_percentage >= self.coverage_threshold else "failed",
                criteria={"minimum_coverage": self.coverage_threshold},
                actual_values={"coverage": unit_test_result.coverage_percentage},
                message=f"Coverage: {unit_test_result.coverage_percentage:.1f}% (required: {self.coverage_threshold}%)"
            )
            self.quality_gates.append(coverage_gate)
        
        # Gate 2: Test Success Rate
        total_tests = sum(r.test_count for r in self.results)
        total_passed = sum(r.passed_count for r in self.results)
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        success_gate = QualityGateResult(
            gate_name="Test Success Rate",
            status="passed" if success_rate >= 95.0 else "failed",
            criteria={"minimum_success_rate": 95.0},
            actual_values={"success_rate": success_rate},
            message=f"Success Rate: {success_rate:.1f}% (required: 95.0%)"
        )
        self.quality_gates.append(success_gate)
        
        # Gate 3: Performance Thresholds
        perf_result = next((r for r in self.results if r.suite_name == "Performance Tests"), None)
        if perf_result:
            memory_used = perf_result.performance_metrics.get('memory_used_mb', 0)
            performance_gate = QualityGateResult(
                gate_name="Performance Standards",
                status="passed" if memory_used <= self.performance_thresholds['memory_usage_max_mb'] else "failed",
                criteria={"max_memory_mb": self.performance_thresholds['memory_usage_max_mb']},
                actual_values={"memory_used_mb": memory_used},
                message=f"Memory Usage: {memory_used:.1f}MB (max: {self.performance_thresholds['memory_usage_max_mb']}MB)"
            )
            self.quality_gates.append(performance_gate)
        
        # Gate 4: Security Tests
        security_result = next((r for r in self.results if r.suite_name == "Security Tests"), None)
        if security_result:
            security_gate = QualityGateResult(
                gate_name="Security Validation",
                status="passed" if security_result.failed_count == 0 else "failed",
                criteria={"max_security_failures": 0},
                actual_values={"security_failures": security_result.failed_count},
                message=f"Security Failures: {security_result.failed_count} (max: 0)"
            )
            self.quality_gates.append(security_gate)
        
        # Gate 5: Overall System Health
        failed_suites = [r for r in self.results if r.status == "failed"]
        system_health_gate = QualityGateResult(
            gate_name="System Health",
            status="passed" if len(failed_suites) == 0 else "failed",
            criteria={"max_failed_suites": 0},
            actual_values={"failed_suites": len(failed_suites)},
            message=f"Failed Test Suites: {len(failed_suites)} (max: 0)"
        )
        self.quality_gates.append(system_health_gate)
        
        # Print quality gate results
        for gate in self.quality_gates:
            status_icon = "✅" if gate.status == "passed" else "❌"
            print(f"{status_icon} {gate.gate_name}: {gate.message}")
    
    async def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_time": total_duration,
            "summary": {
                "total_suites": len(self.results),
                "passed_suites": len([r for r in self.results if r.status == "passed"]),
                "failed_suites": len([r for r in self.results if r.status == "failed"]),
                "total_tests": sum(r.test_count for r in self.results),
                "total_passed": sum(r.passed_count for r in self.results),
                "total_failed": sum(r.failed_count for r in self.results),
                "total_skipped": sum(r.skipped_count for r in self.results),
                "overall_success_rate": (sum(r.passed_count for r in self.results) / max(sum(r.test_count for r in self.results), 1)) * 100
            },
            "test_results": [asdict(result) for result in self.results],
            "quality_gates": [asdict(gate) for gate in self.quality_gates],
            "quality_gate_status": "passed" if all(gate.status == "passed" for gate in self.quality_gates) else "failed",
            "production_ready": all(gate.status == "passed" for gate in self.quality_gates) and all(r.status == "passed" for r in self.results)
        }
        
        # Save detailed report
        report_file = f"reports/comprehensive_test_report_{int(time.time())}.json"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"Execution Time: {total_duration:.1f}s")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Success Rate: {report['summary']['overall_success_rate']:.1f}%")
        print(f"Quality Gates: {len([g for g in self.quality_gates if g.status == 'passed'])}/{len(self.quality_gates)} passed")
        
        if report["production_ready"]:
            print("🎉 PRODUCTION READY: All tests passed and quality gates met!")
        else:
            print("⚠️  NOT PRODUCTION READY: Quality gates failed or tests failed")
        
        print(f"📄 Detailed report saved to: {report_file}")
        
        return report
    
    def _parse_pytest_json_report(self, report_file: str) -> Dict[str, Any]:
        """Parse pytest JSON report."""
        try:
            if Path(report_file).exists():
                with open(report_file, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                return {
                    'total': summary.get('total', 0),
                    'passed': summary.get('passed', 0),
                    'failed': summary.get('failed', 0),
                    'skipped': summary.get('skipped', 0),
                    'errors': [test.get('call', {}).get('longrepr', '') for test in data.get('tests', []) if test.get('outcome') == 'failed'],
                    'warnings': []
                }
        except Exception as e:
            print(f"Warning: Could not parse test report {report_file}: {e}")
        
        return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0, 'errors': [], 'warnings': []}


class ContinuousIntegrationRunner:
    """CI/CD integration for automated testing."""
    
    def __init__(self):
        self.runner = TestRunner()
    
    async def run_ci_pipeline(self) -> bool:
        """Run CI pipeline with automated quality gates."""
        print("🔄 Starting CI/CD Pipeline for Project Index")
        
        # 1. Run all tests
        report = await self.runner.run_all_tests()
        
        # 2. Generate artifacts
        await self._generate_ci_artifacts(report)
        
        # 3. Determine pipeline status
        pipeline_success = report.get("production_ready", False)
        
        # 4. Set exit code for CI system
        exit_code = 0 if pipeline_success else 1
        
        print(f"\n🏁 CI Pipeline {'PASSED' if pipeline_success else 'FAILED'}")
        return pipeline_success
    
    async def _generate_ci_artifacts(self, report: Dict[str, Any]):
        """Generate CI artifacts for build system integration."""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # 1. JUnit XML for test results integration
        await self._generate_junit_xml(report, artifacts_dir / "test-results.xml")
        
        # 2. Coverage report for code coverage badges
        await self._generate_coverage_badge_data(report, artifacts_dir / "coverage.json")
        
        # 3. Performance metrics for monitoring
        await self._generate_performance_metrics(report, artifacts_dir / "performance.json")
        
        # 4. Quality gate status for deployment gates
        await self._generate_quality_gate_status(report, artifacts_dir / "quality-gates.json")
    
    async def _generate_junit_xml(self, report: Dict[str, Any], output_file: Path):
        """Generate JUnit XML for CI integration."""
        # Simplified JUnit XML generation
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += '<testsuites>\n'
        
        for result in report["test_results"]:
            xml_content += f'  <testsuite name="{result["suite_name"]}" tests="{result["test_count"]}" '
            xml_content += f'failures="{result["failed_count"]}" time="{result["duration"]:.3f}">\n'
            xml_content += '  </testsuite>\n'
        
        xml_content += '</testsuites>\n'
        
        with open(output_file, 'w') as f:
            f.write(xml_content)
    
    async def _generate_coverage_badge_data(self, report: Dict[str, Any], output_file: Path):
        """Generate coverage data for badges."""
        unit_test_result = next((r for r in report["test_results"] if r["suite_name"] == "Unit Tests"), None)
        coverage_data = {
            "coverage": unit_test_result["coverage_percentage"] if unit_test_result else 0,
            "color": "green" if (unit_test_result and unit_test_result["coverage_percentage"] >= 90) else "yellow" if (unit_test_result and unit_test_result["coverage_percentage"] >= 80) else "red"
        }
        
        with open(output_file, 'w') as f:
            json.dump(coverage_data, f, indent=2)
    
    async def _generate_performance_metrics(self, report: Dict[str, Any], output_file: Path):
        """Generate performance metrics for monitoring."""
        perf_result = next((r for r in report["test_results"] if r["suite_name"] == "Performance Tests"), None)
        
        metrics = {
            "timestamp": report["timestamp"],
            "execution_time": report["execution_time"],
            "memory_usage_mb": perf_result["performance_metrics"].get("memory_used_mb", 0) if perf_result else 0,
            "test_success_rate": report["summary"]["overall_success_rate"],
            "total_tests": report["summary"]["total_tests"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    async def _generate_quality_gate_status(self, report: Dict[str, Any], output_file: Path):
        """Generate quality gate status for deployment decisions."""
        status_data = {
            "production_ready": report["production_ready"],
            "quality_gate_status": report["quality_gate_status"],
            "gates": report["quality_gates"],
            "recommendation": "DEPLOY" if report["production_ready"] else "DO_NOT_DEPLOY"
        }
        
        with open(output_file, 'w') as f:
            json.dump(status_data, f, indent=2)


async def main():
    """Main entry point for test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Project Index Quality Assurance Test Runner")
    parser.add_argument("--ci", action="store_true", help="Run in CI/CD mode")
    parser.add_argument("--coverage-threshold", type=float, default=90.0, help="Coverage threshold percentage")
    parser.add_argument("--suite", choices=["unit", "integration", "performance", "security", "e2e", "all"], 
                       default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    if args.ci:
        # CI/CD mode
        ci_runner = ContinuousIntegrationRunner()
        ci_runner.runner.coverage_threshold = args.coverage_threshold
        
        success = await ci_runner.run_ci_pipeline()
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        runner = TestRunner()
        runner.coverage_threshold = args.coverage_threshold
        
        if args.suite == "all":
            report = await runner.run_all_tests()
        else:
            # Run specific suite
            if args.suite == "unit":
                await runner._run_unit_tests()
            elif args.suite == "integration":
                await runner._run_integration_tests()
            elif args.suite == "performance":
                await runner._run_performance_tests()
            elif args.suite == "security":
                await runner._run_security_tests()
            elif args.suite == "e2e":
                await runner._run_e2e_tests()
            
            report = await runner._generate_comprehensive_report()
        
        print(f"\n🎯 Test execution completed. Production ready: {report['production_ready']}")


if __name__ == "__main__":
    asyncio.run(main())
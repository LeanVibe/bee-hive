#!/usr/bin/env python3
"""
Comprehensive Test Runner for Context Compression Feature
Executes all test suites and generates detailed reports
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pytest


class ContextCompressionTestRunner:
    """Comprehensive test runner for context compression feature"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {}
        self.start_time = datetime.utcnow()
        
    def run_test_suite(self, test_path: str, suite_name: str, coverage: bool = True) -> Dict[str, Any]:
        """Run a specific test suite and return results"""
        print(f"\n{'='*60}")
        print(f"Running {suite_name}")
        print(f"{'='*60}")
        
        cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]
        
        if coverage:
            cmd.extend([
                "--cov=app.core.context_compression",
                "--cov=app.core.hive_slash_commands", 
                "--cov=app.api.v1.sessions",
                "--cov-report=term-missing",
                "--cov-append"
            ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per suite
            )
            
            execution_time = time.time() - start_time
            
            return {
                "suite_name": suite_name,
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "suite_name": suite_name,
                "success": False,
                "execution_time": 300,
                "stdout": "",
                "stderr": f"Test suite timed out after 300 seconds",
                "returncode": -1
            }
        except Exception as e:
            return {
                "suite_name": suite_name,
                "success": False,
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": f"Test execution failed: {str(e)}",
                "returncode": -1
            }
    
    def run_frontend_tests(self) -> Dict[str, Any]:
        """Run frontend TypeScript tests"""
        print(f"\n{'='*60}")
        print("Running Frontend Tests")
        print(f"{'='*60}")
        
        frontend_path = self.project_root / "mobile-pwa"
        
        if not frontend_path.exists():
            return {
                "suite_name": "Frontend Tests",
                "success": False,
                "execution_time": 0,
                "stdout": "",
                "stderr": "Frontend directory not found",
                "returncode": -1
            }
        
        start_time = time.time()
        
        try:
            # Run npm test (or equivalent)
            result = subprocess.run(
                ["npm", "test"],
                cwd=frontend_path,
                capture_output=True,
                text=True,
                timeout=180  # 3 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "suite_name": "Frontend Tests",
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "suite_name": "Frontend Tests",
                "success": False,
                "execution_time": 180,
                "stdout": "",
                "stderr": "Frontend tests timed out after 180 seconds",
                "returncode": -1
            }
        except Exception as e:
            return {
                "suite_name": "Frontend Tests",
                "success": False,
                "execution_time": time.time() - start_time,
                "stdout": "",
                "stderr": f"Frontend test execution failed: {str(e)}",
                "returncode": -1
            }
    
    def check_test_coverage(self) -> Dict[str, Any]:
        """Generate final coverage report"""
        print(f"\n{'='*60}")
        print("Generating Coverage Report")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov-report=html", "--cov-report=json"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Try to read coverage data
            coverage_file = self.project_root / "coverage.json"
            coverage_data = {}
            
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                except Exception:
                    pass
            
            return {
                "success": result.returncode == 0,
                "coverage_data": coverage_data,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "success": False,
                "coverage_data": {},
                "stdout": "",
                "stderr": f"Coverage generation failed: {str(e)}"
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites in order"""
        print("🧪 Starting Comprehensive Context Compression Test Suite")
        print(f"Start time: {self.start_time.isoformat()}")
        
        # Test suite configuration
        test_suites = [
            {
                "path": "tests/unit/test_context_compression.py",
                "name": "Unit Tests - Context Compression",
                "critical": True
            },
            {
                "path": "tests/unit/test_hive_compact_command.py", 
                "name": "Unit Tests - Hive Compact Command",
                "critical": True
            },
            {
                "path": "tests/integration/test_context_compression_pipeline.py",
                "name": "Integration Tests",
                "critical": True
            },
            {
                "path": "tests/api/test_sessions_context_compression.py",
                "name": "API Endpoint Tests", 
                "critical": True
            },
            {
                "path": "tests/edge_cases/test_context_compression_edge_cases.py",
                "name": "Edge Case Tests",
                "critical": False
            },
            {
                "path": "tests/performance/test_context_compression_performance.py",
                "name": "Performance Tests",
                "critical": True
            },
            {
                "path": "tests/security/test_context_compression_security.py",
                "name": "Security Tests",
                "critical": True
            },
            {
                "path": "tests/production/test_context_compression_production_readiness.py",
                "name": "Production Readiness Tests",
                "critical": False
            }
        ]
        
        results = {
            "start_time": self.start_time.isoformat(),
            "test_suites": [],
            "summary": {
                "total_suites": len(test_suites) + 1,  # +1 for frontend
                "passed": 0,
                "failed": 0,
                "critical_failures": 0,
                "total_execution_time": 0
            }
        }
        
        # Run backend test suites
        for suite_config in test_suites:
            suite_result = self.run_test_suite(
                suite_config["path"],
                suite_config["name"],
                coverage=True
            )
            
            suite_result["critical"] = suite_config["critical"]
            results["test_suites"].append(suite_result)
            
            if suite_result["success"]:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
                if suite_config["critical"]:
                    results["summary"]["critical_failures"] += 1
            
            results["summary"]["total_execution_time"] += suite_result["execution_time"]
        
        # Run frontend tests
        frontend_result = self.run_frontend_tests()
        frontend_result["critical"] = True
        results["test_suites"].append(frontend_result)
        
        if frontend_result["success"]:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
            results["summary"]["critical_failures"] += 1
        
        results["summary"]["total_execution_time"] += frontend_result["execution_time"]
        
        # Generate coverage report
        coverage_result = self.check_test_coverage()
        results["coverage"] = coverage_result
        
        # Calculate final results
        results["end_time"] = datetime.utcnow().isoformat()
        results["success"] = results["summary"]["critical_failures"] == 0
        results["quality_gate_passed"] = self.evaluate_quality_gates(results)
        
        return results
    
    def evaluate_quality_gates(self, results: Dict[str, Any]) -> bool:
        """Evaluate if all quality gates are passed"""
        quality_gates = {
            "no_critical_failures": results["summary"]["critical_failures"] == 0,
            "minimum_pass_rate": (results["summary"]["passed"] / results["summary"]["total_suites"]) >= 0.9,
            "performance_acceptable": results["summary"]["total_execution_time"] < 1800,  # 30 minutes max
        }
        
        # Check coverage if available
        coverage_data = results.get("coverage", {}).get("coverage_data", {})
        if coverage_data and "totals" in coverage_data:
            coverage_percent = coverage_data["totals"].get("percent_covered", 0)
            quality_gates["coverage_threshold"] = coverage_percent >= 90.0
        
        results["quality_gates"] = quality_gates
        return all(quality_gates.values())
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("CONTEXT COMPRESSION FEATURE - COMPREHENSIVE TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        report_lines.append(f"Test Duration: {results['summary']['total_execution_time']:.2f} seconds")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Status: {'✅ PASSED' if results['success'] else '❌ FAILED'}")
        report_lines.append(f"Quality Gates: {'✅ PASSED' if results['quality_gate_passed'] else '❌ FAILED'}")
        report_lines.append(f"Test Suites: {results['summary']['passed']}/{results['summary']['total_suites']} passed")
        report_lines.append(f"Critical Failures: {results['summary']['critical_failures']}")
        report_lines.append("")
        
        # Quality Gates Details
        if "quality_gates" in results:
            report_lines.append("QUALITY GATES")
            report_lines.append("-" * 40)
            for gate_name, passed in results["quality_gates"].items():
                status = "✅ PASS" if passed else "❌ FAIL"
                report_lines.append(f"{gate_name}: {status}")
            report_lines.append("")
        
        # Test Suite Results
        report_lines.append("TEST SUITE RESULTS")
        report_lines.append("-" * 40)
        
        for suite in results["test_suites"]:
            status = "✅ PASS" if suite["success"] else "❌ FAIL"
            critical_flag = " (CRITICAL)" if suite.get("critical", False) else ""
            report_lines.append(f"{suite['suite_name']}{critical_flag}: {status}")
            report_lines.append(f"  Execution Time: {suite['execution_time']:.2f}s")
            
            if not suite["success"] and suite["stderr"]:
                # Show first few lines of error
                error_lines = suite["stderr"].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        report_lines.append(f"  Error: {line.strip()}")
            
            report_lines.append("")
        
        # Coverage Report
        coverage_data = results.get("coverage", {}).get("coverage_data", {})
        if coverage_data and "totals" in coverage_data:
            report_lines.append("CODE COVERAGE")
            report_lines.append("-" * 40)
            totals = coverage_data["totals"]
            report_lines.append(f"Overall Coverage: {totals.get('percent_covered', 0):.1f}%")
            report_lines.append(f"Lines Covered: {totals.get('covered_lines', 0)}/{totals.get('num_statements', 0)}")
            report_lines.append(f"Missing Lines: {totals.get('missing_lines', 0)}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if results["summary"]["critical_failures"] > 0:
            report_lines.append("🚨 CRITICAL: Fix failing critical test suites before deployment")
        
        if results["summary"]["failed"] > 0:
            report_lines.append("⚠️  Review and fix failing test suites")
        
        coverage_percent = 0
        if coverage_data and "totals" in coverage_data:
            coverage_percent = coverage_data["totals"].get("percent_covered", 0)
        
        if coverage_percent < 90:
            report_lines.append(f"📈 Improve test coverage from {coverage_percent:.1f}% to ≥90%")
        
        if results["summary"]["total_execution_time"] > 900:  # 15 minutes
            report_lines.append("⚡ Consider optimizing test execution time")
        
        if results["quality_gate_passed"]:
            report_lines.append("✅ All quality gates passed - Ready for production deployment")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save test results to files"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.project_root / f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save text report
        report = self.generate_report(results)
        report_file = self.project_root / f"test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📊 Results saved:")
        print(f"  JSON: {json_file}")
        print(f"  Report: {report_file}")


def main():
    """Main execution function"""
    runner = ContextCompressionTestRunner()
    
    try:
        results = runner.run_all_tests()
        
        # Print final report
        report = runner.generate_report(results)
        print("\n" + report)
        
        # Save results
        runner.save_results(results)
        
        # Exit with appropriate code
        exit_code = 0 if results["quality_gate_passed"] else 1
        
        if exit_code == 0:
            print("\n🎉 All tests passed! Context compression feature is ready for production.")
        else:
            print("\n❌ Quality gates failed. Please review and fix issues before deployment.")
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
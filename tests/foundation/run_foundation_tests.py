#!/usr/bin/env python3
"""
Foundation Test Runner - Testing Pyramid Base Layer

Executes comprehensive foundation testing that validates imports, configurations,
models, and core dependencies. Foundation tests are designed to be fast, reliable,
and provide confidence in basic system integrity.

TESTING PYRAMID LEVEL: Foundation (Base Layer)
EXECUTION TIME TARGET: <30 seconds
QUALITY GATES: All foundation tests must pass for higher-level testing
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class FoundationTestRunner:
    """Comprehensive foundation test execution and reporting."""
    
    def __init__(self, verbose: bool = True, fast_mode: bool = False):
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.start_time = time.time()
        self.results: Dict[str, Any] = {
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "warnings": 0,
                "execution_time": 0.0
            },
            "test_categories": {},
            "quality_gates": {},
            "recommendations": []
        }
        
    def run_all_foundation_tests(self) -> bool:
        """Run all foundation tests and return success status."""
        if self.verbose:
            print("🏗️  FOUNDATION TESTING - Testing Pyramid Base Layer")
            print("=" * 60)
            print("Validating imports, configurations, models, and core dependencies")
            print()
            
        success = True
        
        # Run each test category
        test_categories = [
            ("Import Resolution", self._run_import_tests),
            ("Configuration Validation", self._run_config_tests),
            ("Model Integrity", self._run_model_tests),
            ("Core Dependencies", self._run_dependency_tests)
        ]
        
        for category_name, test_function in test_categories:
            if self.verbose:
                print(f"🔍 Running {category_name} Tests...")
                
            category_success = test_function()
            
            if not category_success:
                success = False
                
            if self.verbose:
                status = "✅ PASSED" if category_success else "❌ FAILED"
                print(f"   {status}")
                print()
                
        # Validate quality gates
        self._validate_quality_gates()
        
        # Generate final report
        self._generate_final_report()
        
        return success
    
    def _run_import_tests(self) -> bool:
        """Run import resolution tests."""
        test_file = Path(__file__).parent / "test_import_resolution.py"
        return self._run_pytest_category(
            test_file, 
            "import_resolution"
        )
    
    def _run_config_tests(self) -> bool:
        """Run configuration validation tests."""
        test_file = Path(__file__).parent / "test_configuration_validation.py"
        return self._run_pytest_category(
            test_file,
            "configuration_validation"
        )
    
    def _run_model_tests(self) -> bool:
        """Run model integrity tests."""
        test_file = Path(__file__).parent / "test_model_integrity.py"
        return self._run_pytest_category(
            test_file,
            "model_integrity"
        )
    
    def _run_dependency_tests(self) -> bool:
        """Run core dependency tests."""
        test_file = Path(__file__).parent / "test_core_dependencies.py"
        return self._run_pytest_category(
            test_file,
            "core_dependencies"
        )
    
    def _run_pytest_category(self, test_file: Path, category: str, markers: List[str] = None) -> bool:
        """Run a category of pytest tests."""
        if not test_file.exists():
            self.results["test_categories"][category] = {
                "status": "skipped",
                "reason": "Test file not found",
                "tests": 0,
                "passed": 0,
                "failed": 0
            }
            return True
            
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "--tb=short",
            "--no-cov",
            "--disable-warnings",
            "-v" if self.verbose else "-q"
        ]
        
        if self.fast_mode:
            cmd.extend(["--maxfail=3", "-x"])
            
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        try:
            # Run tests
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Hard timeout for the entire category
            )
            execution_time = time.time() - start_time
            
            # Parse results
            success = result.returncode == 0
            tests_run = self._count_tests_in_output(result.stdout)
            
            self.results["test_categories"][category] = {
                "status": "passed" if success else "failed",
                "execution_time": execution_time,
                "tests_run": tests_run,
                "exit_code": result.returncode,
                "stdout": result.stdout if not success or self.verbose else "",
                "stderr": result.stderr if result.stderr else ""
            }
            
            # Update summary
            self.results["summary"]["total_tests"] += tests_run
            if success:
                self.results["summary"]["passed"] += tests_run
            else:
                self.results["summary"]["failed"] += tests_run
                
            return success
            
        except subprocess.TimeoutExpired:
            self.results["test_categories"][category] = {
                "status": "timeout",
                "reason": "Test execution timed out",
                "execution_time": 60.0
            }
            return False
            
        except Exception as e:
            self.results["test_categories"][category] = {
                "status": "error",
                "reason": f"Test execution error: {e}",
                "execution_time": 0.0
            }
            return False
    
    def _count_tests_in_output(self, output: str) -> int:
        """Count number of tests from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if "passed" in line or "failed" in line:
                # Look for pytest summary line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts) and "passed" in parts[i + 1]:
                        return int(part)
        return 0
    
    def _validate_quality_gates(self):
        """Validate foundation testing quality gates."""
        gates = {
            "execution_time": {
                "threshold": 30.0,
                "actual": time.time() - self.start_time,
                "passed": False
            },
            "test_coverage": {
                "threshold": 4,  # All 4 categories
                "actual": len([cat for cat in self.results["test_categories"].values() 
                              if cat.get("status") == "passed"]),
                "passed": False
            },
            "failure_rate": {
                "threshold": 0.0,  # No failures allowed in foundation
                "actual": self.results["summary"]["failed"],
                "passed": False
            }
        }
        
        # Check each gate
        gates["execution_time"]["passed"] = (
            gates["execution_time"]["actual"] <= gates["execution_time"]["threshold"]
        )
        
        gates["test_coverage"]["passed"] = (
            gates["test_coverage"]["actual"] >= gates["test_coverage"]["threshold"]
        )
        
        gates["failure_rate"]["passed"] = (
            gates["failure_rate"]["actual"] <= gates["failure_rate"]["threshold"]
        )
        
        self.results["quality_gates"] = gates
        
        # Generate recommendations based on gate results
        if not gates["execution_time"]["passed"]:
            self.results["recommendations"].append(
                f"Foundation tests took {gates['execution_time']['actual']:.1f}s "
                f"(target: <{gates['execution_time']['threshold']}s). "
                f"Consider optimizing slow tests."
            )
            
        if not gates["test_coverage"]["passed"]:
            self.results["recommendations"].append(
                f"Only {gates['test_coverage']['actual']} of 4 test categories passed. "
                f"Foundation layer requires all categories to pass."
            )
            
        if not gates["failure_rate"]["passed"]:
            self.results["recommendations"].append(
                f"Foundation tests had {gates['failure_rate']['actual']} failures. "
                f"Foundation layer must have zero failures for higher-level testing."
            )
    
    def _generate_final_report(self):
        """Generate final test report."""
        self.results["summary"]["execution_time"] = time.time() - self.start_time
        
        if self.verbose:
            print("📊 FOUNDATION TEST SUMMARY")
            print("=" * 40)
            
            # Summary stats
            summary = self.results["summary"]
            print(f"Total Tests: {summary['total_tests']}")
            print(f"Passed: {summary['passed']}")
            print(f"Failed: {summary['failed']}")
            print(f"Execution Time: {summary['execution_time']:.2f}s")
            print()
            
            # Category results
            print("📋 Category Results:")
            for category, result in self.results["test_categories"].items():
                status_icon = "✅" if result.get("status") == "passed" else "❌"
                time_str = f"({result.get('execution_time', 0):.1f}s)"
                print(f"  {status_icon} {category.replace('_', ' ').title()} {time_str}")
            print()
            
            # Quality gates
            print("🚪 Quality Gates:")
            for gate_name, gate in self.results["quality_gates"].items():
                status_icon = "✅" if gate["passed"] else "❌"
                print(f"  {status_icon} {gate_name.replace('_', ' ').title()}: "
                      f"{gate['actual']} (threshold: {gate['threshold']})")
            print()
            
            # Recommendations
            if self.results["recommendations"]:
                print("💡 Recommendations:")
                for rec in self.results["recommendations"]:
                    print(f"  • {rec}")
                print()
                
        # Save detailed report
        report_file = Path(__file__).parent / "foundation_test_report.json"
        self.results["metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runner_version": "1.0.0",
            "test_level": "foundation",
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        if self.verbose:
            print(f"📄 Detailed report saved to: {report_file}")
    
    def get_results(self) -> Dict[str, Any]:
        """Get test results."""
        return self.results
    
    def all_quality_gates_passed(self) -> bool:
        """Check if all quality gates passed."""
        return all(gate["passed"] for gate in self.results["quality_gates"].values())

def main():
    """Main entry point for foundation test runner."""
    parser = argparse.ArgumentParser(description="Foundation Test Runner")
    parser.add_argument("--quiet", action="store_true", help="Quiet output")
    parser.add_argument("--fast", action="store_true", help="Fast mode (exit on first failure)")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing results")
    
    args = parser.parse_args()
    
    if args.report_only:
        # Load existing results and generate report
        report_file = Path(__file__).parent / "foundation_test_report.json"
        if report_file.exists():
            with open(report_file) as f:
                results = json.load(f)
            print("📊 Foundation Test Report (from existing results)")
            print(json.dumps(results["summary"], indent=2))
            return 0
        else:
            print("❌ No existing test results found")
            return 1
    
    # Run tests
    runner = FoundationTestRunner(verbose=not args.quiet, fast_mode=args.fast)
    success = runner.run_all_foundation_tests()
    
    # Quality gate validation
    if not runner.all_quality_gates_passed():
        if not args.quiet:
            print("❌ FOUNDATION QUALITY GATES FAILED")
            print("Higher-level testing should not proceed until foundation issues are resolved.")
        return 2
    
    if success:
        if not args.quiet:
            print("🎉 ALL FOUNDATION TESTS PASSED")
            print("System foundation is solid - ready for higher-level testing!")
        return 0
    else:
        if not args.quiet:
            print("❌ FOUNDATION TESTS FAILED")
            print("Foundation issues must be resolved before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
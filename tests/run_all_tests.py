#!/usr/bin/env python3
"""
Multi-CLI Agent Testing Suite Runner

Comprehensive test runner for all multi-CLI agent coordination tests.
Executes all test suites in the correct order and generates consolidated reports.
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import subprocess
from dataclasses import dataclass

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import all test frameworks
from multi_cli_agent_testing_framework import MultiCLITestFramework
from git_worktree_isolation_tests import WorktreeSecurityTester
from multi_agent_coordination_scenarios import MultiAgentCoordinationTester
from communication_protocol_tests import CommunicationProtocolTester
from end_to_end_workflow_tests import EndToEndTestSuite
from performance_reliability_tests import PerformanceTestSuite
from security_isolation_tests import SecurityTestSuite

@dataclass
class TestSuiteResult:
    """Results from a test suite execution."""
    suite_name: str
    start_time: float
    end_time: float
    duration: float
    status: str  # "passed", "failed", "error", "skipped"
    tests_executed: int
    tests_passed: int
    tests_failed: int
    details: Dict[str, Any]
    error_message: Optional[str] = None

class MultiCLITestRunner:
    """Orchestrates execution of all multi-CLI agent tests."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.results = []
        self.start_time = 0.0
        self.end_time = 0.0
        self.summary = {}
    
    async def run_all_tests(self, suites_to_run: List[str] = None) -> Dict[str, Any]:
        """Run all test suites or specified subset."""
        print("🚀 Multi-CLI Agent Coordination Testing Suite")
        print("=" * 80)
        
        self.start_time = time.time()
        
        # Define all available test suites
        all_suites = {
            "foundation": self._run_foundation_tests,
            "worktree": self._run_worktree_tests,
            "coordination": self._run_coordination_tests,
            "communication": self._run_communication_tests,
            "e2e": self._run_e2e_tests,
            "performance": self._run_performance_tests,
            "security": self._run_security_tests
        }
        
        # Determine which suites to run
        if suites_to_run is None:
            suites_to_run = list(all_suites.keys())
        
        # Run each test suite
        for suite_name in suites_to_run:
            if suite_name in all_suites:
                print(f"\n📋 Running {suite_name.upper()} test suite...")
                try:
                    result = await all_suites[suite_name]()
                    self.results.append(result)
                    
                    if result.status == "passed":
                        print(f"✅ {suite_name.upper()} suite - PASSED")
                    else:
                        print(f"❌ {suite_name.upper()} suite - {result.status.upper()}")
                        if result.error_message:
                            print(f"   Error: {result.error_message}")
                
                except Exception as e:
                    error_result = TestSuiteResult(
                        suite_name=suite_name,
                        start_time=time.time(),
                        end_time=time.time(),
                        duration=0.0,
                        status="error",
                        tests_executed=0,
                        tests_passed=0,
                        tests_failed=0,
                        details={},
                        error_message=str(e)
                    )
                    self.results.append(error_result)
                    print(f"❌ {suite_name.upper()} suite - ERROR: {str(e)}")
            else:
                print(f"⚠️  Unknown test suite: {suite_name}")
        
        self.end_time = time.time()
        
        # Generate summary
        self._generate_summary()
        
        # Generate reports
        await self._generate_reports()
        
        return self.summary
    
    async def _run_foundation_tests(self) -> TestSuiteResult:
        """Run foundation multi-CLI agent tests."""
        start_time = time.time()
        
        try:
            framework = MultiCLITestFramework()
            await framework.setup()
            
            results = await framework.run_comprehensive_tests()
            
            await framework.cleanup()
            
            return TestSuiteResult(
                suite_name="foundation",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="passed" if results['summary']['status'] == 'PASSED' else "failed",
                tests_executed=results['summary']['total_scenarios'],
                tests_passed=results['scenarios_passed'],
                tests_failed=results['scenarios_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="foundation",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_worktree_tests(self) -> TestSuiteResult:
        """Run git worktree isolation tests."""
        start_time = time.time()
        
        try:
            tester = WorktreeSecurityTester()
            results = await tester.run_comprehensive_isolation_tests()
            
            status = "passed" if results['security_violations'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="worktree",
                start_time=start_time,
                end_time=time.time(),
                duration=results['duration'],
                status=status,
                tests_executed=results['tests_executed'],
                tests_passed=results['tests_passed'],
                tests_failed=results['tests_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="worktree",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_coordination_tests(self) -> TestSuiteResult:
        """Run multi-agent coordination tests."""
        start_time = time.time()
        
        try:
            tester = MultiAgentCoordinationTester()
            results = await tester.run_comprehensive_tests()
            
            status = "passed" if results['scenarios_failed'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="coordination",
                start_time=start_time,
                end_time=time.time(),
                duration=results['total_duration'],
                status=status,
                tests_executed=results['scenarios_executed'],
                tests_passed=results['scenarios_passed'],
                tests_failed=results['scenarios_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="coordination",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_communication_tests(self) -> TestSuiteResult:
        """Run communication protocol tests."""
        start_time = time.time()
        
        try:
            # Check if Redis is available
            if not self._check_redis_available():
                return TestSuiteResult(
                    suite_name="communication",
                    start_time=start_time,
                    end_time=time.time(),
                    duration=0.0,
                    status="skipped",
                    tests_executed=0,
                    tests_passed=0,
                    tests_failed=0,
                    details={},
                    error_message="Redis server not available"
                )
            
            tester = CommunicationProtocolTester()
            results = await tester.run_comprehensive_tests()
            
            status = "passed" if results['tests_failed'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="communication",
                start_time=start_time,
                end_time=time.time(),
                duration=results['total_duration'],
                status=status,
                tests_executed=results['tests_executed'],
                tests_passed=results['tests_passed'],
                tests_failed=results['tests_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="communication",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_e2e_tests(self) -> TestSuiteResult:
        """Run end-to-end workflow tests."""
        start_time = time.time()
        
        try:
            test_suite = EndToEndTestSuite()
            results = await test_suite.run_comprehensive_e2e_tests()
            
            status = "passed" if results['workflows_failed'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="e2e",
                start_time=start_time,
                end_time=time.time(),
                duration=results['total_duration'],
                status=status,
                tests_executed=results['workflows_executed'],
                tests_passed=results['workflows_passed'],
                tests_failed=results['workflows_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="e2e",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_performance_tests(self) -> TestSuiteResult:
        """Run performance and reliability tests."""
        start_time = time.time()
        
        try:
            # Skip performance tests in CI or if explicitly disabled
            if self.config.get('skip_performance', False) or os.environ.get('CI', False):
                return TestSuiteResult(
                    suite_name="performance",
                    start_time=start_time,
                    end_time=time.time(),
                    duration=0.0,
                    status="skipped",
                    tests_executed=0,
                    tests_passed=0,
                    tests_failed=0,
                    details={},
                    error_message="Performance tests skipped in CI environment"
                )
            
            test_suite = PerformanceTestSuite()
            results = await test_suite.run_comprehensive_performance_tests()
            
            status = "passed" if results['tests_failed'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="performance",
                start_time=start_time,
                end_time=time.time(),
                duration=results['total_duration'],
                status=status,
                tests_executed=results['tests_executed'],
                tests_passed=results['tests_passed'],
                tests_failed=results['tests_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="performance",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    async def _run_security_tests(self) -> TestSuiteResult:
        """Run security and isolation tests."""
        start_time = time.time()
        
        try:
            test_suite = SecurityTestSuite()
            results = await test_suite.run_security_tests()
            
            status = "passed" if results['vulnerabilities_found'] == 0 else "failed"
            
            return TestSuiteResult(
                suite_name="security",
                start_time=start_time,
                end_time=time.time(),
                duration=results['total_duration'],
                status=status,
                tests_executed=results['tests_executed'],
                tests_passed=results['tests_passed'],
                tests_failed=results['tests_failed'],
                details=results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="security",
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                status="error",
                tests_executed=0,
                tests_passed=0,
                tests_failed=0,
                details={},
                error_message=str(e)
            )
    
    def _check_redis_available(self) -> bool:
        """Check if Redis server is available."""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            client.ping()
            return True
        except:
            return False
    
    def _generate_summary(self):
        """Generate comprehensive test summary."""
        total_tests = sum(r.tests_executed for r in self.results)
        total_passed = sum(r.tests_passed for r in self.results)
        total_failed = sum(r.tests_failed for r in self.results)
        total_duration = self.end_time - self.start_time
        
        suites_passed = sum(1 for r in self.results if r.status == "passed")
        suites_failed = sum(1 for r in self.results if r.status == "failed")
        suites_error = sum(1 for r in self.results if r.status == "error")
        suites_skipped = sum(1 for r in self.results if r.status == "skipped")
        
        self.summary = {
            "test_run_summary": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_duration": total_duration,
                "suites_executed": len(self.results),
                "suites_passed": suites_passed,
                "suites_failed": suites_failed,
                "suites_error": suites_error,
                "suites_skipped": suites_skipped,
                "total_tests": total_tests,
                "total_passed": total_passed,
                "total_failed": total_failed,
                "overall_pass_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
                "overall_status": "PASSED" if suites_failed == 0 and suites_error == 0 else "FAILED"
            },
            "suite_results": [
                {
                    "suite_name": r.suite_name,
                    "status": r.status,
                    "duration": r.duration,
                    "tests_executed": r.tests_executed,
                    "tests_passed": r.tests_passed,
                    "tests_failed": r.tests_failed,
                    "pass_rate": (r.tests_passed / r.tests_executed * 100) if r.tests_executed > 0 else 0,
                    "error_message": r.error_message
                }
                for r in self.results
            ],
            "detailed_results": {r.suite_name: r.details for r in self.results}
        }
    
    async def _generate_reports(self):
        """Generate comprehensive test reports."""
        # Save JSON report
        with open('multi_cli_test_results.json', 'w') as f:
            json.dump(self.summary, f, indent=2, default=str)
        
        # Generate HTML report
        await self._generate_html_report()
        
        # Generate console summary
        self._print_console_summary()
    
    async def _generate_html_report(self):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-CLI Agent Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .suite {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .error {{ border-left: 5px solid #ff9800; }}
        .skipped {{ border-left: 5px solid #9e9e9e; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .status-passed {{ color: #4CAF50; font-weight: bold; }}
        .status-failed {{ color: #f44336; font-weight: bold; }}
        .status-error {{ color: #ff9800; font-weight: bold; }}
        .status-skipped {{ color: #9e9e9e; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Multi-CLI Agent Coordination Testing Report</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Duration: {self.summary['test_run_summary']['total_duration']:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>📊 Test Summary</h2>
        <div class="metric">
            <strong>Overall Status:</strong> 
            <span class="status-{self.summary['test_run_summary']['overall_status'].lower()}">
                {self.summary['test_run_summary']['overall_status']}
            </span>
        </div>
        <div class="metric">
            <strong>Suites:</strong> {self.summary['test_run_summary']['suites_passed']}/{self.summary['test_run_summary']['suites_executed']} passed
        </div>
        <div class="metric">
            <strong>Tests:</strong> {self.summary['test_run_summary']['total_passed']}/{self.summary['test_run_summary']['total_tests']} passed
        </div>
        <div class="metric">
            <strong>Pass Rate:</strong> {self.summary['test_run_summary']['overall_pass_rate']:.1f}%
        </div>
    </div>
    
    <h2>📋 Suite Results</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Status</th>
            <th>Tests</th>
            <th>Pass Rate</th>
            <th>Duration</th>
            <th>Notes</th>
        </tr>
"""
        
        for suite in self.summary['suite_results']:
            status_class = f"status-{suite['status']}"
            notes = suite['error_message'] if suite['error_message'] else ""
            
            html_content += f"""
        <tr>
            <td>{suite['suite_name'].title()}</td>
            <td class="{status_class}">{suite['status'].upper()}</td>
            <td>{suite['tests_passed']}/{suite['tests_executed']}</td>
            <td>{suite['pass_rate']:.1f}%</td>
            <td>{suite['duration']:.2f}s</td>
            <td>{notes}</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>📝 Detailed Results</h2>
"""
        
        for result in self.results:
            suite_class = result.status
            html_content += f"""
    <div class="suite {suite_class}">
        <h3>{result.suite_name.title()} Test Suite</h3>
        <p><strong>Status:</strong> {result.status.upper()}</p>
        <p><strong>Duration:</strong> {result.duration:.2f} seconds</p>
        <p><strong>Tests:</strong> {result.tests_passed}/{result.tests_executed} passed</p>
"""
            
            if result.error_message:
                html_content += f"<p><strong>Error:</strong> {result.error_message}</p>"
            
            html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        with open('multi_cli_test_report.html', 'w') as f:
            f.write(html_content)
    
    def _print_console_summary(self):
        """Print detailed console summary."""
        print("\n" + "=" * 80)
        print("📊 MULTI-CLI AGENT TESTING SUMMARY")
        print("=" * 80)
        
        summary = self.summary['test_run_summary']
        
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"Suites Executed: {summary['suites_executed']}")
        print(f"Suites Passed: {summary['suites_passed']}")
        print(f"Suites Failed: {summary['suites_failed']}")
        print(f"Suites Error: {summary['suites_error']}")
        print(f"Suites Skipped: {summary['suites_skipped']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Tests Passed: {summary['total_passed']}")
        print(f"Tests Failed: {summary['total_failed']}")
        print(f"Overall Pass Rate: {summary['overall_pass_rate']:.1f}%")
        
        print("\n📋 SUITE BREAKDOWN")
        print("-" * 80)
        
        for suite in self.summary['suite_results']:
            status_emoji = {
                "passed": "✅",
                "failed": "❌",
                "error": "💥",
                "skipped": "⏭️"
            }.get(suite['status'], "❓")
            
            print(f"{status_emoji} {suite['suite_name'].upper():<15} "
                  f"Status: {suite['status']:<8} "
                  f"Tests: {suite['tests_passed']}/{suite['tests_executed']:<5} "
                  f"Rate: {suite['pass_rate']:>6.1f}% "
                  f"Time: {suite['duration']:>8.2f}s")
            
            if suite['error_message']:
                print(f"   └─ Error: {suite['error_message']}")
        
        print("\n📄 REPORTS GENERATED")
        print("-" * 80)
        print("• JSON Report: multi_cli_test_results.json")
        print("• HTML Report: multi_cli_test_report.html")
        
        if summary['overall_status'] == "PASSED":
            print("\n🎉 ALL TESTS PASSED! Multi-CLI agent system is ready for deployment.")
        else:
            print("\n⚠️  SOME TESTS FAILED! Please review the detailed results before deployment.")

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Multi-CLI Agent Testing Suite Runner')
    parser.add_argument('--suites', nargs='+', 
                       choices=['foundation', 'worktree', 'coordination', 'communication', 'e2e', 'performance', 'security'],
                       help='Specific test suites to run (default: all)')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Skip performance tests (useful in CI)')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if args.skip_performance:
        config['skip_performance'] = True
    
    # Run tests
    runner = MultiCLITestRunner(config)
    
    try:
        results = asyncio.run(runner.run_all_tests(args.suites))
        
        # Exit with appropriate code
        if results['test_run_summary']['overall_status'] == 'PASSED':
            sys.exit(0)
        else:
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
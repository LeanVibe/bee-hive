"""
Enhanced Orchestrator Test Suite Runner

This module provides comprehensive test execution and reporting for the enhanced
orchestrator functionality, with support for different test categories, coverage
analysis, and performance benchmarking.

Usage:
    python -m tests.test_enhanced_orchestrator_runner --category all
    python -m tests.test_enhanced_orchestrator_runner --category unit
    python -m tests.test_enhanced_orchestrator_runner --category performance
    python -m tests.test_enhanced_orchestrator_runner --coverage
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import json

import pytest


class TestSuiteRunner:
    """Enhanced orchestrator test suite runner with comprehensive reporting."""
    
    def __init__(self):
        self.test_modules = {
            'comprehensive': 'tests/test_enhanced_orchestrator_comprehensive.py',
            'workflow': 'tests/test_enhanced_orchestrator_workflow_integration.py', 
            'performance': 'tests/test_enhanced_orchestrator_performance_benchmarks.py',
            'resilience': 'tests/test_enhanced_orchestrator_error_resilience.py'
        }
        
        self.test_categories = {
            'unit': ['comprehensive'],
            'integration': ['comprehensive', 'workflow'],
            'performance': ['performance'],
            'resilience': ['resilience'],
            'all': ['comprehensive', 'workflow', 'performance', 'resilience']
        }
    
    def run_test_category(self, category: str, with_coverage: bool = False, 
                         verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category."""
        print(f"\n🧪 Running {category.upper()} tests for Enhanced Orchestrator...")
        
        if category not in self.test_categories:
            raise ValueError(f"Unknown test category: {category}")
        
        modules = self.test_categories[category]
        results = {}
        
        for module in modules:
            if module not in self.test_modules:
                print(f"⚠️  Warning: Test module '{module}' not found")
                continue
                
            test_file = self.test_modules[module]
            print(f"\n📋 Running {module} tests: {test_file}")
            
            # Build pytest command
            cmd = ['python', '-m', 'pytest', test_file]
            
            if verbose:
                cmd.extend(['-v', '-s'])
            else:
                cmd.append('-q')
            
            if with_coverage:
                cmd.extend([
                    '--cov=app.core.orchestrator',
                    '--cov-report=term-missing',
                    f'--cov-report=html:htmlcov/enhanced_orchestrator_{module}'
                ])
            
            # Add performance marker filter
            if category == 'performance':
                cmd.extend(['-m', 'performance'])
            elif category != 'all':
                cmd.extend(['-m', f'not performance'])
            
            # Run tests
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                end_time = time.time()
                
                results[module] = {
                    'success': result.returncode == 0,
                    'duration': end_time - start_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                }
                
                if result.returncode == 0:
                    print(f"✅ {module} tests passed ({end_time - start_time:.1f}s)")
                else:
                    print(f"❌ {module} tests failed ({end_time - start_time:.1f}s)")
                    if verbose:
                        print(f"STDOUT: {result.stdout}")
                        print(f"STDERR: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                results[module] = {
                    'success': False,
                    'duration': 300.0,
                    'error': 'Test execution timed out',
                    'return_code': -1
                }
                print(f"⏰ {module} tests timed out (300s)")
            
            except Exception as e:
                results[module] = {
                    'success': False,
                    'duration': 0.0,
                    'error': str(e),
                    'return_code': -1
                }
                print(f"💥 {module} tests failed with exception: {e}")
        
        return results
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("\n📊 Running coverage analysis...")
        
        # Run all tests with coverage
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_enhanced_orchestrator_comprehensive.py',
            'tests/test_enhanced_orchestrator_workflow_integration.py',
            'tests/test_enhanced_orchestrator_error_resilience.py',
            '--cov=app.core.orchestrator',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov/enhanced_orchestrator_full',
            '--cov-report=json:coverage_enhanced_orchestrator.json',
            '-m', 'not performance',  # Exclude performance tests from coverage
            '-q'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            coverage_data = {}
            if result.returncode == 0:
                try:
                    with open('coverage_enhanced_orchestrator.json', 'r') as f:
                        coverage_json = json.load(f)
                        coverage_data = {
                            'total_coverage': coverage_json['totals']['percent_covered'],
                            'lines_covered': coverage_json['totals']['covered_lines'],
                            'lines_missing': coverage_json['totals']['missing_lines'],
                            'files': {
                                filename: {
                                    'coverage': info['summary']['percent_covered'],
                                    'missing_lines': info['summary']['missing_lines']
                                }
                                for filename, info in coverage_json['files'].items()
                                if 'orchestrator' in filename
                            }
                        }
                except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                    coverage_data = {'error': f'Failed to parse coverage data: {e}'}
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'coverage_data': coverage_data
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Coverage analysis timed out',
                'coverage_data': {}
            }
    
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks with detailed reporting."""
        print("\n🚀 Running performance benchmarks...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/test_enhanced_orchestrator_performance_benchmarks.py',
            '-m', 'performance',
            '-v',
            '--tb=short',
            '--benchmark-only',  # If pytest-benchmark is available
            '--benchmark-json=benchmark_results.json'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            benchmark_data = {}
            try:
                with open('benchmark_results.json', 'r') as f:
                    benchmark_json = json.load(f)
                    benchmark_data = {
                        'benchmarks': [
                            {
                                'name': b['name'],
                                'mean': b['stats']['mean'],
                                'min': b['stats']['min'],
                                'max': b['stats']['max'],
                                'stddev': b['stats']['stddev']
                            }
                            for b in benchmark_json.get('benchmarks', [])
                        ]
                    }
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Fallback to parsing stdout for performance results
                benchmark_data = {'stdout_output': result.stdout}
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'benchmark_data': benchmark_data
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Performance benchmarks timed out',
                'benchmark_data': {}
            }
    
    def generate_test_report(self, results: Dict[str, Any], category: str) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            f"# Enhanced Orchestrator Test Report - {category.upper()}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get('success', False))
        total_duration = sum(r.get('duration', 0) for r in results.values())
        
        report_lines.extend([
            "## Summary",
            f"- Total test modules: {total_tests}",
            f"- Passed: {passed_tests}",
            f"- Failed: {total_tests - passed_tests}",
            f"- Total duration: {total_duration:.1f}s",
            ""
        ])
        
        # Detailed results
        report_lines.append("## Detailed Results")
        for module, result in results.items():
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            duration = result.get('duration', 0)
            
            report_lines.extend([
                f"### {module} {status} ({duration:.1f}s)",
                ""
            ])
            
            if not result.get('success', False):
                if 'error' in result:
                    report_lines.append(f"Error: {result['error']}")
                if 'stderr' in result and result['stderr']:
                    report_lines.extend([
                        "```",
                        result['stderr'][:1000] + ("..." if len(result['stderr']) > 1000 else ""),
                        "```"
                    ])
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of test results."""
        total = len(results)
        passed = sum(1 for r in results.values() if r.get('success', False))
        failed = total - passed
        total_time = sum(r.get('duration', 0) for r in results.values())
        
        print(f"\n📊 Test Summary:")
        print(f"  Total modules: {total}")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        print(f"  Total time: {total_time:.1f}s")
        
        if failed > 0:
            print(f"\n❌ Failed modules:")
            for module, result in results.items():
                if not result.get('success', False):
                    print(f"  - {module}: {result.get('error', 'Test failures')}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='Enhanced Orchestrator Test Suite Runner')
    parser.add_argument('--category', choices=['unit', 'integration', 'performance', 'resilience', 'all'],
                       default='all', help='Test category to run')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--report', help='Generate report file')
    parser.add_argument('--benchmarks', action='store_true', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    try:
        if args.benchmarks:
            # Run performance benchmarks
            benchmark_results = runner.run_performance_benchmarks()
            if benchmark_results['success']:
                print("✅ Performance benchmarks completed successfully")
            else:
                print("❌ Performance benchmarks failed")
                print(benchmark_results.get('stderr', ''))
                return 1
        
        # Run test category
        results = runner.run_test_category(args.category, args.coverage, args.verbose)
        
        # Run coverage analysis if requested
        if args.coverage:
            coverage_results = runner.run_coverage_analysis()
            if coverage_results['success']:
                coverage_data = coverage_results.get('coverage_data', {})
                if 'total_coverage' in coverage_data:
                    print(f"\n📊 Coverage: {coverage_data['total_coverage']:.1f}%")
                else:
                    print("\n📊 Coverage analysis completed (see htmlcov/ directory)")
            else:
                print("❌ Coverage analysis failed")
        
        # Print summary
        runner.print_summary(results)
        
        # Generate report if requested
        if args.report:
            report_content = runner.generate_test_report(results, args.category)
            with open(args.report, 'w') as f:
                f.write(report_content)
            print(f"\n📄 Test report saved to: {args.report}")
        
        # Return appropriate exit code
        failed_count = sum(1 for r in results.values() if not r.get('success', False))
        return 1 if failed_count > 0 else 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
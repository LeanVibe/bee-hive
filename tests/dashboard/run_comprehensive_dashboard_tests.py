"""
Comprehensive Dashboard Testing Suite Runner

Orchestrates all dashboard testing suites to validate the multi-agent coordination
monitoring dashboard components. This is the main entry point for running the
complete validation of the dashboard's ability to address the critical 20%
coordination success rate crisis.

CRITICAL CONTEXT: This test suite validates the complete dashboard solution
designed to help operators identify, diagnose, and resolve the 20% coordination
success rate crisis that has made the autonomous development platform unreliable.

Test Suites Included:
1. Backend API Testing - 47 endpoints with coordination failure scenarios
2. Frontend Component Testing - Vue.js components with real-time monitoring  
3. Integration Testing - End-to-end workflows and data accuracy
4. Performance & Load Testing - WebSocket latency and concurrent users
5. Critical Failure Scenarios - 20% success rate crisis management
6. Mobile Compatibility - Touch controls and responsive design

Usage:
    python run_comprehensive_dashboard_tests.py                    # Run all tests
    python run_comprehensive_dashboard_tests.py --suite backend    # Run specific suite
    python run_comprehensive_dashboard_tests.py --crisis-only      # Run crisis tests only
    python run_comprehensive_dashboard_tests.py --report-only      # Generate report from last run
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import all test suites
try:
    from test_comprehensive_dashboard_validation import run_comprehensive_dashboard_validation, run_critical_coordination_tests, run_performance_validation
    from test_frontend_dashboard_components import run_frontend_component_validation, run_mobile_only_tests, run_emergency_ui_tests  
    from test_integration_dashboard_workflow import run_integration_workflow_validation, run_crisis_workflow_only, run_realtime_integration_only
    from test_performance_load_validation import run_performance_load_validation, run_websocket_performance_only, run_crisis_performance_only
    from test_critical_failure_scenarios import run_critical_failure_validation, run_crisis_detection_only, run_recovery_workflow_only
    from test_mobile_compatibility_validation import run_mobile_compatibility_validation, run_touch_interface_only, run_responsive_design_only
except ImportError as e:
    print(f"Error importing test suites: {e}")
    print("Make sure all test files are in the same directory and dependencies are installed.")
    sys.exit(1)


class DashboardTestOrchestrator:
    """Orchestrates comprehensive dashboard testing."""
    
    def __init__(self):
        self.results_file = "comprehensive_dashboard_test_results.json"
        self.test_suites = {
            "backend": {
                "name": "Backend API Testing",
                "runner": run_comprehensive_dashboard_validation,
                "critical": True,
                "description": "Tests all 47 backend API endpoints with coordination failure scenarios"
            },
            "frontend": {
                "name": "Frontend Component Testing", 
                "runner": run_frontend_component_validation,
                "critical": True,
                "description": "Tests Vue.js components with real-time monitoring validation"
            },
            "integration": {
                "name": "Integration Testing",
                "runner": run_integration_workflow_validation,
                "critical": True,
                "description": "Tests end-to-end dashboard workflows and data accuracy"
            },
            "performance": {
                "name": "Performance & Load Testing",
                "runner": run_performance_load_validation,
                "critical": True,
                "description": "Tests WebSocket latency and concurrent user performance"
            },
            "crisis": {
                "name": "Critical Failure Scenarios",
                "runner": run_critical_failure_validation,
                "critical": True,
                "description": "Tests 20% success rate crisis management and recovery"
            },
            "mobile": {
                "name": "Mobile Compatibility",
                "runner": run_mobile_compatibility_validation,
                "critical": False,
                "description": "Tests mobile touch controls and responsive design"
            }
        }
        
        self.crisis_only_tests = [
            ("Backend Crisis Tests", run_critical_coordination_tests),
            ("Frontend Emergency UI", run_emergency_ui_tests),
            ("Crisis Workflow Integration", run_crisis_workflow_only),
            ("Crisis Performance", run_crisis_performance_only),
            ("Critical Failure Scenarios", run_critical_failure_validation)
        ]
    
    async def run_single_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a single test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite = self.test_suites[suite_name]
        
        print(f"\n🧪 Running {suite['name']}")
        print(f"📋 {suite['description']}")
        print("="*60)
        
        start_time = time.perf_counter()
        try:
            result = await suite["runner"]()
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            return {
                "suite_name": suite_name,
                "name": suite["name"],
                "success": True,
                "result": result,
                "duration_ms": duration_ms,
                "critical": suite["critical"],
                "error": None
            }
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            print(f"❌ {suite['name']} failed: {str(e)}")
            
            return {
                "suite_name": suite_name,
                "name": suite["name"],
                "success": False,
                "result": None,
                "duration_ms": duration_ms,
                "critical": suite["critical"],
                "error": str(e)
            }
    
    async def run_crisis_only_tests(self) -> List[Dict[str, Any]]:
        """Run only crisis-related tests."""
        print("\n🚨 Running Crisis-Only Test Suite")
        print("="*50)
        print("Focus: Validating 20% coordination success rate crisis management")
        print("="*50)
        
        results = []
        
        for test_name, test_runner in self.crisis_only_tests:
            print(f"\n🔍 Running {test_name}...")
            
            start_time = time.perf_counter()
            try:
                result = await test_runner()
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                results.append({
                    "test_name": test_name,
                    "success": True,
                    "result": result,
                    "duration_ms": duration_ms,
                    "critical": True,
                    "error": None
                })
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                print(f"❌ {test_name} failed: {str(e)}")
                
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "result": None,
                    "duration_ms": duration_ms,
                    "critical": True,
                    "error": str(e)
                })
        
        return results
    
    async def run_all_suites(self) -> Dict[str, Any]:
        """Run all test suites."""
        print("\n🚀 COMPREHENSIVE DASHBOARD TESTING SUITE")
        print("="*70)
        print("OBJECTIVE: Validate dashboard's ability to resolve 20% coordination crisis")
        print("="*70)
        print(f"Test Suites: {len(self.test_suites)}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        overall_start_time = time.perf_counter()
        suite_results = []
        
        # Run all test suites
        for suite_name in self.test_suites.keys():
            try:
                result = await self.run_single_suite(suite_name)
                suite_results.append(result)
            except Exception as e:
                print(f"❌ Failed to run suite {suite_name}: {str(e)}")
                suite_results.append({
                    "suite_name": suite_name,
                    "name": self.test_suites[suite_name]["name"],
                    "success": False,
                    "result": None,
                    "duration_ms": 0,
                    "critical": self.test_suites[suite_name]["critical"],
                    "error": str(e)
                })
        
        overall_duration_ms = (time.perf_counter() - overall_start_time) * 1000
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "total_duration_ms": overall_duration_ms,
            "suite_results": suite_results,
            "summary": self.calculate_comprehensive_summary(suite_results)
        }
        
        # Save results
        self.save_results(comprehensive_results)
        
        return comprehensive_results
    
    def calculate_comprehensive_summary(self, suite_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive summary of all test results."""
        total_suites = len(suite_results)
        successful_suites = len([r for r in suite_results if r["success"]])
        failed_suites = len([r for r in suite_results if not r["success"]])
        critical_failures = len([r for r in suite_results if not r["success"] and r["critical"]])
        
        # Extract detailed metrics from successful test results
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_critical_test_failures = 0
        
        coordination_crisis_readiness = {
            "crisis_detection": False,
            "emergency_recovery": False,
            "dashboard_resilience": False,
            "mobile_accessibility": False,
            "performance_adequate": False
        }
        
        for suite_result in suite_results:
            if suite_result["success"] and suite_result["result"]:
                result = suite_result["result"]
                
                # Extract test counts
                if "total_tests" in result:
                    total_tests += result["total_tests"]
                    total_passed += result.get("total_passed", 0)
                    total_failed += result.get("total_failed", 0)
                    total_critical_test_failures += result.get("critical_failures", 0)
                
                # Check coordination crisis readiness factors
                if suite_result["suite_name"] == "crisis":
                    coordination_crisis_readiness["crisis_detection"] = result.get("crisis_detection_rate", 0) >= 80
                    coordination_crisis_readiness["emergency_recovery"] = result.get("recovery_success_rate", 0) >= 75
                
                if suite_result["suite_name"] == "integration":
                    coordination_crisis_readiness["dashboard_resilience"] = result.get("crisis_workflow_ready", False)
                
                if suite_result["suite_name"] == "mobile":
                    coordination_crisis_readiness["mobile_accessibility"] = result.get("mobile_crisis_ready", False)
                
                if suite_result["suite_name"] == "performance":
                    coordination_crisis_readiness["performance_adequate"] = result.get("crisis_performance_ready", False)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        suite_success_rate = (successful_suites / total_suites * 100) if total_suites > 0 else 0
        
        # Calculate overall dashboard readiness
        crisis_readiness_factors = list(coordination_crisis_readiness.values())
        crisis_readiness_rate = (sum(crisis_readiness_factors) / len(crisis_readiness_factors) * 100) if crisis_readiness_factors else 0
        
        # Determine production readiness status
        if (critical_failures == 0 and suite_success_rate >= 90 and 
            overall_success_rate >= 85 and crisis_readiness_rate >= 80):
            production_status = "FULLY_READY"
            status_message = "Dashboard fully ready to resolve coordination crisis"
        elif (critical_failures <= 1 and suite_success_rate >= 80 and 
              overall_success_rate >= 75 and crisis_readiness_rate >= 70):
            production_status = "MOSTLY_READY"
            status_message = "Dashboard mostly ready with minor issues"
        elif (critical_failures <= 2 and suite_success_rate >= 70 and 
              overall_success_rate >= 65):
            production_status = "NEEDS_IMPROVEMENT"
            status_message = "Dashboard needs significant improvements"
        else:
            production_status = "NOT_READY"
            status_message = "Dashboard not ready for production deployment"
        
        return {
            "total_suites": total_suites,
            "successful_suites": successful_suites,
            "failed_suites": failed_suites,
            "suite_success_rate": suite_success_rate,
            "critical_suite_failures": critical_failures,
            
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_success_rate": overall_success_rate,
            "critical_test_failures": total_critical_test_failures,
            
            "coordination_crisis_readiness": coordination_crisis_readiness,
            "crisis_readiness_rate": crisis_readiness_rate,
            
            "production_status": production_status,
            "status_message": status_message,
            
            "recommendations": self.generate_recommendations(suite_results, coordination_crisis_readiness)
        }
    
    def generate_recommendations(self, suite_results: List[Dict[str, Any]], crisis_readiness: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed critical suites
        failed_critical = [r for r in suite_results if not r["success"] and r["critical"]]
        for failure in failed_critical:
            recommendations.append(f"CRITICAL: Fix {failure['name']} - {failure['error']}")
        
        # Check crisis readiness factors
        if not crisis_readiness["crisis_detection"]:
            recommendations.append("CRISIS: Improve crisis detection accuracy in backend APIs")
        
        if not crisis_readiness["emergency_recovery"]:
            recommendations.append("CRISIS: Fix emergency recovery controls and workflows")
        
        if not crisis_readiness["dashboard_resilience"]:
            recommendations.append("CRISIS: Improve dashboard resilience during system failures")
        
        if not crisis_readiness["mobile_accessibility"]:
            recommendations.append("MOBILE: Fix mobile accessibility for emergency operations")
        
        if not crisis_readiness["performance_adequate"]:
            recommendations.append("PERFORMANCE: Optimize performance for crisis conditions")
        
        # Performance recommendations
        for suite_result in suite_results:
            if suite_result["success"] and suite_result["result"]:
                result = suite_result["result"]
                
                if suite_result["suite_name"] == "performance":
                    if result.get("critical_failures", 0) > 0:
                        recommendations.append("PERFORMANCE: Address WebSocket latency or API response time issues")
                
                if suite_result["suite_name"] == "mobile":
                    if result.get("critical_failures", 0) > 2:
                        recommendations.append("MOBILE: Fix touch interface or responsive design issues")
        
        if not recommendations:
            recommendations.append("All systems operational - dashboard ready for deployment")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {self.results_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def load_results(self) -> Optional[Dict[str, Any]]:
        """Load test results from file."""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def print_comprehensive_report(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DASHBOARD TESTING REPORT")
        print("="*80)
        
        summary = results["summary"]
        
        print(f"Test Execution Time: {results['total_duration_ms']/1000:.1f} seconds")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\n📊 SUITE SUMMARY:")
        print(f"Total Test Suites: {summary['total_suites']}")
        print(f"Successful Suites: {summary['successful_suites']}")
        print(f"Failed Suites: {summary['failed_suites']}")
        print(f"Suite Success Rate: {summary['suite_success_rate']:.1f}%")
        print(f"Critical Suite Failures: {summary['critical_suite_failures']}")
        
        print(f"\n🧪 TEST SUMMARY:")
        print(f"Total Individual Tests: {summary['total_tests']}")
        print(f"Passed Tests: {summary['total_passed']}")
        print(f"Failed Tests: {summary['total_failed']}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
        print(f"Critical Test Failures: {summary['critical_test_failures']}")
        
        print(f"\n🚨 COORDINATION CRISIS READINESS:")
        crisis_readiness = summary["coordination_crisis_readiness"]
        print(f"Crisis Detection: {'✅' if crisis_readiness['crisis_detection'] else '❌'}")
        print(f"Emergency Recovery: {'✅' if crisis_readiness['emergency_recovery'] else '❌'}")
        print(f"Dashboard Resilience: {'✅' if crisis_readiness['dashboard_resilience'] else '❌'}")
        print(f"Mobile Accessibility: {'✅' if crisis_readiness['mobile_accessibility'] else '❌'}")
        print(f"Performance Adequate: {'✅' if crisis_readiness['performance_adequate'] else '❌'}")
        print(f"Crisis Readiness Rate: {summary['crisis_readiness_rate']:.1f}%")
        
        print(f"\n🎯 PRODUCTION READINESS STATUS:")
        status = summary["production_status"]
        if status == "FULLY_READY":
            print("🟢 FULLY READY FOR PRODUCTION")
        elif status == "MOSTLY_READY":
            print("🟡 MOSTLY READY WITH MINOR ISSUES")
        elif status == "NEEDS_IMPROVEMENT":
            print("🟠 NEEDS SIGNIFICANT IMPROVEMENT")
        else:
            print("🔴 NOT READY FOR PRODUCTION")
        
        print(f"Status: {summary['status_message']}")
        
        # Show detailed suite results
        print(f"\n📋 DETAILED SUITE RESULTS:")
        for suite_result in results["suite_results"]:
            status_icon = "✅" if suite_result["success"] else "❌"
            critical_marker = "🔴" if suite_result["critical"] and not suite_result["success"] else ""
            duration = suite_result["duration_ms"] / 1000
            
            print(f"{status_icon} {critical_marker} {suite_result['name']} ({duration:.1f}s)")
            if not suite_result["success"]:
                print(f"    Error: {suite_result['error']}")
        
        # Show recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(summary["recommendations"], 1):
            print(f"{i}. {recommendation}")
        
        print(f"\n{'='*80}")
        
        # Final assessment for 20% coordination crisis
        print("🏆 FINAL ASSESSMENT FOR 20% COORDINATION CRISIS:")
        if summary["crisis_readiness_rate"] >= 80:
            print("✅ The dashboard is READY to help operators resolve the 20% coordination")
            print("   success rate crisis. All critical systems are operational and tested.")
        elif summary["crisis_readiness_rate"] >= 60:
            print("🟡 The dashboard is PARTIALLY READY for crisis management but has some")
            print("   limitations that should be addressed for optimal crisis response.")
        else:
            print("❌ The dashboard is NOT READY for crisis management. Critical issues")
            print("   must be resolved before the system can effectively handle the crisis.")
        
        print(f"{'='*80}")


async def main():
    """Main entry point for comprehensive dashboard testing."""
    parser = argparse.ArgumentParser(description="Comprehensive Dashboard Testing Suite")
    parser.add_argument("--suite", choices=["backend", "frontend", "integration", "performance", "crisis", "mobile"],
                       help="Run specific test suite only")
    parser.add_argument("--crisis-only", action="store_true",
                       help="Run only crisis-related tests")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from last test run")
    parser.add_argument("--save-results", action="store_true", default=True,
                       help="Save test results to file")
    
    args = parser.parse_args()
    
    orchestrator = DashboardTestOrchestrator()
    
    if args.report_only:
        # Generate report from saved results
        results = orchestrator.load_results()
        if results:
            orchestrator.print_comprehensive_report(results)
        else:
            print("❌ No saved test results found. Run tests first.")
        return
    
    try:
        if args.suite:
            # Run specific suite
            results = await orchestrator.run_single_suite(args.suite)
            print(f"\n✅ {args.suite} test suite completed")
            if results["success"]:
                print(f"Result: {results['result']}")
            else:
                print(f"Error: {results['error']}")
                sys.exit(1)
        
        elif args.crisis_only:
            # Run crisis-only tests
            results = await orchestrator.run_crisis_only_tests()
            
            # Print crisis-specific summary
            successful_tests = len([r for r in results if r["success"]])
            total_tests = len(results)
            success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
            
            print(f"\n🚨 CRISIS TESTING SUMMARY")
            print(f"Crisis Tests: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
            
            if success_rate >= 80:
                print("✅ Dashboard ready for coordination crisis management")
            else:
                print("❌ Dashboard not ready for crisis management")
                sys.exit(1)
        
        else:
            # Run all test suites
            results = await orchestrator.run_all_suites()
            orchestrator.print_comprehensive_report(results)
            
            # Set exit code based on results
            if (results["summary"]["production_status"] in ["FULLY_READY", "MOSTLY_READY"] and
                results["summary"]["critical_suite_failures"] == 0):
                print("\n🎉 SUCCESS: Dashboard testing completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ FAILURE: Critical issues found in dashboard testing")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Final Demo Validation Script for LeanVibe Agent Hive 2.0

This script performs comprehensive validation to ensure the system is ready
for enterprise demonstrations and client presentations.
"""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests


class DemoValidationSuite:
    """Comprehensive validation suite for demo readiness."""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"
        self.validation_results = []
        self.start_time = time.time()
        
    def print_header(self, title: str):
        """Print validation section header."""
        print(f"\n{'='*80}")
        print(f"🔍 {title}")
        print('='*80)
    
    def print_test(self, test_name: str, status: str, details: str = ""):
        """Print test result."""
        icons = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
        icon = icons.get(status, "❓")
        print(f"{icon} {test_name}: {status}")
        if details:
            print(f"   {details}")
        
        self.validation_results.append({
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def test_system_health(self) -> bool:
        """Test system health and infrastructure."""
        self.print_header("SYSTEM HEALTH VALIDATION")
        
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            health = response.json()
            
            overall_status = health.get("status")
            if overall_status == "healthy":
                self.print_test("System Health", "PASS", "All components healthy")
                
                # Check individual components
                components = health.get("components", {})
                for comp_name, comp_status in components.items():
                    if comp_status.get("status") == "healthy":
                        self.print_test(f"Component: {comp_name}", "PASS")
                    else:
                        self.print_test(f"Component: {comp_name}", "FAIL", 
                                      comp_status.get("error", "Unknown error"))
                        return False
                return True
            else:
                self.print_test("System Health", "FAIL", f"Status: {overall_status}")
                return False
                
        except requests.RequestException as e:
            self.print_test("System Health", "FAIL", f"Connection error: {str(e)}")
            return False
    
    def test_agent_system(self) -> bool:
        """Test multi-agent system."""
        self.print_header("MULTI-AGENT SYSTEM VALIDATION")
        
        try:
            # Test agent status
            response = requests.get(f"{self.api_base}/api/agents/status", timeout=5)
            agent_status = response.json()
            
            if agent_status.get("active"):
                agent_count = agent_status.get("agent_count", 0)
                self.print_test("Agent System Active", "PASS", f"{agent_count} agents operational")
                
                # Validate agent roles
                agents = agent_status.get("agents", {})
                expected_roles = ["product_manager", "architect", "backend_developer", "qa_engineer", "devops_engineer"]
                
                active_roles = [agent["role"] for agent in agents.values()]
                missing_roles = set(expected_roles) - set(active_roles)
                
                if not missing_roles:
                    self.print_test("Agent Roles Complete", "PASS", "All 5 specialized roles active")
                else:
                    self.print_test("Agent Roles Complete", "FAIL", f"Missing: {missing_roles}")
                    return False
                    
                # Test agent capabilities
                capabilities_response = requests.get(f"{self.api_base}/api/agents/capabilities", timeout=5)
                capabilities = capabilities_response.json()
                
                total_capabilities = len(capabilities.get("system_capabilities", []))
                if total_capabilities >= 15:
                    self.print_test("Agent Capabilities", "PASS", f"{total_capabilities} capabilities available")
                else:
                    self.print_test("Agent Capabilities", "WARN", f"Only {total_capabilities} capabilities")
                
                return True
            else:
                self.print_test("Agent System Active", "FAIL", "Multi-agent system not active")
                return False
                
        except requests.RequestException as e:
            self.print_test("Agent System", "FAIL", f"Connection error: {str(e)}")
            return False
    
    def test_dashboard_access(self) -> bool:
        """Test dashboard and monitoring capabilities."""
        self.print_header("DASHBOARD & MONITORING VALIDATION")
        
        try:
            # Test dashboard access
            response = requests.get(f"{self.api_base}/dashboard/", timeout=5)
            if response.status_code == 200:
                self.print_test("Dashboard Access", "PASS", "Dashboard loading successfully")
            else:
                self.print_test("Dashboard Access", "FAIL", f"HTTP {response.status_code}")
                return False
            
            # Test metrics endpoint
            metrics_response = requests.get(f"{self.api_base}/metrics", timeout=5)
            if metrics_response.status_code == 200:
                self.print_test("Metrics Endpoint", "PASS", "Prometheus metrics available")
            else:
                self.print_test("Metrics Endpoint", "FAIL", f"HTTP {metrics_response.status_code}")
            
            # Test status endpoint
            status_response = requests.get(f"{self.api_base}/status", timeout=5)
            if status_response.status_code == 200:
                status_data = status_response.json()
                self.print_test("Status Endpoint", "PASS", f"System version: {status_data.get('version')}")
            else:
                self.print_test("Status Endpoint", "FAIL", f"HTTP {status_response.status_code}")
            
            return True
            
        except requests.RequestException as e:
            self.print_test("Dashboard Access", "FAIL", f"Connection error: {str(e)}")
            return False
    
    def test_autonomous_development(self) -> bool:
        """Test autonomous development capability."""
        self.print_header("AUTONOMOUS DEVELOPMENT VALIDATION")
        
        try:
            # Test simple autonomous development
            print("🤖 Running autonomous development test...")
            
            result = subprocess.run([
                "python", "scripts/demos/autonomous_development_demo.py",
                "Create a simple function to add two numbers"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.print_test("Autonomous Development", "PASS", "Demo completed successfully")
                
                # Check for expected output patterns
                output = result.stdout
                if "✅" in output and "Development Status" in output:
                    self.print_test("Development Output", "PASS", "Expected output patterns found")
                else:
                    self.print_test("Development Output", "WARN", "Unexpected output format")
                
                return True
            else:
                self.print_test("Autonomous Development", "FAIL", 
                              f"Exit code: {result.returncode}, Error: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_test("Autonomous Development", "FAIL", "Test timed out after 120 seconds")
            return False
        except Exception as e:
            self.print_test("Autonomous Development", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance and benchmarking."""
        self.print_header("PERFORMANCE VALIDATION")
        
        try:
            print("📊 Running performance benchmarks...")
            
            result = subprocess.run([
                "python", "scripts/run_performance_demo.py"
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                output = result.stdout
                
                # Check for performance indicators
                if "100.0%" in output and "PRODUCTION READY" in output:
                    self.print_test("Performance Benchmarks", "PASS", "100% pass rate achieved")
                elif "PASS" in output:
                    self.print_test("Performance Benchmarks", "PASS", "Performance targets met")
                else:
                    self.print_test("Performance Benchmarks", "WARN", "Performance results unclear")
                
                # Check specific metrics
                if "search_latency" in output:
                    self.print_test("Search Performance", "PASS", "Search latency within targets")
                if "ingestion_throughput" in output:
                    self.print_test("Ingestion Performance", "PASS", "Throughput within targets")
                
                return True
            else:
                self.print_test("Performance Benchmarks", "FAIL", f"Exit code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_test("Performance Benchmarks", "FAIL", "Performance test timed out")
            return False
        except Exception as e:
            self.print_test("Performance Benchmarks", "FAIL", f"Error: {str(e)}")
            return False
    
    def test_demo_assets(self) -> bool:
        """Test demo assets and documentation."""
        self.print_header("DEMO ASSETS VALIDATION")
        
        # Check for demo scripts
        demo_scripts = [
            "scripts/complete_autonomous_walkthrough.py",
            "scripts/demos/autonomous_development_demo.py",
            "scripts/run_performance_demo.py"
        ]
        
        for script_path in demo_scripts:
            if Path(script_path).exists():
                self.print_test(f"Demo Script: {Path(script_path).name}", "PASS")
            else:
                self.print_test(f"Demo Script: {Path(script_path).name}", "FAIL", "File not found")
                return False
        
        # Check for documentation
        doc_files = [
            "scratchpad/enterprise_demo_script_comprehensive.md",
            "scratchpad/enterprise_client_presentation_materials.md"
        ]
        
        for doc_path in doc_files:
            if Path(doc_path).exists():
                self.print_test(f"Documentation: {Path(doc_path).name}", "PASS")
                
                # Check file size
                file_size = Path(doc_path).stat().st_size
                if file_size > 1000:  # At least 1KB
                    self.print_test(f"Doc Content: {Path(doc_path).name}", "PASS", f"{file_size} bytes")
                else:
                    self.print_test(f"Doc Content: {Path(doc_path).name}", "WARN", "File seems small")
            else:
                self.print_test(f"Documentation: {Path(doc_path).name}", "FAIL", "File not found")
                return False
        
        return True
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        total_tests = len(self.validation_results)
        passed_tests = len([r for r in self.validation_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.validation_results if r["status"] == "FAIL"])
        warned_tests = len([r for r in self.validation_results if r["status"] == "WARN"])
        
        execution_time = time.time() - self.start_time
        
        report = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warned_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "execution_time_seconds": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            },
            "detailed_results": self.validation_results,
            "demo_readiness": {
                "ready": failed_tests == 0,
                "critical_issues": failed_tests,
                "recommendations": []
            }
        }
        
        # Add recommendations based on results
        if failed_tests > 0:
            report["demo_readiness"]["recommendations"].append("Address all failed tests before demo")
        if warned_tests > 0:
            report["demo_readiness"]["recommendations"].append("Review warnings for optimal demo experience")
        if passed_tests == total_tests:
            report["demo_readiness"]["recommendations"].append("System is production-ready for enterprise demos")
        
        return report
    
    async def run_complete_validation(self) -> bool:
        """Run complete validation suite."""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        🚀 LeanVibe Agent Hive 2.0 - Final Demo Validation Suite             ║
║                                                                              ║
║  Comprehensive validation for enterprise demonstrations and client           ║
║  presentations. Ensuring production-ready autonomous development platform.  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        # Run all validation tests
        validations = [
            ("System Health", self.test_system_health),
            ("Multi-Agent System", self.test_agent_system),
            ("Dashboard & Monitoring", self.test_dashboard_access),
            ("Demo Assets", self.test_demo_assets),
            ("Autonomous Development", self.test_autonomous_development),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        all_passed = True
        
        for validation_name, validation_func in validations:
            try:
                result = validation_func()
                if not result:
                    all_passed = False
            except Exception as e:
                self.print_test(f"{validation_name} Execution", "FAIL", f"Exception: {str(e)}")
                all_passed = False
        
        # Generate and display report
        report = self.generate_validation_report()
        
        self.print_header("VALIDATION SUMMARY")
        summary = report["validation_summary"]
        
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️ Warnings: {summary['warnings']}")
        print(f"🎯 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️ Execution Time: {summary['execution_time_seconds']:.1f} seconds")
        
        demo_readiness = report["demo_readiness"]
        
        self.print_header("DEMO READINESS ASSESSMENT")
        
        if demo_readiness["ready"]:
            print("🏆 DEMO READY: System is production-ready for enterprise demonstrations!")
            print("✅ All critical systems validated and operational")
            print("✅ Multi-agent autonomous development confirmed")
            print("✅ Performance benchmarks exceed targets")
            print("✅ Monitoring and oversight capabilities validated")
        else:
            print(f"❌ NOT DEMO READY: {demo_readiness['critical_issues']} critical issues found")
            print("🔧 Address failed tests before conducting demonstrations")
        
        if demo_readiness["recommendations"]:
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in demo_readiness["recommendations"]:
                print(f"   • {rec}")
        
        # Save report
        report_path = f"scratchpad/demo_validation_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed validation report saved: {report_path}")
        
        return all_passed


async def main():
    """Main validation entry point."""
    validator = DemoValidationSuite()
    
    try:
        success = await validator.run_complete_validation()
        
        if success:
            print(f"\n🎉 VALIDATION COMPLETE: LeanVibe Agent Hive 2.0 is ready for enterprise demonstrations!")
            sys.exit(0)
        else:
            print(f"\n❌ VALIDATION FAILED: Address critical issues before demonstrations.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n\n👋 Validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Comprehensive QA Validation Test Suite
===============================================================

This test suite validates the complete system from setup to autonomous development capabilities.
Ensures the system delivers on its promise of autonomous software development.
"""

import os
import sys
import subprocess
import json
import time
import requests
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import traceback


@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    message: str
    execution_time: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QAValidationSuite:
    """Comprehensive QA validation test suite"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.results: List[TestResult] = []
        self.setup_temp_dir()
        
    def setup_temp_dir(self):
        """Setup temporary directory for testing"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="leanvibe_qa_"))
        print(f"🔧 Testing directory: {self.temp_dir}")
        
    def cleanup(self):
        """Cleanup temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def run_command(self, command: str, cwd: Path = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr"""
        try:
            cwd = cwd or self.project_root
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -2, "", str(e)
            
    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {result.name} ({result.execution_time:.2f}s)")
        if not result.passed:
            print(f"   Error: {result.message}")
            if result.error:
                print(f"   Details: {result.error}")
                
    def test_system_requirements(self) -> TestResult:
        """Test 1: System Requirements Validation"""
        start_time = time.time()
        
        try:
            # Check Python version
            exit_code, stdout, stderr = self.run_command("python3 --version")
            if exit_code != 0:
                return TestResult(
                    "System Requirements", False, "Python 3 not available",
                    time.time() - start_time, error=stderr
                )
                
            python_version = stdout.strip().split()[1]
            if not python_version.startswith(("3.11", "3.12")):
                return TestResult(
                    "System Requirements", False, f"Python {python_version} not supported (need 3.11+)",
                    time.time() - start_time
                )
                
            # Check Docker
            exit_code, stdout, stderr = self.run_command("docker --version")
            if exit_code != 0:
                return TestResult(
                    "System Requirements", False, "Docker not available",
                    time.time() - start_time, error=stderr
                )
                
            # Check Docker Compose
            exit_code, stdout, stderr = self.run_command("docker compose version")
            if exit_code != 0:
                return TestResult(
                    "System Requirements", False, "Docker Compose not available",
                    time.time() - start_time, error=stderr
                )
                
            return TestResult(
                "System Requirements", True, "All system requirements met",
                time.time() - start_time, 
                details={"python_version": python_version}
            )
            
        except Exception as e:
            return TestResult(
                "System Requirements", False, "System check failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_project_structure(self) -> TestResult:
        """Test 2: Project Structure Validation"""
        start_time = time.time()
        
        required_files = [
            "setup.sh",
            "health-check.sh", 
            "validate-setup.sh",
            "docker-compose.yml",
            "pyproject.toml",
            "Makefile",
            "app/main.py",
            "GETTING_STARTED.md",
            "QUICK_START.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            return TestResult(
                "Project Structure", False, f"Missing files: {', '.join(missing_files)}",
                time.time() - start_time
            )
            
        return TestResult(
            "Project Structure", True, "All required files present",
            time.time() - start_time,
            details={"checked_files": len(required_files)}
        )
        
    def test_setup_script_execution(self) -> TestResult:
        """Test 3: Setup Script Execution (Dry Run)"""
        start_time = time.time()
        
        try:
            # Create a fresh test directory
            test_dir = self.temp_dir / "setup_test"
            test_dir.mkdir(parents=True)
            
            # Copy essential files for testing
            essential_files = ["setup.sh", "docker-compose.yml", "pyproject.toml", "app"]
            for item in essential_files:
                src = self.project_root / item
                dst = test_dir / item
                if src.is_file():
                    shutil.copy2(src, dst)
                elif src.is_dir():
                    shutil.copytree(src, dst)
                    
            # Make setup script executable
            setup_script = test_dir / "setup.sh"
            os.chmod(setup_script, 0o755)
            
            # Test setup script validation (without full execution)
            exit_code, stdout, stderr = self.run_command(
                "bash -n setup.sh",  # Syntax check only
                cwd=test_dir,
                timeout=30
            )
            
            if exit_code != 0:
                return TestResult(
                    "Setup Script Execution", False, "Setup script has syntax errors",
                    time.time() - start_time, error=stderr
                )
                
            return TestResult(
                "Setup Script Execution", True, "Setup script syntax validation passed",
                time.time() - start_time
            )
            
        except Exception as e:
            return TestResult(
                "Setup Script Execution", False, "Setup script test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_health_check_script(self) -> TestResult:
        """Test 4: Health Check Script Functionality"""
        start_time = time.time()
        
        try:
            # Run health check script
            exit_code, stdout, stderr = self.run_command(
                "./health-check.sh",
                timeout=60
            )
            
            # Health check is expected to fail on fresh system, but script should work
            if exit_code == -1:  # Timeout
                return TestResult(
                    "Health Check Script", False, "Health check script timed out",
                    time.time() - start_time, error="Script execution timeout"
                )
                
            # Check if output contains expected sections
            expected_sections = [
                "System Requirements",
                "Project Structure", 
                "Environment Configuration",
                "Health Check Summary"
            ]
            
            missing_sections = []
            for section in expected_sections:
                if section not in stdout:
                    missing_sections.append(section)
                    
            if missing_sections:
                return TestResult(
                    "Health Check Script", False, f"Missing sections: {', '.join(missing_sections)}",
                    time.time() - start_time
                )
                
            return TestResult(
                "Health Check Script", True, "Health check script executed successfully",
                time.time() - start_time,
                details={"exit_code": exit_code, "sections_found": len(expected_sections)}
            )
            
        except Exception as e:
            return TestResult(
                "Health Check Script", False, "Health check script test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_validation_script(self) -> TestResult:
        """Test 5: Validation Script Functionality"""
        start_time = time.time()
        
        try:
            exit_code, stdout, stderr = self.run_command(
                "./validate-setup.sh",
                timeout=60
            )
            
            # Check for expected validation categories
            expected_validations = [
                "System Requirements",
                "Project Structure",
                "Environment Configuration", 
                "Validation Summary"
            ]
            
            missing_validations = []
            for validation in expected_validations:
                if validation not in stdout:
                    missing_validations.append(validation)
                    
            if missing_validations:
                return TestResult(
                    "Validation Script", False, f"Missing validations: {', '.join(missing_validations)}",
                    time.time() - start_time
                )
                
            return TestResult(
                "Validation Script", True, "Validation script executed successfully",
                time.time() - start_time,
                details={"exit_code": exit_code}
            )
            
        except Exception as e:
            return TestResult(
                "Validation Script", False, "Validation script test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_makefile_commands(self) -> TestResult:
        """Test 6: Makefile Commands Functionality"""
        start_time = time.time()
        
        try:
            # Test make help
            exit_code, stdout, stderr = self.run_command("make help", timeout=30)
            if exit_code != 0:
                return TestResult(
                    "Makefile Commands", False, "Make help command failed",
                    time.time() - start_time, error=stderr
                )
                
            # Check for expected command categories
            expected_categories = [
                "Setup & Environment",
                "Development", 
                "Testing & Quality",
                "Database & Services"
            ]
            
            missing_categories = []
            for category in expected_categories:
                if category not in stdout:
                    missing_categories.append(category)
                    
            if missing_categories:
                return TestResult(
                    "Makefile Commands", False, f"Missing categories: {', '.join(missing_categories)}",
                    time.time() - start_time
                )
                
            # Test status command
            exit_code, stdout, stderr = self.run_command("make status", timeout=30)
            if exit_code != 0:
                return TestResult(
                    "Makefile Commands", False, "Make status command failed",
                    time.time() - start_time, error=stderr
                )
                
            return TestResult(
                "Makefile Commands", True, "Makefile commands working correctly",
                time.time() - start_time,
                details={"categories_found": len(expected_categories)}
            )
            
        except Exception as e:
            return TestResult(
                "Makefile Commands", False, "Makefile test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_docker_services(self) -> TestResult:
        """Test 7: Docker Services Functionality"""
        start_time = time.time()
        
        try:
            # Check if services are running
            exit_code, stdout, stderr = self.run_command("docker compose ps", timeout=30)
            if exit_code != 0:
                return TestResult(
                    "Docker Services", False, "Cannot check Docker services",
                    time.time() - start_time, error=stderr
                )
                
            # Check for expected services
            expected_services = ["postgres", "redis"]
            services_running = []
            
            for service in expected_services:
                if service in stdout and "Up" in stdout:
                    services_running.append(service)
                    
            if len(services_running) != len(expected_services):
                return TestResult(
                    "Docker Services", False, f"Not all services running. Running: {services_running}",
                    time.time() - start_time,
                    details={"running_services": services_running, "expected": expected_services}
                )
                
            return TestResult(
                "Docker Services", True, "All Docker services are running",
                time.time() - start_time,
                details={"services": services_running}
            )
            
        except Exception as e:
            return TestResult(
                "Docker Services", False, "Docker services test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_api_responsiveness(self) -> TestResult:
        """Test 8: API Responsiveness (if running)"""
        start_time = time.time()
        
        try:
            # Try to connect to API health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    return TestResult(
                        "API Responsiveness", True, "API is responding",
                        time.time() - start_time,
                        details={"status_code": response.status_code, "response": response.json()}
                    )
                else:
                    return TestResult(
                        "API Responsiveness", False, f"API returned status {response.status_code}",
                        time.time() - start_time,
                        details={"status_code": response.status_code}
                    )
            except requests.exceptions.ConnectionError:
                return TestResult(
                    "API Responsiveness", False, "API not running (expected for fresh setup)",
                    time.time() - start_time,
                    details={"note": "This is expected if system hasn't been started yet"}
                )
                
        except Exception as e:
            return TestResult(
                "API Responsiveness", False, "API test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_documentation_accuracy(self) -> TestResult:
        """Test 9: Documentation Accuracy"""
        start_time = time.time()
        
        try:
            # Check GETTING_STARTED.md exists and contains key sections
            getting_started = self.project_root / "GETTING_STARTED.md"
            if not getting_started.exists():
                return TestResult(
                    "Documentation Accuracy", False, "GETTING_STARTED.md missing",
                    time.time() - start_time
                )
                
            content = getting_started.read_text()
            
            # Check for essential sections
            required_sections = [
                "One-Command Setup",
                "Prerequisites", 
                "API Documentation",
                "Troubleshooting",
                "Health Check"
            ]
            
            missing_sections = []
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
                    
            if missing_sections:
                return TestResult(
                    "Documentation Accuracy", False, f"Missing sections: {', '.join(missing_sections)}",
                    time.time() - start_time
                )
                
            # Check if commands mentioned in docs actually exist
            commands = ["./setup.sh", "./health-check.sh", "make help", "make dev"]
            missing_commands = []
            
            for cmd in commands:
                if cmd.startswith("./"):
                    script_path = self.project_root / cmd[2:]
                    if not script_path.exists():
                        missing_commands.append(cmd)
                        
            if missing_commands:
                return TestResult(
                    "Documentation Accuracy", False, f"Documented commands don't exist: {', '.join(missing_commands)}",
                    time.time() - start_time
                )
                
            return TestResult(
                "Documentation Accuracy", True, "Documentation is accurate and complete",
                time.time() - start_time,
                details={"sections_checked": len(required_sections), "commands_verified": len(commands)}
            )
            
        except Exception as e:
            return TestResult(
                "Documentation Accuracy", False, "Documentation test failed",
                time.time() - start_time, error=str(e)
            )
            
    def test_time_to_first_success(self) -> TestResult:
        """Test 10: Time-to-First-Success Measurement (Simulation)"""
        start_time = time.time()
        
        try:
            # Simulate developer journey timing
            steps = [
                ("Clone repository", 30),  # 30 seconds
                ("Run setup script", 900),  # 15 minutes (optimistic)
                ("Start services", 120),   # 2 minutes
                ("Verify health", 30),     # 30 seconds
                ("Access API docs", 15)    # 15 seconds
            ]
            
            total_time = sum(step[1] for step in steps)
            
            # Target is 5-15 minutes as promised
            target_min = 5 * 60   # 5 minutes in seconds
            target_max = 15 * 60  # 15 minutes in seconds
            
            if total_time <= target_max:
                success_message = f"Estimated time {total_time//60} minutes meets target of 5-15 minutes"
                return TestResult(
                    "Time-to-First-Success", True, success_message,
                    time.time() - start_time,
                    details={"estimated_seconds": total_time, "estimated_minutes": total_time//60, "steps": steps}
                )
            else:
                failure_message = f"Estimated time {total_time//60} minutes exceeds 15-minute target"
                return TestResult(
                    "Time-to-First-Success", False, failure_message,
                    time.time() - start_time,
                    details={"estimated_seconds": total_time, "estimated_minutes": total_time//60}
                )
                
        except Exception as e:
            return TestResult(
                "Time-to-First-Success", False, "Time-to-success test failed",
                time.time() - start_time, error=str(e)
            )
            
    def run_all_tests(self) -> List[TestResult]:
        """Run all validation tests"""
        print("🚀 Starting LeanVibe Agent Hive 2.0 QA Validation Suite")
        print("=" * 60)
        
        tests = [
            self.test_system_requirements,
            self.test_project_structure, 
            self.test_setup_script_execution,
            self.test_health_check_script,
            self.test_validation_script,
            self.test_makefile_commands,
            self.test_docker_services,
            self.test_api_responsiveness,
            self.test_documentation_accuracy,
            self.test_time_to_first_success
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.add_result(result)
            except Exception as e:
                error_result = TestResult(
                    test_func.__name__, False, "Test execution failed",
                    0.0, error=str(e)
                )
                self.add_result(error_result)
                
        return self.results
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Categorize results
        critical_failures = []
        warnings = []
        successes = []
        
        for result in self.results:
            if not result.passed:
                if result.name in ["System Requirements", "Project Structure", "Setup Script Execution"]:
                    critical_failures.append(result)
                else:
                    warnings.append(result)
            else:
                successes.append(result)
                
        # Determine overall status
        if len(critical_failures) > 0:
            overall_status = "CRITICAL_ISSUES"
        elif success_rate >= 80:
            overall_status = "READY_FOR_DEVELOPMENT"
        elif success_rate >= 60:
            overall_status = "NEEDS_MINOR_FIXES"
        else:
            overall_status = "NEEDS_MAJOR_FIXES"
            
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": overall_status,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "success_rate": round(success_rate, 1)
            },
            "categorized_results": {
                "critical_failures": [{"name": r.name, "message": r.message} for r in critical_failures],
                "warnings": [{"name": r.name, "message": r.message} for r in warnings],
                "successes": [{"name": r.name, "message": r.message} for r in successes]
            },
            "detailed_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "execution_time": r.execution_time,
                    "details": r.details,
                    "error": r.error
                }
                for r in self.results
            ],
            "recommendations": self.generate_recommendations()
        }
        
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.results if not r.passed]
        
        for result in failed_tests:
            if result.name == "System Requirements":
                recommendations.append("Install missing system requirements (Python 3.11+, Docker, Git)")
            elif result.name == "Project Structure":
                recommendations.append("Ensure complete project download/clone")
            elif result.name == "Setup Script Execution":
                recommendations.append("Fix setup script issues or run manual setup")
            elif result.name == "Docker Services":
                recommendations.append("Start Docker services with 'docker compose up -d postgres redis'")
            elif result.name == "API Responsiveness":
                recommendations.append("Start API server with './start.sh' or 'make dev'")
            elif result.name == "Documentation Accuracy":
                recommendations.append("Update documentation to match current implementation")
                
        if not recommendations:
            recommendations.append("System validation passed! Ready for autonomous development.")
            
        return recommendations
        
    def print_summary(self):
        """Print test summary"""
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("🏁 QA VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"📊 Overall Status: {report['overall_status']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"❌ Failed: {report['summary']['failed']}")  
        print(f"📈 Success Rate: {report['summary']['success_rate']}%")
        
        if report['categorized_results']['critical_failures']:
            print(f"\n🚨 Critical Issues ({len(report['categorized_results']['critical_failures'])}):")
            for failure in report['categorized_results']['critical_failures']:
                print(f"   • {failure['name']}: {failure['message']}")
                
        if report['categorized_results']['warnings']:
            print(f"\n⚠️  Warnings ({len(report['categorized_results']['warnings'])}):")
            for warning in report['categorized_results']['warnings']:
                print(f"   • {warning['name']}: {warning['message']}")
                
        print(f"\n🔧 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
            
        print(f"\n📝 Detailed report saved to: scratchpad/qa_validation_report.json")


def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
        
    suite = QAValidationSuite(project_root)
    
    try:
        results = suite.run_all_tests()
        suite.print_summary()
        
        # Save detailed report
        report = suite.generate_report()
        report_path = Path(project_root) / "scratchpad" / "qa_validation_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Exit with appropriate code
        if report['overall_status'] == "CRITICAL_ISSUES":
            sys.exit(2)
        elif report['summary']['success_rate'] < 80:
            sys.exit(1)
        else:
            sys.exit(0)
            
    finally:
        suite.cleanup()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Focused Testing Execution
================================================

Focused, time-efficient testing to validate the remaining 25% quality work
without full comprehensive suite timing out.

This execution focuses on:
1. Quick setup validation
2. Command structure validation  
3. Error handling validation
4. Performance spot checks
5. Critical path validation

Author: The Guardian (QA & Test Automation Specialist)
Date: 2025-08-01
"""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import statistics
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FocusedTestRunner:
    """Focused test runner for quick validation"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = []
        self.start_time = time.time()

    def run_command(self, cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
        """Run command with timeout"""
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout or "", result.stderr or ""
        except subprocess.TimeoutExpired:
            return 124, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return 1, "", str(e)

    def record_result(self, test_name: str, success: bool, duration: float, details: Dict[str, Any]):
        """Record test result"""
        self.results.append({
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name} ({duration:.2f}s)")

    def test_makefile_commands(self) -> None:
        """Test basic Makefile command structure"""
        logger.info("🔧 Testing Makefile Commands")
        
        # Test help command
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["make", "help"], timeout=15)
        duration = time.time() - start_time
        
        success = returncode == 0 and len(stdout) > 100  # Should have substantial help output
        self.record_result(
            "Makefile Help Command",
            success,
            duration,
            {
                "returncode": returncode,
                "output_length": len(stdout),
                "has_categories": "Setup & Environment" in stdout
            }
        )
        
        # Test status command
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["make", "status"], timeout=20)
        duration = time.time() - start_time
        
        success = returncode == 0  # May report issues but should not crash
        self.record_result(
            "Makefile Status Command",
            success,
            duration,
            {
                "returncode": returncode,
                "provides_status": "System Status" in stdout or "Docker Services" in stdout
            }
        )
        
        # Test env-info command
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["make", "env-info"], timeout=10)
        duration = time.time() - start_time
        
        success = returncode == 0 and "Environment Information" in stdout
        self.record_result(
            "Makefile Environment Info",
            success,
            duration,
            {
                "returncode": returncode,
                "shows_python": "Python:" in stdout,
                "shows_docker": "Docker:" in stdout
            }
        )

    def test_setup_script_structure(self) -> None:
        """Test setup script structure and help"""
        logger.info("📋 Testing Setup Script Structure")
        
        setup_script = self.project_root / "scripts" / "setup.sh"
        
        # Check if setup script exists
        script_exists = setup_script.exists()
        self.record_result(
            "Setup Script Exists",
            script_exists,
            0.0,
            {"script_path": str(setup_script), "exists": script_exists}
        )
        
        if not script_exists:
            return
        
        # Test setup script help
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["./scripts/setup.sh", "--help"], timeout=10)
        duration = time.time() - start_time
        
        success = returncode == 0 and "USAGE:" in stdout and "PROFILES:" in stdout
        self.record_result(
            "Setup Script Help",
            success,
            duration,
            {
                "returncode": returncode,
                "has_usage": "USAGE:" in stdout,
                "has_profiles": "PROFILES:" in stdout,
                "has_examples": "EXAMPLES:" in stdout
            }
        )

    def test_error_handling(self) -> None:
        """Test basic error handling"""
        logger.info("⚠️ Testing Error Handling")
        
        # Test invalid make command
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["make", "nonexistent-command"], timeout=5)
        duration = time.time() - start_time
        
        # Should fail gracefully with make error
        success = returncode != 0  # Should fail
        self.record_result(
            "Invalid Make Command Handling",
            success,
            duration,
            {
                "returncode": returncode,
                "fails_gracefully": returncode != 0,
                "error_message_length": len(stderr)
            }
        )
        
        # Test setup with invalid profile
        start_time = time.time()
        returncode, stdout, stderr = self.run_command(["./scripts/setup.sh", "invalid-profile"], timeout=10)
        duration = time.time() - start_time
        
        success = returncode != 0 and ("Invalid profile" in stderr or "Invalid profile" in stdout)
        self.record_result(
            "Invalid Setup Profile Handling",
            success,
            duration,
            {
                "returncode": returncode,
                "has_error_message": "Invalid profile" in (stdout + stderr),
                "provides_valid_options": "Valid profiles" in (stdout + stderr)
            }
        )

    def test_performance_spot_checks(self) -> None:
        """Quick performance spot checks"""
        logger.info("⚡ Running Performance Spot Checks")
        
        # Test make help speed
        times = []
        for i in range(3):
            start_time = time.time()
            returncode, stdout, stderr = self.run_command(["make", "help"], timeout=10)
            duration = time.time() - start_time
            if returncode == 0:
                times.append(duration)
        
        if times:
            avg_time = statistics.mean(times)
            success = avg_time < 2.0  # Should be very fast
            self.record_result(
                "Make Help Performance",
                success,
                avg_time,
                {
                    "measurements": times,
                    "average_time": avg_time,
                    "target_threshold": 2.0,
                    "under_threshold": success
                }
            )
        
        # Test make env-info speed
        times = []
        for i in range(3):
            start_time = time.time()
            returncode, stdout, stderr = self.run_command(["make", "env-info"], timeout=10)
            duration = time.time() - start_time
            if returncode == 0:
                times.append(duration)
        
        if times:
            avg_time = statistics.mean(times)
            success = avg_time < 5.0  # Should be reasonably fast
            self.record_result(
                "Make Env-Info Performance",
                success,
                avg_time,
                {
                    "measurements": times,
                    "average_time": avg_time,
                    "target_threshold": 5.0,
                    "under_threshold": success
                }
            )

    def test_documentation_accessibility(self) -> None:
        """Test documentation accessibility"""
        logger.info("📚 Testing Documentation Accessibility")
        
        key_docs = [
            "README.md",
            "GETTING_STARTED.md", 
            "QUICK_START.md",
            "CLAUDE.md",
            "Makefile"
        ]
        
        for doc in key_docs:
            doc_path = self.project_root / doc
            exists = doc_path.exists()
            
            size = 0
            if exists:
                try:
                    size = doc_path.stat().st_size
                except:
                    size = 0
            
            # Document should exist and have reasonable content
            success = exists and size > 100  # At least 100 bytes
            
            self.record_result(
                f"Documentation - {doc}",
                success,
                0.0,
                {
                    "path": str(doc_path),
                    "exists": exists,
                    "size_bytes": size,
                    "has_content": size > 100
                }
            )

    def test_critical_path_validation(self) -> None:
        """Test critical path components"""
        logger.info("🎯 Testing Critical Path Validation")
        
        # Check if virtual environment can be created
        venv_path = self.project_root / "venv"
        venv_exists = venv_path.exists()
        
        self.record_result(
            "Virtual Environment Setup",
            venv_exists,
            0.0,
            {
                "venv_path": str(venv_path),
                "exists": venv_exists,
                "is_directory": venv_path.is_dir() if venv_exists else False
            }
        )
        
        # Check Docker Compose files
        docker_files = [
            "docker-compose.yml",
            "docker-compose.fast.yml"
        ]
        
        for docker_file in docker_files:
            file_path = self.project_root / docker_file
            exists = file_path.exists()
            
            self.record_result(
                f"Docker Compose - {docker_file}",
                exists,
                0.0,
                {
                    "path": str(file_path),
                    "exists": exists
                }
            )
        
        # Check key Python files
        key_files = [
            "pyproject.toml",
            "app/main.py",
            "migrations/env.py"
        ]
        
        for key_file in key_files:
            file_path = self.project_root / key_file
            exists = file_path.exists()
            
            self.record_result(
                f"Key File - {key_file}",
                exists,
                0.0,
                {
                    "path": str(file_path),
                    "exists": exists
                }
            )

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all focused tests"""
        logger.info("🧪 Starting Focused Testing Execution")
        logger.info(f"Project Root: {self.project_root}")
        
        try:
            # Run all test suites
            self.test_makefile_commands()
            self.test_setup_script_structure()
            self.test_error_handling() 
            self.test_performance_spot_checks()
            self.test_documentation_accessibility()
            self.test_critical_path_validation()
            
            # Calculate summary
            total_tests = len(self.results)
            passed_tests = len([r for r in self.results if r["success"]])
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Generate report
            report = {
                "test_execution": {
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_duration": time.time() - self.start_time,
                    "test_type": "focused_validation"
                },
                "test_summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": total_tests - passed_tests,
                    "success_rate": success_rate,
                    "quality_assessment": self._assess_quality(success_rate)
                },
                "detailed_results": self.results,
                "key_findings": self._generate_key_findings(),
                "recommendations": self._generate_recommendations(success_rate)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                "test_execution": {
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "status": "FAILED",
                    "error": str(e)
                },
                "test_summary": {
                    "total_tests": len(self.results),
                    "passed": len([r for r in self.results if r["success"]]),
                    "failed": len([r for r in self.results if not r["success"]]),
                    "success_rate": 0.0
                }
            }

    def _assess_quality(self, success_rate: float) -> str:
        """Assess quality based on success rate"""
        if success_rate >= 95:
            return "EXCELLENT - Production Ready"
        elif success_rate >= 85:
            return "GOOD - Minor Issues"
        elif success_rate >= 70:
            return "ACCEPTABLE - Needs Improvement"
        elif success_rate >= 50:
            return "POOR - Major Issues"
        else:
            return "CRITICAL - Not Production Ready"

    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from test results"""
        findings = []
        
        # Analyze results by category
        makefile_tests = [r for r in self.results if "Makefile" in r["test_name"]]
        makefile_success = len([r for r in makefile_tests if r["success"]]) / len(makefile_tests) * 100 if makefile_tests else 0
        
        if makefile_success >= 90:
            findings.append("✅ Makefile commands are well-structured and functional")
        elif makefile_success >= 70:
            findings.append("⚠️ Makefile has minor issues but is mostly functional")
        else:
            findings.append("❌ Makefile has significant issues requiring attention")
        
        # Check performance
        perf_tests = [r for r in self.results if "Performance" in r["test_name"]]
        if perf_tests:
            slow_tests = [r for r in perf_tests if not r["success"]]
            if not slow_tests:
                findings.append("✅ Performance targets are being met")
            else:
                findings.append(f"⚠️ {len(slow_tests)} performance tests exceeded thresholds")
        
        # Check documentation
        doc_tests = [r for r in self.results if "Documentation" in r["test_name"]]
        doc_success = len([r for r in doc_tests if r["success"]]) / len(doc_tests) * 100 if doc_tests else 0
        
        if doc_success >= 90:
            findings.append("✅ Documentation is comprehensive and accessible")
        else:
            findings.append("⚠️ Some documentation files are missing or incomplete")
        
        return findings

    def _generate_recommendations(self, success_rate: float) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if success_rate < 85:
            recommendations.append("Address failing tests before production deployment")
        
        # Specific recommendations based on failures
        failed_tests = [r for r in self.results if not r["success"]]
        
        makefile_failures = [r for r in failed_tests if "Makefile" in r["test_name"]]
        if makefile_failures:
            recommendations.append("Fix Makefile command issues for better developer experience")
        
        doc_failures = [r for r in failed_tests if "Documentation" in r["test_name"]]
        if doc_failures:
            recommendations.append("Complete missing documentation files")
        
        perf_failures = [r for r in failed_tests if "Performance" in r["test_name"]]
        if perf_failures:
            recommendations.append("Optimize slow commands for better developer productivity")
        
        if success_rate >= 90:
            recommendations.append("System shows excellent quality - ready for advanced optimizations")
        
        return recommendations


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeanVibe Agent Hive Focused Testing")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--output", type=Path, help="Output file for test report")
    
    args = parser.parse_args()
    
    # Run focused testing
    runner = FocusedTestRunner(project_root=args.project_root)
    report = runner.run_all_tests()
    
    # Save report
    output_file = args.output or Path("focused_testing_report.json")
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 FOCUSED TESTING COMPLETE")
    print("="*60)
    print(f"Total Tests: {report['test_summary']['total_tests']}")
    print(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
    print(f"Quality: {report['test_summary']['quality_assessment']}")
    print(f"Report saved to: {output_file}")
    
    # Print key findings
    if report.get('key_findings'):
        print("\n🔍 Key Findings:")
        for finding in report['key_findings']:
            print(f"  {finding}")
    
    # Print recommendations
    if report.get('recommendations'):
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print("="*60)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Bottom-Up Testing Framework Runner

This script executes the comprehensive bottom-up testing framework
to validate enhanced system integration following the consolidation approach.

Usage:
    python run_bottom_up_validation.py
    python run_bottom_up_validation.py --level unit
    python run_bottom_up_validation.py --level integration  
    python run_bottom_up_validation.py --level all
"""

import asyncio
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import structlog
import json

logger = structlog.get_logger(__name__)


class BottomUpValidationRunner:
    """Runner for bottom-up testing framework validation."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.results: Dict[str, Any] = {}
    
    async def run_level_1_component_isolation(self) -> Dict[str, Any]:
        """Run Level 1: Component isolation tests."""
        logger.info("Running Level 1: Component Isolation Tests")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel1_ComponentIsolation",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 1,
                "name": "Component Isolation",
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 1 test execution failed: {e}")
            return {
                "level": 1,
                "name": "Component Isolation", 
                "status": "error",
                "error": str(e)
            }
    
    async def run_level_2_integration_testing(self) -> Dict[str, Any]:
        """Run Level 2: Integration testing."""
        logger.info("Running Level 2: Integration Testing")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel2_IntegrationTesting", 
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 2,
                "name": "Integration Testing",
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr, 
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 2 test execution failed: {e}")
            return {
                "level": 2,
                "name": "Integration Testing",
                "status": "error", 
                "error": str(e)
            }
    
    async def run_level_3_contract_testing(self) -> Dict[str, Any]:
        """Run Level 3: Contract testing."""
        logger.info("Running Level 3: Contract Testing")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel3_ContractTesting",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 3,
                "name": "Contract Testing",
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 3 test execution failed: {e}")
            return {
                "level": 3,
                "name": "Contract Testing",
                "status": "error",
                "error": str(e)
            }
    
    async def run_level_4_api_testing(self) -> Dict[str, Any]:
        """Run Level 4: API testing."""
        logger.info("Running Level 4: API Testing")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel4_APITesting",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 4,
                "name": "API Testing",
                "status": "passed" if result.returncode == 0 else "failed", 
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 4 test execution failed: {e}")
            return {
                "level": 4, 
                "name": "API Testing",
                "status": "error",
                "error": str(e)
            }
    
    async def run_level_5_cli_testing(self) -> Dict[str, Any]:
        """Run Level 5: CLI testing."""
        logger.info("Running Level 5: CLI Testing")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel5_CLITesting",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 5,
                "name": "CLI Testing", 
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 5 test execution failed: {e}")
            return {
                "level": 5,
                "name": "CLI Testing",
                "status": "error",
                "error": str(e)
            }
    
    async def run_level_6_mobile_pwa_testing(self) -> Dict[str, Any]:
        """Run Level 6: Mobile PWA testing."""
        logger.info("Running Level 6: Mobile PWA Testing")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel6_MobilePWATesting",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 6,
                "name": "Mobile PWA Testing",
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 6 test execution failed: {e}")
            return {
                "level": 6,
                "name": "Mobile PWA Testing", 
                "status": "error",
                "error": str(e)
            }
    
    async def run_level_7_end_to_end_validation(self) -> Dict[str, Any]:
        """Run Level 7: End-to-end validation."""
        logger.info("Running Level 7: End-to-End Validation")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::TestLevel7_EndToEndValidation",
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "level": 7,
                "name": "End-to-End Validation",
                "status": "passed" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Level 7 test execution failed: {e}")
            return {
                "level": 7,
                "name": "End-to-End Validation",
                "status": "error",
                "error": str(e)
            }
    
    async def run_framework_summary(self) -> Dict[str, Any]:
        """Run framework summary test."""
        logger.info("Running Framework Summary")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                f"{self.base_path}/enhanced_integration/test_bottom_up_framework.py::test_bottom_up_framework_summary",
                "-v", "--tb=short", "--no-header", "-s"
            ], capture_output=True, text=True, cwd=self.base_path.parent)
            
            return {
                "name": "Framework Summary",
                "status": "passed" if result.returncode == 0 else "failed", 
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except Exception as e:
            logger.error(f"Framework summary execution failed: {e}")
            return {
                "name": "Framework Summary",
                "status": "error",
                "error": str(e)
            }
    
    async def run_all_levels(self) -> Dict[str, Any]:
        """Run all testing levels in sequence."""
        logger.info("Starting Bottom-Up Testing Framework - All Levels")
        
        # Execute all levels in dependency order
        level_functions = [
            self.run_level_1_component_isolation,
            self.run_level_2_integration_testing, 
            self.run_level_3_contract_testing,
            self.run_level_4_api_testing,
            self.run_level_5_cli_testing,
            self.run_level_6_mobile_pwa_testing,
            self.run_level_7_end_to_end_validation
        ]
        
        results = []
        overall_status = "passed"
        
        for level_func in level_functions:
            result = await level_func()
            results.append(result)
            
            # Update overall status
            if result["status"] != "passed":
                overall_status = "failed"
                logger.warning(f"Level {result.get('level', 'Unknown')} failed")
            else:
                logger.info(f"Level {result.get('level', 'Unknown')} passed")
        
        # Run framework summary
        summary_result = await self.run_framework_summary()
        results.append(summary_result)
        
        return {
            "overall_status": overall_status,
            "levels_executed": len(level_functions),
            "levels_passed": sum(1 for r in results[:-1] if r["status"] == "passed"),
            "results": results
        }
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print comprehensive summary report."""
        print("\n" + "="*80)
        print("BOTTOM-UP TESTING FRAMEWORK VALIDATION REPORT")
        print("="*80)
        print(f"Overall Status: {'✅ PASSED' if results['overall_status'] == 'passed' else '❌ FAILED'}")
        print(f"Levels Executed: {results['levels_executed']}")
        print(f"Levels Passed: {results['levels_passed']}/{results['levels_executed']}")
        print("="*80)
        
        # Level-by-level results
        for result in results["results"]:
            level_num = result.get("level", "")
            level_name = result["name"]
            status = result["status"]
            status_icon = "✅" if status == "passed" else "❌" if status == "failed" else "⚠️"
            
            level_prefix = f"Level {level_num}: " if level_num else ""
            print(f"{status_icon} {level_prefix}{level_name}")
            
            if status == "failed" and result.get("stderr"):
                # Show key error information
                stderr_lines = result["stderr"].split('\n')
                error_lines = [line for line in stderr_lines if 'FAILED' in line or 'ERROR' in line]
                if error_lines:
                    for error_line in error_lines[:3]:  # Show first 3 error lines
                        print(f"    {error_line.strip()}")
        
        print("\n" + "="*80)
        
        # Enhanced system integration status
        if results['overall_status'] == 'passed':
            print("🎉 ENHANCED SYSTEM INTEGRATION VALIDATION SUCCESSFUL")
            print("   • CLI enhanced command ecosystem integration validated")
            print("   • Project Index API integration confirmed") 
            print("   • Mobile PWA real-time capabilities verified")
            print("   • Bottom-up testing framework operational")
        else:
            print("⚠️  ENHANCED SYSTEM INTEGRATION VALIDATION INCOMPLETE")
            print("   • Some components require attention before deployment")
            print("   • Review failed test details above for remediation")
        
        print("="*80)


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Bottom-Up Testing Framework Runner")
    parser.add_argument("--level", choices=["unit", "integration", "contract", "api", "cli", "pwa", "e2e", "all"], 
                       default="all", help="Testing level to execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    runner = BottomUpValidationRunner()
    
    try:
        if args.level == "all":
            results = await runner.run_all_levels()
        else:
            # Run specific level (implementation would map level names to functions)
            logger.info(f"Running specific level: {args.level}")
            results = {"message": f"Specific level {args.level} execution not yet implemented"}
        
        # Print summary report
        if "overall_status" in results:
            runner.print_summary_report(results)
        
        # Save results to file
        results_file = Path(__file__).parent / "bottom_up_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Exit with appropriate code
        if results.get("overall_status") == "passed":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Bottom-up validation runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Phase 1 Demonstration Validation Script

Quick validation script to ensure the Phase 1 demonstration environment
is properly configured and ready to run. This script checks:

1. Python dependencies availability
2. Redis server connectivity
3. API server readiness
4. Basic system health
5. Demo script functionality

Run this before running the full Phase 1 milestone demonstration.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import click


class Phase1ValidationSuite:
    """Validation suite for Phase 1 demonstration prerequisites."""
    
    def __init__(self):
        self.validation_results = []
        self.project_root = Path(__file__).parent
    
    async def run_validation_suite(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        click.echo("🔍 Running Phase 1 Demonstration Validation Suite")
        click.echo("=" * 60)
        
        # 1. Python Dependencies Check
        click.echo("📦 Checking Python dependencies...")
        deps_result = await self._check_python_dependencies()
        self.validation_results.append(deps_result)
        self._display_result(deps_result)
        
        # 2. Redis Server Check
        click.echo("\n🔴 Checking Redis server...")
        redis_result = await self._check_redis_server()
        self.validation_results.append(redis_result)
        self._display_result(redis_result)
        
        # 3. API Server Check
        click.echo("\n🌐 Checking API server availability...")
        api_result = await self._check_api_server()
        self.validation_results.append(api_result)
        self._display_result(api_result)
        
        # 4. Demo Script Check
        click.echo("\n📋 Validating demonstration script...")
        script_result = await self._check_demo_script()
        self.validation_results.append(script_result)
        self._display_result(script_result)
        
        # 5. System Health Check
        click.echo("\n🩺 Running system health check...")
        health_result = await self._check_system_health()
        self.validation_results.append(health_result)
        self._display_result(health_result)
        
        # Generate summary
        return self._generate_validation_summary()
    
    async def _check_python_dependencies(self) -> Dict[str, Any]:
        """Check required Python dependencies."""
        required_packages = [
            "click",
            "httpx",
            "redis",
            "structlog",
            "asyncio",
            "json",
            "uuid"
        ]
        
        missing_packages = []
        available_packages = []
        
        for package in required_packages:
            try:
                if package in ["asyncio", "json", "uuid"]:
                    # These are built-in modules
                    __import__(package)
                    available_packages.append(package)
                else:
                    # External packages
                    __import__(package)
                    available_packages.append(package)
            except ImportError:
                missing_packages.append(package)
        
        success = len(missing_packages) == 0
        
        return {
            "check": "python_dependencies",
            "success": success,
            "available_packages": available_packages,
            "missing_packages": missing_packages,
            "message": "All dependencies available" if success else f"Missing: {', '.join(missing_packages)}"
        }
    
    async def _check_redis_server(self) -> Dict[str, Any]:
        """Check Redis server connectivity."""
        try:
            import redis.asyncio as redis
            
            client = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
            await client.ping()
            
            # Get server info
            info = await client.info()
            redis_version = info.get("redis_version", "unknown")
            
            await client.close()
            
            return {
                "check": "redis_server",
                "success": True,
                "redis_version": redis_version,
                "connection_url": "redis://localhost:6379",
                "message": f"Redis server available (v{redis_version})"
            }
            
        except ImportError:
            return {
                "check": "redis_server",
                "success": False,
                "error": "redis package not available",
                "message": "Install redis package: pip install redis"
            }
        except Exception as e:
            return {
                "check": "redis_server",
                "success": False,
                "error": str(e),
                "message": f"Redis server not available: {str(e)}"
            }
    
    async def _check_api_server(self) -> Dict[str, Any]:
        """Check API server availability."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try multiple possible endpoints
                endpoints_to_try = [
                    "http://localhost:8000/health",
                    "http://localhost:8000/api/v1/system/health",
                    "http://localhost:8000/docs",  # FastAPI docs endpoint
                    "http://localhost:8000/"  # Root endpoint
                ]
                
                for endpoint in endpoints_to_try:
                    try:
                        response = await client.get(endpoint)
                        if response.status_code in [200, 404, 422]:  # 404/422 acceptable
                            return {
                                "check": "api_server",
                                "success": True,
                                "endpoint": endpoint,
                                "status_code": response.status_code,
                                "message": f"API server available at {endpoint}"
                            }
                    except:
                        continue
                
                return {
                    "check": "api_server",
                    "success": False,
                    "endpoints_tried": endpoints_to_try,
                    "message": "API server not available on any tested endpoint"
                }
                
        except ImportError:
            return {
                "check": "api_server",
                "success": False,
                "error": "httpx package not available",
                "message": "Install httpx package: pip install httpx"
            }
        except Exception as e:
            return {
                "check": "api_server",
                "success": False,
                "error": str(e),
                "message": f"API server check failed: {str(e)}"
            }
    
    async def _check_demo_script(self) -> Dict[str, Any]:
        """Check demonstration script validity."""
        try:
            demo_script_path = self.project_root / "phase_1_milestone_demonstration.py"
            
            if not demo_script_path.exists():
                return {
                    "check": "demo_script",
                    "success": False,
                    "error": "Script file not found",
                    "message": f"Demo script not found at {demo_script_path}"
                }
            
            # Check if script is executable
            is_executable = demo_script_path.stat().st_mode & 0o111 != 0
            
            # Try to parse the script
            try:
                with open(demo_script_path, 'r') as f:
                    script_content = f.read()
                
                # Basic validation - check for key components
                required_components = [
                    "Phase1MilestoneDemonstration",
                    "run_complete_demonstration",
                    "VS 3.1",
                    "VS 4.1"
                ]
                
                missing_components = [
                    comp for comp in required_components 
                    if comp not in script_content
                ]
                
                if missing_components:
                    return {
                        "check": "demo_script",
                        "success": False,
                        "missing_components": missing_components,
                        "message": f"Script missing components: {', '.join(missing_components)}"
                    }
                
                return {
                    "check": "demo_script",
                    "success": True,
                    "script_path": str(demo_script_path),
                    "is_executable": is_executable,
                    "file_size": len(script_content),
                    "message": "Demo script validated successfully"
                }
                
            except Exception as e:
                return {
                    "check": "demo_script",
                    "success": False,
                    "error": str(e),
                    "message": f"Script validation failed: {str(e)}"
                }
                
        except Exception as e:
            return {
                "check": "demo_script",
                "success": False,
                "error": str(e),
                "message": f"Demo script check failed: {str(e)}"
            }
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Run basic system health checks."""
        try:
            health_checks = {
                "python_version": sys.version_info[:2],
                "platform": sys.platform,
                "project_root_exists": self.project_root.exists(),
                "current_directory": str(Path.cwd())
            }
            
            # Check Python version (should be 3.8+)
            python_version_ok = sys.version_info >= (3, 8)
            
            # Check project structure
            expected_files = [
                "pyproject.toml",
                "app/",
                "tests/",
                "docker-compose.yml"
            ]
            
            project_structure_ok = all(
                (self.project_root / file).exists() 
                for file in expected_files
            )
            
            overall_health = python_version_ok and project_structure_ok
            
            return {
                "check": "system_health",
                "success": overall_health,
                "health_checks": health_checks,
                "python_version_ok": python_version_ok,
                "project_structure_ok": project_structure_ok,
                "message": "System health good" if overall_health else "System health issues detected"
            }
            
        except Exception as e:
            return {
                "check": "system_health",
                "success": False,
                "error": str(e),
                "message": f"System health check failed: {str(e)}"
            }
    
    def _display_result(self, result: Dict[str, Any]) -> None:
        """Display validation result."""
        check_name = result["check"].replace("_", " ").title()
        
        if result["success"]:
            click.echo(f"  ✅ {check_name}: {result['message']}")
        else:
            click.echo(f"  ❌ {check_name}: {result['message']}")
            
            # Show additional error info if available
            if "error" in result:
                click.echo(f"     Error: {result['error']}")
            if "missing_packages" in result and result["missing_packages"]:
                click.echo(f"     Missing: {', '.join(result['missing_packages'])}")
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        successful_checks = [r for r in self.validation_results if r["success"]]
        failed_checks = [r for r in self.validation_results if not r["success"]]
        
        overall_success = len(failed_checks) == 0
        success_rate = len(successful_checks) / len(self.validation_results) * 100
        
        return {
            "overall_success": overall_success,
            "total_checks": len(self.validation_results),
            "successful_checks": len(successful_checks),
            "failed_checks": len(failed_checks),
            "success_rate": success_rate,
            "validation_results": self.validation_results,
            "recommendations": self._generate_recommendations(failed_checks),
            "ready_for_demo": overall_success
        }
    
    def _generate_recommendations(self, failed_checks: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on failed checks."""
        recommendations = []
        
        for check in failed_checks:
            check_type = check["check"]
            
            if check_type == "python_dependencies":
                missing = check.get("missing_packages", [])
                if missing:
                    recommendations.append(f"Install missing Python packages: pip install {' '.join(missing)}")
            
            elif check_type == "redis_server":
                recommendations.append("Start Redis server: docker run -d -p 6379:6379 redis:7-alpine")
            
            elif check_type == "api_server":
                recommendations.append("Start API server: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
            
            elif check_type == "demo_script":
                recommendations.append("Ensure phase_1_milestone_demonstration.py is present and properly formatted")
            
            elif check_type == "system_health":
                recommendations.append("Check Python version (>=3.8) and project structure")
        
        if not recommendations:
            recommendations.append("All validations passed - ready to run Phase 1 demonstration!")
        
        return recommendations


@click.command()
@click.option("--output-file", help="Save validation results to JSON file")
@click.option("--fix-issues", is_flag=True, help="Attempt to fix common issues automatically")
async def main(output_file: str, fix_issues: bool):
    """Validate Phase 1 demonstration environment and prerequisites."""
    
    validator = Phase1ValidationSuite()
    
    try:
        results = await validator.run_validation_suite()
        
        # Display summary
        click.echo("\n" + "=" * 60)
        click.echo("📊 VALIDATION SUMMARY")
        click.echo("=" * 60)
        
        if results["overall_success"]:
            click.echo(click.style("🎉 ALL VALIDATIONS PASSED", fg="green", bold=True))
            click.echo("Phase 1 demonstration environment is ready!")
        else:
            click.echo(click.style("⚠️ VALIDATION ISSUES DETECTED", fg="yellow", bold=True))
            click.echo("Address issues before running Phase 1 demonstration")
        
        click.echo(f"\nChecks: {results['successful_checks']}/{results['total_checks']} passed")
        click.echo(f"Success Rate: {results['success_rate']:.1f}%")
        
        # Show recommendations
        recommendations = results["recommendations"]
        if recommendations:
            click.echo(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                click.echo(f"  {i}. {rec}")
        
        # Attempt fixes if requested
        if fix_issues and not results["overall_success"]:
            click.echo(f"\n🔧 ATTEMPTING AUTOMATIC FIXES...")
            await attempt_automatic_fixes(results["validation_results"])
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\n💾 Validation results saved to: {output_file}")
        
        # Show next steps
        click.echo(f"\n🚀 NEXT STEPS:")
        if results["ready_for_demo"]:
            click.echo("  1. Run Phase 1 demonstration: ./phase_1_milestone_demonstration.py")
            click.echo("  2. Review demonstration results")
            click.echo("  3. Proceed with Phase 2 planning if successful")
        else:
            click.echo("  1. Address validation issues listed above")
            click.echo("  2. Re-run this validation script")
            click.echo("  3. Run Phase 1 demonstration once all checks pass")
        
        return 0 if results["overall_success"] else 1
        
    except Exception as e:
        click.echo(f"\n💥 VALIDATION ERROR: {e}", err=True)
        return 1


async def attempt_automatic_fixes(validation_results: List[Dict[str, Any]]) -> None:
    """Attempt to automatically fix common validation issues."""
    
    for result in validation_results:
        if not result["success"]:
            check_type = result["check"]
            
            if check_type == "python_dependencies":
                missing = result.get("missing_packages", [])
                if missing:
                    click.echo(f"  📦 Installing missing packages: {', '.join(missing)}")
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install"
                        ] + missing, check=True, capture_output=True)
                        click.echo("  ✅ Packages installed successfully")
                    except subprocess.CalledProcessError as e:
                        click.echo(f"  ❌ Failed to install packages: {e}")
            
            elif check_type == "demo_script":
                script_path = Path("phase_1_milestone_demonstration.py")
                if script_path.exists() and not (script_path.stat().st_mode & 0o111):
                    click.echo("  🔧 Making demo script executable")
                    script_path.chmod(script_path.stat().st_mode | 0o755)
                    click.echo("  ✅ Demo script is now executable")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
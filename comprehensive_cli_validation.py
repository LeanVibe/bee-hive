#!/usr/bin/env python3
"""
Comprehensive CLI System Validation for LeanVibe Agent Hive 2.0

This script systematically tests all CLI functionality to establish the current
operational state and identify gaps that need to be addressed.

Author: Claude (Strategic Analysis Agent)
Date: August 23, 2025
Purpose: Epic 1 Phase 1.1 - CLI functionality assessment
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

@dataclass
class TestResult:
    name: str
    success: bool
    output: str = ""
    error: str = ""
    duration: float = 0.0
    details: Dict = field(default_factory=dict)

class CLIValidator:
    """Comprehensive CLI validation system."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
        self.project_root = Path(__file__).parent
        
    def run_cli_command(self, cmd_args: List[str], timeout: int = 10) -> TestResult:
        """Run a CLI command and capture results."""
        start_time = time.time()
        cmd_name = " ".join(cmd_args)
        
        try:
            # Run the command through Python import system
            result = subprocess.run(
                [
                    sys.executable, "-c", f"""
import sys
sys.argv = {cmd_args}
from app.hive_cli import hive
try:
    hive()
except SystemExit as e:
    if e.code != 0:
        sys.exit(e.code)
"""
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            return TestResult(
                name=cmd_name,
                success=success,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                details={"returncode": result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                name=cmd_name,
                success=False,
                error="Command timed out",
                duration=timeout,
                details={"timeout": True}
            )
        except Exception as e:
            return TestResult(
                name=cmd_name,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )

    def test_basic_cli_functionality(self):
        """Test basic CLI commands and help systems."""
        console.print("\n🔍 [bold blue]Phase 1: Basic CLI Functionality[/bold blue]")
        
        basic_tests = [
            (["hive", "--help"], "CLI help system"),
            (["hive", "--version"], "Version information"),
            (["hive", "doctor"], "System diagnostics"),
            (["hive", "status"], "System status"),
            (["hive", "agent", "--help"], "Agent command help"),
        ]
        
        for cmd_args, description in basic_tests:
            result = self.run_cli_command(cmd_args)
            self.results.append(result)
            
            status = "✅" if result.success else "❌"
            console.print(f"  {status} {description}: {result.duration:.2f}s")
            if not result.success and result.error:
                console.print(f"    [red]Error: {result.error[:100]}...[/red]")

    def test_agent_management(self):
        """Test agent management functionality."""
        console.print("\n🤖 [bold blue]Phase 2: Agent Management[/bold blue]")
        
        agent_tests = [
            (["hive", "agent", "list"], "List agents"),
            (["hive", "agent", "status"], "Agent status"),
            # Skip create/deploy tests to avoid side effects
        ]
        
        for cmd_args, description in agent_tests:
            result = self.run_cli_command(cmd_args)
            self.results.append(result)
            
            status = "✅" if result.success else "❌"
            console.print(f"  {status} {description}: {result.duration:.2f}s")

    def test_system_operations(self):
        """Test system operation commands."""
        console.print("\n⚙️ [bold blue]Phase 3: System Operations[/bold blue]")
        
        # Note: Skip start/stop to avoid interfering with running system
        system_tests = [
            (["hive", "version"], "Version display"),
            (["hive", "id", "generate"], "ID generation"),
            # Note: logs command might fail if no API server
        ]
        
        for cmd_args, description in system_tests:
            result = self.run_cli_command(cmd_args)
            self.results.append(result)
            
            status = "✅" if result.success else "❌"
            console.print(f"  {status} {description}: {result.duration:.2f}s")

    def test_api_dependent_commands(self):
        """Test commands that depend on API connectivity."""
        console.print("\n🌐 [bold blue]Phase 4: API-Dependent Commands[/bold blue]")
        
        # These commands require the API server to be running
        api_tests = [
            (["hive", "logs"], "System logs"),
            (["hive", "agent", "deploy", "--help"], "Agent deployment help"),
        ]
        
        for cmd_args, description in api_tests:
            result = self.run_cli_command(cmd_args, timeout=5)  # Shorter timeout
            self.results.append(result)
            
            status = "✅" if result.success else "⚠️"  # Warning for API-dependent
            console.print(f"  {status} {description}: {result.duration:.2f}s")
            
            if not result.success:
                # This is expected if API server isn't fully ready
                console.print(f"    [yellow]Note: May require running API server[/yellow]")

    def test_performance_benchmarks(self):
        """Test CLI performance requirements."""
        console.print("\n⚡ [bold blue]Phase 5: Performance Benchmarks[/bold blue]")
        
        # Test response time requirements
        performance_tests = [
            (["hive", "--help"], "Help response time", 1.0),
            (["hive", "status"], "Status response time", 2.0),
            (["hive", "doctor"], "Doctor response time", 3.0),
        ]
        
        for cmd_args, description, max_time in performance_tests:
            result = self.run_cli_command(cmd_args)
            self.results.append(result)
            
            meets_requirement = result.success and result.duration <= max_time
            status = "✅" if meets_requirement else "❌"
            
            console.print(f"  {status} {description}: {result.duration:.2f}s (max: {max_time}s)")

    def analyze_results(self) -> Dict:
        """Analyze test results and generate summary."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Performance analysis
        avg_response_time = sum(r.duration for r in self.results) / len(self.results)
        slow_commands = [r for r in self.results if r.duration > 2.0]
        
        # Error analysis
        failed_tests = [r for r in self.results if not r.success]
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "slow_commands": len(slow_commands),
            "failed_tests": len(failed_tests),
            "failures": [{"name": r.name, "error": r.error} for r in failed_tests]
        }

    def generate_report(self):
        """Generate comprehensive validation report."""
        analysis = self.analyze_results()
        
        # Summary panel
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            f"🎯 [bold green]CLI VALIDATION COMPLETE[/bold green]\n"
            f"Success Rate: {analysis['success_rate']:.1f}% ({analysis['successful_tests']}/{analysis['total_tests']})\n"
            f"Average Response Time: {analysis['avg_response_time']:.2f}s\n"
            f"Total Duration: {time.time() - self.start_time:.1f}s",
            title="Validation Summary"
        ))
        
        # Results table
        table = Table(title="Test Results Details")
        table.add_column("Test", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Notes")
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            duration = f"{result.duration:.2f}s"
            notes = ""
            
            if result.error and not result.success:
                notes = result.error[:50] + "..." if len(result.error) > 50 else result.error
            elif result.duration > 2.0:
                notes = "Slow response"
            
            table.add_row(result.name, status, duration, notes)
        
        console.print(table)
        
        # Failure analysis
        if analysis['failed_tests'] > 0:
            console.print("\n❌ [bold red]Failed Tests Analysis[/bold red]")
            for failure in analysis['failures']:
                console.print(f"  • {failure['name']}: {failure['error']}")
        
        # Performance analysis
        if analysis['slow_commands'] > 0:
            console.print(f"\n⚠️ [bold yellow]Performance Issues: {analysis['slow_commands']} slow commands[/bold yellow]")
        
        # Recommendations
        console.print("\n💡 [bold blue]Recommendations[/bold blue]")
        if analysis['success_rate'] < 80:
            console.print("  • Low success rate indicates significant CLI issues")
        if analysis['avg_response_time'] > 1.5:
            console.print("  • Average response time exceeds optimal threshold")
        if analysis['failed_tests'] > 0:
            console.print("  • Failed tests require immediate attention")
        
        console.print("\n🚀 [bold green]Next Steps[/bold green]")
        console.print("  • Address failed tests to improve success rate")
        console.print("  • Optimize slow commands for better user experience") 
        console.print("  • Implement missing functionality for 100% CLI coverage")
        
        return analysis

    def run_validation(self):
        """Run complete CLI validation suite."""
        console.print("🚀 [bold green]Starting Comprehensive CLI Validation[/bold green]")
        console.print(f"📍 Project Root: {self.project_root}")
        
        # Run all test phases
        self.test_basic_cli_functionality()
        self.test_agent_management()
        self.test_system_operations()
        self.test_api_dependent_commands()
        self.test_performance_benchmarks()
        
        # Generate and display report
        return self.generate_report()

def main():
    """Main execution function."""
    validator = CLIValidator()
    analysis = validator.run_validation()
    
    # Exit with appropriate code
    success_rate = analysis['success_rate']
    if success_rate >= 90:
        console.print("✅ [bold green]CLI system is highly functional[/bold green]")
        return 0
    elif success_rate >= 75:
        console.print("⚠️ [bold yellow]CLI system has some issues but is mostly functional[/bold yellow]")
        return 1
    else:
        console.print("❌ [bold red]CLI system has significant issues requiring attention[/bold red]")
        return 2

if __name__ == "__main__":
    sys.exit(main())
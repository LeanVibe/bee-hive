"""
LeanVibe Unified CLI - Developer Experience Enhancement

The `lv` command provides a unified interface for all LeanVibe Agent Hive operations
with intelligent auto-completion, context-aware suggestions, and advanced debugging.
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.live import Live
from rich.layout import Layout
from rich.tree import Tree
from rich.columns import Columns
from rich.status import Status
from rich import print as rprint

# Import existing CLI functionality
from app.cli import AgentHiveCLI, AgentHiveConfig

# Import DX enhancement modules
from app.core.onboarding_automation import run_zero_setup_onboarding
from app.core.productivity_intelligence import (
    get_productivity_metrics,
    get_productivity_recommendations,
    run_system_health_check
)

console = Console()


@dataclass
class DeveloperContext:
    """Smart context detection for enhanced developer experience."""
    
    project_type: Optional[str] = None
    git_status: Optional[str] = None
    last_command: Optional[str] = None
    working_directory: Path = Path.cwd()
    agent_hive_status: str = "unknown"
    recent_errors: List[str] = None
    productivity_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recent_errors is None:
            self.recent_errors = []
        if self.productivity_metrics is None:
            self.productivity_metrics = {}


class IntelligentCommandSuggester:
    """AI-powered command suggestions based on context and usage patterns."""
    
    def __init__(self):
        self.usage_history = self._load_usage_history()
        self.context_patterns = {
            "setup": ["start", "status", "develop"],
            "error": ["debug", "logs", "health", "reset"],
            "development": ["develop", "debug", "test", "deploy"],
            "monitoring": ["status", "logs", "dashboard", "agents"]
        }
    
    def _load_usage_history(self) -> Dict[str, Any]:
        """Load command usage history for intelligent suggestions."""
        config_dir = Path.home() / ".config" / "leanvibe"
        usage_file = config_dir / "usage_history.json"
        
        if usage_file.exists():
            try:
                with open(usage_file) as f:
                    return json.load(f)
            except:
                pass
        
        return {"commands": {}, "sequences": [], "last_updated": time.time()}
    
    def record_command_usage(self, command: str, context: DeveloperContext):
        """Record command usage for future suggestions."""
        timestamp = time.time()
        
        # Update command frequency
        self.usage_history["commands"][command] = self.usage_history["commands"].get(command, 0) + 1
        
        # Record command sequence
        if len(self.usage_history["sequences"]) > 100:
            self.usage_history["sequences"] = self.usage_history["sequences"][-50:]
        
        self.usage_history["sequences"].append({
            "command": command,
            "timestamp": timestamp,
            "context": {
                "project_type": context.project_type,
                "working_dir": str(context.working_directory),
                "agent_status": context.agent_hive_status
            }
        })
        
        self._save_usage_history()
    
    def suggest_next_commands(self, current_context: DeveloperContext) -> List[Tuple[str, str, float]]:
        """Suggest next commands based on context and patterns."""
        suggestions = []
        
        # Context-based suggestions
        if current_context.agent_hive_status == "not_running":
            suggestions.append(("start", "Start LeanVibe Agent Hive services", 0.9))
            suggestions.append(("setup", "Run system setup if needed", 0.8))
        elif current_context.agent_hive_status == "running":
            suggestions.append(("develop", "Start autonomous development", 0.9))
            suggestions.append(("dashboard", "Open monitoring dashboard", 0.7))
            suggestions.append(("status", "Check detailed system status", 0.6))
        
        # Error-based suggestions
        if current_context.recent_errors:
            suggestions.append(("debug", "Debug recent errors", 0.95))
            suggestions.append(("logs", "View system logs", 0.8))
            suggestions.append(("health", "Run comprehensive health check", 0.7))
        
        # Usage pattern suggestions
        for cmd, count in sorted(self.usage_history["commands"].items(), key=lambda x: x[1], reverse=True)[:3]:
            suggestions.append((cmd, f"Frequently used command", 0.5 + count * 0.1))
        
        # Remove duplicates and sort by confidence
        unique_suggestions = {}
        for cmd, desc, conf in suggestions:
            if cmd not in unique_suggestions or unique_suggestions[cmd][1] < conf:
                unique_suggestions[cmd] = (desc, conf)
        
        return [(cmd, desc, conf) for cmd, (desc, conf) in unique_suggestions.items()]
    
    def _save_usage_history(self):
        """Save usage history to file."""
        config_dir = Path.home() / ".config" / "leanvibe"
        config_dir.mkdir(parents=True, exist_ok=True)
        usage_file = config_dir / "usage_history.json"
        
        try:
            with open(usage_file, 'w') as f:
                json.dump(self.usage_history, f, indent=2)
        except Exception as e:
            # Silent fail for usage tracking
            pass


class AdvancedDebuggingInterface:
    """Advanced debugging suite with visual agent flow tracking."""
    
    def __init__(self, api_base: str):
        self.api_base = api_base
    
    def show_agent_flow_visualization(self) -> bool:
        """Display real-time agent workflow visualization."""
        try:
            response = requests.get(f"{self.api_base}/api/debug/agent-flows", timeout=5)
            if response.status_code != 200:
                return False
            
            flows = response.json()
            
            # Create visual tree of agent flows
            tree = Tree("🤖 [bold blue]Active Agent Workflows[/bold blue]")
            
            for flow in flows.get("active_flows", []):
                flow_node = tree.add(f"📊 {flow['name']} ({flow['status']})")
                
                for agent in flow.get("agents", []):
                    agent_node = flow_node.add(f"🔧 {agent['role']}: {agent['current_task']}")
                    
                    for step in agent.get("recent_steps", [])[-3:]:
                        agent_node.add(f"• {step['action']} ({step['duration']}ms)")
            
            console.print(Panel(tree, title="Agent Flow Visualization", border_style="blue"))
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to fetch agent flows: {e}[/red]")
            return False
    
    def diagnose_system_issues(self) -> Dict[str, Any]:
        """Intelligent system diagnosis with suggested fixes."""
        issues = []
        suggestions = []
        
        try:
            # Check system health
            health_response = requests.get(f"{self.api_base}/health", timeout=3)
            if health_response.status_code != 200:
                issues.append("API server not responding")
                suggestions.append("Run: lv start")
            
            # Check agent status
            agent_response = requests.get(f"{self.api_base}/api/agents/status", timeout=3)
            if agent_response.status_code == 200:
                agent_data = agent_response.json()
                if not agent_data.get("active"):
                    issues.append("No active agents")
                    suggestions.append("Run: lv develop <project_description>")
            
            # Check for common configuration issues
            config_path = Path.home() / ".config" / "agent-hive" / "config.json"
            if not config_path.exists():
                issues.append("Missing configuration file")
                suggestions.append("Run: lv setup")
                
        except requests.RequestException:
            issues.append("Cannot connect to LeanVibe services")
            suggestions.append("Run: lv start")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "status": "healthy" if not issues else "needs_attention"
        }
    
    def show_performance_profiling(self) -> bool:
        """Display performance metrics and profiling information."""
        try:
            response = requests.get(f"{self.api_base}/api/debug/performance", timeout=5)
            if response.status_code != 200:
                return False
            
            perf_data = response.json()
            
            # Create performance metrics table
            perf_table = Table(title="🚀 Performance Metrics")
            perf_table.add_column("Metric", style="cyan")
            perf_table.add_column("Current", style="green")
            perf_table.add_column("Average", style="yellow")
            perf_table.add_column("Target", style="blue")
            perf_table.add_column("Status", style="bold")
            
            for metric in perf_data.get("metrics", []):
                status = "✅ Good" if metric.get("status") == "good" else "⚠️ Attention"
                perf_table.add_row(
                    metric["name"],
                    metric["current"],
                    metric["average"],
                    metric["target"],
                    status
                )
            
            console.print(perf_table)
            
            # Show recommendations if any
            recommendations = perf_data.get("recommendations", [])
            if recommendations:
                console.print("\n💡 [bold yellow]Performance Recommendations:[/bold yellow]")
                for rec in recommendations:
                    console.print(f"   • {rec}")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to fetch performance data: {e}[/red]")
            return False


class UnifiedLeanVibeCLI:
    """Unified CLI interface with enhanced developer experience."""
    
    def __init__(self):
        self.agent_hive_cli = AgentHiveCLI()
        self.config = AgentHiveConfig()
        self.suggester = IntelligentCommandSuggester()
        self.debugger = AdvancedDebuggingInterface(self.agent_hive_cli.api_base)
        self.context = self._detect_context()
    
    def _detect_context(self) -> DeveloperContext:
        """Detect current development context for intelligent suggestions."""
        context = DeveloperContext()
        
        # Detect project type
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists():
            context.project_type = "python"
        elif (cwd / "package.json").exists():
            context.project_type = "nodejs"
        elif (cwd / "Cargo.toml").exists():
            context.project_type = "rust"
        elif (cwd / "go.mod").exists():
            context.project_type = "go"
        
        # Check git status
        try:
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True, cwd=cwd)
            if result.returncode == 0:
                changes = result.stdout.strip()
                if changes:
                    context.git_status = "modified"
                else:
                    context.git_status = "clean"
        except:
            context.git_status = "no_git"
        
        # Check LeanVibe status
        if self.agent_hive_cli.check_system_health():
            context.agent_hive_status = "running"
        else:
            context.agent_hive_status = "not_running"
        
        return context
    
    def show_intelligent_help(self, command_context: str = None):
        """Show context-aware help with intelligent suggestions."""
        console.print(Panel.fit(
            "🚀 [bold blue]LeanVibe Unified CLI (lv)[/bold blue]\n"
            "Intelligent autonomous development platform",
            border_style="blue"
        ))
        
        # Show context information
        context_info = []
        context_info.append(f"📁 Project: {self.context.project_type or 'unknown'}")
        context_info.append(f"🌿 Git: {self.context.git_status}")
        context_info.append(f"⚙️ LeanVibe: {self.context.agent_hive_status}")
        
        console.print("📊 [bold]Current Context:[/bold]")
        console.print("   " + " | ".join(context_info))
        
        # Show intelligent command suggestions
        suggestions = self.suggester.suggest_next_commands(self.context)
        if suggestions:
            console.print("\n💡 [bold yellow]Suggested Commands:[/bold yellow]")
            
            suggestions_table = Table()
            suggestions_table.add_column("Command", style="green")
            suggestions_table.add_column("Description", style="white")
            suggestions_table.add_column("Confidence", style="cyan")
            
            for cmd, desc, confidence in suggestions[:5]:
                conf_bar = "█" * int(confidence * 10)
                suggestions_table.add_row(f"lv {cmd}", desc, f"{conf_bar} {confidence:.1%}")
            
            console.print(suggestions_table)
        
        # Show available command categories
        console.print("\n📋 [bold]Command Categories:[/bold]")
        categories = {
            "🚀 Essential": ["start", "setup", "develop", "status"],
            "🔧 Development": ["debug", "test", "logs", "health"],
            "🎛️ Monitoring": ["dashboard", "agents", "metrics"],
            "⚙️ Configuration": ["config", "update", "reset"]
        }
        
        for category, commands in categories.items():
            console.print(f"   {category}: {', '.join([f'[green]lv {cmd}[/green]' for cmd in commands])}")
        
        console.print("\n💬 For detailed help on any command: [green]lv <command> --help[/green]")
    
    def enhanced_start_command(self, **kwargs):
        """Enhanced start command with intelligent pre-checks and setup."""
        with console.status("[bold green]Preparing to start LeanVibe...") as status:
            # Pre-flight checks
            status.update("Running pre-flight checks...")
            
            # Check for common issues
            diagnosis = self.debugger.diagnose_system_issues()
            if diagnosis["status"] == "needs_attention":
                console.print("\n⚠️ [yellow]Issues detected:[/yellow]")
                for issue in diagnosis["issues"]:
                    console.print(f"   • {issue}")
                
                console.print("\n💡 [bold]Suggested fixes:[/bold]")
                for suggestion in diagnosis["suggestions"]:
                    console.print(f"   • {suggestion}")
                
                if not click.confirm("Continue with startup anyway?"):
                    return False
            
            status.update("Starting services...")
        
        # Use existing start functionality
        return self.agent_hive_cli.start_services(**kwargs)
    
    def advanced_debug_command(self):
        """Advanced debugging command with visual flow tracking."""
        console.print("🔍 [bold blue]LeanVibe Advanced Debugging Suite[/bold blue]")
        
        # System diagnosis
        console.print("\n📊 Running intelligent system diagnosis...")
        diagnosis = self.debugger.diagnose_system_issues()
        
        if diagnosis["status"] == "healthy":
            console.print("✅ [green]System appears healthy[/green]")
        else:
            console.print("⚠️ [yellow]Issues detected requiring attention[/yellow]")
            
            for issue in diagnosis["issues"]:
                console.print(f"   🔴 {issue}")
            
            console.print("\n💡 [bold]Recommended actions:[/bold]")
            for suggestion in diagnosis["suggestions"]:
                console.print(f"   ⚡ {suggestion}")
        
        # Show agent flow visualization if system is running
        if self.context.agent_hive_status == "running":
            console.print("\n🤖 Agent workflow visualization:")
            if not self.debugger.show_agent_flow_visualization():
                console.print("   [dim]No active agent workflows found[/dim]")
            
            console.print("\n🚀 Performance profiling:")
            if not self.debugger.show_performance_profiling():
                console.print("   [dim]Performance metrics not available[/dim]")
        else:
            console.print("\n[dim]Start LeanVibe services to view agent debugging information[/dim]")
    
    def smart_develop_command(self, project_description: str, **kwargs):
        """Enhanced develop command with intelligent project analysis."""
        console.print(f"💻 [bold]Smart Development Mode[/bold]")
        console.print(f"📋 Project: {project_description}")
        
        # Analyze project context for smart suggestions
        context_analysis = []
        
        if self.context.project_type:
            context_analysis.append(f"Detected {self.context.project_type} project")
        
        if self.context.git_status == "modified":
            context_analysis.append("Uncommitted changes detected")
            if not click.confirm("Continue with uncommitted changes?"):
                return
        
        if context_analysis:
            console.print("🔍 [bold]Context Analysis:[/bold]")
            for analysis in context_analysis:
                console.print(f"   • {analysis}")
        
        # Record command usage for future suggestions
        self.suggester.record_command_usage("develop", self.context)
        
        # Use existing develop functionality  
        return self.agent_hive_cli.execute_hive_command(f'/hive:develop "{project_description}"')


# Command group for the unified CLI
@click.group(invoke_without_command=True)
@click.option('--version', is_flag=True, help='Show version information')
@click.pass_context
def lv(ctx, version):
    """
    🚀 LeanVibe Unified CLI - Intelligent Autonomous Development Platform
    
    The unified interface for all LeanVibe operations with smart suggestions,
    context awareness, and advanced debugging capabilities.
    """
    cli = UnifiedLeanVibeCLI()
    
    if version:
        console.print("🚀 [bold blue]LeanVibe Unified CLI[/bold blue] v2.0.0")
        console.print("Built on LeanVibe Agent Hive 2.0 - Autonomous Development Platform")
        return
    
    if ctx.invoked_subcommand is None:
        cli.show_intelligent_help()


@lv.command()
@click.option('--quick', is_flag=True, help='Quick background start')
@click.option('--dashboard', is_flag=True, help='Open dashboard after start')
def start(quick, dashboard):
    """🚀 Start LeanVibe with intelligent pre-checks"""
    cli = UnifiedLeanVibeCLI()
    success = cli.enhanced_start_command(quick=quick)
    
    if success and dashboard:
        import webbrowser
        webbrowser.open(f"{cli.agent_hive_cli.api_base}/dashboard/")


@lv.command()
@click.argument('project_description')
@click.option('--dashboard', is_flag=True, help='Open monitoring dashboard')
@click.option('--timeout', default=300, help='Development timeout in seconds')
def develop(project_description, dashboard, timeout):
    """💻 Smart autonomous development with context analysis"""
    cli = UnifiedLeanVibeCLI()
    cli.smart_develop_command(project_description, dashboard=dashboard, timeout=timeout)


@lv.command()
def debug():
    """🔍 Advanced debugging with agent flow visualization"""
    cli = UnifiedLeanVibeCLI()
    cli.advanced_debug_command()


@lv.command()
def status():
    """📊 Intelligent system status with recommendations"""
    cli = UnifiedLeanVibeCLI()
    
    # Use existing status but with enhanced context
    console.print("📊 [bold blue]LeanVibe System Status[/bold blue]")
    console.print(f"🏠 Working Directory: {cli.context.working_directory}")
    console.print(f"📁 Project Type: {cli.context.project_type or 'Unknown'}")
    console.print(f"🌿 Git Status: {cli.context.git_status}")
    
    # Show existing status information
    cli.agent_hive_cli.status(detailed=True, agents=False)


@lv.command()
def dashboard():
    """🎛️ Open monitoring dashboard"""
    cli = UnifiedLeanVibeCLI()
    cli.agent_hive_cli.dashboard(mobile_info=True)


@lv.command()
def setup():
    """🛠️ Setup LeanVibe with intelligent configuration"""
    cli = UnifiedLeanVibeCLI()
    
    console.print("🛠️ [bold blue]LeanVibe Intelligent Setup[/bold blue]")
    console.print("This will configure LeanVibe with smart defaults for your environment")
    
    # Detect and suggest optimal configuration
    if cli.context.project_type:
        console.print(f"💡 Detected {cli.context.project_type} project - will optimize for this environment")
    
    # Use existing setup with enhancements
    cli.agent_hive_cli.setup(skip_deps=False, docker_only=False)


# Additional utility commands
@lv.command()
def health():
    """🏥 Comprehensive system health check"""
    cli = UnifiedLeanVibeCLI()
    diagnosis = cli.debugger.diagnose_system_issues()
    
    if diagnosis["status"] == "healthy":
        console.print("✅ [green]All systems healthy![/green]")
    else:
        console.print("⚠️ [yellow]System needs attention[/yellow]")
        
        for issue in diagnosis["issues"]:
            console.print(f"   🔴 {issue}")
        
        console.print("\n💡 Recommended actions:")
        for suggestion in diagnosis["suggestions"]:
            console.print(f"   ⚡ {suggestion}")


@lv.command()
def logs():
    """📜 View system logs with intelligent filtering"""
    console.print("📜 [bold]System Logs[/bold]")
    console.print("Use 'docker compose logs -f' for real-time logs")
    subprocess.run(["docker", "compose", "logs", "--tail", "50"])


@lv.command()
def reset():
    """🔄 Reset system to clean state"""
    if click.confirm("This will stop all services and reset the system. Continue?"):
        console.print("🔄 [bold]Resetting LeanVibe system...[/bold]")
        subprocess.run(["docker", "compose", "down", "-v"])
        console.print("✅ System reset completed. Run 'lv setup' to reconfigure.")


# DX Enhancement Commands

@lv.command()
@click.argument('project_name', required=False)
@click.option('--template', help='Specify project template')
def init(project_name, template):
    """🚀 Initialize new project with zero-setup environment"""
    console.print("🚀 [bold blue]LeanVibe Zero-Setup Project Initialization[/bold blue]")
    
    # Run async onboarding
    try:
        import asyncio
        success = asyncio.run(run_zero_setup_onboarding(project_name))
        
        if success:
            console.print("\n✅ [green]Project initialized successfully![/green]")
            console.print("💡 Next: cd into your project and run 'lv start'")
        else:
            console.print("\n❌ [red]Project initialization failed[/red]")
            console.print("💡 Try: lv debug for troubleshooting")
            
    except Exception as e:
        console.print(f"\n❌ [red]Initialization error: {e}[/red]")
        console.print("💡 Please check the logs and try again")


@lv.command()
@click.option('--developer-id', help='Developer ID for metrics (defaults to system user)')
def metrics(developer_id):
    """📊 Show developer productivity metrics and insights"""
    if not developer_id:
        developer_id = os.getenv('USER', 'default_user')
    
    console.print(f"📊 [bold blue]Productivity Metrics for {developer_id}[/bold blue]")
    
    try:
        import asyncio
        
        # Get metrics and recommendations
        dev_metrics = asyncio.run(get_productivity_metrics(developer_id))
        recommendations = asyncio.run(get_productivity_recommendations(developer_id))
        
        # Show metrics table
        metrics_table = Table(title="📈 Your Productivity Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Status", style="yellow")
        
        # Productivity score with status
        score_status = "🔥 Excellent" if dev_metrics.productivity_score >= 80 else \
                      "✅ Good" if dev_metrics.productivity_score >= 60 else \
                      "⚠️ Needs Improvement"
        
        metrics_table.add_row(
            "Productivity Score", 
            f"{dev_metrics.productivity_score:.1f}/100",
            score_status
        )
        
        metrics_table.add_row(
            "Development Sessions",
            str(dev_metrics.session_count),
            "📈 Active" if dev_metrics.session_count > 5 else "📊 Getting Started"
        )
        
        metrics_table.add_row(
            "Total Dev Time",
            f"{dev_metrics.total_development_time_hours:.1f} hours",
            "💪 Dedicated" if dev_metrics.total_development_time_hours > 10 else "🌱 Building"
        )
        
        metrics_table.add_row(
            "Tasks Completed",
            str(dev_metrics.tasks_completed),
            "🎯 Productive" if dev_metrics.tasks_completed > 10 else "🚀 Growing"
        )
        
        if dev_metrics.average_task_completion_time_minutes > 0:
            metrics_table.add_row(
                "Avg Task Time",
                f"{dev_metrics.average_task_completion_time_minutes:.1f} minutes",
                "⚡ Fast" if dev_metrics.average_task_completion_time_minutes < 45 else "🐢 Room for improvement"
            )
        
        metrics_table.add_row(
            "Success Rate",
            f"{dev_metrics.success_rate:.1%}",
            "💯 Excellent" if dev_metrics.success_rate > 0.9 else \
            "👍 Good" if dev_metrics.success_rate > 0.7 else "📚 Learning"
        )
        
        console.print(metrics_table)
        
        # Show preferred technologies
        if dev_metrics.preferred_technologies:
            console.print(f"\n🛠️ [bold]Your Tech Stack:[/bold] {', '.join(dev_metrics.preferred_technologies)}")
        
        console.print(f"\n📈 [bold]Trend:[/bold] {dev_metrics.improvement_trend.title()}")
        
        # Show recommendations
        if recommendations:
            console.print("\n💡 [bold yellow]Personalized Recommendations:[/bold yellow]")
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                priority_icon = "🔥" if rec.priority == "high" else "⭐" if rec.priority == "medium" else "💡"
                
                console.print(f"\n{i}. {priority_icon} [bold]{rec.title}[/bold]")
                console.print(f"   {rec.description}")
                console.print(f"   📋 Actions:")
                for action in rec.action_items[:2]:  # Show first 2 actions
                    console.print(f"      • {action}")
                
                console.print(f"   ⏱️ Estimated time: {rec.estimated_time_minutes} minutes")
                console.print(f"   🎯 Impact: {rec.estimated_impact}")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to retrieve metrics: {e}[/red]")
        console.print("💡 Try: lv start (to ensure services are running)")


@lv.command()
def optimize():
    """⚡ Get intelligent productivity optimization recommendations"""
    console.print("⚡ [bold blue]LeanVibe Productivity Optimizer[/bold blue]")
    
    developer_id = os.getenv('USER', 'default_user')
    
    try:
        import asyncio
        recommendations = asyncio.run(get_productivity_recommendations(developer_id))
        
        if not recommendations:
            console.print("✅ [green]Your productivity is already optimized![/green]")
            console.print("💡 Keep up the great work with 'lv develop' and 'lv dashboard'")
            return
        
        console.print("🎯 [bold]Personalized Optimization Recommendations:[/bold]")
        
        # Group by priority
        high_priority = [r for r in recommendations if r.priority == "high"]
        medium_priority = [r for r in recommendations if r.priority == "medium"]
        low_priority = [r for r in recommendations if r.priority == "low"]
        
        for priority_group, title, icon in [
            (high_priority, "🔥 High Priority", "🔥"),
            (medium_priority, "⭐ Medium Priority", "⭐"), 
            (low_priority, "💡 Low Priority", "💡")
        ]:
            if priority_group:
                console.print(f"\n{title}:")
                for rec in priority_group:
                    console.print(f"\n{icon} [bold]{rec.title}[/bold]")
                    console.print(f"   {rec.description}")
                    
                    if rec.action_items:
                        console.print("   📋 Action Plan:")
                        for action in rec.action_items:
                            console.print(f"      • {action}")
                    
                    console.print(f"   ⏱️ Time investment: {rec.estimated_time_minutes} minutes")
                    console.print(f"   🎯 Expected impact: {rec.estimated_impact}")
        
        # Show quick wins
        quick_wins = [r for r in recommendations if r.estimated_time_minutes <= 30]
        if quick_wins:
            console.print("\n⚡ [bold yellow]Quick Wins (≤30 minutes):[/bold yellow]")
            for rec in quick_wins:
                console.print(f"   • {rec.title} ({rec.estimated_time_minutes}min)")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to generate recommendations: {e}[/red]")
        console.print("💡 Try: lv start (to ensure services are running)")


@lv.command()
def health():
    """🏥 Enhanced system health check with intelligent diagnostics"""
    console.print("🏥 [bold blue]LeanVibe System Health Check[/bold blue]")
    
    try:
        import asyncio
        health_report = asyncio.run(run_system_health_check())
        
        # Overall status
        status_icon = "✅" if health_report["overall_status"] == "healthy" else "❌"
        console.print(f"\n{status_icon} [bold]Overall Status: {health_report['overall_status'].upper()}[/bold]")
        
        # Component status
        console.print("\n🔧 [bold]Component Health:[/bold]")
        components_table = Table()
        components_table.add_column("Component", style="cyan")
        components_table.add_column("Status", style="green")
        components_table.add_column("Details", style="white")
        
        for component, info in health_report["components"].items():
            status_display = "✅ Healthy" if info["status"] == "healthy" else "❌ Unhealthy"
            components_table.add_row(
                component.title(),
                status_display,
                info.get("details", "")
            )
        
        console.print(components_table)
        
        # Critical issues
        if health_report["critical_issues"]:
            console.print("\n🚨 [bold red]Critical Issues:[/bold red]")
            for issue in health_report["critical_issues"]:
                console.print(f"   • {issue}")
        
        # Recommendations
        if health_report["recommendations"]:
            console.print("\n💡 [bold yellow]Recommendations:[/bold yellow]")
            for rec in health_report["recommendations"]:
                console.print(f"   • {rec}")
        
        # Performance metrics if available
        if "performance" in health_report:
            console.print("\n📊 [bold]Performance Metrics:[/bold]")
            perf_data = health_report["performance"]
            for metric, value in perf_data.items():
                console.print(f"   {metric}: {value}")
        
    except Exception as e:
        console.print(f"❌ [red]Health check failed: {e}[/red]")
        console.print("💡 Basic checks you can try:")
        console.print("   • docker compose ps  (check container status)")
        console.print("   • lv logs            (check for errors)")
        console.print("   • lv reset           (if all else fails)")


def main():
    """Main entry point for the unified CLI."""
    lv()


if __name__ == "__main__":
    main()
import asyncio
"""
Agent Hive CLI - Professional Command Line Interface

This is the main CLI module for LeanVibe Agent Hive 2.0, designed for
professional installation and system-wide availability.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from app.core.updater import AgentHiveUpdater, UpdateChannel

console = Console()


class AgentHiveConfig:
    """Configuration management for Agent Hive."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".config" / "agent-hive"
        self.config_file = self.config_dir / "config.json"
        self.workspaces_dir = self.config_dir / "workspaces"
        self.integrations_dir = self.config_dir / "integrations"
        
        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.workspaces_dir.mkdir(exist_ok=True)
        self.integrations_dir.mkdir(exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default configuration
        return {
            "version": "2.0.0",
            "api_base": "http://localhost:8000",
            "workspace_dir": str(self.workspaces_dir),
            "integrations": {
                "claude_code": True,
                "gemini_cli": True,
                "opencode": True
            },
            "services": {
                "auto_start": True,
                "use_docker": True,
                "postgres_port": 5432,
                "redis_port": 6379
            }
        }
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_workspace_path(self, name: str) -> Path:
        """Get path for a workspace."""
        return self.workspaces_dir / name


class AgentHiveCLI:
    """Main CLI controller."""
    
    def __init__(self):
        self.config = AgentHiveConfig()
        self.settings = self.config.load_config()
        self.api_base = self.settings.get("api_base", "http://localhost:8000")
        self.updater = AgentHiveUpdater()
    
    def check_system_health(self) -> bool:
        """Check if the Agent Hive system is running."""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=3)
            return response.status_code == 200
        except:
            return False

    def check_pwa_dev(self) -> Optional[str]:
        """Detect running PWA dev server and return URL if available."""
        for url in [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code < 500:
                    return url
            except Exception:
                continue
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {"status": "unhealthy", "error": "System not responding"}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent system status."""
        try:
            response = requests.get(f"{self.api_base}/api/agents/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {"active": False, "agent_count": 0, "agents": {}}
    
    def execute_hive_command(self, command: str) -> Dict[str, Any]:
        """Execute a hive slash command."""
        try:
            payload = {"command": command}
            response = requests.post(
                f"{self.api_base}/api/hive/execute", 
                json=payload, 
                timeout=60
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def start_services(self, quick: bool = False) -> bool:
        """Start Agent Hive services."""
        try:
            # Try to find the project directory
            project_dirs = [
                Path.cwd(),
                Path.home() / "agent-hive",
                Path.home() / "work" / "agent-hive",
                Path("/opt/agent-hive")
            ]
            
            project_dir = None
            for dir_path in project_dirs:
                if (dir_path / "pyproject.toml").exists():
                    project_dir = dir_path
                    break
            
            if not project_dir:
                console.print("[red]❌ Agent Hive installation not found[/red]")
                console.print("💡 Run 'agent-hive setup' first")
                return False
            
            # Change to project directory and start services
            os.chdir(project_dir)
            
            if quick:
                subprocess.run(["make", "start-bg"], check=True)
            else:
                subprocess.run(["make", "start"], check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to start services: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            return False
    
    def check_for_updates_notification(self):
        """Check for updates and show notification if available."""
        try:
            update_info = self.updater.auto_check_for_updates()
            if update_info:
                console.print(Panel(
                    f"🚀 [bold green]Update Available![/bold green]\n"
                    f"New version {update_info.version} is available on {update_info.channel.value} channel.\n"
                    f"Run 'agent-hive update' to upgrade.",
                    title="Agent Hive Update",
                    border_style="green"
                ))
        except Exception:
            # Silent fail for update notifications
            pass


# Global CLI instance
cli_instance = AgentHiveCLI()


@click.group()
@click.version_option(version="2.0.0", prog_name="agent-hive")
@click.pass_context
def cli(ctx):
    """
    🤖 LeanVibe Agent Hive 2.0 - Autonomous Development Platform
    
    Professional multi-agent development system with remote oversight capabilities.
    """
    # Show update notifications for certain commands
    if ctx.invoked_subcommand in ['start', 'develop', 'status', 'dashboard']:
        # Check if auto-update notifications are enabled
        config = cli_instance.config.load_config()
        auto_update_config = config.get("auto_update", {})
        
        if auto_update_config.get("enabled", True):
            cli_instance.check_for_updates_notification()


@cli.command()
@click.option('--skip-deps', is_flag=True, help='Skip dependency installation')
@click.option('--docker-only', is_flag=True, help='Use Docker for all services')
def setup(skip_deps: bool, docker_only: bool):
    """
    🚀 Set up Agent Hive for the first time
    
    This command handles complete system setup including dependencies,
    configuration, and service initialization.
    """
    console.print(Panel.fit(
        "🚀 [bold blue]LeanVibe Agent Hive 2.0 Setup[/bold blue]\n"
        "Professional autonomous development platform installation",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        
        # Step 1: Check system requirements
        task1 = progress.add_task("Checking system requirements...", total=None)
        
        # Check for required tools
        required_tools = ["python3", "docker", "git"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, "--version"], 
                             capture_output=True, check=True)
            except:
                missing_tools.append(tool)
        
        if missing_tools and not skip_deps:
            progress.update(task1, description="❌ Missing required tools")
            console.print(f"[red]Missing tools: {', '.join(missing_tools)}[/red]")
            console.print("💡 Install missing tools and run setup again")
            return
        
        progress.update(task1, description="✅ System requirements met")
        progress.remove_task(task1)
        
        # Step 2: Create configuration
        task2 = progress.add_task("Creating configuration...", total=None)
        
        config = cli_instance.config.load_config()
        config["setup_completed"] = True
        config["setup_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cli_instance.config.save_config(config)
        
        progress.update(task2, description="✅ Configuration created")
        progress.remove_task(task2)
        
        # Step 3: Set up AI tool integrations
        task3 = progress.add_task("Setting up AI tool integrations...", total=None)
        
        # Check for existing AI tools
        ai_tools = {
            "claude_code": Path.home() / ".claude",
            "gemini_cli": Path.home() / ".config" / "gemini",
            "opencode": Path.home() / ".opencode"
        }
        
        integrations = {}
        for tool, config_path in ai_tools.items():
            integrations[tool] = config_path.exists()
        
        # Save integration status
        integration_config = cli_instance.config.integrations_dir / "status.json"
        with open(integration_config, 'w') as f:
            json.dump(integrations, f, indent=2)
        
        progress.update(task3, description="✅ AI tool integrations configured")
        progress.remove_task(task3)
        
        # Step 4: Install Agent Hive if needed
        task4 = progress.add_task("Checking Agent Hive installation...", total=None)
        
        # Check if already installed
        installation_found = False
        project_dirs = [
            Path.home() / "agent-hive",
            Path.home() / "work" / "agent-hive", 
            Path("/opt/agent-hive")
        ]
        
        for dir_path in project_dirs:
            if (dir_path / "pyproject.toml").exists():
                installation_found = True
                break
        
        if not installation_found:
            progress.update(task4, description="Installing Agent Hive...")
            
            # Clone repository
            install_dir = Path.home() / "agent-hive"
            try:
                repo_url = os.environ.get("AGENT_HIVE_REPO_URL", "https://github.com/LeanVibe/bee-hive.git")
                subprocess.run([
                    "git", "clone", 
                    repo_url,
                    str(install_dir)
                ], check=True, capture_output=True)
                
                # Run setup
                os.chdir(install_dir)
                subprocess.run(["make", "setup"], check=True)
                
            except subprocess.CalledProcessError as e:
                progress.update(task4, description="❌ Installation failed")
                console.print(f"[red]Installation failed: {e}[/red]")
                return
        
        progress.update(task4, description="✅ Agent Hive installation ready")
        progress.remove_task(task4)
    
    # Setup complete
    console.print("\n🎉 [green]Agent Hive setup completed successfully![/green]")
    console.print("\n📋 Next steps:")
    console.print("   • agent-hive start     # Start the platform")
    console.print("   • agent-hive status    # Check system status") 
    console.print("   • agent-hive develop   # Start autonomous development")
    console.print("\n🎛️ Dashboard will be available at: http://localhost:8000/dashboard/")


@cli.command()
@click.option('--quick', is_flag=True, help='Quick start in background')
@click.option('--dashboard', is_flag=True, help='Open dashboard after start')
def start(quick: bool, dashboard: bool):
    """
    🚀 Start the Agent Hive platform
    
    Starts all services including the multi-agent system, database,
    and web dashboard.
    """
    console.print("🚀 Starting LeanVibe Agent Hive 2.0...")
    
    # Check if already running
    if cli_instance.check_system_health():
        console.print("✅ [green]Agent Hive is already running[/green]")
        if dashboard:
            webbrowser.open(f"{cli_instance.api_base}/dashboard/")
        return
    
    # Start services
    with console.status("[bold green]Starting services...") as status:
        success = cli_instance.start_services(quick=quick)
        
        if success:
            status.update("[bold green]Waiting for services to be ready...")
            
            # Wait for system to be ready
            for i in range(30):
                if cli_instance.check_system_health():
                    break
                time.sleep(1)
            else:
                console.print("⚠️ [yellow]Services started but health check failed[/yellow]")
    
    if success and cli_instance.check_system_health():
        console.print("✅ [green]Agent Hive started successfully![/green]")
        
        # Show access URLs
        table = Table(title="🌐 Access URLs")
        table.add_column("Service", style="cyan")
        table.add_column("URL", style="green")
        
        pwa_url = cli_instance.check_pwa_dev()
        table.add_row("📱 Mobile PWA", pwa_url or "(start via: cd mobile-pwa && npm run dev)")
        table.add_row("📊 API Docs", f"{cli_instance.api_base}/docs")
        table.add_row("🏥 Health", f"{cli_instance.api_base}/health")
        
        console.print(table)
        
        if dashboard:
            console.print("\n🖥️ Opening dashboard...")
            if pwa_url:
                webbrowser.open(pwa_url)
            else:
                webbrowser.open(f"{cli_instance.api_base}/docs")
    else:
        console.print("❌ [red]Failed to start Agent Hive[/red]")
        console.print("💡 Try: agent-hive setup")


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed status information')
@click.option('--agents', is_flag=True, help='Show agent status only')
def status(detailed: bool, agents: bool):
    """
    📊 Show Agent Hive system status
    
    Displays current status of the platform, services, and agents.
    """
    # System health
    if not agents:
        console.print("🏥 [bold]System Health[/bold]")
        
        if cli_instance.check_system_health():
            system_status = cli_instance.get_system_status()
            
            status_table = Table()
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="green")
            status_table.add_column("Details")
            
            components = system_status.get("components", {})
            for name, info in components.items():
                status_icon = "✅" if info.get("status") == "healthy" else "❌"
                status_table.add_row(
                    name.title(),
                    f"{status_icon} {info.get('status', 'unknown')}",
                    info.get("details", "")
                )
            
            console.print(status_table)
        else:
            console.print("❌ [red]System is not responding[/red]")
            console.print("💡 Try: agent-hive start")
            return
    
    # Agent status
    console.print("\n🤖 [bold]Agent System[/bold]")
    agent_status = cli_instance.get_agent_status()
    
    if agent_status.get("active"):
        agent_count = agent_status.get("agent_count", 0)
        console.print(f"✅ [green]{agent_count} agents active[/green]")
        
        if detailed and "agents" in agent_status:
            agent_table = Table(title="Active Agents")
            agent_table.add_column("Role", style="cyan")
            agent_table.add_column("Status", style="green")
            agent_table.add_column("Tasks")
            agent_table.add_column("Capabilities")
            
            for agent_id, info in agent_status["agents"].items():
                role = info.get("role", "unknown")
                status = info.get("status", "unknown")
                tasks = str(info.get("assigned_tasks", 0))
                capabilities = ", ".join(info.get("capabilities", [])[:3])
                
                agent_table.add_row(role, status, tasks, capabilities + "...")
            
            console.print(agent_table)
    else:
        console.print("❌ [red]No agents active[/red]")
        console.print("💡 Try: agent-hive start")


@cli.command()
@click.argument('project_description')
@click.option('--dashboard', is_flag=True, help='Open dashboard during development')
@click.option('--timeout', default=300, help='Timeout in seconds')
def develop(project_description: str, dashboard: bool, timeout: int):
    """
    💻 Start autonomous development
    
    Launch multi-agent development workflow for the specified project.
    
    Example: agent-hive develop "Build authentication API with JWT"
    """
    console.print(f"💻 [bold]Starting autonomous development[/bold]")
    console.print(f"📋 Project: {project_description}")
    
    # Check if system is running
    if not cli_instance.check_system_health():
        console.print("⚠️ [yellow]System not running, starting now...[/yellow]")
        if not cli_instance.start_services(quick=True):
            console.print("❌ [red]Failed to start system[/red]")
            return
    
    # Execute development command
    command = f"/hive:develop \"{project_description}\""
    if dashboard:
        command += " --dashboard"
    if timeout != 300:
        command += f" --timeout={timeout}"
    
    with console.status(f"[bold green]Agents working on: {project_description}..."):
        result = cli_instance.execute_hive_command(command)
    
    if result.get("success"):
        console.print("✅ [green]Development completed successfully![/green]")
        
        command_result = result.get("result", {})
        if "agents_involved" in command_result:
            console.print(f"👥 Agents involved: {command_result['agents_involved']}")
        
        if dashboard:
            console.print("🎛️ Dashboard: http://localhost:8000/dashboard/")
    else:
        console.print("❌ [red]Development failed[/red]")
        error = result.get("error", "Unknown error")
        console.print(f"💥 Error: {error}")


@cli.command()
@click.option('--mobile-info', is_flag=True, help='Include mobile access information')
def dashboard(mobile_info: bool):
    """
    🎛️ Open the remote oversight dashboard
    
    Opens the web-based dashboard for real-time agent monitoring.
    """
    if not cli_instance.check_system_health():
        console.print("❌ [red]System not running[/red]")
        console.print("💡 Try: agent-hive start")
        return
    
    # Prefer PWA dev if available, otherwise open API docs
    pwa = cli_instance.check_pwa_dev()
    target_url = pwa or f"{cli_instance.api_base}/docs"
    console.print("🎛️ [bold]Opening Agent Hive Dashboard[/bold]")
    console.print(f"🌐 URL: {target_url}")
    
    try:
        webbrowser.open(target_url)
        console.print("✅ [green]Dashboard opened in browser[/green]")
    except Exception as e:
        console.print(f"⚠️ Could not auto-open browser: {e}")
        console.print(f"Please manually visit: {target_url}")
    
    if mobile_info:
        console.print("\n📱 [bold]Mobile Access[/bold]")
        console.print("Use the same URL on your mobile device for remote oversight")
        console.print("Features: Real-time monitoring, agent status, task progress")


@cli.command()
@click.option('--open', 'open_browser', is_flag=True, help='Open PWA if detected')
def up(open_browser: bool):
    """
    ⬆️ Start services quickly and optionally open the PWA.
    """
    start(quick=True, dashboard=open_browser)


@cli.command()
def down():
    """
    ⬇️ Stop all services via Makefile/docker compose.
    """
    try:
        subprocess.run(["make", "stop"], check=True)
        console.print("✅ [green]All services stopped[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ [red]Failed to stop services: {e}[/red]")


@cli.command()
def doctor():
    """
    🩺 Diagnose environment and provide actionable fixes.
    """
    checks = []
    def add(name, ok, hint=""):
        checks.append((name, ok, hint))

    # Tooling
    for tool in ["python3", "git", "docker", "node", "npm"]:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            add(f"{tool}", True)
        except Exception:
            add(f"{tool}", False, f"Install {tool}")

    # Ports
    def port_free(host, port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex((host, port)) != 0
    add("Port 8000 available", port_free("127.0.0.1", 8000), "API port in use")
    add("Port 5173 optional (PWA)", True if port_free("127.0.0.1", 5173) or cli_instance.check_pwa_dev() else False, "PWA dev port busy")

    # Backend health
    add("Backend health", cli_instance.check_system_health(), "Run: make dev or agent-hive start")

    # WS metrics endpoint
    try:
        r = requests.get(f"{cli_instance.api_base}/api/dashboard/metrics/websockets", timeout=2)
        add("WS metrics endpoint", r.status_code == 200, "Check backend routes")
    except Exception:
        add("WS metrics endpoint", False, "Backend not reachable or route missing")

    # Render
    table = Table(title="Environment Diagnostics")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Hint")
    for name, ok, hint in checks:
        table.add_row(name, "✅" if ok else "❌", hint)
    console.print(table)


@cli.command()
@click.option('--agents-only', is_flag=True, help='Stop only agents, keep services')
@click.option('--force', is_flag=True, help='Force shutdown')
def stop(agents_only: bool, force: bool):
    """
    🛑 Stop the Agent Hive platform
    
    Gracefully stops all services and agents.
    """
    console.print("🛑 [bold]Stopping Agent Hive...[/bold]")
    
    command = "/hive:stop"
    if agents_only:
        command += " --agents-only"
    if force:
        command += " --force"
    
    result = cli_instance.execute_hive_command(command)
    
    if result.get("success"):
        console.print("✅ [green]Agent Hive stopped successfully[/green]")
    else:
        console.print("⚠️ [yellow]Stop command completed with issues[/yellow]")


@cli.command()
@click.option('--show-all', is_flag=True, help='Show all configuration details')
def config(show_all: bool):
    """
    ⚙️ Show Agent Hive configuration
    
    Displays current configuration and settings.
    """
    config = cli_instance.config.load_config()
    
    if show_all:
        console.print("⚙️ [bold]Agent Hive Configuration[/bold]")
        console.print(json.dumps(config, indent=2))
    else:
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Version", config.get("version", "unknown"))
        table.add_row("API Base", config.get("api_base", "unknown"))
        table.add_row("Config Dir", str(cli_instance.config.config_dir))
        table.add_row("Setup Complete", "✅" if config.get("setup_completed") else "❌")
        
        console.print(table)


@cli.command()
@click.option('--check', is_flag=True, help='Check for updates without installing')
@click.option('--channel', type=click.Choice(['stable', 'beta', 'dev', 'homebrew']), 
              default='stable', help='Update channel to use')
@click.option('--force', is_flag=True, help='Force update without safety checks')
@click.option('--yes', is_flag=True, help='Skip confirmation prompts')
def update(check: bool, channel: str, force: bool, yes: bool):
    """
    🔄 Update Agent Hive to the latest version
    
    Supports multiple update channels:
    - stable: PyPI releases (default)
    - beta: GitHub pre-releases  
    - dev: Git repository development builds
    - homebrew: System package manager
    """
    # Map string to enum
    channel_map = {
        'stable': UpdateChannel.STABLE,
        'beta': UpdateChannel.BETA,
        'dev': UpdateChannel.DEVELOPMENT,
        'homebrew': UpdateChannel.HOMEBREW
    }
    update_channel = channel_map[channel]
    
    console.print(f"🔄 [bold]Agent Hive Update Manager[/bold]")
    console.print(f"📡 Channel: {update_channel.value}")
    
    # Check for updates
    update_info = cli_instance.updater.check_for_updates(update_channel)
    
    if not update_info:
        console.print("✅ [green]You're already running the latest version![/green]")
        return
    
    # Display update information
    update_table = Table(title="Available Update")
    update_table.add_column("Property", style="cyan")
    update_table.add_column("Value", style="green")
    
    update_table.add_row("Current Version", cli_instance.updater.current_version)
    update_table.add_row("New Version", update_info.version)
    update_table.add_row("Channel", update_info.channel.value)
    update_table.add_row("Release Date", update_info.release_date)
    
    if update_info.changelog_url:
        update_table.add_row("Changelog", update_info.changelog_url)
    
    console.print(update_table)
    
    # If just checking, stop here
    if check:
        console.print("\n💡 Run 'agent-hive update' to install this update")
        return
    
    # Confirm update
    if not yes:
        if not click.confirm(f"\nProceed with update to {update_info.version}?"):
            console.print("Update cancelled")
            return
    
    # Perform update
    console.print("\n🚀 Starting update process...")
    result = cli_instance.updater.perform_update(update_info, force=force)
    
    if result.success:
        console.print(Panel.fit(
            f"✅ [bold green]Update Successful![/bold green]\n"
            f"Agent Hive updated from {result.old_version} to {result.new_version}\n"
            f"Channel: {result.channel.value}",
            border_style="green"
        ))
        
        console.print("\n💡 You may need to restart any running Agent Hive services")
        console.print("   Run: agent-hive restart")
    else:
        console.print(Panel.fit(
            f"❌ [bold red]Update Failed[/bold red]\n"
            f"Error: {result.error_message}\n"
            f"{'Rollback completed' if result.rollback_available else 'Manual recovery may be needed'}",
            border_style="red"
        ))


@cli.command()
@click.option('--detailed', is_flag=True, help='Show detailed version information')
@click.option('--check-updates', is_flag=True, help='Check for available updates')
@click.option('--history', is_flag=True, help='Show update history')
def version(detailed: bool, check_updates: bool, history: bool):
    """
    📋 Show version information and update status
    
    Display current version, build information, and optionally check for updates.
    """
    console.print("📋 [bold]Agent Hive Version Information[/bold]")
    
    if history:
        # Show update history
        update_history = cli_instance.updater.get_update_history()
        
        if update_history:
            history_table = Table(title="Update History")
            history_table.add_column("Date", style="cyan")
            history_table.add_column("From", style="yellow")
            history_table.add_column("To", style="green")
            history_table.add_column("Channel", style="blue")
            history_table.add_column("Status", style="green")
            
            for entry in update_history[-10:]:  # Show last 10 updates
                date = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry["timestamp"]))
                status = "✅ Success" if entry.get("success") else "❌ Failed"
                
                history_table.add_row(
                    date,
                    entry["old_version"],
                    entry["new_version"],
                    entry["channel"],
                    status
                )
            
            console.print(history_table)
        else:
            console.print("📝 No update history available")
        
        return
    
    # Basic version information
    version_table = Table()
    version_table.add_column("Property", style="cyan")
    version_table.add_column("Value", style="green")
    
    version_table.add_row("Version", cli_instance.updater.current_version)
    version_table.add_row("Installation", "PyPI Package" if detailed else "Standard")
    
    if detailed:
        try:
            # Add more detailed information
            import app
            version_table.add_row("Python Version", sys.version.split()[0])
            version_table.add_row("Platform", sys.platform)
            version_table.add_row("Installation Path", str(Path(app.__file__).parent))
            
            # Git information if available
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    capture_output=True, text=True, cwd=Path(app.__file__).parent.parent
                )
                if result.returncode == 0:
                    version_table.add_row("Git Commit", result.stdout.strip())
            except:
                pass
                
        except:
            pass
    
    console.print(version_table)
    
    # Check for updates if requested
    if check_updates:
        console.print("\n🔍 Checking for updates...")
        
        for channel in [UpdateChannel.STABLE, UpdateChannel.BETA]:
            update_info = cli_instance.updater.check_for_updates(channel)
            if update_info:
                console.print(f"📦 Update available on {channel.value}: v{update_info.version}")
                console.print(f"   Run: agent-hive update --channel={channel.value}")
                break
        else:
            console.print("✅ No updates available")


@cli.command()
@click.option('--channel', type=click.Choice(['stable', 'beta', 'dev']), 
              default='stable', help='Channel to configure for auto-updates')
@click.option('--enable/--disable', default=True, help='Enable or disable auto-update notifications')
@click.option('--frequency', type=click.Choice(['daily', 'weekly', 'never']), 
              default='daily', help='Auto-check frequency')
def auto_update(channel: str, enable: bool, frequency: str):
    """
    ⚙️ Configure automatic update checking and notifications
    
    Set up automatic update checking to stay informed about new releases.
    """
    console.print("⚙️ [bold]Auto-Update Configuration[/bold]")
    
    config = cli_instance.config.load_config()
    
    # Update auto-update settings
    config.setdefault("auto_update", {})
    config["auto_update"]["enabled"] = enable
    config["auto_update"]["channel"] = channel
    config["auto_update"]["frequency"] = frequency
    
    cli_instance.config.save_config(config)
    
    if enable:
        console.print(f"✅ [green]Auto-update notifications enabled[/green]")
        console.print(f"📡 Channel: {channel}")
        console.print(f"⏰ Frequency: {frequency}")
        console.print("\nAgent Hive will check for updates and notify you when new versions are available.")
    else:
        console.print("❌ [yellow]Auto-update notifications disabled[/yellow]")
        console.print("You can manually check for updates with 'agent-hive update --check'")


@cli.command()
@click.option('--version', help='Specific version to rollback to')
@click.option('--steps', type=int, default=1, help='Number of versions to rollback')
@click.option('--list', 'list_versions', is_flag=True, help='List available rollback versions')
@click.option('--yes', is_flag=True, help='Skip confirmation prompts')
def rollback(version: str, steps: int, list_versions: bool, yes: bool):
    """
    ⏪ Rollback to a previous version
    
    Safely rollback to a previous version using update history and backups.
    """
    console.print("⏪ [bold]Agent Hive Rollback Manager[/bold]")
    
    if list_versions:
        # Show available versions for rollback
        history = cli_instance.updater.get_update_history()
        
        if history:
            rollback_table = Table(title="Available Rollback Versions")
            rollback_table.add_column("Version", style="cyan")
            rollback_table.add_column("Date", style="yellow")
            rollback_table.add_column("Channel", style="blue")
            
            # Show unique versions from history
            seen_versions = set()
            for entry in reversed(history[-20:]):  # Last 20 entries
                ver = entry["old_version"]
                if ver not in seen_versions and ver != cli_instance.updater.current_version:
                    seen_versions.add(ver)
                    date = time.strftime("%Y-%m-%d", time.localtime(entry["timestamp"]))
                    rollback_table.add_row(ver, date, entry["channel"])
            
            console.print(rollback_table)
            console.print("\n💡 Use --version to rollback to a specific version")
        else:
            console.print("📝 No rollback history available")
        
        return
    
    # Implement rollback logic
    console.print("🚧 [yellow]Rollback functionality is under development[/yellow]")
    console.print("💡 For now, you can reinstall a specific version:")
    console.print("   pip install leanvibe-agent-hive==<version>")
    console.print("   agent-hive setup  # Re-run setup after downgrade")


@cli.command()
@click.option('--report', is_flag=True, help='Show detailed port usage report')
@click.option('--scan', is_flag=True, help='Scan for port conflicts')
@click.option('--fix', is_flag=True, help='Suggest fixes for port conflicts')
@click.option('--validate', is_flag=True, help='Validate current port configuration')
def ports(report: bool, scan: bool, fix: bool, validate: bool):
    """
    🔌 Manage and monitor port configuration
    
    View port status, detect conflicts, and manage service ports.
    """
    try:
        # Import port management
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root / "config"))
        
        from port_management import PortManager, PortStatus
        
        manager = PortManager()
        
        if report or (not scan and not fix and not validate):
            # Default action: show report
            console.print("🔌 [bold blue]LeanVibe Port Configuration Report[/bold blue]")
            print(manager.generate_status_report())
            
        elif scan:
            console.print("🔍 [bold]Scanning for port conflicts...[/bold]")
            results = manager.scan_all_ports()
            conflicts = manager.find_conflicts()
            
            if conflicts:
                console.print("⚠️ [yellow]Port conflicts detected:[/yellow]")
                for service1, service2 in conflicts:
                    port = manager.ports[service1].port
                    console.print(f"   • {service1} and {service2} both use port {port}")
            else:
                console.print("✅ [green]No port conflicts found[/green]")
                
            # Show ports in use
            in_use = [(name, config) for name, config in manager.ports.items() 
                     if config.status == PortStatus.IN_USE]
            if in_use:
                console.print(f"\n🔴 {len(in_use)} ports currently in use:")
                for name, config in in_use[:5]:  # Show first 5
                    console.print(f"   • {config.port} - {config.description}")
                if len(in_use) > 5:
                    console.print(f"   ... and {len(in_use) - 5} more")
        
        elif fix:
            conflicts = manager.find_conflicts()
            if conflicts:
                console.print("🔧 [bold]Suggested fixes for port conflicts:[/bold]")
                suggestions = manager.suggest_port_fixes()
                for service, new_port in suggestions.items():
                    old_port = manager.ports[service].port
                    console.print(f"   {service}: {old_port} → {new_port}")
                    
                console.print("\n💡 To apply fixes, update your .env.ports file manually")
            else:
                console.print("✅ [green]No port conflicts to fix[/green]")
        
        elif validate:
            is_valid, issues = manager.validate_configuration()
            if is_valid:
                console.print("✅ [green]Port configuration is valid[/green]")
            else:
                console.print("❌ [red]Port configuration has issues:[/red]")
                for issue in issues:
                    console.print(f"   • {issue}")
                    
        console.print(f"\n📋 Configuration file: {manager.config_file}")
        console.print("💡 Edit .env.ports to modify port assignments")
        
    except Exception as e:
        console.print(f"❌ [red]Error managing ports: {e}[/red]")
        console.print("💡 Ensure the port management system is properly configured")


def main():
    """Main entry point for the CLI."""
    cli()


# REFACTORED: Import shared patterns to eliminate simple main() duplication  
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.common.utilities.shared_patterns import simple_main_wrapper


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class CliScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            # REFACTORED: Use shared simple_main_wrapper instead of direct main() call
            # This replaces: main()
            # With standardized error handling and script name tracking
            simple_main_wrapper(main, "agent-hive-cli")
            
            return {"status": "completed"}
    
    script_main(CliScript)
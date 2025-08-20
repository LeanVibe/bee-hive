#!/usr/bin/env python3
"""
LeanVibe Agent Hive - Unified CLI
Professional command-line interface following Unix philosophies.

Usage:
    hive <command> [options]
    
Examples:
    hive start                    # Start the system
    hive status                   # Show system status
    hive agent list               # List all agents
    hive agent deploy backend     # Deploy a backend agent
    hive logs -f                  # Follow logs
    hive --help                   # Show help
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import asyncio
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    # Import our existing CLI components
    from .cli.unix_commands import unix_commands
    # Import from project root
    from deploy_agent_cli import AgentDeploymentCLI

    console = Console()

    @click.group()
    @click.version_option(version="2.0.0", prog_name="hive")
    def hive():
        """🤖 LeanVibe Agent Hive - Autonomous Development Platform
        
        Professional multi-agent development system following Unix philosophies.
        
        Core Commands:
          start      Start the platform services
          stop       Stop all services  
          status     Show system status
          logs       View system logs
          
        Agent Management:
          agent      Agent lifecycle commands
          task       Task management
          workflow   Workflow operations
          
        Monitoring:
          dashboard  Open monitoring dashboard
          metrics    View system metrics
          doctor     System diagnostics
        """
        pass

    # === SYSTEM COMMANDS ===
    @hive.command()
    @click.option('--background', '-d', is_flag=True, help='Run in background')
    def start(background):
        """🚀 Start the Agent Hive platform services"""
        console.print("🚀 [bold blue]Starting LeanVibe Agent Hive 2.0...[/bold blue]")
        
        # Check if FastAPI server is running (using non-standard port)
        api_port = os.getenv("API_PORT", "18080")
        try:
            import requests
            response = requests.get(f"http://localhost:{api_port}/health", timeout=2)
            if response.status_code == 200:
                console.print("✅ [green]System is already running[/green]")
                console.print(f"🌐 API: http://localhost:{api_port}")
                console.print(f"📊 Docs: http://localhost:{api_port}/docs")
                return
        except:
            pass
        
        # Start the FastAPI server with non-standard port
        if background:
            console.print("🔄 Starting services in background...")
            subprocess.Popen([
                sys.executable, "-m", "uvicorn", "app.main:app", 
                "--host", "0.0.0.0", "--port", api_port, "--reload"
            ], cwd=project_root)
            console.print("✅ Services started in background")
            console.print(f"🌐 API will be available at: http://localhost:{api_port}")
        else:
            console.print("🔄 Starting services...")
            console.print("💡 Press Ctrl+C to stop")
            try:
                subprocess.run([
                    sys.executable, "-m", "uvicorn", "app.main:app",
                    "--host", "0.0.0.0", "--port", api_port, "--reload"
                ], cwd=project_root)
            except KeyboardInterrupt:
                console.print("\n🛑 Stopping services...")

    @hive.command()
    def stop():
        """🛑 Stop all platform services"""
        console.print("🛑 [bold red]Stopping Agent Hive services...[/bold red]")
        
        # Find and kill uvicorn processes
        try:
            result = subprocess.run(
                ["pkill", "-f", "uvicorn.*app.main:app"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                console.print("✅ [green]Services stopped successfully[/green]")
            else:
                console.print("ℹ️ [yellow]No running services found[/yellow]")
        except FileNotFoundError:
            console.print("⚠️ [yellow]pkill not available, services may still be running[/yellow]")

    @hive.command()
    @click.option('--watch', '-w', is_flag=True, help='Watch for changes')
    @click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
    def status(watch, output_json):
        """📊 Show system and agent status"""
        deployment_cli = AgentDeploymentCLI()
        
        if watch:
            import time
            console.print("👁️ Watching system status (Press Ctrl+C to stop)...")
            try:
                while True:
                    click.clear()
                    console.print(f"🕒 {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    asyncio.run(deployment_cli.system_status())
                    time.sleep(2)
            except KeyboardInterrupt:
                console.print("\n🛑 Stopped watching")
                return
        else:
            asyncio.run(deployment_cli.system_status())

    @hive.command()
    @click.option('--follow', '-f', is_flag=True, help='Follow log output')
    @click.option('--lines', '-n', default=50, help='Number of lines to show')
    def logs(follow, lines):
        """📋 View system logs"""
        console.print("📋 [bold]System Logs[/bold]")
        
        if follow:
            console.print("👁️ Following logs (Press Ctrl+C to stop)...")
            try:
                # In a real implementation, this would stream from log files
                subprocess.run(["tail", "-f", "/var/log/agent-hive.log"])
            except (FileNotFoundError, KeyboardInterrupt):
                console.print("📝 [yellow]Log streaming not available[/yellow]")
        else:
            console.print(f"📄 Last {lines} log entries:")
            console.print("ℹ️ [blue]Log aggregation system coming soon[/blue]")

    # === AGENT MANAGEMENT ===
    @hive.group()
    def agent():
        """🤖 Agent lifecycle management"""
        pass

    @agent.command('list')
    @click.option('--format', '-o', type=click.Choice(['table', 'json']), default='table')
    def agent_list(format):
        """📋 List all agents (alias: ls)"""
        deployment_cli = AgentDeploymentCLI()
        asyncio.run(deployment_cli.list_agents())

    @agent.command('ls')
    @click.option('--format', '-o', type=click.Choice(['table', 'json']), default='table') 
    def agent_ls(format):
        """📋 List all agents (alias for list)"""
        deployment_cli = AgentDeploymentCLI()
        asyncio.run(deployment_cli.list_agents())

    @agent.command('deploy')
    @click.argument('role', type=click.Choice(['backend-developer', 'frontend-developer', 'qa-engineer', 'devops-engineer', 'meta-agent']))
    @click.option('--task', '-t', default='Autonomous development task')
    @click.option('--name', help='Custom agent name')
    def agent_deploy(role, task, name):
        """🚀 Deploy a new agent"""
        deployment_cli = AgentDeploymentCLI()
        
        console.print(f"🚀 [bold blue]Deploying {role} agent...[/bold blue]")
        agent_id = asyncio.run(deployment_cli.deploy_agent(role, task, True))
        
        if agent_id:
            console.print(f"✅ [green]Agent deployed: {agent_id}[/green]")
        else:
            console.print("❌ [red]Deployment failed[/red]")

    @agent.command('run')
    @click.argument('role', type=click.Choice(['backend-developer', 'frontend-developer', 'qa-engineer', 'devops-engineer', 'meta-agent']))
    @click.option('--task', '-t', default='Autonomous development task')
    def agent_run(role, task):
        """🏃 Run an agent (alias for deploy)"""
        deployment_cli = AgentDeploymentCLI()
        agent_id = asyncio.run(deployment_cli.deploy_agent(role, task, True))
        
        if agent_id:
            console.print(f"✅ [green]Agent running: {agent_id}[/green]")

    @agent.command('ps')
    def agent_ps():
        """📊 Show running agents (docker ps style)"""
        deployment_cli = AgentDeploymentCLI()
        asyncio.run(deployment_cli.list_agents())

    # === QUICK ACTIONS ===
    @hive.command()
    def up():
        """⬆️ Quick start (docker-compose up style)"""
        console.print("⬆️ [bold blue]Quick starting Agent Hive...[/bold blue]")
        start.callback(background=True)

    @hive.command()
    def down():
        """⬇️ Quick stop (docker-compose down style)"""
        stop.callback()

    @hive.command()
    @click.option('--demo', is_flag=True, help='Run demonstration mode')
    def dashboard(demo):
        """🎛️ Open monitoring dashboard"""
        import webbrowser
        
        console.print("🎛️ [bold]Opening Agent Hive Dashboard[/bold]")
        
        # Use non-standard ports
        api_port = os.getenv("API_PORT", "18080")
        pwa_port = os.getenv("PWA_DEV_PORT", "18443")
        
        try:
            # Check if PWA is running (on non-standard port)
            import requests
            response = requests.get(f"http://localhost:{pwa_port}", timeout=2)
            if response.status_code == 200:
                console.print("📱 Opening PWA Dashboard...")
                webbrowser.open(f"http://localhost:{pwa_port}")
            else:
                console.print("🌐 Opening API Dashboard...")
                webbrowser.open(f"http://localhost:{api_port}/docs")
        except:
            console.print("🌐 Opening API Dashboard...")
            webbrowser.open(f"http://localhost:{api_port}/docs")

    @hive.command()
    def demo():
        """🎭 Run complete system demonstration"""
        deployment_cli = AgentDeploymentCLI()
        
        console.print("🎭 [bold blue]Running Agent Hive Demo[/bold blue]")
        console.print("This will deploy multiple agents and show system capabilities")
        
        if click.confirm("Continue with demo?"):
            # Run the demo from deployment CLI
            import inspect
            demo_method = getattr(deployment_cli.__class__, 'demo_async', None)
            if demo_method:
                # Get the demo implementation from deploy_agent_cli
                asyncio.run(demo_agent_deployment())

    async def demo_agent_deployment():
        """Demo function for agent deployment"""
        deployment_cli = AgentDeploymentCLI()
        
        console.print("🔄 [bold]Phase 1: Deploying Backend Developer[/bold]")
        backend_agent = await deployment_cli.deploy_agent(
            'backend-developer', 
            'Implement missing PWA backend API endpoints',
            True
        )
        
        console.print("🔄 [bold]Phase 2: Deploying QA Engineer[/bold]")
        qa_agent = await deployment_cli.deploy_agent(
            'qa-engineer',
            'Create comprehensive tests for new backend endpoints',
            True
        )
        
        console.print("🔄 [bold]Phase 3: Deploying Meta-Agent[/bold]")
        meta_agent = await deployment_cli.deploy_agent(
            'meta-agent',
            'Analyze system architecture and recommend optimizations',
            True
        )
        
        console.print("\n🎉 [green]Demo Complete![/green]")
        console.print("Deployed agents:")
        if backend_agent: console.print(f"  - Backend Developer: {backend_agent}")
        if qa_agent: console.print(f"  - QA Engineer: {qa_agent}")
        if meta_agent: console.print(f"  - Meta-Agent: {meta_agent}")
        
        await deployment_cli.system_status()

    @hive.command()
    def doctor():
        """🩺 System diagnostics and health check"""
        console.print("🩺 [bold]Agent Hive System Diagnostics[/bold]")
        
        checks = []
        
        # Check Python and dependencies
        console.print("\n🐍 Python Environment:")
        console.print(f"  Python: {sys.version.split()[0]} ✅")
        
        # Check required packages
        required_packages = ['fastapi', 'uvicorn', 'click', 'rich', 'pydantic']
        for package in required_packages:
            try:
                __import__(package)
                console.print(f"  {package}: ✅")
            except ImportError:
                console.print(f"  {package}: ❌ Missing")
        
        # Check ports (using non-standard ports)
        console.print("\n🔌 Port Status:")
        import socket
        api_port = int(os.getenv("API_PORT", "18080"))
        postgres_port = int(os.getenv("POSTGRES_PORT", "15432"))
        redis_port = int(os.getenv("REDIS_PORT", "16379"))
        pwa_port = int(os.getenv("PWA_DEV_PORT", "18443"))
        
        ports_to_check = {
            api_port: "API Server",
            postgres_port: "PostgreSQL", 
            redis_port: "Redis",
            pwa_port: "PWA Dev Server"
        }
        
        for port, service in ports_to_check.items():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                result = s.connect_ex(('127.0.0.1', port))
                if result == 0:
                    console.print(f"  Port {port} ({service}): 🔴 In use")
                else:
                    console.print(f"  Port {port} ({service}): 🟢 Available")
        
        # Check system health (using non-standard port)
        console.print("\n🏥 System Health:")
        try:
            import requests
            response = requests.get(f"http://localhost:{api_port}/health", timeout=2)
            if response.status_code == 200:
                console.print("  API Health: ✅ Healthy")
            else:
                console.print("  API Health: ⚠️ Issues detected")
        except:
            console.print("  API Health: ❌ Not running")
        
        console.print("\n💡 [blue]Recommendations:[/blue]")
        console.print("  • Run 'hive start' to start services")
        console.print("  • Run 'hive agent deploy backend' to deploy agents")
        console.print("  • Run 'hive dashboard' to open monitoring")

    @hive.command()
    def version():
        """📋 Show version information"""
        console.print("📋 [bold]LeanVibe Agent Hive Version Information[/bold]")
        
        version_table = Table()
        version_table.add_column("Component", style="cyan")
        version_table.add_column("Version", style="green")
        
        version_table.add_row("Agent Hive", "2.0.0")
        version_table.add_row("Python", sys.version.split()[0])
        version_table.add_row("Platform", sys.platform)
        
        console.print(version_table)

    def main():
        """Main entry point for the hive CLI."""
        hive()

    # Main entry point
    if __name__ == "__main__":
        main()

except ImportError as e:
    # Fallback for minimal dependencies
    print(f"🚨 Missing dependencies: {e}")
    print("💡 Run: pip install fastapi uvicorn click rich pydantic")
    sys.exit(1)
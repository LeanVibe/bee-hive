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

    # Import performance caching system for <500ms response times
    from .cli.performance_cache import (
        get_cached_config, 
        get_cached_orchestrator, 
        get_cached_initialized_orchestrator,
        CLIPerformanceTracker,
        measure_cli_performance
    )

    # Import our existing CLI components
    try:
        from .cli.unix_commands import HiveContext, ctx
        unix_commands_available = True
    except ImportError:
        from pathlib import Path
        import requests
        from dataclasses import dataclass
        
        @dataclass
        class HiveContext:
            api_base: str = "http://localhost:18080"
            config_dir: Path = Path.home() / ".config" / "agent-hive"
            
            def __post_init__(self):
                self.config_dir.mkdir(parents=True, exist_ok=True)
            
            def api_call(self, endpoint: str, method: str = "GET", data: dict = None):
                try:
                    url = f"{self.api_base}/{endpoint.lstrip('/')}"
                    if method == "GET":
                        response = requests.get(url, timeout=5)
                    elif method == "POST":
                        response = requests.post(url, json=data, timeout=5)
                    else:
                        response = requests.request(method, url, json=data, timeout=5)
                    
                    if response.status_code == 200:
                        return response.json() if response.content else {}
                    return None
                except Exception:
                    return None
        
        ctx = HiveContext()
        unix_commands_available = False
    
    # Import from project root
    try:
        from deploy_agent_cli import AgentDeploymentCLI
        agent_cli_available = True
    except ImportError:
        agent_cli_available = False
    
    # Import existing CLI modules
    try:
        from .cli.project_management_commands import project_management
        project_management_available = True
    except ImportError:
        project_management_available = False
    
    try:
        from .cli.enhanced_project_commands import enhanced_project_commands
        enhanced_project_available = True
    except ImportError:
        enhanced_project_available = False
    
    try:
        from .cli.short_id_commands import short_id
        short_id_available = True
    except ImportError:
        short_id_available = False
    
    try:
        from .cli.agent_session_commands import agent as agent_session_commands
        agent_session_available = True
    except ImportError:
        agent_session_available = False

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

    @hive.command("status")
    @click.option('--watch', '-w', is_flag=True, help='Watch for changes')
    @click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
    @measure_cli_performance("status")
    def status_cmd(watch, output_json):
        """📊 Show system and agent status - optimized for <500ms response time"""
        with CLIPerformanceTracker("hive status"):
            if agent_cli_available:
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
                    # Use cached orchestrator for improved performance
                    try:
                        orchestrator = get_cached_orchestrator()
                        if orchestrator:
                            status_info = asyncio.run(orchestrator.get_system_status())
                            
                            # Display enhanced status with caching metrics
                            console.print("╭────────────────────────────── 🖥️  System Status ──────────────────────────────╮")
                            console.print("│                                                                              │")
                            console.print(f"│ System Health: {status_info.get('health', 'unknown')}                     │")
                            console.print(f"│ Total Agents: {status_info.get('agents', {}).get('total', 0)}                   │")
                            console.print(f"│ Timestamp: {status_info.get('timestamp', 'unknown')}      │")
                            console.print(f"│ Performance: {status_info.get('performance', {}).get('response_time_ms', 'unknown')}ms response    │")
                            console.print("│                                                                              │")
                            console.print("╰──────────────────────────────────────────────────────────────────────────────╯")
                            
                            if output_json:
                                import json
                                print(json.dumps(status_info, indent=2))
                        else:
                            # Fallback to deployment CLI
                            asyncio.run(deployment_cli.system_status())
                    except Exception as e:
                        console.print(f"⚠️ Cached status failed ({e}), falling back...")
                        asyncio.run(deployment_cli.system_status())
            else:
                # Fallback status check using cached config
                try:
                    config_service = get_cached_config()
                    if config_service:
                        api_port = config_service.get_settings().api_prefix or "18080"
                    else:
                        api_port = os.getenv("API_PORT", "18080")
                    
                    import requests
                    response = requests.get(f"http://localhost:{api_port}/health", timeout=2)
                    if response.status_code == 200:
                        console.print("✅ [green]System is running[/green]")
                        if output_json:
                            import json
                            print(json.dumps({"status": "healthy", "api_port": api_port}))
                    else:
                        console.print("❌ [red]System is not responding[/red]")
                except:
                    console.print("❌ [red]System is not running[/red]")
                    console.print(f"💡 Run 'hive start' to start services")

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
    @measure_cli_performance("agent list")
    def agent_list(format):
        """📋 List all agents (alias: ls) - optimized for <500ms response time"""
        with CLIPerformanceTracker("hive agent list"):
            try:
                # Use cached orchestrator for fast agent listing
                orchestrator = get_cached_orchestrator()
                if orchestrator:
                    status_info = asyncio.run(orchestrator.get_system_status())
                    agents = status_info.get('agents', {})
                    
                    if agents.get('total', 0) == 0:
                        console.print("No agents currently deployed")
                    else:
                        if format == 'json':
                            import json
                            print(json.dumps(agents, indent=2))
                        else:
                            # Table format
                            table = Table()
                            table.add_column("Agent ID", style="cyan")
                            table.add_column("Status", style="green")
                            table.add_column("Role", style="yellow")
                            table.add_column("Created", style="dim")
                            
                            for agent_id, agent_info in agents.get('details', {}).items():
                                table.add_row(
                                    agent_id[:8] + "...",
                                    agent_info.get('status', 'unknown'),
                                    agent_info.get('role', 'unknown'),
                                    agent_info.get('created_at', 'unknown')[:19]
                                )
                            
                            console.print(table)
                    return
                
                # Fallback to deployment CLI if orchestrator not available
                if agent_cli_available:
                    deployment_cli = AgentDeploymentCLI()
                    asyncio.run(deployment_cli.list_agents())
                else:
                    console.print("[yellow]Agent CLI not available[/yellow]")
                    console.print("Available agents: Run 'hive doctor' for system status")
                    
            except Exception as e:
                console.print(f"⚠️ Fast agent list failed ({e}), falling back...")
                if agent_cli_available:
                    deployment_cli = AgentDeploymentCLI()
                    asyncio.run(deployment_cli.list_agents())
                else:
                    console.print("[yellow]Agent CLI not available[/yellow]")

    @agent.command('ls')
    @click.option('--format', '-o', type=click.Choice(['table', 'json']), default='table') 
    @measure_cli_performance("agent ls")
    def agent_ls(format):
        """📋 List all agents (alias for list) - optimized for <500ms response time"""
        with CLIPerformanceTracker("hive agent ls"):
            # Reuse the same optimized logic as agent_list
            agent_list.callback(format)

    @agent.command('deploy')
    @click.argument('role', type=click.Choice(['backend-developer', 'frontend-developer', 'qa-engineer', 'devops-engineer', 'meta-agent']))
    @click.option('--task', '-t', default='Autonomous development task')
    @click.option('--name', help='Custom agent name')
    def agent_deploy(role, task, name):
        """🚀 Deploy a new agent"""
        if agent_cli_available:
            deployment_cli = AgentDeploymentCLI()
            
            console.print(f"🚀 [bold blue]Deploying {role} agent...[/bold blue]")
            agent_id = asyncio.run(deployment_cli.deploy_agent(role, task, True))
            
            if agent_id:
                console.print(f"✅ [green]Agent deployed: {agent_id}[/green]")
            else:
                console.print("❌ [red]Deployment failed[/red]")
        else:
            console.print("[red]Agent deployment CLI not available[/red]")
            console.print("💡 Make sure the system is properly configured")

    @agent.command('run')
    @click.argument('role', type=click.Choice(['backend-developer', 'frontend-developer', 'qa-engineer', 'devops-engineer', 'meta-agent']))
    @click.option('--task', '-t', default='Autonomous development task')
    def agent_run(role, task):
        """🏃 Run an agent (alias for deploy)"""
        if agent_cli_available:
            deployment_cli = AgentDeploymentCLI()
            agent_id = asyncio.run(deployment_cli.deploy_agent(role, task, True))
            
            if agent_id:
                console.print(f"✅ [green]Agent running: {agent_id}[/green]")
        else:
            console.print("[red]Agent deployment CLI not available[/red]")

    @agent.command('ps')
    def agent_ps():
        """📊 Show running agents (docker ps style)"""
        if agent_cli_available:
            deployment_cli = AgentDeploymentCLI()
            asyncio.run(deployment_cli.list_agents())
        else:
            console.print("[yellow]Agent CLI not available[/yellow]")

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
        if agent_cli_available:
            deployment_cli = AgentDeploymentCLI()
            
            console.print("🎭 [bold blue]Running Agent Hive Demo[/bold blue]")
            console.print("This will deploy multiple agents and show system capabilities")
            
            if click.confirm("Continue with demo?"):
                # Run the demo from deployment CLI
                asyncio.run(demo_agent_deployment())
        else:
            console.print("[red]Demo requires agent deployment CLI[/red]")
            console.print("💡 Make sure the system is properly configured")

    async def demo_agent_deployment():
        """Demo function for agent deployment"""
        if agent_cli_available:
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
        else:
            console.print("[red]Demo requires agent deployment CLI[/red]")

    @hive.command()
    @click.option('--detailed', is_flag=True, help='Show detailed performance metrics')
    def metrics(detailed):
        """📊 Show CLI performance metrics and cache statistics"""
        from .cli.performance_cache import get_cli_performance_metrics
        
        try:
            metrics_data = get_cli_performance_metrics()
            
            console.print("📊 [bold]CLI Performance Metrics[/bold]")
            console.print()
            
            # Cache statistics
            cache_stats = metrics_data.get('cache_stats', {})
            console.print("🚀 [bold]Cache Performance:[/bold]")
            console.print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
            console.print(f"  Total Requests: {cache_stats.get('total_requests', 0)}")
            console.print(f"  Cache Size: {cache_stats.get('cache_size', 0)} entries")
            
            # Orchestrator statistics  
            orch_stats = metrics_data.get('orchestrator_stats', {})
            console.print("\n🤖 [bold]Orchestrator Performance:[/bold]")
            console.print(f"  Cached: {'✅' if orch_stats.get('cached') else '❌'}")
            console.print(f"  Healthy: {'✅' if orch_stats.get('healthy') else '❌'}")
            if orch_stats.get('init_time_ms'):
                console.print(f"  Init Time: {orch_stats['init_time_ms']:.1f}ms")
            
            # System statistics
            sys_stats = metrics_data.get('system_stats', {})
            console.print(f"\n💻 [bold]System Resources:[/bold]")
            console.print(f"  Memory: {sys_stats.get('memory_usage_mb', 0):.1f}MB")
            console.print(f"  CPU: {sys_stats.get('cpu_percent', 0):.1f}%")
            console.print(f"  Uptime: {sys_stats.get('uptime_seconds', 0):.1f}s")
            
            # Performance targets
            targets = metrics_data.get('performance_targets', {})
            console.print(f"\n🎯 [bold]Performance Targets:[/bold]")
            console.print(f"  Target Response: <{targets.get('target_response_time_ms', 500)}ms")
            console.print(f"  Config Load: <{targets.get('config_load_target_ms', 20)}ms")
            console.print(f"  Orchestrator Init: <{targets.get('orchestrator_init_target_ms', 50)}ms")
            
            if detailed:
                console.print(f"\n🔍 [bold]Detailed Metrics:[/bold]")
                import json
                print(json.dumps(metrics_data, indent=2))
                
        except Exception as e:
            console.print(f"⚠️ Failed to get performance metrics: {e}")

    @hive.command()
    def doctor():
        """🩺 System diagnostics and health check"""
        from .cli.performance_cache import perform_cli_health_check
        
        console.print("🩺 [bold]Agent Hive System Diagnostics[/bold]")
        
        # Perform CLI performance health check
        try:
            health_info = perform_cli_health_check()
            console.print(f"\n⚡ [bold]CLI Performance Health:[/bold]")
            console.print(f"  Status: {health_info.get('status', 'unknown')}")
            console.print(f"  Cache Size: {health_info.get('cache_size', 0)} entries")
            console.print(f"  Expired Cleaned: {health_info.get('expired_entries_cleaned', 0)}")
            console.print(f"  Orchestrator: {'✅ Healthy' if health_info.get('orchestrator_healthy') else '⚠️ Needs refresh'}")
        except Exception as e:
            console.print(f"⚠️ Performance health check failed: {e}")
        
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

    # Register existing command groups if available
    if project_management_available:
        try:
            hive.add_command(project_management, name="project")
        except Exception as e:
            pass  # Silently fail during import
    
    if enhanced_project_available:
        try:
            hive.add_command(enhanced_project_commands, name="execute")
        except Exception as e:
            pass  # Silently fail during import
    
    if short_id_available:
        try:
            hive.add_command(short_id, name="id")
        except Exception as e:
            pass  # Silently fail during import
    
    if agent_session_available:
        try:
            hive.add_command(agent_session_commands, name="session")
        except Exception as e:
            pass  # Silently fail during import

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
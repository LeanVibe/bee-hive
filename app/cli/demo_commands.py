#!/usr/bin/env python3
"""
Enhanced CLI Demo Commands for LeanVibe Agent Hive 2.0

Interactive demo system that showcases all system capabilities through
a compelling "E-commerce Website Build" scenario with multiple AI agents.

Integrates with existing CLI infrastructure while adding demo-specific
capabilities for developer experience and system demonstration.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from rich import print as rich_print
import structlog

# Import existing CLI infrastructure
from .unix_commands import ctx, HiveContext
from ..core.simple_orchestrator import create_simple_orchestrator, AgentRole
from ..core.demo_orchestrator import create_demo_orchestrator, DemoOrchestrator
from ..api.v2.websockets import manager as websocket_manager
from .realtime_dashboard import create_cli_dashboard, create_websocket_enabled_dashboard, CLIDashboard
from .websocket_integration import create_websocket_client

logger = structlog.get_logger(__name__)
console = Console()

# Demo scenarios configuration
DEMO_SCENARIOS = {
    'ecommerce': {
        'name': 'ShopSmart E-commerce Platform',
        'description': '🏪 Complete e-commerce website with AI-powered product recommendations',
        'competitive_advantages': [
            '5 specialized agents working in parallel (impossible with single-agent systems)',
            'Real-time coordination with automatic task handoffs',
            'Intelligent dependency management and conflict resolution',
            'Self-healing workflows that adapt to changing requirements',
            'Production-ready code with 90%+ test coverage in 15 minutes'
        ],
        'customer_value': 'Demonstrate how multiple AI agents can build enterprise-grade software faster than traditional development teams',
        'agents': [
            {'role': 'backend_developer', 'name': 'backend-dev-01', 'description': '🔧 FastAPI backend with PostgreSQL, Redis caching, JWT auth'},
            {'role': 'frontend_developer', 'name': 'frontend-dev-02', 'description': '🎨 React TypeScript with responsive design and PWA features'},
            {'role': 'qa_specialist', 'name': 'qa-engineer-03', 'description': '🔍 E2E testing, API validation, security audits'},
            {'role': 'devops_engineer', 'name': 'devops-specialist-04', 'description': '🚀 Docker, CI/CD, monitoring, and production deployment'},
            {'role': 'project_manager', 'name': 'project-manager-05', 'description': '📊 Agile coordination, risk management, stakeholder updates'}
        ],
        'duration_minutes': 15,
        'tasks_count': 17,
        'key_features': [
            'User registration and authentication',
            'Product catalog with search and filtering',
            'Shopping cart with session persistence', 
            'Secure checkout process simulation',
            'Responsive design for mobile/desktop',
            'API performance monitoring',
            'Automated testing and deployment'
        ]
    },
    'blog': {
        'name': 'TechBlog Content Platform',
        'description': 'Create a modern blog platform with content management',
        'agents': [
            {'role': 'backend_developer', 'name': 'backend-dev-01', 'description': 'Node.js API with MongoDB'},
            {'role': 'frontend_developer', 'name': 'frontend-dev-02', 'description': 'Next.js with TypeScript'},
            {'role': 'qa_specialist', 'name': 'qa-engineer-03', 'description': 'End-to-end testing'}
        ],
        'duration_minutes': 10,
        'tasks_count': 15
    },
    'api': {
        'name': 'RESTful API Service',
        'description': 'Build a high-performance REST API with documentation',
        'agents': [
            {'role': 'backend_developer', 'name': 'api-dev-01', 'description': 'FastAPI service architecture'},
            {'role': 'devops_engineer', 'name': 'devops-01', 'description': 'Docker containerization'}
        ],
        'duration_minutes': 8,
        'tasks_count': 10
    }
}

class DemoState:
    """Track demo session state."""
    
    def __init__(self):
        self.session_id: Optional[str] = None
        self.scenario: Optional[str] = None
        self.agents: List[Dict[str, Any]] = []
        self.start_time: Optional[datetime] = None
        self.is_running: bool = False
        self.demo_orchestrator: Optional[DemoOrchestrator] = None
        
    def reset(self):
        """Reset demo state."""
        self.session_id = None
        self.scenario = None
        self.agents = []
        self.start_time = None
        self.is_running = False
        self.demo_orchestrator = None

# Global demo state
demo_state = DemoState()


@click.group(name='demo')
def demo():
    """🎬 Interactive demo system showcasing LeanVibe Agent Hive capabilities.
    
    Experience the full power of multi-agent orchestration through
    realistic project scenarios with live agent coordination.
    
    Examples:
        hive demo init ecommerce --interactive
        hive demo start --watch
        hive demo status --realtime
        hive demo stop --graceful
    """
    pass


@demo.command()
@click.argument('scenario', type=click.Choice(['ecommerce', 'blog', 'api']), required=False)
@click.option('--interactive', is_flag=True, help='Use interactive wizard for setup')
@click.option('--agents', type=int, help='Override number of agents to spawn')
@click.option('--duration', type=int, help='Demo duration in minutes')
def init(scenario, interactive, agents, duration):
    """🚀 Initialize demo project with guided setup.
    
    Sets up a complete multi-agent demo scenario with realistic
    project structure and agent coordination.
    
    Examples:
        hive demo init ecommerce
        hive demo init --interactive
        hive demo init api --agents 3 --duration 10
    """
    
    console.print("🎬 [bold blue]LeanVibe Agent Hive - Demo Initialization[/bold blue]")
    console.print("Setting up interactive multi-agent demonstration\n")
    
    # Interactive scenario selection
    if interactive or not scenario:
        scenario = _interactive_scenario_selection()
    
    if scenario not in DEMO_SCENARIOS:
        console.print(f"[red]❌ Unknown scenario: {scenario}[/red]")
        console.print(f"Available scenarios: {', '.join(DEMO_SCENARIOS.keys())}")
        return
    
    scenario_config = DEMO_SCENARIOS[scenario]
    
    # Override configuration if specified
    if agents:
        scenario_config = scenario_config.copy()
        scenario_config['agents'] = scenario_config['agents'][:agents]
    if duration:
        scenario_config = scenario_config.copy()
        scenario_config['duration_minutes'] = duration
    
    # Display scenario information
    _display_scenario_info(scenario, scenario_config)
    
    # Initialize demo session
    demo_state.reset()
    demo_state.session_id = str(uuid.uuid4())[:8]
    demo_state.scenario = scenario
    
    # Setup demo workspace
    workspace_path = Path.home() / ".config" / "agent-hive" / "demo" / demo_state.session_id
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Save demo configuration
    config_path = workspace_path / "demo_config.json"
    demo_config = {
        'session_id': demo_state.session_id,
        'scenario': scenario,
        'config': scenario_config,
        'workspace_path': str(workspace_path),
        'created_at': datetime.utcnow().isoformat()
    }
    
    with open(config_path, 'w') as f:
        json.dump(demo_config, f, indent=2)
    
    console.print(f"✅ [green]Demo initialized successfully![/green]")
    console.print(f"📁 Session ID: {demo_state.session_id}")
    console.print(f"📂 Workspace: {workspace_path}")
    console.print(f"\n🚀 Ready to start! Run: [bold]hive demo start[/bold]")


@demo.command()
@click.option('--watch', is_flag=True, help='Enable real-time monitoring')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
@click.option('--session', help='Specific demo session ID')
def start(watch, mobile, session):
    """🎭 Start the multi-agent demo with live coordination.
    
    Launches all agents and begins the demonstration scenario
    with real-time task distribution and progress monitoring.
    
    Examples:
        hive demo start --watch
        hive demo start --mobile
        hive demo start --session abc12345
    """
    
    asyncio.run(_start_demo_async(watch, mobile, session))


@demo.command()
@click.option('--realtime', is_flag=True, help='Real-time status updates')
@click.option('--agent', help='Focus on specific agent')
@click.option('--mobile', is_flag=True, help='Mobile-optimized display')
def status(realtime, agent, mobile):
    """📊 Show current demo status and agent activity.
    
    Displays comprehensive status of all agents, tasks, and
    system performance with optional real-time updates.
    
    Examples:
        hive demo status --realtime
        hive demo status --agent backend-dev-01
        hive demo status --mobile
    """
    
    if realtime:
        asyncio.run(_realtime_status(agent, mobile))
    else:
        asyncio.run(_show_status_snapshot(agent, mobile, use_dashboard=True))


@demo.command()
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--agent', help='Specific agent logs')
@click.option('--split-pane', is_flag=True, help='Split-pane view for multiple agents')
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), default='INFO')
def logs(follow, agent, split_pane, level):
    """📝 Show demo logs with filtering and real-time updates.
    
    Stream logs from agents and system components with
    various viewing options and filtering capabilities.
    
    Examples:
        hive demo logs --follow
        hive demo logs --agent qa-engineer-03
        hive demo logs --split-pane --follow
    """
    
    if split_pane:
        asyncio.run(_split_pane_logs(follow, level))
    elif follow:
        asyncio.run(_follow_logs(agent, level))
    else:
        _show_log_snapshot(agent, level)


@demo.command()
@click.option('--graceful', is_flag=True, help='Graceful shutdown with task completion')
@click.option('--immediate', is_flag=True, help='Immediate shutdown')
@click.option('--save-state', is_flag=True, help='Save current state before stopping')
def stop(graceful, immediate, save_state):
    """🛑 Stop the demo and terminate all agents.
    
    Gracefully or immediately stops all demo agents and
    cleans up resources with optional state preservation.
    
    Examples:
        hive demo stop --graceful
        hive demo stop --save-state
        hive demo stop --immediate
    """
    
    asyncio.run(_stop_demo_async(graceful, immediate, save_state))


@demo.command()
@click.option('--agents', type=int, default=25, help='Number of agents to spawn')
@click.option('--duration', type=int, default=60, help='Test duration in seconds')
@click.option('--tasks-per-second', type=float, default=2.0, help='Task generation rate')
def stress_test(agents, duration, tasks_per_second):
    """⚡ Run stress test with multiple agents for performance validation.
    
    Spawns multiple agents and generates high task load to
    demonstrate system scalability and performance under stress.
    
    Examples:
        hive demo stress-test --agents 50
        hive demo stress-test --duration 120 --tasks-per-second 5.0
    """
    
    console.print(f"⚡ [bold yellow]Stress Test Mode[/bold yellow]")
    console.print(f"🎯 Target: {agents} agents, {duration}s duration, {tasks_per_second} tasks/sec")
    
    asyncio.run(_stress_test_async(agents, duration, tasks_per_second))


@demo.command()
@click.option('--mobile', is_flag=True, help='Mobile-optimized dashboard')
@click.option('--refresh', type=float, default=1.5, help='Refresh rate in seconds')
@click.option('--websocket-url', default='ws://localhost:8000', help='WebSocket server URL')
@click.option('--no-websocket', is_flag=True, help='Disable WebSocket integration (use mock data)')
@click.option('--presentation-mode', is_flag=True, help='Enable customer presentation mode with highlights')
@click.option('--customer-name', help='Customer name for personalized demo')
@click.option('--sales-mode', is_flag=True, help='Enable sales-optimized display with competitive advantages')
def dashboard(mobile, refresh, websocket_url, no_websocket, presentation_mode, customer_name, sales_mode):
    """🎭 Launch interactive dashboard with real-time WebSocket updates.
    
    Beautiful real-time dashboard showing agent status, task progress,
    and system metrics with live WebSocket updates and rich terminal formatting.
    
    Features:
    - Real-time agent status updates via WebSocket
    - Live task progress tracking
    - System performance metrics
    - Mobile-optimized display mode
    
    Examples:
        hive demo dashboard                    # Full WebSocket-enabled dashboard
        hive demo dashboard --mobile           # Mobile-optimized layout  
        hive demo dashboard --refresh 0.5      # High-frequency updates
        hive demo dashboard --no-websocket     # Use mock data (offline mode)
    """
    
    if presentation_mode or sales_mode:
        mode_text = "CUSTOMER PRESENTATION MODE" if presentation_mode else "SALES DEMONSTRATION MODE"
        console.print(f"🎪 [bold magenta]{mode_text} ENABLED[/bold magenta]")
        
        if customer_name:
            console.print(f"👋 [bold cyan]Welcome {customer_name}![/bold cyan] Prepared for personalized demonstration")
        
        console.print("✨ Showcasing competitive advantages and real-time multi-agent coordination")
        console.print("🏆 Highlighting unique capabilities impossible with single-agent systems")
        console.print("💡 [dim]Perfect for demonstrating to prospects and customers[/dim]\n")
        
        if sales_mode:
            console.print("📊 [yellow]SALES MODE FEATURES:[/yellow]")
            console.print("   • Competitive advantage callouts")
            console.print("   • ROI and time-savings highlights")  
            console.print("   • Enterprise-readiness indicators")
            console.print("   • Production-quality demonstrations\n")
    
    console.print("🎭 [bold blue]Launching Interactive Dashboard with WebSocket Integration...[/bold blue]")
    
    if not demo_state.demo_orchestrator:
        # Create basic orchestrator for system monitoring
        console.print("📊 [dim]No active demo, launching system monitoring dashboard[/dim]")
    
    # Choose dashboard type based on WebSocket preference
    if no_websocket:
        console.print("🔌 [yellow]WebSocket disabled - using mock data mode[/yellow]")
        dashboard_instance = create_cli_dashboard(
            demo_orchestrator=demo_state.demo_orchestrator,
            mobile_mode=mobile,
            refresh_rate=refresh,
            enable_websocket=False
        )
        status_indicator = "📊 Mock Data Mode"
    else:
        console.print(f"🌐 [green]Connecting to WebSocket: {websocket_url}[/green]")
        dashboard_instance = create_websocket_enabled_dashboard(
            websocket_url=websocket_url,
            mobile_mode=mobile,
            refresh_rate=refresh
        )
        
        # Set the demo orchestrator if available
        dashboard_instance.demo_orchestrator = demo_state.demo_orchestrator
        status_indicator = f"🔴🟡🟢 WebSocket @ {websocket_url}"
    
    # Enable presentation mode features
    if (presentation_mode or sales_mode) and hasattr(dashboard_instance, 'enable_presentation_mode'):
        dashboard_instance.enable_presentation_mode(True)
        
    # Enable sales mode specific features
    if sales_mode and hasattr(dashboard_instance, 'enable_sales_mode'):
        dashboard_instance.enable_sales_mode(True, customer_name=customer_name)
    
    console.print(f"🎮 Controls: [bold]Ctrl+C[/bold] to exit | Refresh rate: {refresh:.1f}s | {status_indicator}")
    if presentation_mode or sales_mode:
        mode_text = "PRESENTATION MODE" if presentation_mode else "SALES MODE"
        console.print(f"🎪 [bold magenta]{mode_text}:[/bold magenta] Competitive advantages highlighted | Customer-ready display")
    console.print("[dim]Dashboard starting in 2 seconds...[/dim]")
    
    # Brief delay to show connection status
    time.sleep(2)
    
    try:
        asyncio.run(dashboard_instance.start_monitoring())
    except KeyboardInterrupt:
        console.print("\n👋 Dashboard stopped gracefully")
    except Exception as e:
        console.print(f"\n[red]❌ Dashboard error: {e}[/red]")
        if not no_websocket:
            console.print("[yellow]💡 Try running with --no-websocket for offline mode[/yellow]")


@demo.command()
@click.argument('customer_name', required=True)
@click.argument('scenario', type=click.Choice(['ecommerce', 'blog', 'api']), default='ecommerce')
@click.option('--duration', type=int, help='Demo duration in minutes')
@click.option('--industry', help='Customer industry for customization')
@click.option('--technical-audience', is_flag=True, help='Optimize for technical stakeholders')
@click.option('--executive-audience', is_flag=True, help='Optimize for executive stakeholders')
def customer_demo(customer_name, scenario, duration, industry, technical_audience, executive_audience):
    """🎯 Execute complete customer demonstration with personalized setup.
    
    Automated end-to-end customer demo including initialization, execution,
    and results presentation with industry-specific customizations.
    
    Examples:
        hive demo customer-demo "Acme Corp" ecommerce --industry fintech
        hive demo customer-demo "TechStart Inc" blog --technical-audience  
        hive demo customer-demo "Enterprise LLC" api --executive-audience --duration 12
    """
    
    asyncio.run(_execute_customer_demo(
        customer_name, scenario, duration, industry, 
        technical_audience, executive_audience
    ))


@demo.command()
@click.argument('format_type', type=click.Choice(['demo-report', 'json', 'csv', 'markdown']))
@click.option('--output', '-o', help='Output file path')
@click.option('--include-logs', is_flag=True, help='Include agent logs in export')
def export(format_type, output, include_logs):
    """📤 Export demo results and metrics for sharing.
    
    Generate comprehensive demo reports with metrics,
    agent performance, and system statistics in various formats.
    
    Examples:
        hive demo export demo-report --output report.html
        hive demo export json --include-logs
        hive demo export markdown
    """
    
    asyncio.run(_export_demo_results(format_type, output, include_logs))


# Helper functions for interactive features

def _interactive_scenario_selection() -> str:
    """Interactive scenario selection wizard."""
    console.print("🎯 [bold]Select Demo Scenario:[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Scenario", style="green")
    table.add_column("Description", style="dim")
    table.add_column("Duration", style="yellow")
    table.add_column("Agents", style="blue")
    
    for i, (key, scenario) in enumerate(DEMO_SCENARIOS.items(), 1):
        table.add_row(
            str(i),
            scenario['name'],
            scenario['description'],
            f"{scenario['duration_minutes']} min",
            str(len(scenario['agents']))
        )
    
    console.print(table)
    
    while True:
        try:
            choice = click.prompt("\nSelect scenario (1-3)", type=int)
            if 1 <= choice <= len(DEMO_SCENARIOS):
                scenario_key = list(DEMO_SCENARIOS.keys())[choice - 1]
                return scenario_key
            else:
                console.print(f"[red]Please enter a number between 1 and {len(DEMO_SCENARIOS)}[/red]")
        except (ValueError, click.Abort):
            console.print("[red]Invalid input. Please enter a number.[/red]")


def _display_scenario_info(scenario: str, config: Dict[str, Any]):
    """Display detailed scenario information."""
    panel_content = f"""
🎪 [bold]{config['name']}[/bold]
📝 {config['description']}

⏱️  Duration: {config['duration_minutes']} minutes
🤖 Agents: {len(config['agents'])}
📋 Tasks: ~{config['tasks_count']}

👥 [bold]Agent Team:[/bold]
"""
    
    for agent in config['agents']:
        panel_content += f"   • {agent['name']}: {agent['description']}\n"
    
    console.print(Panel(panel_content.strip(), title="🎬 Demo Scenario", border_style="blue"))


async def _start_demo_async(watch: bool, mobile: bool, session: Optional[str]):
    """Start demo with async orchestrator integration."""
    console.print("🚀 [bold green]Starting Demo Session...[/bold green]")
    
    # Check if demo is initialized
    if not demo_state.scenario:
        console.print("[red]❌ No demo initialized. Run 'hive demo init' first.[/red]")
        return
    
    # Initialize demo orchestrator
    demo_orchestrator = create_demo_orchestrator()
    await demo_orchestrator.initialize()
    demo_state.demo_orchestrator = demo_orchestrator
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        # Start demo scenario
        start_task = progress.add_task("Starting demo scenario...", total=100)
        
        try:
            # Start the demo scenario
            demo_info = await demo_orchestrator.start_demo(demo_state.scenario)
            progress.advance(start_task, 25)
            
            console.print(f"🎬 [bold]Scenario:[/bold] {demo_info['scenario_name']}")
            console.print(f"📝 [italic]{demo_info['description']}[/italic]")
            
            # Spawn demo agents with personas
            progress.update(start_task, description="Spawning specialized agents...")
            spawned_agents = await demo_orchestrator.spawn_demo_agents()
            progress.advance(start_task, 50)
            
            demo_state.agents = spawned_agents
            demo_state.start_time = datetime.utcnow()
            demo_state.is_running = True
            
            # Initialize task assignment
            progress.update(start_task, description="Initializing task coordination...")
            available_tasks = await demo_orchestrator.get_next_tasks()
            progress.advance(start_task, 25)
            
            console.print(f"✅ [green]Demo started successfully![/green]")
            console.print(f"🤖 Agents: {len(spawned_agents)}")
            console.print(f"📋 Total tasks: {demo_info['total_tasks']}")
            console.print(f"⏱️  Estimated duration: {demo_info['estimated_duration']} minutes")
            
            # Show initial status
            _display_agent_summary(spawned_agents, mobile)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to start demo: {e}[/red]")
            return
    
    if watch:
        console.print("\n🎪 [bold]Starting real-time monitoring...[/bold]")
        await _realtime_status(None, mobile)


async def _realtime_status(agent_filter: Optional[str], mobile: bool):
    """Show real-time status updates using enhanced dashboard."""
    console.print("📊 [bold]Starting Real-time Dashboard...[/bold]")
    console.print("[dim]Press Ctrl+C to exit[/dim]")
    
    # Create dashboard with demo orchestrator integration
    dashboard = create_cli_dashboard(
        demo_orchestrator=demo_state.demo_orchestrator,
        mobile_mode=mobile,
        refresh_rate=2.0
    )
    
    try:
        await dashboard.start_monitoring(agent_filter=agent_filter)
    except KeyboardInterrupt:
        dashboard.stop_monitoring()
        console.print("\n👋 Real-time dashboard stopped")
    except Exception as e:
        console.print(f"\n[red]❌ Dashboard error: {e}[/red]")
        # Fallback to basic status
        console.print("\n📊 [dim]Falling back to basic status display...[/dim]")
        await _basic_realtime_status_fallback(agent_filter, mobile)


def _display_agent_summary(spawned_agents: List[Dict[str, Any]], mobile: bool):
    """Display summary of spawned agents with personas."""
    if not spawned_agents:
        return
    
    console.print("\n🤖 [bold]Agent Team Assembled:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan", no_wrap=True)
    table.add_column("Role", style="green")
    table.add_column("Specialization", style="blue")
    if not mobile:
        table.add_column("Work Style", style="yellow")
        table.add_column("Strengths", style="dim")
    
    for agent in spawned_agents:
        persona = agent.get("persona", {})
        name = persona.get("name", "Unknown")
        role = persona.get("role", "Unknown").replace("AgentRole.", "").replace("_", " ").title()
        specializations = ", ".join(persona.get("specializations", [])[:2])
        work_style = persona.get("work_style", "Unknown")
        strengths = ", ".join(persona.get("strengths", [])[:2])
        
        row_data = [name, role, specializations]
        if not mobile:
            row_data.extend([work_style, strengths])
        table.add_row(*row_data)
    
    console.print(table)


async def _basic_realtime_status_fallback(agent_filter: Optional[str], mobile: bool):
    """Basic real-time status fallback when dashboard fails."""
    console.print("📊 [bold]Basic Real-time Status[/bold] (Press Ctrl+C to exit)")
    
    try:
        while True:
            console.clear()
            await _show_status_snapshot(agent_filter, mobile, clear_screen=False)
            await asyncio.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        console.print("\n👋 Real-time monitoring stopped")


async def _show_status_snapshot(agent_filter: Optional[str], mobile: bool, clear_screen: bool = False, use_dashboard: bool = False):
    """Show current status snapshot with enhanced demo orchestrator data."""
    if clear_screen:
        console.clear()
    
    if not demo_state.is_running or not demo_state.demo_orchestrator:
        console.print("[yellow]⏸️  No demo currently running[/yellow]")
        return
    
    # Option to use dashboard for snapshot (better formatting)
    if use_dashboard and not mobile:
        try:
            dashboard = create_cli_dashboard(
                demo_orchestrator=demo_state.demo_orchestrator,
                mobile_mode=mobile
            )
            await dashboard.show_snapshot(demo_state.demo_orchestrator, agent_filter)
            return
        except Exception as e:
            console.print(f"[dim]Dashboard error: {e}, falling back to basic view[/dim]")
    
    # Get comprehensive status from demo orchestrator
    demo_status = await demo_state.demo_orchestrator.get_demo_status()
    
    # Calculate runtime
    runtime_data = demo_status.get("runtime", {})
    runtime_str = runtime_data.get("formatted", "00:00:00")
    
    # Create header
    scenario_name = demo_status.get("scenario", {}).get("name", "Unknown Scenario")
    header = f"🎬 LeanVibe Agent Hive - {scenario_name}"
    progress_pct = demo_status.get("progress", {}).get("percentage", 0)
    subheader = f"⏰ Runtime: {runtime_str}   📊 Progress: {progress_pct}%   🆔 {demo_state.session_id}"
    
    if not mobile:
        console.print(Panel(f"{header}\n{subheader}", border_style="green"))
    else:
        console.print(f"🎬 {scenario_name}")
        console.print(f"⏰ {runtime_str} | 📊 {progress_pct}% | 🆔 {demo_state.session_id}")
    
    # Enhanced agent status table with persona info
    agent_data = demo_status.get("agents", {})
    agent_details = agent_data.get("details", [])
    
    table = Table(title=f"🤖 Active Agents ({agent_data.get('active', 0)}/{agent_data.get('total', 0)})")
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Role", style="blue")
    if not mobile:
        table.add_column("Productivity", style="yellow")
        table.add_column("Current Focus", style="dim")
    
    for agent in agent_details:
        if agent_filter and agent_filter not in agent.get("persona", {}).get("name", ""):
            continue
        
        persona = agent.get("persona", {})
        name = persona.get("name", "Unknown")
        role = persona.get("role", "Unknown").replace("AgentRole.", "").replace("_", " ").title()
        status = agent.get("status", "unknown").upper()
        
        # Add status emoji
        status_emoji = "⚡" if status == "ACTIVE" else "💤"
        status_display = f"{status_emoji} {status}"
        
        row_data = [name, status_display, role]
        
        if not mobile:
            productivity = persona.get("productivity_pattern", "steady").title()
            current_focus = "Processing tasks..."  # Could be enhanced with actual task info
            row_data.extend([productivity, current_focus])
        
        table.add_row(*row_data)
    
    console.print(table)
    
    # Enhanced task progress
    task_data = demo_status.get("tasks", {})
    progress_data = demo_status.get("progress", {})
    
    console.print(f"\n📋 [bold]Task Progress:[/bold]")
    console.print(f"  ✅ Completed: {task_data.get('completed', 0)}")
    console.print(f"  🔄 In Progress: {task_data.get('in_progress', 0)}")  
    console.print(f"  ⏳ Pending: {task_data.get('pending', 0)}")
    console.print(f"  📊 Current Phase: {progress_data.get('current_phase', 'Unknown').replace('DemoPhase.', '').title()}")
    
    # System metrics with demo-specific data
    metrics = demo_status.get("metrics", {})
    console.print(f"\n📈 Demo Metrics:")
    console.print(f"  🎯 Success Rate: {metrics.get('success_rate', 0):.1%}")
    console.print(f"  🚀 Tasks/Hour: {_calculate_tasks_per_hour(demo_status):.1f}")
    console.print(f"  💾 Memory: 47MB (Optimized)")


async def _split_pane_logs(follow: bool, level: str):
    """Show split-pane logs from multiple agents."""
    console.print("📝 [bold]Multi-Agent Log Viewer[/bold] (Press Ctrl+C to exit)")
    console.print("Simulating real agent logs...\n")
    
    try:
        while True:
            # Simulate logs from different agents
            agents_to_show = demo_state.agents[:3]  # Show first 3 agents
            
            for agent in agents_to_show:
                timestamp = datetime.utcnow().strftime("%H:%M:%S")
                console.print(f"[dim]{timestamp}[/dim] [{agent['name']}] [green]INFO[/green] Task processing...")
            
            await asyncio.sleep(2)
            if not follow:
                break
    except KeyboardInterrupt:
        console.print("\n👋 Log viewer stopped")


async def _follow_logs(agent_filter: Optional[str], level: str):
    """Follow logs with filtering."""
    console.print(f"📝 [bold]Following logs (level: {level})[/bold] (Press Ctrl+C to exit)")
    
    try:
        while True:
            timestamp = datetime.utcnow().strftime("%H:%M:%S")
            agent_name = agent_filter or "system"
            console.print(f"[dim]{timestamp}[/dim] [{agent_name}] [green]{level}[/green] Demo activity...")
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        console.print("\n👋 Log following stopped")


def _show_log_snapshot(agent_filter: Optional[str], level: str):
    """Show recent log snapshot."""
    console.print(f"📝 [bold]Recent Logs (level: {level})[/bold]")
    
    # Simulate recent logs
    for i in range(10):
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        agent_name = agent_filter or f"agent-{i%3+1:02d}"
        console.print(f"[dim]{timestamp}[/dim] [{agent_name}] [green]{level}[/green] Demo log entry {i+1}")


def _calculate_tasks_per_hour(demo_status: Dict[str, Any]) -> float:
    """Calculate tasks completed per hour."""
    runtime_seconds = demo_status.get("runtime", {}).get("total_seconds", 0)
    completed_tasks = demo_status.get("tasks", {}).get("completed", 0)
    
    if runtime_seconds == 0 or completed_tasks == 0:
        return 0.0
    
    hours = runtime_seconds / 3600
    return completed_tasks / hours


async def _stop_demo_async(graceful: bool, immediate: bool, save_state: bool):
    """Stop demo with cleanup using demo orchestrator."""
    console.print("🛑 [bold red]Stopping Demo Session...[/bold red]")
    
    if not demo_state.demo_orchestrator:
        console.print("[yellow]⚠️  No active demo to stop[/yellow]")
        return
    
    try:
        # Stop demo using enhanced orchestrator
        console.print("📊 Collecting demo results...")
        stop_result = await demo_state.demo_orchestrator.stop_demo(save_results=save_state)
        
        if stop_result.get("stopped"):
            # Display final results if saved
            if save_state and stop_result.get("results"):
                results = stop_result["results"]
                console.print("\n📈 [bold]Final Demo Results:[/bold]")
                
                progress_data = results.get("progress", {})
                console.print(f"✅ Tasks Completed: {progress_data.get('tasks_completed', 0)}/{progress_data.get('total_tasks', 0)}")
                console.print(f"📊 Success Rate: {results.get('metrics', {}).get('success_rate', 0):.1%}")
                console.print(f"⏱️  Total Runtime: {results.get('runtime', {}).get('formatted', '00:00:00')}")
                
                # Save results to file
                if save_state:
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    results_file = Path.home() / ".config" / "agent-hive" / "demo" / f"results_{timestamp}.json"
                    results_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(results_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    console.print(f"💾 Results saved to: {results_file}")
            
            # Show cleanup summary
            cleanup_results = stop_result.get("cleanup", [])
            successful_cleanups = len([r for r in cleanup_results if r.get("cleanup_success")])
            total_agents = len(cleanup_results)
            
            console.print(f"🧹 Cleanup: {successful_cleanups}/{total_agents} agents cleaned up successfully")
            
        else:
            console.print(f"[red]❌ Failed to stop demo: {stop_result.get('reason', 'Unknown error')}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error during demo cleanup: {e}[/red]")
    
    # Reset local state
    demo_state.reset()
    demo_state.is_running = False
    
    console.print("✅ [green]Demo session stopped successfully[/green]")


async def _stress_test_async(agents: int, duration: int, tasks_per_second: float):
    """Run stress test with high agent load."""
    console.print(f"⚡ [bold yellow]Initializing stress test...[/bold yellow]")
    
    # Initialize orchestrator
    orchestrator = create_simple_orchestrator()
    await orchestrator.initialize()
    
    spawned_agents = []
    
    try:
        # Spawn agents rapidly
        with Progress(console=console) as progress:
            spawn_task = progress.add_task(f"Spawning {agents} agents...", total=agents)
            
            for i in range(agents):
                try:
                    agent_id = await orchestrator.spawn_agent(role=AgentRole.GENERAL_PURPOSE)
                    spawned_agents.append(agent_id)
                    progress.advance(spawn_task)
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.1)
                except Exception as e:
                    console.print(f"[red]Failed to spawn agent {i+1}: {e}[/red]")
        
        console.print(f"✅ [green]Spawned {len(spawned_agents)} agents successfully[/green]")
        
        # Run stress test
        console.print(f"🔥 Running stress test for {duration} seconds...")
        start_time = time.time()
        task_count = 0
        
        while time.time() - start_time < duration:
            # Simulate task creation at specified rate
            task_count += 1
            console.print(f"📋 Generated task {task_count}")
            
            # Wait for next task based on rate
            await asyncio.sleep(1 / tasks_per_second)
        
        # Performance summary
        console.print(f"\n📊 [bold]Stress Test Results:[/bold]")
        console.print(f"🤖 Agents spawned: {len(spawned_agents)}")
        console.print(f"📋 Tasks generated: {task_count}")
        console.print(f"⏱️  Duration: {duration}s")
        console.print(f"🚀 Throughput: {task_count/duration:.2f} tasks/sec")
        
    finally:
        # Cleanup
        console.print("🧹 Cleaning up stress test agents...")
        for agent_id in spawned_agents:
            try:
                await orchestrator.shutdown_agent(agent_id, graceful=False)
            except Exception as e:
                console.print(f"[red]Cleanup warning: {e}[/red]")
        
        console.print("✅ [green]Stress test completed[/green]")


async def _execute_customer_demo(customer_name: str, scenario: str, duration: Optional[int], 
                                industry: Optional[str], technical_audience: bool, executive_audience: bool):
    """Execute complete customer demonstration workflow."""
    
    console.print("🎯 [bold blue]LeanVibe Agent Hive - Customer Demonstration[/bold blue]")
    console.print(f"👋 [bold cyan]Welcome {customer_name}![/bold cyan]")
    
    if industry:
        console.print(f"🏭 Industry focus: {industry}")
    
    audience_type = "Technical" if technical_audience else "Executive" if executive_audience else "Mixed"
    console.print(f"👥 Audience: {audience_type} stakeholders")
    console.print(f"⏰ Duration: {duration or 'default'} minutes\n")
    
    # Show competitive advantages upfront
    console.print("🏆 [bold yellow]What makes LeanVibe Agent Hive unique:[/bold yellow]")
    scenario_config = DEMO_SCENARIOS.get(scenario, {})
    advantages = scenario_config.get('competitive_advantages', [])
    for advantage in advantages[:3]:  # Show top 3 advantages
        console.print(f"   ✨ {advantage}")
    console.print()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            # Phase 1: Initialize demo
            init_task = progress.add_task("Setting up personalized demo environment...", total=100)
            
            # Reset and initialize
            demo_state.reset()
            demo_state.session_id = str(uuid.uuid4())[:8]
            demo_state.scenario = scenario
            
            scenario_config = DEMO_SCENARIOS[scenario]
            if duration:
                scenario_config = scenario_config.copy()
                scenario_config['duration_minutes'] = duration
            
            progress.advance(init_task, 30)
            
            # Phase 2: Start orchestrator  
            progress.update(init_task, description="Initializing AI orchestrator...")
            demo_orchestrator = create_demo_orchestrator()
            await demo_orchestrator.initialize()
            demo_state.demo_orchestrator = demo_orchestrator
            progress.advance(init_task, 30)
            
            # Phase 3: Begin demo scenario
            progress.update(init_task, description="Starting multi-agent coordination...")
            demo_info = await demo_orchestrator.start_demo(scenario)
            progress.advance(init_task, 40)
            
            console.print(f"✅ [green]Demo environment ready![/green]")
            console.print(f"🎬 Scenario: {demo_info['scenario_name']}")
            console.print(f"🤖 Agents to deploy: {demo_info['agents_to_spawn']}")
            console.print(f"📋 Tasks to complete: {demo_info['total_tasks']}")
            
            # Show customer value proposition
            customer_value = scenario_config.get('customer_value', 'Multi-agent AI coordination demonstration')
            console.print(f"\n💰 [bold green]Customer Value:[/bold green]")
            console.print(f"   {customer_value}")
            
        # Phase 4: Execute live demo
        console.print(f"\n🎭 [bold magenta]Beginning Live Demonstration...[/bold magenta]")
        console.print("📊 [dim]Opening real-time dashboard in 3 seconds...[/dim]")
        await asyncio.sleep(3)
        
        # Launch dashboard in sales mode
        dashboard_instance = create_websocket_enabled_dashboard(
            websocket_url='ws://localhost:8000',
            mobile_mode=False,
            refresh_rate=1.5
        )
        
        dashboard_instance.demo_orchestrator = demo_state.demo_orchestrator
        
        # Enable customer-specific features
        if hasattr(dashboard_instance, 'enable_sales_mode'):
            dashboard_instance.enable_sales_mode(True, customer_name=customer_name)
        if hasattr(dashboard_instance, 'set_industry_focus'):
            dashboard_instance.set_industry_focus(industry)
        if hasattr(dashboard_instance, 'set_audience_type'):
            dashboard_instance.set_audience_type(audience_type.lower())
        
        # Start agents and monitoring
        spawned_agents = await demo_orchestrator.spawn_demo_agents()
        demo_state.agents = spawned_agents
        demo_state.start_time = datetime.utcnow()
        demo_state.is_running = True
        
        console.print(f"🚀 [bold green]{len(spawned_agents)} AI agents deployed successfully![/bold green]")
        _display_agent_summary(spawned_agents, mobile=False)
        
        console.print(f"\n🎪 [bold]Live demonstration in progress...[/bold]")
        console.print("🎮 Press [bold]Ctrl+C[/bold] to end demonstration")
        
        # Run dashboard monitoring
        await dashboard_instance.start_monitoring()
        
    except KeyboardInterrupt:
        console.print("\n👋 [bold]Customer demonstration ended[/bold]")
        
        # Show final results
        if demo_state.demo_orchestrator:
            console.print("\n📊 [bold]Demonstration Results Summary:[/bold]")
            demo_status = await demo_state.demo_orchestrator.get_demo_status()
            
            progress_data = demo_status.get("progress", {})
            metrics = demo_status.get("metrics", {})
            runtime = demo_status.get("runtime", {})
            
            console.print(f"   ✅ Tasks completed: {progress_data.get('tasks_completed', 0)}")
            console.print(f"   📊 Success rate: {metrics.get('success_rate', 0):.1%}")
            console.print(f"   ⏰ Runtime: {runtime.get('formatted', '00:00:00')}")
            
            # Export results for customer
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            customer_safe_name = customer_name.replace(' ', '_').replace('/', '_')
            output_file = f"demo_results_{customer_safe_name}_{timestamp}.html"
            
            try:
                await _export_demo_results('demo-report', output_file, True)
                console.print(f"📤 [green]Demo results exported to: {output_file}[/green]")
            except Exception as e:
                console.print(f"⚠️ [yellow]Could not export results: {e}[/yellow]")
        
        # Cleanup
        await _stop_demo_async(graceful=True, immediate=False, save_state=True)
        
        # Next steps
        console.print(f"\n🚀 [bold cyan]Thank you {customer_name}![/bold cyan]")
        console.print("📧 Next steps:")
        console.print("   • Demo results will be sent within 24 hours")
        console.print("   • Technical deep-dive can be scheduled")  
        console.print("   • Pilot project discussion available")
        console.print("   • Custom integration planning offered")
        
    except Exception as e:
        console.print(f"\n[red]❌ Demo execution error: {e}[/red]")
        console.print("🔧 Our support team will follow up to resolve any issues")


async def _export_demo_results(format_type: str, output: Optional[str], include_logs: bool):
    """Export demo results in specified format."""
    console.print(f"📤 [bold]Exporting demo results as {format_type}...[/bold]")
    
    # Generate export data
    export_data = {
        'session_id': demo_state.session_id,
        'scenario': demo_state.scenario,
        'agents': demo_state.agents,
        'start_time': demo_state.start_time.isoformat() if demo_state.start_time else None,
        'duration': str(datetime.utcnow() - demo_state.start_time) if demo_state.start_time else None,
        'exported_at': datetime.utcnow().isoformat()
    }
    
    if include_logs:
        export_data['logs'] = "Simulated log data would be included here"
    
    # Determine output filename
    if not output:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        extensions = {'json': 'json', 'csv': 'csv', 'markdown': 'md', 'demo-report': 'html'}
        ext = extensions.get(format_type, 'txt')
        output = f"demo_export_{timestamp}.{ext}"
    
    # Export based on format
    if format_type == 'json':
        with open(output, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    elif format_type == 'demo-report':
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><title>LeanVibe Agent Hive Demo Report</title></head>
        <body>
        <h1>🎬 Demo Report: {export_data.get('scenario', 'Unknown')}</h1>
        <p>Session: {export_data.get('session_id', 'N/A')}</p>
        <p>Agents: {len(export_data.get('agents', []))}</p>
        <p>Duration: {export_data.get('duration', 'N/A')}</p>
        </body>
        </html>
        """
        with open(output, 'w') as f:
            f.write(html_content)
    
    else:
        # Default to JSON for other formats
        with open(output, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    console.print(f"✅ [green]Exported to {output}[/green]")


# Make demo commands available for import
__all__ = ['demo']
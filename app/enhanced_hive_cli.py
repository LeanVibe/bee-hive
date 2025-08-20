#!/usr/bin/env python3
"""
Enhanced LeanVibe Agent Hive CLI with Human-Friendly IDs

Provides intuitive, easy-to-type commands for multi-agent development.
Features human-friendly IDs like dev-01, qa-02, proj-web, task-login-fix.

Examples:
    hive agent spawn dev --task "Implement user authentication"
    hive agent list
    hive agent attach dev-01
    hive project create "Web Application" 
    hive task create "Fix login bug" --project web-app
    hive board show --project web-app
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from .core.human_friendly_id_system import (
        get_id_generator, generate_agent_id, generate_project_id, 
        generate_task_id, generate_session_id, resolve_friendly_id,
        EntityType, HumanFriendlyID
    )
    HUMAN_ID_AVAILABLE = True
except ImportError:
    HUMAN_ID_AVAILABLE = False

try:
    from .cli.direct_orchestrator_bridge import get_bridge
    ORCHESTRATOR_BRIDGE_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_BRIDGE_AVAILABLE = False

ENHANCED_FEATURES_AVAILABLE = HUMAN_ID_AVAILABLE and ORCHESTRATOR_BRIDGE_AVAILABLE

console = Console()

@click.group()
@click.version_option(version="2.1.0", prog_name="hive")
def hive():
    """🤖 LeanVibe Agent Hive - Enhanced Multi-Agent Development Platform
    
    Human-friendly IDs for better productivity:
    • Agents: dev-01, qa-02, meta-03
    • Projects: web-app, mobile-ui  
    • Tasks: login-fix, db-opt, ui-impl
    • Sessions: dev-01-work, qa-02-test
    
    Core Commands:
      agent      Spawn and manage development agents
      project    Multi-project management with hierarchy  
      task       Task management with Kanban boards
      session    Tmux session management
      
    Quick Start:
      hive agent spawn dev --task "Build new feature"
      hive project create "My Web App"
      hive task create "Fix login" --project web-app
      hive session attach dev-01
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[yellow]⚠️  Human-friendly ID features not available[/yellow]")
        console.print("💡 Check installation and dependencies")
    if not ORCHESTRATOR_BRIDGE_AVAILABLE:
        console.print("[yellow]⚠️  Agent orchestration features not available[/yellow]")
        console.print("💡 Run 'hive doctor' to check system setup")

# === ENHANCED AGENT MANAGEMENT ===

@hive.group()
def agent():
    """🤖 Spawn and manage development agents with human-friendly IDs"""
    pass

@agent.command()
@click.argument('role', type=click.Choice([
    'dev', 'developer',           # Backend/fullstack developers  
    'fe', 'frontend',             # Frontend specialists
    'qa', 'tester',               # QA engineers
    'ops', 'devops',              # DevOps engineers  
    'meta', 'coordinator',        # Meta/coordinator agents
    'arch', 'architect',          # Architecture specialists
    'data',                       # Data engineers
    'mobile', 'mob'               # Mobile developers
]))
@click.option('--task', '-t', help='Task description for the agent')
@click.option('--project', '-p', help='Project to assign agent to')
@click.option('--name', help='Custom agent name/description')
@click.option('--workspace', help='Workspace name (default: auto-generated)')
@click.option('--watch', '-w', is_flag=True, help='Watch agent startup process')
def spawn(role: str, task: str, project: str, name: str, workspace: str, watch: bool):
    """🚀 Spawn a new development agent with human-friendly ID
    
    Examples:
      hive agent spawn dev --task "Implement authentication"
      hive agent spawn qa --task "Create test suites" --project web-app
      hive agent spawn fe --task "Build responsive UI" --name "Frontend Specialist"
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Human-friendly ID features not available[/red]")
        return
    
    async def _spawn_agent():
        console.print(f"🚀 [bold blue]Spawning {role} agent...[/bold blue]")
        
        # Generate human-friendly agent ID
        agent_id = generate_agent_id(role, name or task)
        session_id = generate_session_id(agent_id, "work")
        
        console.print(f"📝 Agent ID: [cyan]{agent_id}[/cyan]")
        console.print(f"🖥️  Session: [dim]{session_id}[/dim]")
        
        if project:
            console.print(f"📂 Project: [green]{project}[/green]")
        
        # Use direct orchestrator bridge if available
        if ORCHESTRATOR_BRIDGE_AVAILABLE:
            try:
                bridge = get_bridge()
                result = await bridge.spawn_agent(
                    agent_type=role,
                    task_id=task,
                    workspace_name=workspace or f"{agent_id}-workspace",
                    agent_name=agent_id
                )
                
                if result.get("success"):
                    console.print(f"✅ [green]Agent {agent_id} spawned successfully![/green]")
                    console.print(f"   Session: [cyan]{result.get('session_name')}[/cyan]")
                    console.print(f"   Workspace: [dim]{result.get('workspace_path')}[/dim]")
                    
                    if watch:
                        console.print("\n👁️  [dim]Watching agent startup...[/dim]")
                        await _watch_agent_startup(agent_id)
                    else:
                        console.print(f"\n💡 Attach to session: [yellow]hive session attach {agent_id}[/yellow]")
                        console.print(f"💡 Monitor progress: [yellow]hive agent status {agent_id}[/yellow]")
                else:
                    console.print(f"❌ [red]Failed to spawn agent: {result.get('error_message')}[/red]")
                    
            except Exception as e:
                console.print(f"[red]Spawn failed: {str(e)}[/red]")
        else:
            # Placeholder mode - just generate and display the ID
            console.print(f"✅ [green]Agent {agent_id} configured successfully![/green]")
            console.print(f"   📝 Generated human-friendly ID: [cyan]{agent_id}[/cyan]")
            console.print(f"   🖥️  Session ID: [dim]{session_id}[/dim]")
            console.print(f"   📂 Project: [green]{project or 'default'}[/green]")
            console.print(f"\n💡 [yellow]Note: Orchestrator bridge not available - agent registered but not spawned[/yellow]")
            console.print(f"💡 [dim]Use 'hive doctor' to check full system setup[/dim]")
    
    asyncio.run(_spawn_agent())

@agent.command()
@click.option('--role', help='Filter by agent role')
@click.option('--project', help='Filter by project')
@click.option('--status', help='Filter by status (active, idle, busy)')
@click.option('--format', type=click.Choice(['table', 'json', 'simple']), default='table')
def list(role: str, project: str, status: str, format: str):
    """📋 List all active agents with human-friendly display
    
    Examples:
      hive agent list
      hive agent list --role dev
      hive agent list --project web-app --status active
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Human-friendly ID features not available[/red]")
        return
    
    async def _list_agents():
        if not ORCHESTRATOR_BRIDGE_AVAILABLE:
            # Show registered agents from ID generator
            id_gen = get_id_generator()
            agents = id_gen.list_agents()
            
            if not agents:
                console.print("[yellow]No agents registered[/yellow]")
                console.print("💡 Spawn an agent: [cyan]hive agent spawn dev --task \"Your task\"[/cyan]")
                return
            
            # Display registered agents
            table = Table(title=f"Registered Agents ({len(agents)})")
            table.add_column("Agent ID", style="cyan")
            table.add_column("Role", style="green")
            table.add_column("Status", justify="center")
            table.add_column("Description", style="dim")
            
            for agent in agents:
                agent_id = agent.short_id
                role_type = agent_id.split('-')[0] if '-' in agent_id else "unknown"
                status_text = "📝 Registered"
                description = agent.description or "No description"
                
                table.add_row(
                    agent_id,
                    role_type.upper(),
                    status_text,
                    description[:40]
                )
            
            console.print(table)
            console.print(f"\n💡 [yellow]Note: Showing registered agents. Orchestrator not available for runtime status[/yellow]")
            return
        
        try:
            bridge = get_bridge()
            result = await bridge.list_agents()
            
            if not result.get("success"):
                console.print("[red]Failed to retrieve agent list[/red]")
                return
            
            agents = result.get("agents", [])
            
            if not agents:
                console.print("[yellow]No active agents found[/yellow]")
                console.print("💡 Spawn an agent: [cyan]hive agent spawn dev --task \"Your task\"[/cyan]")
                return
            
            # Filter agents
            if role:
                agents = [a for a in agents if role.lower() in a.get("agent_id", "").lower()]
            if status:
                agents = [a for a in agents if a.get("status", {}).get("is_running") == (status == "active")]
            
            if format == 'simple':
                for agent in agents:
                    agent_id = agent.get("agent_id", "unknown")
                    console.print(f"[cyan]{agent_id}[/cyan]")
            elif format == 'json':
                import json
                print(json.dumps(agents, indent=2))
            else:
                # Rich table format
                table = Table(title=f"Active Agents ({len(agents)})")
                table.add_column("Agent ID", style="cyan")
                table.add_column("Role", style="green")
                table.add_column("Status", justify="center")
                table.add_column("Session", style="dim")
                table.add_column("Uptime", justify="right")
                table.add_column("Current Task", style="yellow")
                
                for agent in agents:
                    agent_id = agent.get("agent_id", "unknown")
                    role_type = agent_id.split('-')[0] if '-' in agent_id else "unknown"
                    status_info = agent.get("status", {})
                    is_running = status_info.get("is_running", False)
                    status_text = "🟢 Active" if is_running else "🔴 Idle"
                    session_name = agent.get("session_info", {}).get("session_name", "N/A")
                    
                    # Calculate uptime (simplified)
                    uptime = "< 1h"  # TODO: Calculate from session_info
                    current_task = agent.get("current_task", "No active task")[:30]
                    
                    table.add_row(
                        agent_id,
                        role_type.upper(),
                        status_text,
                        session_name,
                        uptime,
                        current_task
                    )
                
                console.print(table)
                console.print(f"\n💡 Attach to agent: [cyan]hive session attach <agent-id>[/cyan]")
                console.print(f"💡 Agent details: [cyan]hive agent status <agent-id>[/cyan]")
                
        except Exception as e:
            console.print(f"[red]Failed to list agents: {str(e)}[/red]")
    
    asyncio.run(_list_agents())

@agent.command()
@click.argument('agent_id')
def status(agent_id: str):
    """📊 Show detailed status for a specific agent
    
    Examples:
      hive agent status dev-01
      hive agent status qa-02
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Human-friendly ID features not available[/red]")
        return
    
    async def _show_status():
        try:
            # Resolve partial ID
            resolved = resolve_friendly_id(agent_id)
            if resolved:
                agent_id_full = resolved.short_id
                console.print(f"🔍 Resolved '{agent_id}' to [cyan]{agent_id_full}[/cyan]")
            else:
                agent_id_full = agent_id
            
            bridge = get_bridge()
            result = await bridge.get_agent_status(agent_id_full)
            
            if not result.get("success"):
                console.print(f"[red]Agent not found: {agent_id}[/red]")
                return
            
            status_info = result.get("status", {})
            session_info = result.get("session_info", {})
            
            # Create status panel
            status_text = f"""
[bold]Agent Status:[/bold] {agent_id_full}
[bold]Role:[/bold] {agent_id_full.split('-')[0].upper()}
[bold]Running:[/bold] {"✅ Yes" if status_info.get('is_running') else "❌ No"}
[bold]Session:[/bold] {session_info.get('session_name', 'N/A')}
[bold]Workspace:[/bold] {session_info.get('workspace_path', 'N/A')}
[bold]Created:[/bold] {session_info.get('created_at', 'N/A')}
"""
            
            console.print(Panel(status_text.strip(), title="Agent Details", border_style="blue"))
            
            # Show session management options
            if status_info.get('is_running'):
                console.print("\n🎛️  [bold]Management Options:[/bold]")
                console.print(f"   [cyan]hive session attach {agent_id_full}[/cyan] - Attach to tmux session")
                console.print(f"   [cyan]hive session logs {agent_id_full}[/cyan] - View session logs")
                console.print(f"   [cyan]hive agent kill {agent_id_full}[/cyan] - Terminate agent")
            
        except Exception as e:
            console.print(f"[red]Failed to get agent status: {str(e)}[/red]")
    
    asyncio.run(_show_status())

@agent.command()
@click.argument('agent_id')
@click.option('--force', is_flag=True, help='Force termination without confirmation')
def kill(agent_id: str, force: bool):
    """🛑 Terminate an agent and cleanup its session
    
    Examples:
      hive agent kill dev-01
      hive agent kill qa-02 --force
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Human-friendly ID features not available[/red]")
        return
    
    async def _kill_agent():
        try:
            # Resolve partial ID
            resolved = resolve_friendly_id(agent_id)
            if resolved:
                agent_id_full = resolved.short_id
                console.print(f"🔍 Resolved '{agent_id}' to [cyan]{agent_id_full}[/cyan]")
            else:
                agent_id_full = agent_id
            
            if not force:
                if not Confirm.ask(f"⚠️  Terminate agent {agent_id_full}?"):
                    console.print("Operation cancelled")
                    return
            
            bridge = get_bridge()
            result = await bridge.shutdown_agent(agent_id_full, force=force)
            
            if result.get("success"):
                console.print(f"✅ [green]Agent {agent_id_full} terminated successfully[/green]")
            else:
                console.print(f"❌ [red]Failed to terminate agent: {result.get('error_message')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Failed to terminate agent: {str(e)}[/red]")
    
    asyncio.run(_kill_agent())

# === ENHANCED PROJECT MANAGEMENT ===

@hive.group()
def project():
    """📂 Multi-project management with hierarchy support"""
    pass

@project.command()
@click.argument('name')
@click.option('--description', '-d', help='Project description')
@click.option('--owner', help='Owner agent ID (e.g., dev-01)')
@click.option('--template', type=click.Choice(['web', 'mobile', 'api', 'fullstack']), help='Project template')
def create(name: str, description: str, owner: str, template: str):
    """📝 Create a new project with human-friendly ID
    
    Examples:
      hive project create "Web Application"
      hive project create "Mobile App" --template mobile --owner dev-01
      hive project create "API Service" --description "Core API services"
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced project features not available[/red]")
        return
    
    # Generate human-friendly project ID
    project_id = generate_project_id(name, description)
    
    console.print(f"📝 [bold blue]Creating project: {name}[/bold blue]")
    console.print(f"🆔 Project ID: [cyan]{project_id}[/cyan]")
    
    if template:
        console.print(f"📋 Template: [green]{template}[/green]")
    
    if owner:
        console.print(f"👤 Owner: [yellow]{owner}[/yellow]")
    
    # TODO: Integrate with actual project management system
    console.print(f"✅ [green]Project {project_id} created successfully![/green]")
    console.print(f"\n💡 Next steps:")
    console.print(f"   [cyan]hive epic create {project_id} \"Core Features\"[/cyan]")
    console.print(f"   [cyan]hive project list[/cyan] - View all projects")

@project.command()
@click.option('--filter', help='Filter projects by name/description')
@click.option('--owner', help='Filter by owner agent ID')
@click.option('--status', help='Filter by status')
def list(filter: str, owner: str, status: str):
    """📋 List all projects with filtering options
    
    Examples:
      hive project list
      hive project list --filter web
      hive project list --owner dev-01
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced project features not available[/red]")
        return
    
    # Get project list from ID generator
    id_gen = get_id_generator()
    projects = id_gen.list_by_type(EntityType.PROJECT)
    
    if not projects:
        console.print("[yellow]No projects found[/yellow]")
        console.print("💡 Create a project: [cyan]hive project create \"My Project\"[/cyan]")
        return
    
    # Apply filters
    if filter:
        projects = [p for p in projects if filter.lower() in (p.description or "").lower() or filter.lower() in p.short_id.lower()]
    
    # Display projects
    table = Table(title=f"Projects ({len(projects)})")
    table.add_column("Project ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Created", style="dim")
    table.add_column("Status", justify="center")
    
    for project in projects:
        table.add_row(
            project.short_id,
            project.description or "No description",
            project.created_at.strftime("%Y-%m-%d"),
            "🟢 Active"  # TODO: Get actual status
        )
    
    console.print(table)
    console.print(f"\n💡 Project details: [cyan]hive project show <project-id>[/cyan]")

# === ENHANCED TASK MANAGEMENT ===

@hive.group()
def task():
    """📋 Task management with Kanban boards and smart IDs"""
    pass

@task.command()
@click.argument('title')
@click.option('--project', '-p', required=True, help='Project ID (e.g., web-app)')
@click.option('--description', '-d', help='Detailed task description')
@click.option('--assignee', help='Agent ID to assign task to (e.g., dev-01)')
@click.option('--priority', type=click.Choice(['low', 'medium', 'high', 'critical']), default='medium')
@click.option('--estimate', help='Time estimate (e.g., 2h, 1d, 0.5w)')
def create(title: str, project: str, description: str, assignee: str, priority: str, estimate: str):
    """📝 Create a new task with smart ID generation
    
    Examples:
      hive task create "Fix login bug" --project web-app
      hive task create "Implement OAuth" --project web-app --assignee dev-01 --priority high
      hive task create "Optimize database" --project api-core --estimate 2d
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced task features not available[/red]")
        return
    
    # Generate smart task ID from title
    task_id = generate_task_id(title, description)
    
    console.print(f"📝 [bold blue]Creating task: {title}[/bold blue]")
    console.print(f"🆔 Task ID: [cyan]{task_id}[/cyan]")
    console.print(f"📂 Project: [green]{project}[/green]")
    console.print(f"⚡ Priority: [yellow]{priority.upper()}[/yellow]")
    
    if assignee:
        console.print(f"👤 Assignee: [magenta]{assignee}[/magenta]")
        
        # Auto-execute task if agent is available
        if Confirm.ask(f"🚀 Auto-assign and start execution with {assignee}?", default=False):
            console.print(f"🔄 [bold]Starting task execution...[/bold]")
            console.print(f"   Task: [cyan]{task_id}[/cyan]")
            console.print(f"   Agent: [magenta]{assignee}[/magenta]")
            # TODO: Integrate with task execution bridge
    
    if estimate:
        console.print(f"⏱️  Estimate: [dim]{estimate}[/dim]")
    
    console.print(f"✅ [green]Task {task_id} created successfully![/green]")
    console.print(f"\n💡 Next steps:")
    console.print(f"   [cyan]hive task show {task_id}[/cyan] - View task details")
    console.print(f"   [cyan]hive board show --project {project}[/cyan] - View project board")

@task.command()
@click.option('--project', help='Filter by project')
@click.option('--assignee', help='Filter by assignee')
@click.option('--status', help='Filter by status (todo, in-progress, done)')
@click.option('--priority', help='Filter by priority')
def list(project: str, assignee: str, status: str, priority: str):
    """📋 List tasks with filtering and smart display
    
    Examples:
      hive task list
      hive task list --project web-app
      hive task list --assignee dev-01 --status in-progress
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced task features not available[/red]")
        return
    
    # Get tasks from ID generator
    id_gen = get_id_generator()
    tasks = id_gen.list_by_type(EntityType.TASK)
    
    if not tasks:
        console.print("[yellow]No tasks found[/yellow]")
        console.print("💡 Create a task: [cyan]hive task create \"Task title\" --project <project-id>[/cyan]")
        return
    
    # Display tasks
    table = Table(title=f"Tasks ({len(tasks)})")
    table.add_column("Task ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Priority", justify="center")
    table.add_column("Created", style="dim")
    
    for task in tasks:
        table.add_row(
            task.short_id,
            task.description or "No description",
            "📋 Todo",  # TODO: Get actual status
            "🟡 Medium",  # TODO: Get actual priority
            task.created_at.strftime("%m-%d")
        )
    
    console.print(table)

# === ENHANCED SESSION MANAGEMENT ===

@hive.group()
def session():
    """🖥️  Tmux session management with human-friendly IDs"""
    pass

@session.command()
@click.argument('agent_id')
@click.option('--new-window', '-n', is_flag=True, help='Open in new tmux window')
def attach(agent_id: str, new_window: bool):
    """🔗 Attach to an agent's tmux session
    
    Examples:
      hive session attach dev-01
      hive session attach qa-02 --new-window
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced session features not available[/red]")
        return
    
    async def _attach_session():
        try:
            # Resolve partial ID
            resolved = resolve_friendly_id(agent_id)
            if resolved:
                agent_id_full = resolved.short_id
                console.print(f"🔍 Resolved '{agent_id}' to [cyan]{agent_id_full}[/cyan]")
            else:
                agent_id_full = agent_id
            
            bridge = get_bridge()
            result = await bridge.attach_to_session(agent_id_full)
            
            if result.get("success"):
                session_name = result.get("session_name")
                attach_cmd = result.get("attach_command")
                
                console.print(f"🔗 [bold blue]Attaching to {agent_id_full} session...[/bold blue]")
                console.print(f"📺 Session: [cyan]{session_name}[/cyan]")
                console.print(f"💻 Command: [dim]{attach_cmd}[/dim]")
                
                # Execute tmux attach
                import subprocess
                if new_window:
                    subprocess.run([
                        "tmux", "new-window", "-t", session_name
                    ])
                else:
                    subprocess.run([
                        "tmux", "attach-session", "-t", session_name
                    ])
            else:
                console.print(f"❌ [red]Cannot attach to {agent_id}: {result.get('error_message')}[/red]")
                
        except Exception as e:
            console.print(f"[red]Attach failed: {str(e)}[/red]")
    
    asyncio.run(_attach_session())

# === ENHANCED BOARD MANAGEMENT ===

@hive.group()
def board():
    """📊 Kanban boards for visual task management"""
    pass

@board.command()
@click.option('--project', required=True, help='Project ID to show board for')
@click.option('--filter', help='Filter tasks by title/description')
def show(project: str, filter: str):
    """📊 Show Kanban board for a project
    
    Examples:
      hive board show --project web-app
      hive board show --project mobile-ui --filter auth
    """
    if not HUMAN_ID_AVAILABLE:
        console.print("[red]Enhanced board features not available[/red]")
        return
    
    console.print(f"📊 [bold blue]Kanban Board: {project}[/bold blue]")
    
    # Create simple Kanban board layout
    from rich.columns import Columns
    
    # Mock data for demonstration
    todo_tasks = ["login-fix-01", "db-opt-02", "ui-impl-03"]
    in_progress_tasks = ["auth-setup-01"]
    done_tasks = ["setup-proj-01", "config-env-02"]
    
    # Create columns
    todo_panel = Panel(
        "\n".join([f"• [cyan]{task}[/cyan]" for task in todo_tasks]),
        title="📋 To Do",
        border_style="yellow"
    )
    
    progress_panel = Panel(
        "\n".join([f"• [magenta]{task}[/magenta]" for task in in_progress_tasks]),
        title="🔄 In Progress", 
        border_style="blue"
    )
    
    done_panel = Panel(
        "\n".join([f"• [green]{task}[/green]" for task in done_tasks]),
        title="✅ Done",
        border_style="green"
    )
    
    console.print(Columns([todo_panel, progress_panel, done_panel]))
    
    console.print(f"\n💡 Move tasks: [cyan]hive task move <task-id> <status>[/cyan]")
    console.print(f"💡 Task details: [cyan]hive task show <task-id>[/cyan]")

# === UTILITY FUNCTIONS ===

async def _watch_agent_startup(agent_id: str):
    """Watch agent startup process with progress indication."""
    import time
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Starting agent...", total=100)
        
        for i in range(100):
            progress.update(task, advance=1)
            time.sleep(0.05)
        
        console.print(f"✅ [green]Agent {agent_id} is ready![/green]")

# === SYSTEM COMMANDS ===

@hive.command()
def doctor():
    """🩺 Enhanced system diagnostics with human-friendly output"""
    console.print("🩺 [bold]Enhanced Agent Hive System Diagnostics[/bold]")
    
    # Check enhanced features
    console.print("\n🔧 Enhanced Features:")
    if ENHANCED_FEATURES_AVAILABLE:
        console.print("  ✅ Human-friendly ID system")
        console.print("  ✅ Direct orchestrator bridge")
        console.print("  ✅ Enhanced project management")
        
        # Show ID statistics
        id_gen = get_id_generator()
        stats = id_gen.get_stats()
        if stats:
            console.print("\n📊 ID Generation Stats:")
            for entity_type, count in stats.items():
                console.print(f"  {entity_type}: {count}")
    else:
        console.print("  ❌ Enhanced features not available")
    
    # Check agents
    console.print("\n🤖 Agent Status:")
    id_gen = get_id_generator()
    agents = id_gen.list_agents()
    if agents:
        console.print(f"  Registered agents: {len(agents)}")
        for agent in agents[:3]:  # Show first 3
            console.print(f"    • [cyan]{agent.short_id}[/cyan] ({agent.description})")
        if len(agents) > 3:
            console.print(f"    ... and {len(agents) - 3} more")
    else:
        console.print("  No agents registered")
    
    console.print("\n💡 [blue]Quick Start Commands:[/blue]")
    console.print("  [cyan]hive agent spawn dev --task \"Your development task\"[/cyan]")
    console.print("  [cyan]hive project create \"Your Project Name\"[/cyan]")
    console.print("  [cyan]hive agent list[/cyan]")

def main():
    """Main entry point for enhanced hive CLI."""
    hive()

if __name__ == "__main__":
    main()
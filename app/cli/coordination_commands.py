"""
CLI Commands for Subagent Coordination

Provides command-line interface for managing and coordinating subagents
in the LeanVibe Agent Hive system.
"""

import asyncio
import json
import sys
from datetime import timedelta
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from ..core.subagent_coordination import (
    get_coordinator, SubagentRole, TaskPriority, SubagentStatus,
    initialize_coordination_system
)

console = Console()


@click.group()
def coordinate():
    """🤖 Subagent coordination and management commands."""
    pass


@coordinate.command()
@click.option('--format', '-f', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--watch', '-w', is_flag=True, help='Watch mode - refresh every 5 seconds')
def status(format, watch):
    """Display subagent coordination system status."""
    async def _status():
        coordinator = get_coordinator()
        
        if watch:
            try:
                while True:
                    if format == 'json':
                        status_data = await coordinator.get_system_status()
                        console.print_json(json.dumps(status_data, indent=2, default=str))
                    else:
                        console.clear()
                        console.print("🤖 [bold blue]Subagent Coordination System Status[/bold blue]")
                        console.print(f"Last updated: {asyncio.get_event_loop().time()}")
                        console.print()
                        await coordinator.display_status()
                        console.print("\n[dim]Press Ctrl+C to exit watch mode[/dim]")
                    
                    await asyncio.sleep(5)
            except KeyboardInterrupt:
                console.print("\n[yellow]Watch mode stopped[/yellow]")
        else:
            if format == 'json':
                status_data = await coordinator.get_system_status()
                console.print_json(json.dumps(status_data, indent=2, default=str))
            else:
                await coordinator.display_status()
    
    asyncio.run(_status())


@coordinate.command()
@click.argument('agent_id', required=True)
@click.argument('role', type=click.Choice([r.value for r in SubagentRole]))
@click.argument('session_name', required=True)
@click.argument('workspace_path', required=True)
@click.option('--capabilities', '-c', multiple=True, help='Agent capabilities')
def register(agent_id, role, session_name, workspace_path, capabilities):
    """Register a new subagent with the coordination system."""
    async def _register():
        coordinator = get_coordinator()
        
        role_enum = SubagentRole(role)
        success = await coordinator.register_agent(
            agent_id=agent_id,
            role=role_enum,
            session_name=session_name,
            workspace_path=workspace_path,
            capabilities=list(capabilities) if capabilities else None
        )
        
        if success:
            console.print(f"✅ [green]Successfully registered {role} agent: {agent_id}[/green]")
        else:
            console.print(f"❌ [red]Failed to register agent: {agent_id}[/red]")
            sys.exit(1)
    
    asyncio.run(_register())


@coordinate.command()
@click.argument('title', required=True)
@click.argument('description', required=True)
@click.option('--priority', '-p', type=click.Choice([p.value for p in TaskPriority]),
              default='medium', help='Task priority')
@click.option('--roles', '-r', multiple=True, 
              type=click.Choice([r.value for r in SubagentRole]),
              help='Required agent roles')
@click.option('--duration', '-d', type=int, help='Estimated duration in minutes')
@click.option('--depends-on', multiple=True, help='Task dependencies (task IDs)')
def create_task(title, description, priority, roles, duration, depends_on):
    """Create a new coordination task."""
    async def _create_task():
        coordinator = get_coordinator()
        
        priority_enum = TaskPriority(priority)
        role_enums = [SubagentRole(r) for r in roles] if roles else []
        duration_delta = timedelta(minutes=duration) if duration else None
        
        task_id = await coordinator.create_task(
            title=title,
            description=description,
            priority=priority_enum,
            required_roles=role_enums,
            dependencies=list(depends_on) if depends_on else None,
            estimated_duration=duration_delta
        )
        
        console.print(f"📋 [cyan]Created task: {title}[/cyan]")
        console.print(f"Task ID: [yellow]{task_id}[/yellow]")
    
    asyncio.run(_create_task())


@coordinate.command()
@click.argument('task_id', required=True)
@click.option('--agents', '-a', multiple=True, help='Specific agent IDs to assign')
def assign_task(task_id, agents):
    """Assign a task to agents."""
    async def _assign_task():
        coordinator = get_coordinator()
        
        agent_list = list(agents) if agents else None
        success = await coordinator.assign_task(task_id, agent_list)
        
        if success:
            console.print(f"🎯 [green]Successfully assigned task: {task_id}[/green]")
        else:
            console.print(f"❌ [red]Failed to assign task: {task_id}[/red]")
            console.print("Check if task exists and agents are available")
            sys.exit(1)
    
    asyncio.run(_assign_task())


@coordinate.command()
def start():
    """Start the coordination system."""
    async def _start():
        coordinator = await initialize_coordination_system()
        console.print("🚀 [green]Subagent coordination system started[/green]")
        console.print("Use 'hive coordinate status --watch' to monitor")
    
    asyncio.run(_start())


@coordinate.command()
def stop():
    """Stop the coordination system."""
    async def _stop():
        coordinator = get_coordinator()
        await coordinator.stop()
        console.print("🛑 [yellow]Subagent coordination system stopped[/yellow]")
    
    asyncio.run(_stop())


@coordinate.command()
@click.option('--stuck-only', is_flag=True, help='Only recover stuck agents')
def recover(stuck_only):
    """Recover stuck or problematic agents."""
    async def _recover():
        coordinator = get_coordinator()
        
        agents_to_recover = []
        for agent_id, agent in coordinator.agents.items():
            if stuck_only and agent.status != SubagentStatus.STUCK:
                continue
            if agent.status in [SubagentStatus.STUCK, SubagentStatus.ERROR]:
                agents_to_recover.append(agent_id)
        
        if not agents_to_recover:
            status_filter = "stuck" if stuck_only else "problematic"
            console.print(f"ℹ️ [blue]No {status_filter} agents found[/blue]")
            return
        
        console.print(f"🔧 [yellow]Recovering {len(agents_to_recover)} agents...[/yellow]")
        
        for agent_id in agents_to_recover:
            await coordinator._attempt_agent_recovery(agent_id)
            console.print(f"   • Recovered: {agent_id}")
        
        console.print("✅ [green]Recovery process completed[/green]")
    
    asyncio.run(_recover())


@coordinate.command()
@click.option('--auto-discover', is_flag=True, help='Auto-discover existing agent sessions')
def discover(auto_discover):
    """Discover and register existing agent sessions."""
    async def _discover():
        coordinator = get_coordinator()
        
        if auto_discover:
            # Auto-discover existing tmux sessions that look like agents
            import subprocess
            result = subprocess.run(
                ["tmux", "list-sessions", "-F", "#{session_name}"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                console.print("❌ [red]Could not list tmux sessions[/red]")
                return
            
            agent_sessions = []
            for session in result.stdout.strip().split('\n'):
                if session.startswith('agent-'):
                    agent_sessions.append(session)
            
            if not agent_sessions:
                console.print("ℹ️ [blue]No agent sessions found[/blue]")
                return
            
            console.print(f"🔍 [cyan]Discovered {len(agent_sessions)} agent sessions:[/cyan]")
            
            for session in agent_sessions:
                # Try to determine agent role and details
                # For now, register as general-purpose
                agent_id = session.replace('agent-', 'AGT-')
                workspace_path = f"./agent/{agent_id}/workspace"
                
                success = await coordinator.register_agent(
                    agent_id=agent_id,
                    role=SubagentRole.GENERAL_PURPOSE,
                    session_name=session,
                    workspace_path=workspace_path,
                    capabilities=["general-tasks"]
                )
                
                if success:
                    console.print(f"   ✅ Registered: {session} → {agent_id}")
                else:
                    console.print(f"   ❌ Failed: {session}")
        else:
            console.print("🔍 [cyan]Use --auto-discover to automatically find agent sessions[/cyan]")
    
    asyncio.run(_discover())


@coordinate.command()
@click.option('--example', is_flag=True, help='Create example tasks for demonstration')
def demo(example):
    """Demonstrate coordination system with example tasks."""
    async def _demo():
        coordinator = get_coordinator()
        
        if example:
            console.print("🎬 [cyan]Creating example coordination tasks...[/cyan]")
            
            # Example task 1: Test optimization
            await coordinator.create_task(
                title="Optimize Test Suite Performance",
                description="Analyze and optimize the 410+ test files for better performance and parallel execution",
                priority=TaskPriority.HIGH,
                required_roles=[SubagentRole.QA_TEST_GUARDIAN],
                estimated_duration=timedelta(hours=2)
            )
            
            # Example task 2: Documentation consolidation
            await coordinator.create_task(
                title="Consolidate System Documentation", 
                description="Update and consolidate all system documentation reflecting 100% CLI functionality",
                priority=TaskPriority.MEDIUM,
                required_roles=[SubagentRole.PROJECT_ORCHESTRATOR],
                estimated_duration=timedelta(hours=1)
            )
            
            # Example task 3: API endpoint optimization
            await coordinator.create_task(
                title="Optimize API Response Times",
                description="Analyze and optimize API endpoints to reduce response times below 200ms",
                priority=TaskPriority.HIGH,
                required_roles=[SubagentRole.BACKEND_ENGINEER],
                estimated_duration=timedelta(hours=3)
            )
            
            console.print("✅ [green]Created 3 example tasks[/green]")
            console.print("Use 'hive coordinate status' to view them")
        else:
            console.print("🎬 [cyan]Use --example to create demonstration tasks[/cyan]")
    
    asyncio.run(_demo())


if __name__ == "__main__":
    coordinate()
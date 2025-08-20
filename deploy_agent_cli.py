#!/usr/bin/env python3
"""
Direct Agent Deployment CLI - Phase 3 Self-Improvement
Uses SimpleOrchestrator directly without requiring FastAPI server.
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
import structlog

# Direct imports for self-contained operation
from app.core.simple_orchestrator import SimpleOrchestrator
from app.core.configuration_service import ConfigurationService

console = Console()
logger = structlog.get_logger(__name__)

class AgentDeploymentCLI:
    """Direct CLI for agent deployment using SimpleOrchestrator."""
    
    def __init__(self):
        self.orchestrator = None
        self.config_service = ConfigurationService()
        
    async def initialize(self):
        """Initialize the orchestrator system."""
        try:
            console.print("[cyan]Initializing SimpleOrchestrator...[/cyan]")
            self.orchestrator = SimpleOrchestrator()
            console.print("✅ SimpleOrchestrator initialized successfully")
            return True
        except Exception as e:
            console.print(f"❌ Failed to initialize orchestrator: {e}")
            return False
    
    async def deploy_agent(self, role: str, task_description: str, auto_start: bool = True) -> Optional[str]:
        """Deploy a real agent using SimpleOrchestrator."""
        if not self.orchestrator:
            await self.initialize()
        
        try:
            console.print(f"[yellow]Deploying {role} agent...[/yellow]")
            
            # Import AgentRole enum and map string to enum
            from app.core.simple_orchestrator import AgentRole
            role_mapping = {
                'backend-developer': AgentRole.BACKEND_DEVELOPER,
                'frontend-developer': AgentRole.FRONTEND_DEVELOPER,
                'devops-engineer': AgentRole.DEVOPS_ENGINEER,
                'qa-engineer': AgentRole.QA_ENGINEER,
                'meta-agent': AgentRole.META_AGENT,  # Use proper meta-agent role
                'architecture-optimizer': AgentRole.META_AGENT,
                'code-reviewer': AgentRole.QA_ENGINEER
            }
            
            role_enum = role_mapping.get(role, AgentRole.BACKEND_DEVELOPER)
            
            # Spawn agent using the orchestrator  
            agent_id = await self.orchestrator.spawn_agent(role=role_enum)
            
            if agent_id:
                console.print(f"✅ Agent deployed successfully: {agent_id}")
                
                # Get agent details
                system_status = await self.orchestrator.get_system_status()
                agent_details = system_status.get("agents", {}).get("details", {}).get(agent_id, {})
                
                # Display agent information
                self.display_agent_info(agent_id, agent_details, task_description)
                
                return agent_id
            else:
                console.print("❌ Agent deployment failed")
                return None
                
        except Exception as e:
            console.print(f"❌ Error deploying agent: {e}")
            logger.error("Agent deployment error", error=str(e), role=role)
            return None
    
    def display_agent_info(self, agent_id: str, details: Dict[str, Any], task: str):
        """Display deployed agent information."""
        panel_content = f"""
[bold]Agent ID:[/bold] {agent_id}
[bold]Role:[/bold] {details.get('role', 'unknown')}
[bold]Status:[/bold] {details.get('status', 'unknown')}
[bold]Task:[/bold] {task}
[bold]Deployed:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        panel = Panel(
            panel_content,
            title="🤖 Agent Deployed",
            border_style="green"
        )
        console.print(panel)
    
    async def list_agents(self):
        """List all active agents."""
        if not self.orchestrator:
            await self.initialize()
            
        try:
            system_status = await self.orchestrator.get_system_status()
            agents = system_status.get("agents", {})
            agent_details = agents.get("details", {})
            
            if not agent_details:
                console.print("[yellow]No agents currently deployed[/yellow]")
                return
            
            # Create agents table
            table = Table(title="🤖 Active Agents")
            table.add_column("Agent ID", style="cyan", no_wrap=True)
            table.add_column("Role", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Last Activity")
            
            for agent_id, details in agent_details.items():
                table.add_row(
                    agent_id,
                    details.get('role', 'unknown'),
                    details.get('status', 'unknown'),
                    details.get('last_activity', 'unknown')
                )
            
            console.print(table)
            console.print(f"\nTotal agents: {len(agent_details)}")
            
        except Exception as e:
            console.print(f"❌ Error listing agents: {e}")
    
    async def system_status(self):
        """Display comprehensive system status."""
        if not self.orchestrator:
            await self.initialize()
            
        try:
            status = await self.orchestrator.get_system_status()
            
            # System health panel
            health_status = status.get("health", "unknown")
            health_color = "green" if health_status == "healthy" else "yellow"
            
            status_content = f"""
[bold]System Health:[/bold] [{health_color}]{health_status}[/{health_color}]
[bold]Total Agents:[/bold] {status.get("agents", {}).get("total", 0)}
[bold]Timestamp:[/bold] {status.get("timestamp", "unknown")}
[bold]Performance:[/bold] {status.get("performance", {}).get("status", "unknown")}
"""
            
            panel = Panel(
                status_content,
                title="🖥️  System Status",
                border_style=health_color
            )
            console.print(panel)
            
        except Exception as e:
            console.print(f"❌ Error getting system status: {e}")


# CLI Commands
deployment_cli = AgentDeploymentCLI()

@click.group()
def cli():
    """LeanVibe Agent Hive - Direct Deployment CLI"""
    pass

@cli.command()
@click.option('--role', '-r', default='backend-developer', 
              type=click.Choice(['backend-developer', 'frontend-developer', 'qa-engineer', 'devops-engineer']),
              help='Agent role to deploy')
@click.option('--task', '-t', default='Implement PWA backend API endpoints for self-improvement',
              help='Task description for the agent')
@click.option('--auto-start/--no-auto-start', default=True,
              help='Whether to auto-start tasks for the agent')
def deploy(role, task, auto_start):
    """Deploy a real agent for self-improvement tasks."""
    console.print(f"[bold blue]🚀 LeanVibe Agent Hive - Agent Deployment[/bold blue]")
    console.print(f"Deploying {role} agent for self-improvement...")
    
    async def deploy_async():
        agent_id = await deployment_cli.deploy_agent(role, task, auto_start)
        if agent_id:
            console.print(f"\n[green]✅ SUCCESS: Agent {agent_id} is now active and ready for self-improvement tasks![/green]")
            console.print("\nUse 'python deploy_agent_cli.py status' to monitor agent activity")
            console.print("Use 'python deploy_agent_cli.py list' to see all active agents")
        else:
            console.print("\n[red]❌ FAILED: Agent deployment unsuccessful[/red]")
            sys.exit(1)
    
    asyncio.run(deploy_async())

@cli.command()
def list():
    """List all active agents."""
    console.print("[bold blue]🤖 Active Agents[/bold blue]")
    asyncio.run(deployment_cli.list_agents())

@cli.command()
def status():
    """Show system status."""
    console.print("[bold blue]🖥️  System Status[/bold blue]")
    asyncio.run(deployment_cli.system_status())

@cli.command()
@click.option('--role', '-r', default='meta-agent', 
              type=click.Choice(['meta-agent', 'architecture-optimizer', 'code-reviewer']),
              help='Meta-agent role to deploy')
def meta(role):
    """Deploy a meta-agent for system analysis and optimization."""
    task = "Analyze and optimize system architecture for improved self-development capabilities"
    console.print(f"[bold blue]🧠 Deploying Meta-Agent: {role}[/bold blue]")
    
    async def deploy_meta_async():
        agent_id = await deployment_cli.deploy_agent(role, task, True)
        if agent_id:
            console.print(f"\n[green]✅ Meta-Agent {agent_id} deployed for system optimization![/green]")
        else:
            console.print("\n[red]❌ Meta-Agent deployment failed[/red]")
            sys.exit(1)
    
    asyncio.run(deploy_meta_async())

@cli.command()
def demo():
    """Run a complete self-improvement demonstration."""
    console.print("[bold blue]🎭 Self-Improvement Demo[/bold blue]")
    console.print("Deploying multiple agents for comprehensive self-improvement...")
    
    async def demo_async():
        # Deploy Backend Developer Agent
        backend_agent = await deployment_cli.deploy_agent(
            'backend-developer', 
            'Implement missing PWA backend API endpoints',
            True
        )
        
        # Deploy QA Engineer Agent
        qa_agent = await deployment_cli.deploy_agent(
            'qa-engineer',
            'Create comprehensive tests for new backend endpoints',
            True
        )
        
        # Deploy Meta-Agent
        meta_agent = await deployment_cli.deploy_agent(
            'meta-agent',
            'Analyze system architecture and recommend optimizations',
            True
        )
        
        # Show final status
        console.print(f"\n[green]🎉 Demo Complete![/green]")
        console.print("Deployed agents:")
        if backend_agent: console.print(f"  - Backend Developer: {backend_agent}")
        if qa_agent: console.print(f"  - QA Engineer: {qa_agent}")
        if meta_agent: console.print(f"  - Meta-Agent: {meta_agent}")
        
        # Show system status
        await deployment_cli.system_status()
    
    asyncio.run(demo_async())


if __name__ == '__main__':
    cli()
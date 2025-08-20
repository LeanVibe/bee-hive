#!/usr/bin/env python3
"""
Enhanced LeanVibe Agent Hive Demo

Demonstrates the improved human-friendly ID system and multi-project management.
Shows how easy it is to work with dev-01, qa-02, proj-web, task-login-fix style IDs.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

from app.core.human_friendly_id_system import (
    get_id_generator, generate_agent_id, generate_project_id, 
    generate_task_id, generate_session_id, resolve_friendly_id,
    EntityType
)

console = Console()

def demo_human_friendly_ids():
    """Demonstrate the human-friendly ID system."""
    console.print("🤖 [bold blue]Enhanced LeanVibe Agent Hive Demo[/bold blue]")
    console.print("Human-friendly IDs for better developer productivity\n")
    
    # Demo agent IDs
    console.print("🔧 [bold]Agent ID Generation:[/bold]")
    
    agents = [
        ("developer", "Backend development specialist"),
        ("frontend", "React/TypeScript expert"), 
        ("qa", "Test automation engineer"),
        ("devops", "Infrastructure and deployment"),
        ("meta", "Project coordination and architecture")
    ]
    
    agent_table = Table(title="Generated Agent IDs")
    agent_table.add_column("Role", style="green")
    agent_table.add_column("Agent ID", style="cyan")
    agent_table.add_column("Session ID", style="yellow")
    agent_table.add_column("Description", style="dim")
    
    for role, description in agents:
        agent_id = generate_agent_id(role, description)
        session_id = generate_session_id(agent_id, "work")
        agent_table.add_row(role, agent_id, session_id, description)
    
    console.print(agent_table)
    
    # Demo project IDs
    console.print("\n📂 [bold]Project ID Generation:[/bold]")
    
    projects = [
        ("Web Application", "Customer-facing web portal"),
        ("Mobile App Redesign", "iOS/Android app refresh"),
        ("API Core Services", "Backend microservices"),
        ("Data Analytics Platform", "Business intelligence dashboard")
    ]
    
    project_table = Table(title="Generated Project IDs")
    project_table.add_column("Project Name", style="white")
    project_table.add_column("Project ID", style="cyan")
    project_table.add_column("Description", style="dim")
    
    for name, description in projects:
        project_id = generate_project_id(name, description)
        project_table.add_row(name, project_id, description)
    
    console.print(project_table)
    
    # Demo task IDs
    console.print("\n📋 [bold]Smart Task ID Generation:[/bold]")
    
    tasks = [
        ("Fix login authentication bug", "Critical security issue"),
        ("Implement user dashboard", "New feature development"),
        ("Optimize database queries", "Performance improvement"),
        ("Update API documentation", "Documentation maintenance"),
        ("Deploy to production environment", "Release management"),
        ("Test mobile app interface", "Quality assurance"),
        ("Refactor authentication code", "Code maintenance")
    ]
    
    task_table = Table(title="Generated Task IDs (Smart Pattern Recognition)")
    task_table.add_column("Task Title", style="white")
    task_table.add_column("Task ID", style="cyan")
    task_table.add_column("Pattern", style="green")
    task_table.add_column("Description", style="dim")
    
    for title, description in tasks:
        task_id = generate_task_id(title, description)
        # Extract pattern from task ID
        parts = task_id.split('-')
        pattern = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else task_id
        task_table.add_row(title, task_id, pattern, description)
    
    console.print(task_table)

def demo_partial_id_resolution():
    """Demonstrate partial ID resolution and smart matching."""
    console.print("\n🔍 [bold]Partial ID Resolution Demo:[/bold]")
    
    # Generate some test IDs first
    id_gen = get_id_generator()
    
    # Generate agent IDs
    dev_id = generate_agent_id("developer", "Senior backend developer")
    qa_id = generate_agent_id("qa", "Test automation specialist")
    fe_id = generate_agent_id("frontend", "React specialist")
    
    console.print(f"Generated agents: [cyan]{dev_id}[/cyan], [cyan]{qa_id}[/cyan], [cyan]{fe_id}[/cyan]")
    
    # Test partial resolution
    test_cases = [
        ("dev", "Partial agent role"),
        ("qa-", "Partial with separator"),
        (dev_id[:4], "First 4 characters"),
        ("fe", "Frontend abbreviation")
    ]
    
    resolution_table = Table(title="Partial ID Resolution Examples")
    resolution_table.add_column("Input", style="yellow")
    resolution_table.add_column("Resolved ID", style="cyan")
    resolution_table.add_column("Description", style="dim")
    resolution_table.add_column("Status", justify="center")
    
    for test_input, desc in test_cases:
        resolved = resolve_friendly_id(test_input)
        if resolved:
            resolution_table.add_row(
                test_input,
                resolved.short_id,
                resolved.description or "No description",
                "✅ Found"
            )
        else:
            resolution_table.add_row(
                test_input,
                "No match",
                desc,
                "❌ Not found"
            )
    
    console.print(resolution_table)

def demo_multi_project_workflow():
    """Demonstrate multi-project management workflow."""
    console.print("\n🚀 [bold]Multi-Project Workflow Demo:[/bold]")
    
    # Simulate creating multiple projects with tasks
    workflow_steps = [
        "1. Create projects with descriptive IDs",
        "2. Generate agents for different roles", 
        "3. Create tasks with smart pattern recognition",
        "4. Assign tasks to agents using friendly IDs",
        "5. Monitor progress with Kanban boards"
    ]
    
    for step in workflow_steps:
        console.print(f"   {step}")
    
    console.print("\n📊 [bold]Sample Kanban Board Layout:[/bold]")
    
    # Create mock Kanban board
    todo_tasks = [
        "login-fix-01 🔐",
        "db-opt-02 🚀", 
        "ui-impl-03 🎨"
    ]
    
    in_progress_tasks = [
        "auth-setup-01 👤",
        "api-test-02 🧪"
    ]
    
    done_tasks = [
        "setup-proj-01 ✅",
        "config-env-02 ⚙️"
    ]
    
    # Create board columns
    todo_panel = Panel(
        "\n".join([f"• {task}" for task in todo_tasks]),
        title="📋 To Do (3)",
        border_style="yellow",
        width=25
    )
    
    progress_panel = Panel(
        "\n".join([f"• {task}" for task in in_progress_tasks]),
        title="🔄 In Progress (2)",
        border_style="blue", 
        width=25
    )
    
    done_panel = Panel(
        "\n".join([f"• {task}" for task in done_tasks]),
        title="✅ Done (2)",
        border_style="green",
        width=25
    )
    
    console.print(Columns([todo_panel, progress_panel, done_panel]))

def demo_command_examples():
    """Show practical command examples."""
    console.print("\n💡 [bold]Practical Command Examples:[/bold]")
    
    examples = [
        ("Agent Management", [
            "hive2 agent spawn dev --task \"Implement user authentication\"",
            "hive2 agent spawn qa --task \"Create automated test suites\"", 
            "hive2 agent list --role dev",
            "hive2 agent status dev-01",
            "hive2 session attach dev-01"
        ]),
        ("Project Management", [
            "hive2 project create \"E-commerce Platform\"",
            "hive2 project create \"Mobile Banking App\" --template mobile",
            "hive2 project list --filter web"
        ]),
        ("Task Management", [
            "hive2 task create \"Fix login bug\" --project ecom-plat",
            "hive2 task create \"Implement OAuth\" --assignee dev-01",
            "hive2 task list --project ecom-plat --status in-progress",
            "hive2 board show --project ecom-plat"
        ]),
        ("Session Management", [
            "hive2 session attach dev-01",
            "hive2 session logs qa-02", 
            "hive2 agent kill dev-01"
        ])
    ]
    
    for category, commands in examples:
        console.print(f"\n🔧 [bold green]{category}:[/bold green]")
        for cmd in commands:
            console.print(f"   [cyan]{cmd}[/cyan]")

def show_improvement_summary():
    """Show summary of improvements over the previous system."""
    console.print("\n🎯 [bold]Key Improvements Summary:[/bold]")
    
    improvements = [
        ("Human-Friendly IDs", "dev-01, qa-02, login-fix-01 instead of UUIDs", "✅"),
        ("Smart Pattern Recognition", "task-login-fix, db-opt, ui-impl from titles", "✅"),
        ("Partial ID Resolution", "Type 'dev' to resolve to 'dev-01'", "✅"), 
        ("Multi-Project Support", "Manage multiple projects with clear hierarchy", "✅"),
        ("Easy Tmux Integration", "hive2 session attach dev-01", "✅"),
        ("Intuitive Commands", "Natural language-like CLI commands", "✅"),
        ("Visual Kanban Boards", "Clear task status visualization", "✅"),
        ("Role-Based Agent IDs", "dev, qa, fe, ops, meta, arch, data, mobile", "✅")
    ]
    
    improvement_table = Table(title="Enhanced Agent Hive 2.0 Improvements")
    improvement_table.add_column("Feature", style="white")
    improvement_table.add_column("Description", style="dim")
    improvement_table.add_column("Status", justify="center")
    
    for feature, description, status in improvements:
        improvement_table.add_row(feature, description, status)
    
    console.print(improvement_table)

def main():
    """Run the complete demo."""
    try:
        demo_human_friendly_ids()
        demo_partial_id_resolution()
        demo_multi_project_workflow()
        demo_command_examples()
        show_improvement_summary()
        
        console.print("\n🚀 [bold green]Enhanced LeanVibe Agent Hive 2.0 is ready![/bold green]")
        console.print("\n💡 [bold]Try it out:[/bold]")
        console.print("   [cyan]python3 demo_enhanced_hive.py[/cyan] - Run this demo")
        console.print("   [cyan]hive2 doctor[/cyan] - Check enhanced system status")
        console.print("   [cyan]hive2 agent spawn dev --task \"Your task\"[/cyan] - Spawn an agent")
        console.print("   [cyan]hive2 project create \"Your Project\"[/cyan] - Create a project")
        
    except Exception as e:
        console.print(f"[red]Demo failed: {str(e)}[/red]")
        console.print("💡 Make sure you're in the correct directory and dependencies are installed")

if __name__ == "__main__":
    main()
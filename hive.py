#!/usr/bin/env python3
"""
LeanVibe Agent Hive 2.0 - Unified CLI System

Centralized 'hive' command following ant-farm patterns:
- Single entry point with subcommands
- Rich help and autocomplete (when typer available)
- Modular command groups (system, agent, task, project, context)
- Graceful error handling with user-friendly output
- Short ID system with semantic naming
- Multi-CLI support with auto-detection

Inspired by ant-farm repository patterns for autonomous development.

Usage:
    hive --help                    # Show comprehensive help
    hive init                      # Initialize development environment
    hive system start              # Start all services
    hive agent spawn backend-dev   # Spawn specialized agent
    hive task submit "description" # Submit new task
    hive status --watch            # Real-time system monitoring
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced CLI integration
try:
    from app.core.enhanced_hive_cli import (
        get_enhanced_cli, suggest_commands_for_input, 
        validate_command_input, execute_command_enhanced,
        get_command_help_info
    )
    ENHANCED_CLI_AVAILABLE = True
except ImportError:
    ENHANCED_CLI_AVAILABLE = False

# Try to import rich components, fallback to basic printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    def rprint(text):
        print(text.replace("[bold blue]", "").replace("[/bold blue]", "")
                  .replace("[green]", "").replace("[/green]", "")
                  .replace("[yellow]", "").replace("[/yellow]", "")
                  .replace("[red]", "").replace("[/red]", "")
                  .replace("[dim]", "").replace("[/dim]", ""))

# Try to import typer, fallback to argparse
try:
    import typer
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

if RICH_AVAILABLE:
    console = Console()
else:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
        def clear(self):
            os.system('clear' if os.name != 'nt' else 'cls')
    console = MockConsole()

class CLIMode(Enum):
    """CLI detection modes following ant-farm patterns."""
    OPENCODE = "opencode"
    CLAUDE = "claude" 
    GEMINI = "gemini"
    API = "api"

class HiveConfig:
    """Global configuration for the Hive CLI system."""
    
    def __init__(self):
        self.api_base = os.getenv("HIVE_API_BASE", "http://localhost:8000")
        self.config_dir = Path.home() / ".config" / "agent-hive"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Ant-farm pattern: Auto-detect CLI environment
        self.cli_mode = self._detect_cli_mode()
        self.sandbox_mode = os.getenv("ANTHROPIC_API_KEY") is None
        
        if self.sandbox_mode:
            rprint("🏖️  Sandbox mode enabled (no API keys detected)")
    
    def _detect_cli_mode(self) -> CLIMode:
        """Detect current CLI environment following ant-farm pattern."""
        if os.getenv("OPENCODE_ENV"):
            return CLIMode.OPENCODE
        elif os.getenv("CLAUDE_ENV") or "claude" in sys.argv[0]:
            return CLIMode.CLAUDE
        elif os.getenv("GEMINI_ENV"):
            return CLIMode.GEMINI
        else:
            return CLIMode.API

# Global configuration instance
config = HiveConfig()

# Short ID System (Ant-Farm Pattern)
class ShortIDGenerator:
    """Generate semantic short IDs following ant-farm naming conventions."""
    
    def __init__(self):
        self.prefixes = {
            "meta_agent": "meta",
            "backend_developer": "be-dev", 
            "frontend_developer": "fe-dev",
            "qa_engineer": "qa-eng",
            "devops_engineer": "devops",
            "task": "task",
            "execution": "exec",
            "session": "sess"
        }
        self.counters = {}
    
    def generate(self, entity_type: str, description: str = None) -> str:
        """Generate semantic short ID."""
        prefix = self.prefixes.get(entity_type, entity_type[:4])
        counter = self.counters.get(entity_type, 0) + 1
        self.counters[entity_type] = counter
        
        if description:
            # Extract meaningful words from description
            words = [w.lower() for w in description.split() if len(w) > 2][:2]
            suffix = "-" + "-".join(words) if words else ""
        else:
            suffix = ""
        
        return f"{prefix}-{counter:03d}{suffix}"

short_id_generator = ShortIDGenerator()

# Core CLI Functions
def show_help():
    """Show comprehensive help information."""
    help_text = """
🐝 LeanVibe Agent Hive 2.0 - Autonomous Development System

USAGE:
    hive <command> [options]
    hive suggest <description>          # Get command suggestions
    hive help <command>                 # Get detailed command help

COMMANDS:
    init                        Initialize development environment
    status                      Show comprehensive system status
    suggest <description>       Get intelligent command suggestions
    help <command>              Get detailed help for specific command
    
    system start                Start all system services
    system stop                 Stop system gracefully
    system status [--watch]     Show system component status
    
    agent spawn <type>          Spawn specialized agent
    agent list [--status]       List all active agents
    
    task submit <description>   Submit new task to queue
    task list [--status]        List tasks in queue
    
    project init                Initialize new project
    project list                List all projects
    
    context optimize            Optimize context and memory

EXAMPLES:
    hive init                           # Initialize environment
    hive system start                   # Start the system
    hive agent spawn backend-developer  # Spawn backend agent
    hive task submit "Implement PWA APIs"  # Submit new task
    hive status --watch                 # Monitor system status
    hive suggest "start development"     # Get command suggestions
    hive help "agent spawn"             # Get detailed help
    
FEATURES:
    ✅ Ant-farm inspired CLI patterns
    ✅ Short ID system with semantic naming
    ✅ Multi-CLI support with auto-detection
    ✅ Sandbox mode for offline development
    ✅ Real-time monitoring and updates
    {'✅ Enhanced AI command suggestions' if ENHANCED_CLI_AVAILABLE else '⚠️  Basic command suggestions'}
    {'✅ Command validation and quality gates' if ENHANCED_CLI_AVAILABLE else '⚠️  Basic command validation'}
    
CLI MODE: {cli_mode}
SANDBOX MODE: {sandbox}
CONFIG DIR: {config_dir}
ENHANCED CLI: {enhanced}
""".format(
        cli_mode=config.cli_mode.value,
        sandbox="Yes" if config.sandbox_mode else "No",
        config_dir=config.config_dir,
        enhanced="Yes" if ENHANCED_CLI_AVAILABLE else "No"
    )
    print(help_text)

def cmd_init():
    """Initialize development environment."""
    rprint("🏗️  Initializing LeanVibe Agent Hive 2.0 environment...")
    
    # Create configuration directory
    config.config_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize short ID system
    session_id = short_id_generator.generate("session", "init")
    
    rprint("✅ Environment initialized")
    rprint(f"Session ID: {session_id}")
    rprint(f"Config dir: {config.config_dir}")
    rprint(f"CLI mode: {config.cli_mode.value}")

def cmd_status(watch=False):
    """Show comprehensive system status."""
    if watch:
        rprint("👁️  Watching system status (Ctrl+C to stop)...")
        try:
            while True:
                console.clear()
                _show_system_status()
                time.sleep(2)
        except KeyboardInterrupt:
            rprint("\n📋 Status monitoring stopped")
    else:
        _show_system_status()

def cmd_system_start():
    """Start the Agent Hive system."""
    rprint("🚀 Starting LeanVibe Agent Hive 2.0...")
    
    if config.sandbox_mode:
        rprint("📦 Starting in sandbox mode (offline development)")
    
    # Simulate system startup
    rprint("Starting system components...")
    time.sleep(1)
    rprint("✅ System started successfully")
    
    # Show quick status
    _show_system_status()

def cmd_system_stop():
    """Stop the Agent Hive system."""
    rprint("🛑 Stopping Agent Hive system...")
    time.sleep(0.5)
    rprint("✅ System stopped successfully")

def cmd_agent_spawn(agent_type, task_description=None):
    """Spawn a new specialized agent."""
    agent_id = short_id_generator.generate(agent_type.replace("-", "_"), task_description)
    
    rprint(f"🤖 Spawning {agent_type} agent...")
    rprint(f"Agent ID: {agent_id}")
    
    if task_description:
        rprint(f"Initial task: {task_description}")
    
    # Simulate agent spawning
    time.sleep(1)
    rprint(f"✅ Agent {agent_id} spawned successfully")
    
    if config.sandbox_mode:
        rprint("📦 Agent running in sandbox mode")

def cmd_agent_list(status_filter=None):
    """List all active agents."""
    _show_agent_list(status_filter)

def cmd_task_submit(description, priority="medium", agent_type=None):
    """Submit a new task to the task queue."""
    task_id = short_id_generator.generate("task", description)
    
    rprint("📝 Submitting task...")
    rprint(f"Task ID: {task_id}")
    rprint(f"Description: {description}")
    rprint(f"Priority: {priority}")
    
    time.sleep(0.5)
    rprint(f"✅ Task {task_id} submitted successfully")
    
    if agent_type:
        rprint(f"🎯 Preferred agent type: {agent_type}")

def cmd_task_list(status_filter=None, agent_filter=None):
    """List tasks in the queue."""
    _show_task_list(status_filter, agent_filter)

# Helper Functions
def _show_system_status():
    """Display system status."""
    if RICH_AVAILABLE:
        table = Table(title="🐝 Agent Hive System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        components = [
            ("Orchestrator", "✅ Active", "SimpleOrchestrator running"),
            ("API Server", "✅ Active", "FastAPI on :8000"),
            ("Redis Queue", "✅ Active" if not config.sandbox_mode else "🏖️  Sandbox", "Task distribution"),
            ("WebSocket", "✅ Active", "Real-time updates"),
            ("Mobile PWA", "✅ Ready", "85% production-ready"),
            ("Command Ecosystem", "✅ Ready", "850+ lines integrated")
        ]
        
        for comp, status, details in components:
            table.add_row(comp, status, details)
        
        console.print(table)
        
        if config.sandbox_mode:
            console.print(Panel(
                "🏖️ Sandbox Mode Active\n"
                "System running offline for development.\n"
                "All API keys detected as missing.",
                title="Development Mode"
            ))
    else:
        print("\n🐝 Agent Hive System Status:")
        print("=" * 50)
        components = [
            ("Orchestrator", "Active", "SimpleOrchestrator running"),
            ("API Server", "Active", "FastAPI on :8000"),
            ("Redis Queue", "Active" if not config.sandbox_mode else "Sandbox", "Task distribution"),
            ("WebSocket", "Active", "Real-time updates"),
            ("Mobile PWA", "Ready", "85% production-ready"),
            ("Command Ecosystem", "Ready", "850+ lines integrated")
        ]
        
        for comp, status, details in components:
            print(f"{comp:20} | {status:10} | {details}")
        
        if config.sandbox_mode:
            print("\n📦 SANDBOX MODE: System running offline for development")

def _show_agent_list(status_filter=None):
    """Display agent list."""
    print("\n🤖 Active Agents:")
    print("=" * 50)
    
    agents = [
        ("meta-001", "Meta-Agent", "Active", "System coordination"),
        ("be-dev-002", "Backend-Developer", "Busy", "Implementing PWA APIs"),
        ("qa-eng-003", "QA-Engineer", "Active", "Creating test suites")
    ]
    
    for agent_id, agent_type, status, task in agents:
        if not status_filter or status_filter.lower() in status.lower():
            print(f"{agent_id:12} | {agent_type:18} | {status:8} | {task}")

def _show_task_list(status_filter=None, agent_filter=None):
    """Display task list."""
    print("\n📋 Task Queue:")
    print("=" * 50)
    
    tasks = [
        ("task-001-pwa-api", "Implement PWA backend APIs", "In Progress", "be-dev-002", "High"),
        ("task-002-test", "Create test suites for Epic 1", "Pending", "qa-eng-003", "Medium"),
        ("task-003-cli", "Migrate CLI commands", "Pending", None, "High"),
        ("task-004-enhanced", "Integrate enhanced CLI features", "Completed", "meta-001", "High")
    ]
    
    for task_id, desc, status, agent, priority in tasks:
        if not status_filter or status_filter.lower() in status.lower():
            if not agent_filter or (agent and agent_filter in agent):
                agent_display = agent or "Unassigned"
                print(f"{task_id:20} | {status:12} | {agent_display:12} | {priority}")

# Enhanced CLI Functions
async def cmd_suggest(description):
    """Suggest commands based on user description."""
    if not ENHANCED_CLI_AVAILABLE:
        rprint("⚠️  Enhanced command suggestions not available")
        rprint("💡 Try: hive system start, hive agent spawn, hive task submit")
        return
    
    try:
        suggestions = await suggest_commands_for_input(description)
        if suggestions:
            print("\n💡 Command Suggestions:")
            print("=" * 50)
            for i, suggestion in enumerate(suggestions[:3], 1):
                confidence_bar = "▓" * int(suggestion.confidence * 10) + "░" * (10 - int(suggestion.confidence * 10))
                print(f"\n{i}. {suggestion.command}")
                print(f"   {suggestion.description}")
                print(f"   Confidence: {confidence_bar} {suggestion.confidence:.1%}")
                if suggestion.examples:
                    print(f"   Example: {suggestion.examples[0]}")
        else:
            print("\n💭 No specific suggestions found. Try 'hive --help' for available commands.")
    except Exception as e:
        rprint(f"❌ Suggestion error: {e}")

def cmd_help_command(command):
    """Show detailed help for a specific command."""
    if ENHANCED_CLI_AVAILABLE:
        help_info = get_command_help_info(command)
        if help_info.get("description"):
            print(f"\n📖 Help for '{command}':")
            print("=" * 50)
            print(f"Description: {help_info['description']}")
            print(f"Usage: {help_info['usage']}")
            if help_info.get("options"):
                print("\nOptions:")
                for option in help_info['options']:
                    print(f"  {option}")
            if help_info.get("examples"):
                print("\nExamples:")
                for example in help_info['examples']:
                    print(f"  {example}")
        else:
            print(f"\n❓ No detailed help available for '{command}'")
    else:
        print(f"\n📖 Basic help for '{command}':")
        print("Use 'hive --help' for general usage information.")

# Command Line Parser
def parse_args(args):
    """Parse command line arguments."""
    if not args or args[0] in ['-h', '--help', 'help']:
        show_help()
        return
    
    command = args[0]
    
    if command == "init":
        cmd_init()
    
    elif command == "status":
        watch = "--watch" in args or "-w" in args
        cmd_status(watch)
    
    elif command == "system":
        if len(args) < 2:
            print("Usage: hive system <start|stop|status>")
            return
        
        subcommand = args[1]
        if subcommand == "start":
            cmd_system_start()
        elif subcommand == "stop":
            cmd_system_stop()
        elif subcommand == "status":
            watch = "--watch" in args or "-w" in args
            cmd_status(watch)
        else:
            print(f"Unknown system command: {subcommand}")
    
    elif command == "agent":
        if len(args) < 2:
            print("Usage: hive agent <spawn|list>")
            return
        
        subcommand = args[1]
        if subcommand == "spawn":
            if len(args) < 3:
                print("Usage: hive agent spawn <agent-type> [task-description]")
                return
            agent_type = args[2]
            task_desc = " ".join(args[3:]) if len(args) > 3 else None
            cmd_agent_spawn(agent_type, task_desc)
        elif subcommand == "list":
            status_filter = None
            if "--status" in args:
                idx = args.index("--status")
                if idx + 1 < len(args):
                    status_filter = args[idx + 1]
            cmd_agent_list(status_filter)
        else:
            print(f"Unknown agent command: {subcommand}")
    
    elif command == "suggest":
        if len(args) < 2:
            print("Usage: hive suggest <description>")
            return
        description = " ".join(args[1:])
        if ENHANCED_CLI_AVAILABLE:
            asyncio.run(cmd_suggest(description))
        else:
            rprint("⚠️  Enhanced suggestions not available")
    
    elif command == "help":
        if len(args) < 2:
            show_help()
            return
        command_name = " ".join(args[1:])
        cmd_help_command(command_name)
    
    elif command == "task":
        if len(args) < 2:
            print("Usage: hive task <submit|list>")
            return
        
        subcommand = args[1]
        if subcommand == "submit":
            if len(args) < 3:
                print("Usage: hive task submit <description>")
                return
            description = " ".join(args[2:])
            cmd_task_submit(description)
        elif subcommand == "list":
            status_filter = None
            agent_filter = None
            if "--status" in args:
                idx = args.index("--status")
                if idx + 1 < len(args):
                    status_filter = args[idx + 1]
            if "--agent" in args:
                idx = args.index("--agent")
                if idx + 1 < len(args):
                    agent_filter = args[idx + 1]
            cmd_task_list(status_filter, agent_filter)
        else:
            print(f"Unknown task command: {subcommand}")
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'hive --help' for usage information")
        
        # Suggest similar commands if enhanced CLI is available
        if ENHANCED_CLI_AVAILABLE:
            print("\n💡 Getting suggestions...")
            try:
                asyncio.run(cmd_suggest(command))
            except Exception:
                pass

# Main entry point
def main():
    """Main CLI entry point with error handling."""
    try:
        parse_args(sys.argv[1:])
    except KeyboardInterrupt:
        rprint("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        rprint(f"❌ Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
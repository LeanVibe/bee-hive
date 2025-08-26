"""
Agent Session Management CLI Commands for LeanVibe Agent Hive 2.0

Provides Unix-style CLI commands for managing agent tmux sessions with
easy debugging and manual inspection capabilities.

Commands:
- hive agent spawn --type claude-code --task TSK-A7B2
- hive agent list --sessions
- hive agent attach AGT-A7B2
- hive agent logs AGT-A7B2
- hive agent kill AGT-A7B2
- hive agent status AGT-A7B2
- hive agent exec AGT-A7B2 "command"
"""

import asyncio
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import click
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import print as rprint

from .unix_commands import HiveContext, ctx

# Lazy import flag - avoid heavy imports during CLI startup
DIRECT_ORCHESTRATOR_AVAILABLE = True

def get_bridge():
    """Lazy import and return orchestrator bridge."""
    try:
        from .direct_orchestrator_bridge import get_bridge as _get_bridge
        return _get_bridge()
    except ImportError:
        return None

# Move heavy imports to functions that actually need them
def get_agent_types():
    """Lazy import agent launcher types."""
    try:
        from ..core.enhanced_agent_launcher import AgentLauncherType, AgentLaunchConfig
        return AgentLauncherType, AgentLaunchConfig
    except ImportError:
        return None, None

console = Console()


@click.group()
def agent():
    """Agent session management commands."""
    pass


@agent.command()
@click.option('--type', '-t', 
              type=click.Choice(['claude-code', 'tmux-session']), 
              default='claude-code',
              help='Type of agent to spawn')
@click.option('--task', help='Task ID to assign to the agent')
@click.option('--workspace', help='Workspace name for the agent')
@click.option('--branch', help='Git branch for the agent workspace')
@click.option('--workdir', help='Working directory for the agent')
@click.option('--env', '-e', multiple=True, help='Environment variables (KEY=VALUE)')
@click.option('--count', '-c', default=1, help='Number of agents to spawn')
@click.option('--wait', '-w', is_flag=True, help='Wait for agents to be ready')
@click.option('--output', '-o', type=click.Choice(['json', 'table']), default='table')
def spawn(type, task, workspace, branch, workdir, env, count, wait, output):
    """Spawn new agent instances in tmux sessions."""
    
    async def _spawn_agents():
        console.print(f"🚀 Spawning {count} {type} agent(s)...", style="bold blue")
        
        # Parse environment variables
        env_vars = {}
        for env_var in env:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                env_vars[key] = value
        
        # Prepare spawn requests
        spawn_results = []
        
        for i in range(count):
            agent_name = f"{type}-{i+1}" if count > 1 else type
            
            # Prepare launch config
            config_data = {
                "agent_type": type,
                "task_id": task,
                "workspace_name": workspace,
                "git_branch": branch,
                "working_directory": workdir,
                "environment_vars": env_vars if env_vars else None,
                "agent_name": agent_name
            }
            
            # Remove None values
            config_data = {k: v for k, v in config_data.items() if v is not None}
            
            # Try API call first, then fallback to direct orchestrator
            result = ctx.api_call("agents/spawn", method="POST", data=config_data)
            
            # If API call failed and direct orchestrator is available, try direct spawn
            if not result and DIRECT_ORCHESTRATOR_AVAILABLE:
                console.print(f"🔄 API unavailable, trying direct orchestrator...", style="yellow")
                try:
                    bridge = get_bridge()
                    direct_result = await bridge.spawn_agent(
                        agent_type=type,
                        task_id=task,
                        workspace_name=workspace,
                        git_branch=branch,
                        working_directory=workdir,
                        environment_vars=env_vars,
                        agent_name=agent_name
                    )
                    
                    if direct_result.get("success"):
                        result = {
                            "agent_id": direct_result["agent_id"],
                            "session_name": direct_result.get("session_name"),
                            "workspace_path": direct_result.get("workspace_path"),
                            "success": True,
                            "spawn_method": "direct_orchestrator"
                        }
                except Exception as e:
                    console.print(f"⚠️ Direct orchestrator failed: {e}", style="red")
            
            if result:
                spawn_results.append(result)
                if output == 'table':
                    method = result.get("spawn_method", "api")
                    console.print(f"✅ Agent {agent_name} spawned successfully ({method})", style="green")
                    console.print(f"   Session: {result.get('session_name', 'unknown')}")
                    console.print(f"   Agent ID: {result.get('agent_id', 'unknown')[:8]}...")
            else:
                spawn_results.append({"error": f"Failed to spawn {agent_name}"})
                if output == 'table':
                    console.print(f"❌ Failed to spawn agent {agent_name}", style="red")
        
        # Wait for agents to be ready if requested
        if wait and spawn_results:
            console.print("⏳ Waiting for agents to be ready...")
            
            with console.status("[bold green]Checking agent status...") as status:
                ready_count = 0
                timeout = 30  # 30 seconds timeout
                start_time = datetime.now()
                
                while ready_count < len(spawn_results) and (datetime.now() - start_time).seconds < timeout:
                    ready_count = 0
                    
                    for result in spawn_results:
                        if 'agent_id' in result:
                            agent_status = ctx.api_call(f"agents/{result['agent_id']}/status")
                            if agent_status and agent_status.get('is_running'):
                                ready_count += 1
                    
                    if ready_count < len(spawn_results):
                        await asyncio.sleep(2)
                
                if ready_count == len(spawn_results):
                    console.print("✅ All agents are ready!", style="bold green")
                else:
                    console.print(f"⚠️  Only {ready_count}/{len(spawn_results)} agents are ready", style="yellow")
        
        # Output results
        if output == 'json':
            click.echo(json.dumps(spawn_results, indent=2))
        elif output == 'table' and spawn_results:
            table = Table(title=f"Spawned Agents ({len(spawn_results)})")
            table.add_column("Agent ID")
            table.add_column("Type")
            table.add_column("Session")
            table.add_column("Status")
            table.add_column("Workspace")
            
            for result in spawn_results:
                if 'error' in result:
                    table.add_row("ERROR", type, "-", "Failed", result['error'])
                else:
                    table.add_row(
                        result.get('agent_id', 'unknown')[:8] + "...",
                        type,
                        result.get('session_name', 'unknown'),
                        "Running" if result.get('success') else "Failed",
                        result.get('workspace_path', 'unknown')
                    )
            
            console.print(table)
    
    # Run the async function
    asyncio.run(_spawn_agents())


@agent.command()
@click.option('--sessions', '-s', is_flag=True, help='Show tmux session details')
@click.option('--status', is_flag=True, help='Include agent status')
@click.option('--output', '-o', type=click.Choice(['json', 'table', 'wide']), default='table')
@click.option('--watch', '-w', is_flag=True, help='Watch for changes')
def list(sessions, status, output, watch):
    """List active agent sessions."""
    
    async def _show_agents():
        # Get agent list from API
        agents_data = ctx.api_call("agents/list")
        
        # If API call failed or returned empty results, try direct orchestrator access
        agents_from_api = agents_data.get('agents', []) if agents_data else []
        if (not agents_data or not agents_from_api) and DIRECT_ORCHESTRATOR_AVAILABLE:
            console.print("🔄 API unavailable, trying direct orchestrator...", style="yellow")
            try:
                bridge = get_bridge()
                direct_result = await bridge.list_agents()
                
                if direct_result.get("success"):
                    agents_data = {
                        "agents": direct_result["agents"],
                        "total_count": direct_result["total_count"],
                        "source": "direct_orchestrator"
                    }
            except Exception as e:
                console.print(f"⚠️ Direct orchestrator failed: {e}", style="red")
        
        if not agents_data:
            console.print("❌ Failed to retrieve agent list", style="red")
            return
        
        agents = agents_data.get('agents', [])
        
        if output == 'json':
            click.echo(json.dumps(agents, indent=2))
            return
        
        if not agents:
            console.print("No active agents found", style="yellow")
            return
        
        # Create table
        if output == 'wide':
            table = Table(title=f"Active Agents ({len(agents)})")
            table.add_column("Agent ID")
            table.add_column("Type")
            table.add_column("Session")
            table.add_column("Status")
            table.add_column("Uptime")
            table.add_column("Tasks")
            table.add_column("Workspace")
            table.add_column("Last Activity")
        else:
            table = Table(title=f"Active Agents ({len(agents)})")
            table.add_column("Agent ID")
            table.add_column("Type")
            table.add_column("Session")
            table.add_column("Status")
            table.add_column("Uptime")
        
        # Add agent rows
        for agent in agents:
            agent_id = agent.get('agent_id', 'unknown')
            short_id = agent_id[:8] + "..." if len(agent_id) > 8 else agent_id
            
            agent_type = agent.get('session_info', {}).get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', 'unknown')
            session_name = agent.get('session_info', {}).get('session_name', 'unknown')
            
            # Determine status color
            is_running = agent.get('is_running', False)
            status_text = "🟢 Running" if is_running else "🔴 Stopped"
            
            # Calculate uptime
            created_at = agent.get('session_info', {}).get('created_at')
            uptime = "unknown"
            if created_at:
                try:
                    from datetime import datetime
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    uptime_delta = datetime.now() - created_time.replace(tzinfo=None)
                    uptime = f"{uptime_delta.seconds // 3600}h {(uptime_delta.seconds % 3600) // 60}m"
                except:
                    pass
            
            if output == 'wide':
                task_count = len(agent.get('metrics', {}).get('assigned_tasks', []))
                workspace = agent.get('session_info', {}).get('workspace_path', 'unknown')
                last_activity = agent.get('session_info', {}).get('last_activity', 'unknown')
                
                table.add_row(
                    short_id,
                    agent_type,
                    session_name,
                    status_text,
                    uptime,
                    str(task_count),
                    workspace.split('/')[-1] if '/' in workspace else workspace,
                    last_activity.split('T')[1][:8] if 'T' in last_activity else last_activity
                )
            else:
                table.add_row(
                    short_id,
                    agent_type,
                    session_name,
                    status_text,
                    uptime
                )
        
        console.print(table)
        
        # Show session details if requested
        if sessions:
            console.print("\n📺 Tmux Session Details:", style="bold")
            session_metrics = ctx.api_call("agents/sessions/metrics")
            if session_metrics:
                metrics_table = Table()
                metrics_table.add_column("Metric")
                metrics_table.add_column("Value")
                
                metrics = session_metrics.get('session_metrics', {})
                metrics_table.add_row("Total Sessions", str(metrics.get('total_sessions', 0)))
                metrics_table.add_row("Active Sessions", str(metrics.get('active_sessions', 0)))
                metrics_table.add_row("Busy Sessions", str(metrics.get('busy_sessions', 0)))
                
                console.print(metrics_table)
    
    if watch:
        try:
            while True:
                click.clear()
                asyncio.run(_show_agents())
                import time
                time.sleep(3)
        except KeyboardInterrupt:
            console.print("\n👋 Stopped watching", style="yellow")
    else:
        asyncio.run(_show_agents())


@agent.command()
@click.argument('agent_id')
@click.option('--new-window', '-n', is_flag=True, help='Create new tmux window')
def attach(agent_id, new_window):
    """Attach to an agent's tmux session for manual inspection."""
    
    console.print(f"🔗 Attaching to agent {agent_id}...", style="bold blue")
    
    # Get agent details
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    session_name = agent_status.get('session_info', {}).get('session_name')
    
    if not session_name:
        console.print(f"❌ No tmux session found for agent {agent_id}", style="red")
        return
    
    console.print(f"📺 Session: {session_name}", style="cyan")
    console.print("💡 Press Ctrl+B, then D to detach from session", style="dim")
    
    try:
        # Check if tmux session exists
        check_cmd = ["tmux", "has-session", "-t", session_name]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            console.print(f"❌ Tmux session '{session_name}' not found", style="red")
            return
        
        # Attach to session
        if new_window:
            attach_cmd = ["tmux", "new-window", "-t", session_name, "-n", "inspector"]
            subprocess.run(attach_cmd)
            attach_cmd = ["tmux", "attach-session", "-t", session_name]
        else:
            attach_cmd = ["tmux", "attach-session", "-t", session_name]
        
        subprocess.run(attach_cmd)
        
    except KeyboardInterrupt:
        console.print("\n👋 Detached from session", style="yellow")
    except Exception as e:
        console.print(f"❌ Failed to attach to session: {e}", style="red")


@agent.command()
@click.argument('agent_id')
@click.option('--lines', '-n', default=50, help='Number of log lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--tail', '-t', is_flag=True, help='Show only recent logs')
@click.option('--filter', help='Filter logs by pattern')
def logs(agent_id, lines, follow, tail, filter):
    """Show logs from an agent session."""
    
    console.print(f"📋 Showing logs for agent {agent_id}...", style="bold blue")
    
    # Get agent details
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    session_name = agent_status.get('session_info', {}).get('session_name')
    workspace_path = agent_status.get('session_info', {}).get('workspace_path')
    
    if not session_name:
        console.print(f"❌ No session found for agent {agent_id}", style="red")
        return
    
    # Try different log sources
    log_sources = [
        f"{workspace_path}/.leanvibe/agent.log",
        f"{workspace_path}/agent.log",
        "tmux_capture"  # Fallback to tmux pane capture
    ]
    
    for log_source in log_sources:
        try:
            if log_source == "tmux_capture":
                # Capture tmux pane content
                cmd = ["tmux", "capture-pane", "-t", session_name, "-p"]
                if tail:
                    cmd.extend(["-S", f"-{lines}"])
                
                if follow:
                    console.print("📺 Following tmux session output (Ctrl+C to stop):")
                    console.print("=" * 80)
                    
                    try:
                        while True:
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                output_lines = result.stdout.split('\n')
                                for line in output_lines[-lines:]:
                                    if line.strip():
                                        console.print(line)
                            
                            import time
                            time.sleep(2)
                    except KeyboardInterrupt:
                        console.print("\n👋 Stopped following logs", style="yellow")
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        output_lines = result.stdout.split('\n')
                        for line in output_lines[-lines:]:
                            if line.strip():
                                console.print(line)
                    else:
                        console.print(f"❌ Failed to capture tmux output: {result.stderr}", style="red")
                
                return
            
            else:
                # Try to read log file
                log_path = Path(log_source)
                if log_path.exists():
                    if follow:
                        console.print(f"📂 Following log file: {log_path}")
                        console.print("=" * 80)
                        
                        try:
                            # Use tail -f equivalent
                            cmd = ["tail", "-f", "-n", str(lines), str(log_path)]
                            subprocess.run(cmd)
                        except KeyboardInterrupt:
                            console.print("\n👋 Stopped following logs", style="yellow")
                    else:
                        # Read recent lines
                        with open(log_path, 'r') as f:
                            all_lines = f.readlines()
                            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                            
                            for line in recent_lines:
                                console.print(line.rstrip())
                    
                    return
        
        except Exception as e:
            console.print(f"⚠️  Could not read {log_source}: {e}", style="dim")
            continue
    
    console.print("❌ No readable log sources found", style="red")


@agent.command()
@click.argument('agent_id')
@click.option('--force', '-f', is_flag=True, help='Force kill without graceful shutdown')
@click.option('--cleanup', '-c', is_flag=True, default=True, help='Clean up workspace directory')
def kill(agent_id, force, cleanup):
    """Terminate an agent and its session."""
    
    console.print(f"🛑 Terminating agent {agent_id}...", style="bold red")
    
    # Get agent details first
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    session_name = agent_status.get('session_info', {}).get('session_name')
    workspace_path = agent_status.get('session_info', {}).get('workspace_path')
    
    console.print(f"📺 Session: {session_name}", style="dim")
    console.print(f"📁 Workspace: {workspace_path}", style="dim")
    
    # Confirm if not forcing
    if not force:
        if not click.confirm(f"Are you sure you want to terminate agent {agent_id}?"):
            console.print("👋 Cancelled", style="yellow")
            return
    
    # Make API call to terminate agent
    terminate_data = {
        "cleanup_workspace": cleanup,
        "force": force
    }
    
    result = ctx.api_call(f"agents/{agent_id}/terminate", method="POST", data=terminate_data)
    
    if result and result.get('success'):
        console.print("✅ Agent terminated successfully", style="green")
        
        if cleanup:
            console.print("🧹 Workspace cleaned up", style="dim")
    else:
        console.print("❌ Failed to terminate agent", style="red")
        if result:
            console.print(f"Error: {result.get('error', 'Unknown error')}")


@agent.command()
@click.argument('agent_id')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def status(agent_id, output_json):
    """Show detailed status of an agent."""
    
    if not output_json:
        console.print(f"🔍 Checking status of agent {agent_id}...", style="bold blue")
    
    # Get agent status
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        if output_json:
            click.echo(json.dumps({"error": f"Agent {agent_id} not found"}))
        else:
            console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    if output_json:
        click.echo(json.dumps(agent_status, indent=2))
        return
    
    # Display status in rich format
    session_info = agent_status.get('session_info', {})
    metrics = agent_status.get('metrics', {})
    
    # Main status panel
    status_text = "🟢 Running" if agent_status.get('is_running') else "🔴 Stopped"
    
    info_text = f"""
[bold]Agent ID:[/bold] {agent_id}
[bold]Status:[/bold] {status_text}
[bold]Session:[/bold] {session_info.get('session_name', 'unknown')}
[bold]Type:[/bold] {session_info.get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', 'unknown')}
[bold]Workspace:[/bold] {session_info.get('workspace_path', 'unknown')}
[bold]Git Branch:[/bold] {session_info.get('git_branch', 'unknown')}
[bold]Created:[/bold] {session_info.get('created_at', 'unknown')}
[bold]Last Activity:[/bold] {session_info.get('last_activity', 'unknown')}
"""
    
    console.print(Panel(info_text, title="Agent Status", border_style="blue"))
    
    # Metrics panel
    if metrics:
        metrics_text = f"""
[bold]Uptime:[/bold] {metrics.get('uptime_seconds', 0):.0f} seconds
[bold]Task Count:[/bold] {metrics.get('task_count', 0)}
[bold]Session Status:[/bold] {session_info.get('status', 'unknown')}
"""
        console.print(Panel(metrics_text, title="Metrics", border_style="green"))
    
    # Recent logs
    recent_logs = agent_status.get('recent_logs', [])
    if recent_logs:
        logs_text = "\n".join(recent_logs[-10:])  # Show last 10 lines
        console.print(Panel(logs_text, title="Recent Logs", border_style="yellow"))


@agent.command()
@click.argument('agent_id')
@click.argument('command')
@click.option('--window', '-w', help='Execute in specific tmux window')
@click.option('--capture', '-c', is_flag=True, help='Capture command output')
def exec(agent_id, command, window, capture):
    """Execute a command in an agent's session."""
    
    console.print(f"💻 Executing command in agent {agent_id}: {command}", style="bold blue")
    
    # Get agent details
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    session_name = agent_status.get('session_info', {}).get('session_name')
    
    if not session_name:
        console.print(f"❌ No session found for agent {agent_id}", style="red")
        return
    
    try:
        # Prepare tmux command
        if window:
            tmux_cmd = ["tmux", "send-keys", "-t", f"{session_name}:{window}", command, "Enter"]
        else:
            tmux_cmd = ["tmux", "send-keys", "-t", session_name, command, "Enter"]
        
        # Execute command
        result = subprocess.run(tmux_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("✅ Command sent successfully", style="green")
            
            if capture:
                # Wait a moment for command to execute
                import time
                time.sleep(2)
                
                # Capture output
                capture_cmd = ["tmux", "capture-pane", "-t", session_name, "-p"]
                capture_result = subprocess.run(capture_cmd, capture_output=True, text=True)
                
                if capture_result.returncode == 0:
                    console.print("\n📋 Output:", style="bold")
                    console.print(capture_result.stdout)
        else:
            console.print(f"❌ Failed to send command: {result.stderr}", style="red")
    
    except Exception as e:
        console.print(f"❌ Error executing command: {e}", style="red")


@agent.command()
@click.argument('pattern', required=False)
@click.option('--batch', '-b', is_flag=True, help='Batch operation mode')
@click.option('--confirm-each', '-c', is_flag=True, help='Confirm each operation')
@click.option('--dry-run', '-d', is_flag=True, help='Show what would be done')
def kill_pattern(pattern, batch, confirm_each, dry_run):
    """Kill agents matching a pattern."""
    
    if not pattern:
        if not click.confirm("⚠️  No pattern specified. Kill ALL agents?"):
            console.print("👋 Cancelled", style="yellow")
            return
        pattern = "*"
    
    console.print(f"🎯 Killing agents matching pattern: {pattern}", style="bold red")
    
    # This would integrate with the enhanced CLI integration
    console.print("⚠️  Pattern-based operations require enhanced CLI integration", style="yellow")


@agent.command()
@click.argument('agent_id')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml', 'summary']), default='summary')
@click.option('--include-health', is_flag=True, help='Include health check information')
@click.option('--include-logs', is_flag=True, help='Include recent logs')
def info(agent_id, format, include_health, include_logs):
    """Show comprehensive agent information."""
    
    console.print(f"🔍 Getting comprehensive info for agent {agent_id}...", style="bold blue")
    
    # Get agent details first
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    if format == 'json':
        click.echo(json.dumps(agent_status, indent=2))
        return
    
    # Show comprehensive information
    session_info = agent_status.get('session_info', {})
    
    info_text = f"""
[bold]Agent Information[/bold]
Agent ID: {agent_id}
Type: {session_info.get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', 'unknown')}
Session: {session_info.get('session_name', 'unknown')}
Status: {'🟢 Running' if agent_status.get('is_running') else '🔴 Stopped'}
Workspace: {session_info.get('workspace_path', 'unknown')}
Git Branch: {session_info.get('git_branch', 'unknown')}
Created: {session_info.get('created_at', 'unknown')}
Last Activity: {session_info.get('last_activity', 'unknown')}

[bold]Environment Variables[/bold]
"""
    
    env_vars = session_info.get('environment_vars', {})
    for key, value in env_vars.items():
        if key.startswith('LEANVIBE_'):
            info_text += f"{key}: {value}\n"
    
    if include_health:
        info_text += "\n[bold]Health Status[/bold]\n"
        info_text += "⚠️  Health monitoring integration pending\n"
    
    if include_logs:
        info_text += "\n[bold]Recent Logs (last 10 lines)[/bold]\n"
        try:
            logs_result = ctx.api_call(f"agents/{agent_id}/logs?lines=10")
            if logs_result and logs_result.get('logs'):
                for log_line in logs_result['logs']:
                    info_text += f"{log_line}\n"
            else:
                info_text += "No recent logs available\n"
        except Exception as e:
            info_text += f"Failed to retrieve logs: {e}\n"
    
    console.print(Panel(info_text.strip(), title=f"Agent {agent_id} Information", border_style="blue"))


@agent.command()
@click.option('--pattern', '-p', help='Filter agents by pattern')
@click.option('--format', '-f', type=click.Choice(['tree', 'table']), default='tree')
def tree(pattern, format):
    """Show agents in a tree structure."""
    
    console.print("🌳 Agent Tree View", style="bold green")
    
    # Get agent list
    agents_data = ctx.api_call("agents/list")
    
    if not agents_data:
        console.print("❌ Failed to retrieve agent list", style="red")
        return
    
    agents = agents_data.get('agents', [])
    
    if pattern:
        # Simple pattern filtering
        filtered_agents = []
        for agent in agents:
            agent_info = agent.get('session_info', {})
            if (pattern.lower() in agent.get('agent_id', '').lower() or
                pattern.lower() in agent_info.get('session_name', '').lower() or
                pattern.lower() in agent_info.get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', '').lower()):
                filtered_agents.append(agent)
        agents = filtered_agents
    
    if not agents:
        console.print("No agents found" + (f" matching pattern '{pattern}'" if pattern else ""), style="yellow")
        return
    
    if format == 'tree':
        from rich.tree import Tree
        
        tree = Tree("🤖 Agent Hive")
        
        # Group by agent type
        type_groups = {}
        for agent in agents:
            agent_type = agent.get('session_info', {}).get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', 'unknown')
            if agent_type not in type_groups:
                type_groups[agent_type] = []
            type_groups[agent_type].append(agent)
        
        for agent_type, type_agents in type_groups.items():
            type_branch = tree.add(f"📦 {agent_type} ({len(type_agents)} agents)")
            
            for agent in type_agents:
                agent_id = agent.get('agent_id', 'unknown')[:8] + "..."
                session_name = agent.get('session_info', {}).get('session_name', 'unknown')
                status = "🟢" if agent.get('is_running') else "🔴"
                
                type_branch.add(f"{status} {agent_id} | {session_name}")
        
        console.print(tree)
    
    else:
        # Use existing table format
        list.callback(False, False, 'table', False)


@agent.command()
@click.argument('name')
@click.argument('agent_id')
@click.option('--description', '-d', help='Bookmark description')
def bookmark(name, agent_id, description):
    """Create a bookmark for quick agent access."""
    
    # Verify agent exists
    agent_status = ctx.api_call(f"agents/{agent_id}/status")
    
    if not agent_status:
        console.print(f"❌ Agent {agent_id} not found", style="red")
        return
    
    # For now, store bookmarks in a simple local file
    from pathlib import Path
    import json
    
    bookmarks_file = Path.home() / ".config" / "agent-hive" / "bookmarks.json"
    bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing bookmarks
    bookmarks = {}
    if bookmarks_file.exists():
        try:
            with open(bookmarks_file, 'r') as f:
                bookmarks = json.load(f)
        except:
            pass
    
    # Add new bookmark
    bookmarks[name] = {
        "agent_id": agent_id,
        "description": description or "",
        "created_at": datetime.now().isoformat()
    }
    
    # Save bookmarks
    try:
        with open(bookmarks_file, 'w') as f:
            json.dump(bookmarks, f, indent=2)
        
        console.print(f"📖 Bookmark '{name}' created for agent {agent_id}", style="green")
        
        if description:
            console.print(f"   Description: {description}", style="dim")
    
    except Exception as e:
        console.print(f"❌ Failed to save bookmark: {e}", style="red")


@agent.command()
@click.option('--remove', '-r', help='Remove a bookmark')
def bookmarks(remove):
    """List or manage agent bookmarks."""
    
    from pathlib import Path
    import json
    
    bookmarks_file = Path.home() / ".config" / "agent-hive" / "bookmarks.json"
    
    if not bookmarks_file.exists():
        console.print("No bookmarks found", style="yellow")
        return
    
    try:
        with open(bookmarks_file, 'r') as f:
            bookmarks = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed to load bookmarks: {e}", style="red")
        return
    
    if remove:
        if remove in bookmarks:
            del bookmarks[remove]
            
            # Save updated bookmarks
            with open(bookmarks_file, 'w') as f:
                json.dump(bookmarks, f, indent=2)
            
            console.print(f"📖 Bookmark '{remove}' removed", style="green")
        else:
            console.print(f"❌ Bookmark '{remove}' not found", style="red")
        return
    
    if not bookmarks:
        console.print("No bookmarks found", style="yellow")
        return
    
    # List bookmarks
    table = Table(title="Agent Bookmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Agent ID", style="green")
    table.add_column("Description")
    table.add_column("Created")
    
    for name, bookmark in bookmarks.items():
        created_at = datetime.fromisoformat(bookmark["created_at"])
        created_str = created_at.strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            name,
            bookmark["agent_id"][:8] + "...",
            bookmark.get("description", ""),
            created_str
        )
    
    console.print(table)
    console.print("\n💡 Use 'hive agent attach @bookmark_name' to quickly attach to a bookmarked agent")


@agent.command()
@click.option('--health', is_flag=True, help='Include health status')
@click.option('--performance', is_flag=True, help='Include performance metrics')
@click.option('--export', type=click.Path(), help='Export dashboard to file')
def dashboard(health, performance, export):
    """Show comprehensive agent dashboard."""
    
    console.print("📊 Agent Hive Dashboard", style="bold blue")
    console.print("=" * 80)
    
    # Get system status
    system_status = ctx.api_call("status")
    if system_status:
        console.print("🟢 System Status: Healthy", style="green")
    else:
        console.print("🔴 System Status: Unhealthy", style="red")
        return
    
    # Get agent overview
    agents_data = ctx.api_call("agents/list")
    if not agents_data:
        console.print("❌ Failed to retrieve agent data", style="red")
        return
    
    agents = agents_data.get('agents', [])
    
    # Agent summary
    running_count = len([a for a in agents if a.get('is_running')])
    stopped_count = len(agents) - running_count
    
    summary_table = Table(title="Agent Summary")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", style="cyan")
    summary_table.add_column("Percentage")
    
    total_agents = len(agents)
    if total_agents > 0:
        running_pct = (running_count / total_agents) * 100
        stopped_pct = (stopped_count / total_agents) * 100
        
        summary_table.add_row("🟢 Running", str(running_count), f"{running_pct:.1f}%")
        summary_table.add_row("🔴 Stopped", str(stopped_count), f"{stopped_pct:.1f}%")
        summary_table.add_row("📊 Total", str(total_agents), "100.0%")
    
    console.print(summary_table)
    
    # Recent activity
    console.print("\n📈 Recent Activity", style="bold")
    
    if agents:
        # Sort by last activity
        recent_agents = sorted(agents, 
                             key=lambda x: x.get('session_info', {}).get('last_activity', ''), 
                             reverse=True)[:5]
        
        activity_table = Table()
        activity_table.add_column("Agent", style="cyan")
        activity_table.add_column("Type", style="green")
        activity_table.add_column("Status")
        activity_table.add_column("Last Activity")
        
        for agent in recent_agents:
            agent_id = agent.get('agent_id', 'unknown')[:8] + "..."
            agent_type = agent.get('session_info', {}).get('environment_vars', {}).get('LEANVIBE_AGENT_TYPE', 'unknown')
            status = "🟢 Running" if agent.get('is_running') else "🔴 Stopped"
            last_activity = agent.get('session_info', {}).get('last_activity', 'unknown')
            
            if 'T' in last_activity:
                # Format datetime
                try:
                    dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    last_activity = dt.strftime("%H:%M:%S")
                except:
                    pass
            
            activity_table.add_row(agent_id, agent_type, status, last_activity)
        
        console.print(activity_table)
    
    if health:
        console.print("\n🏥 Health Status", style="bold")
        console.print("⚠️  Health monitoring integration pending")
    
    if performance:
        console.print("\n⚡ Performance Metrics", style="bold")
        console.print("⚠️  Performance monitoring integration pending")
    
    if export:
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "system_status": system_status,
            "agents": agents,
            "summary": {
                "total": total_agents,
                "running": running_count,
                "stopped": stopped_count
            }
        }
        
        try:
            with open(export, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            console.print(f"📄 Dashboard exported to {export}", style="green")
        except Exception as e:
            console.print(f"❌ Failed to export dashboard: {e}", style="red")


# Add the agent group to the main CLI
if __name__ == "__main__":
    agent()
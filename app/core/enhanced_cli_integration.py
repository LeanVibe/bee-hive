"""
Enhanced CLI Integration for LeanVibe Agent Hive 2.0

Provides seamless integration between the CLI commands and the agent management
system with short ID support for easy debugging and manual inspection.

Features:
- Short ID resolution for easy agent access
- Tab completion for agent IDs and commands
- Rich output formatting with context-aware information
- Batch operations on multiple agents
- Session bookmarking and quick access
- Integration with external tools (tmux, git, editors)
"""

import asyncio
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree

from .short_id_generator import ShortIDGenerator
from .tmux_session_manager import TmuxSessionManager
from .enhanced_agent_launcher import EnhancedAgentLauncher, AgentLauncherType
from .agent_redis_bridge import AgentRedisBridge
from .session_health_monitor import SessionHealthMonitor

console = Console()


class AgentReference:
    """Helper class for resolving agent references (short IDs, full IDs, names)."""
    
    def __init__(
        self,
        tmux_manager: TmuxSessionManager,
        agent_launcher: EnhancedAgentLauncher,
        short_id_generator: ShortIDGenerator
    ):
        self.tmux_manager = tmux_manager
        self.agent_launcher = agent_launcher
        self.short_id_generator = short_id_generator
        
        # Cache for quick lookups
        self._agent_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry = None
    
    async def resolve_agent(self, reference: str) -> Optional[Dict[str, Any]]:
        """
        Resolve an agent reference to full agent information.
        
        Args:
            reference: Can be short ID, full ID, session name, or agent name
            
        Returns:
            Dictionary with agent information if found
        """
        await self._refresh_cache_if_needed()
        
        # Try exact matches first
        for agent_id, agent_info in self._agent_cache.items():
            if reference in [
                agent_id,
                agent_info.get("short_id"),
                agent_info.get("session_name"),
                agent_info.get("agent_name")
            ]:
                return agent_info
        
        # Try partial matches for short IDs
        if len(reference) >= 3:  # Minimum 3 characters for partial match
            matches = []
            for agent_id, agent_info in self._agent_cache.items():
                short_id = agent_info.get("short_id", "")
                if short_id.lower().startswith(reference.lower()):
                    matches.append(agent_info)
            
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                console.print(f"⚠️  Ambiguous reference '{reference}'. Multiple matches found:", style="yellow")
                for match in matches:
                    console.print(f"  - {match.get('short_id')} ({match.get('session_name')})")
                return None
        
        return None
    
    async def list_agents(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by pattern."""
        await self._refresh_cache_if_needed()
        
        agents = list(self._agent_cache.values())
        
        if pattern:
            pattern_lower = pattern.lower()
            filtered_agents = []
            
            for agent in agents:
                if any(pattern_lower in str(value).lower() for value in [
                    agent.get("short_id", ""),
                    agent.get("session_name", ""),
                    agent.get("agent_name", ""),
                    agent.get("agent_type", "")
                ]):
                    filtered_agents.append(agent)
            
            return filtered_agents
        
        return agents
    
    async def get_agent_suggestions(self, prefix: str) -> List[str]:
        """Get agent ID suggestions for tab completion."""
        await self._refresh_cache_if_needed()
        
        suggestions = []
        prefix_lower = prefix.lower()
        
        for agent_info in self._agent_cache.values():
            short_id = agent_info.get("short_id", "")
            session_name = agent_info.get("session_name", "")
            
            if short_id.lower().startswith(prefix_lower):
                suggestions.append(short_id)
            elif session_name.lower().startswith(prefix_lower):
                suggestions.append(session_name)
        
        return sorted(suggestions)
    
    async def _refresh_cache_if_needed(self) -> None:
        """Refresh agent cache if expired."""
        now = datetime.utcnow()
        
        if (self._cache_expiry is None or 
            now > self._cache_expiry or 
            not self._agent_cache):
            
            await self._refresh_cache()
            self._cache_expiry = now.replace(second=now.second + 30)  # 30 second cache
    
    async def _refresh_cache(self) -> None:
        """Refresh the agent cache."""
        self._agent_cache.clear()
        
        try:
            # Get all active agents
            active_agents = await self.agent_launcher.list_active_agents()
            
            for agent_status in active_agents:
                agent_id = agent_status.get("agent_id")
                session_info = agent_status.get("session_info", {})
                
                # Extract short ID from agent ID or session name
                short_id = self._extract_short_id(agent_id, session_info)
                
                self._agent_cache[agent_id] = {
                    "agent_id": agent_id,
                    "short_id": short_id,
                    "session_name": session_info.get("session_name"),
                    "agent_name": session_info.get("environment_vars", {}).get("LEANVIBE_AGENT_TYPE"),
                    "agent_type": session_info.get("environment_vars", {}).get("LEANVIBE_AGENT_TYPE"),
                    "workspace_path": session_info.get("workspace_path"),
                    "status": "running" if agent_status.get("is_running") else "stopped",
                    "session_info": session_info,
                    "full_status": agent_status
                }
                
        except Exception as e:
            console.print(f"⚠️  Failed to refresh agent cache: {e}", style="yellow")
    
    def _extract_short_id(self, agent_id: str, session_info: Dict[str, Any]) -> str:
        """Extract or generate short ID from agent information."""
        # Try to extract from session name (format: agent-SHORT_ID)
        session_name = session_info.get("session_name", "")
        if session_name.startswith("agent-"):
            short_id = session_name.replace("agent-", "")
            if len(short_id) <= 8:
                return short_id
        
        # Try to extract from environment variables
        env_vars = session_info.get("environment_vars", {})
        if "LEANVIBE_AGENT_SHORT_ID" in env_vars:
            return env_vars["LEANVIBE_AGENT_SHORT_ID"]
        
        # Fallback: use first 8 characters of agent ID
        return agent_id[:8] if agent_id else "unknown"


class BookmarkManager:
    """Manages session bookmarks for quick access."""
    
    def __init__(self):
        self.bookmarks_file = Path.home() / ".config" / "agent-hive" / "bookmarks.json"
        self.bookmarks_file.parent.mkdir(parents=True, exist_ok=True)
        self._bookmarks: Dict[str, Dict[str, Any]] = {}
        self._load_bookmarks()
    
    def add_bookmark(self, name: str, agent_id: str, description: str = "") -> None:
        """Add a session bookmark."""
        self._bookmarks[name] = {
            "agent_id": agent_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "access_count": 0
        }
        self._save_bookmarks()
    
    def remove_bookmark(self, name: str) -> bool:
        """Remove a session bookmark."""
        if name in self._bookmarks:
            del self._bookmarks[name]
            self._save_bookmarks()
            return True
        return False
    
    def get_bookmark(self, name: str) -> Optional[Dict[str, Any]]:
        """Get bookmark by name."""
        bookmark = self._bookmarks.get(name)
        if bookmark:
            bookmark["access_count"] += 1
            self._save_bookmarks()
        return bookmark
    
    def list_bookmarks(self) -> Dict[str, Dict[str, Any]]:
        """List all bookmarks."""
        return self._bookmarks.copy()
    
    def _load_bookmarks(self) -> None:
        """Load bookmarks from file."""
        try:
            if self.bookmarks_file.exists():
                with open(self.bookmarks_file, 'r') as f:
                    self._bookmarks = json.load(f)
        except Exception:
            self._bookmarks = {}
    
    def _save_bookmarks(self) -> None:
        """Save bookmarks to file."""
        try:
            with open(self.bookmarks_file, 'w') as f:
                json.dump(self._bookmarks, f, indent=2)
        except Exception as e:
            console.print(f"⚠️  Failed to save bookmarks: {e}", style="yellow")


class EnhancedCLIIntegration:
    """
    Enhanced CLI integration with short ID support and rich formatting.
    """
    
    def __init__(
        self,
        tmux_manager: TmuxSessionManager,
        agent_launcher: EnhancedAgentLauncher,
        redis_bridge: AgentRedisBridge,
        health_monitor: SessionHealthMonitor,
        short_id_generator: ShortIDGenerator
    ):
        self.tmux_manager = tmux_manager
        self.agent_launcher = agent_launcher
        self.redis_bridge = redis_bridge
        self.health_monitor = health_monitor
        self.short_id_generator = short_id_generator
        
        self.agent_ref = AgentReference(tmux_manager, agent_launcher, short_id_generator)
        self.bookmarks = BookmarkManager()
    
    async def smart_agent_list(
        self,
        pattern: Optional[str] = None,
        format_type: str = "table",
        show_health: bool = False,
        show_tasks: bool = False
    ) -> None:
        """Enhanced agent listing with smart formatting."""
        agents = await self.agent_ref.list_agents(pattern)
        
        if not agents:
            if pattern:
                console.print(f"No agents found matching pattern: {pattern}", style="yellow")
            else:
                console.print("No active agents found", style="yellow")
            return
        
        if format_type == "json":
            click.echo(json.dumps([agent["full_status"] for agent in agents], indent=2))
            return
        
        # Create rich table
        table = Table(title=f"Active Agents ({len(agents)})")
        table.add_column("Short ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Session", style="blue")
        table.add_column("Status", style="bold")
        table.add_column("Uptime")
        
        if show_health:
            table.add_column("Health", style="yellow")
        
        if show_tasks:
            table.add_column("Tasks", style="magenta")
        
        if format_type == "wide":
            table.add_column("Workspace")
            table.add_column("Last Activity")
        
        for agent in agents:
            short_id = agent.get("short_id", "unknown")
            agent_type = agent.get("agent_type", "unknown")
            session_name = agent.get("session_name", "unknown")
            status = agent.get("status", "unknown")
            
            # Calculate uptime
            session_info = agent.get("session_info", {})
            created_at = session_info.get("created_at")
            uptime = "unknown"
            if created_at:
                try:
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    uptime_delta = datetime.now() - created_time.replace(tzinfo=None)
                    hours = uptime_delta.seconds // 3600
                    minutes = (uptime_delta.seconds % 3600) // 60
                    uptime = f"{hours}h {minutes}m"
                except:
                    pass
            
            # Status with emoji
            status_display = "🟢 Running" if status == "running" else "🔴 Stopped"
            
            row = [short_id, agent_type, session_name, status_display, uptime]
            
            if show_health:
                # Get health status (simplified for now)
                health_display = "🟢 Healthy"  # Would integrate with health monitor
                row.append(health_display)
            
            if show_tasks:
                # Get task count (simplified for now)
                task_count = "0"  # Would integrate with task tracking
                row.append(task_count)
            
            if format_type == "wide":
                workspace = agent.get("workspace_path", "unknown")
                workspace_display = workspace.split('/')[-1] if '/' in workspace else workspace
                
                last_activity = session_info.get("last_activity", "unknown")
                activity_display = last_activity.split('T')[1][:8] if 'T' in last_activity else last_activity
                
                row.extend([workspace_display, activity_display])
            
            table.add_row(*row)
        
        console.print(table)
    
    async def smart_attach(self, reference: str, new_window: bool = False) -> None:
        """Smart attach to agent session with reference resolution."""
        # Check if reference is a bookmark
        bookmark = self.bookmarks.get_bookmark(reference)
        if bookmark:
            console.print(f"📖 Using bookmark '{reference}'", style="cyan")
            reference = bookmark["agent_id"]
        
        agent = await self.agent_ref.resolve_agent(reference)
        if not agent:
            console.print(f"❌ Agent '{reference}' not found", style="red")
            
            # Suggest similar agents
            suggestions = await self.agent_ref.get_agent_suggestions(reference)
            if suggestions:
                console.print("🔍 Did you mean:", style="cyan")
                for suggestion in suggestions[:5]:
                    console.print(f"  - {suggestion}")
            return
        
        session_name = agent["session_name"]
        agent_id = agent["agent_id"]
        short_id = agent["short_id"]
        
        console.print(f"🔗 Attaching to agent {short_id} ({session_name})...", style="bold blue")
        console.print(f"📁 Workspace: {agent['workspace_path']}", style="dim")
        console.print("💡 Press Ctrl+B, then D to detach from session", style="dim")
        
        # Create info panel in session
        info_commands = [
            f"echo '🤖 Agent: {short_id} ({agent['agent_type']})'",
            f"echo '📺 Session: {session_name}'",
            f"echo '📁 Workspace: {agent['workspace_path']}'",
            f"echo '⏰ Attached at: {datetime.now().strftime('%H:%M:%S')}'",
            "echo '💡 Type \"hive agent status $(echo $LEANVIBE_AGENT_SHORT_ID)\" for agent info'"
        ]
        
        try:
            # Send info commands to session
            for cmd in info_commands:
                subprocess.run(["tmux", "send-keys", "-t", session_name, cmd, "Enter"], 
                             check=False, capture_output=True)
            
            # Attach to session
            if new_window:
                subprocess.run(["tmux", "new-window", "-t", session_name, "-n", "inspector"])
                subprocess.run(["tmux", "attach-session", "-t", session_name])
            else:
                subprocess.run(["tmux", "attach-session", "-t", session_name])
                
        except KeyboardInterrupt:
            console.print("\n👋 Detached from session", style="yellow")
        except Exception as e:
            console.print(f"❌ Failed to attach to session: {e}", style="red")
    
    async def smart_logs(
        self,
        reference: str,
        lines: int = 50,
        follow: bool = False,
        filter_pattern: Optional[str] = None
    ) -> None:
        """Enhanced log viewing with filtering and formatting."""
        agent = await self.agent_ref.resolve_agent(reference)
        if not agent:
            console.print(f"❌ Agent '{reference}' not found", style="red")
            return
        
        agent_id = agent["agent_id"]
        short_id = agent["short_id"]
        session_name = agent["session_name"]
        
        console.print(f"📋 Showing logs for agent {short_id} ({session_name})", style="bold blue")
        
        if follow:
            console.print("📡 Following logs (Ctrl+C to stop)...", style="cyan")
        
        # Get logs from agent launcher
        try:
            if follow:
                # Implement real-time log following
                console.print("⚠️  Real-time log following not yet implemented", style="yellow")
                console.print("Using tmux capture as fallback...")
                
                try:
                    while True:
                        result = subprocess.run(
                            ["tmux", "capture-pane", "-t", session_name, "-p"],
                            capture_output=True, text=True, check=True
                        )
                        
                        output_lines = result.stdout.split('\n')
                        recent_lines = output_lines[-lines:] if len(output_lines) > lines else output_lines
                        
                        # Apply filter if specified
                        if filter_pattern:
                            recent_lines = [line for line in recent_lines if filter_pattern.lower() in line.lower()]
                        
                        console.clear()
                        console.print(f"📋 Live logs for {short_id} (last {len(recent_lines)} lines)")
                        console.print("=" * 80)
                        
                        for line in recent_lines:
                            if line.strip():
                                # Simple syntax highlighting for log levels
                                if any(level in line.upper() for level in ['ERROR', 'FAIL']):
                                    console.print(line, style="red")
                                elif any(level in line.upper() for level in ['WARN', 'WARNING']):
                                    console.print(line, style="yellow")
                                elif any(level in line.upper() for level in ['INFO']):
                                    console.print(line, style="blue")
                                elif any(level in line.upper() for level in ['DEBUG']):
                                    console.print(line, style="dim")
                                else:
                                    console.print(line)
                        
                        await asyncio.sleep(2)
                        
                except KeyboardInterrupt:
                    console.print("\n👋 Stopped following logs", style="yellow")
            
            else:
                # Get static logs
                logs = await self.agent_launcher.get_agent_logs(agent_id, lines)
                
                if not logs:
                    # Fallback to tmux capture
                    result = subprocess.run(
                        ["tmux", "capture-pane", "-t", session_name, "-p"],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        logs = result.stdout.split('\n')[-lines:]
                    else:
                        console.print("❌ No logs available", style="red")
                        return
                
                # Apply filter if specified
                if filter_pattern:
                    logs = [line for line in logs if filter_pattern.lower() in line.lower()]
                
                console.print(f"📄 Last {len(logs)} log lines:")
                console.print("=" * 80)
                
                for line in logs:
                    if line.strip():
                        # Enhanced syntax highlighting
                        if any(level in line.upper() for level in ['ERROR', 'FAIL', '❌']):
                            console.print(line, style="bold red")
                        elif any(level in line.upper() for level in ['WARN', 'WARNING', '⚠️']):
                            console.print(line, style="bold yellow")
                        elif any(level in line.upper() for level in ['INFO', '✅', '🟢']):
                            console.print(line, style="bold blue")
                        elif any(level in line.upper() for level in ['DEBUG', '🔍']):
                            console.print(line, style="dim cyan")
                        elif line.startswith('['):
                            # Structured log entries
                            console.print(line, style="green")
                        else:
                            console.print(line)
                
        except Exception as e:
            console.print(f"❌ Failed to retrieve logs: {e}", style="red")
    
    async def batch_operation(
        self,
        pattern: str,
        operation: str,
        confirm_each: bool = False,
        dry_run: bool = False
    ) -> None:
        """Perform batch operations on multiple agents."""
        agents = await self.agent_ref.list_agents(pattern)
        
        if not agents:
            console.print(f"No agents found matching pattern: {pattern}", style="yellow")
            return
        
        console.print(f"🎯 Found {len(agents)} agents matching pattern '{pattern}':")
        
        # Show matched agents
        for agent in agents:
            console.print(f"  - {agent['short_id']} ({agent['session_name']})")
        
        if dry_run:
            console.print(f"\n🧪 DRY RUN: Would perform '{operation}' on {len(agents)} agents", style="cyan")
            return
        
        if not Confirm.ask(f"\nProceed with '{operation}' on {len(agents)} agents?"):
            console.print("👋 Operation cancelled", style="yellow")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            task = progress.add_task(f"Performing {operation}...", total=len(agents))
            results = []
            
            for agent in agents:
                short_id = agent['short_id']
                agent_id = agent['agent_id']
                
                if confirm_each:
                    if not Confirm.ask(f"Perform '{operation}' on {short_id}?"):
                        progress.advance(task)
                        results.append({"agent": short_id, "status": "skipped"})
                        continue
                
                try:
                    if operation == "kill":
                        success = await self.agent_launcher.terminate_agent(agent_id)
                        results.append({"agent": short_id, "status": "success" if success else "failed"})
                    
                    elif operation == "restart":
                        # This would require orchestrator integration
                        console.print(f"⚠️  Restart not yet implemented for {short_id}")
                        results.append({"agent": short_id, "status": "not_implemented"})
                    
                    else:
                        console.print(f"⚠️  Unknown operation '{operation}' for {short_id}")
                        results.append({"agent": short_id, "status": "unknown_operation"})
                
                except Exception as e:
                    console.print(f"❌ Failed {operation} on {short_id}: {e}")
                    results.append({"agent": short_id, "status": "error", "error": str(e)})
                
                progress.advance(task)
        
        # Show results summary
        console.print(f"\n📊 Batch operation '{operation}' completed:")
        
        success_count = len([r for r in results if r["status"] == "success"])
        failed_count = len([r for r in results if r["status"] == "failed"])
        error_count = len([r for r in results if r["status"] == "error"])
        
        console.print(f"  ✅ Success: {success_count}")
        console.print(f"  ❌ Failed: {failed_count}")
        console.print(f"  🚨 Errors: {error_count}")
        
        if failed_count > 0 or error_count > 0:
            console.print("\n🔍 Detailed results:")
            for result in results:
                if result["status"] in ["failed", "error"]:
                    error_msg = result.get("error", "Unknown error")
                    console.print(f"  {result['agent']}: {result['status']} - {error_msg}")
    
    async def show_agent_tree(self) -> None:
        """Show agents in a tree structure grouped by type and status."""
        agents = await self.agent_ref.list_agents()
        
        if not agents:
            console.print("No active agents found", style="yellow")
            return
        
        # Group agents by type and status
        tree = Tree("🤖 Agent Hive Overview")
        
        # Group by agent type
        type_groups = {}
        for agent in agents:
            agent_type = agent.get("agent_type", "unknown")
            if agent_type not in type_groups:
                type_groups[agent_type] = []
            type_groups[agent_type].append(agent)
        
        for agent_type, type_agents in type_groups.items():
            type_branch = tree.add(f"📦 {agent_type} ({len(type_agents)} agents)")
            
            # Group by status within type
            status_groups = {}
            for agent in type_agents:
                status = agent.get("status", "unknown")
                if status not in status_groups:
                    status_groups[status] = []
                status_groups[status].append(agent)
            
            for status, status_agents in status_groups.items():
                status_icon = "🟢" if status == "running" else "🔴"
                status_branch = type_branch.add(f"{status_icon} {status} ({len(status_agents)})")
                
                for agent in status_agents:
                    short_id = agent.get("short_id", "unknown")
                    session_name = agent.get("session_name", "unknown")
                    workspace = agent.get("workspace_path", "unknown")
                    workspace_name = workspace.split('/')[-1] if '/' in workspace else workspace
                    
                    agent_node = status_branch.add(
                        f"🔧 {short_id} | {session_name} | 📁 {workspace_name}"
                    )
        
        console.print(tree)
    
    def register_bookmark(self, name: str, reference: str, description: str = "") -> None:
        """Register a session bookmark."""
        # Resolve reference to get agent ID
        # For now, assume reference is agent ID or short ID
        self.bookmarks.add_bookmark(name, reference, description)
        console.print(f"📖 Bookmark '{name}' added for agent {reference}", style="green")
    
    def list_bookmarks(self) -> None:
        """List all session bookmarks."""
        bookmarks = self.bookmarks.list_bookmarks()
        
        if not bookmarks:
            console.print("No bookmarks found", style="yellow")
            return
        
        table = Table(title="Session Bookmarks")
        table.add_column("Name", style="cyan")
        table.add_column("Agent", style="green")
        table.add_column("Description")
        table.add_column("Access Count", style="blue")
        table.add_column("Created")
        
        for name, bookmark in bookmarks.items():
            created_at = datetime.fromisoformat(bookmark["created_at"])
            created_str = created_at.strftime("%Y-%m-%d %H:%M")
            
            table.add_row(
                name,
                bookmark["agent_id"][:8] + "...",
                bookmark.get("description", ""),
                str(bookmark["access_count"]),
                created_str
            )
        
        console.print(table)
    
    async def export_agent_info(self, reference: str, format_type: str = "json") -> None:
        """Export detailed agent information."""
        agent = await self.agent_ref.resolve_agent(reference)
        if not agent:
            console.print(f"❌ Agent '{reference}' not found", style="red")
            return
        
        # Get comprehensive agent information
        agent_id = agent["agent_id"]
        full_status = agent["full_status"]
        
        # Add health information if available
        try:
            health_check = await self.health_monitor.check_session_health(
                agent["session_info"]["session_id"]
            )
            full_status["health_check"] = health_check.to_dict()
        except:
            pass
        
        if format_type == "json":
            click.echo(json.dumps(full_status, indent=2))
        
        elif format_type == "yaml":
            try:
                import yaml
                click.echo(yaml.dump(full_status, default_flow_style=False))
            except ImportError:
                console.print("⚠️  PyYAML not installed, falling back to JSON", style="yellow")
                click.echo(json.dumps(full_status, indent=2))
        
        elif format_type == "summary":
            short_id = agent["short_id"]
            console.print(Panel(
                f"""
Agent ID: {agent_id}
Short ID: {short_id}
Type: {agent.get('agent_type', 'unknown')}
Session: {agent.get('session_name', 'unknown')}
Status: {agent.get('status', 'unknown')}
Workspace: {agent.get('workspace_path', 'unknown')}
                """.strip(),
                title=f"Agent {short_id} Summary",
                border_style="blue"
            ))


# Factory function
async def create_enhanced_cli_integration(
    tmux_manager: TmuxSessionManager,
    agent_launcher: EnhancedAgentLauncher,
    redis_bridge: AgentRedisBridge,
    health_monitor: SessionHealthMonitor,
    short_id_generator: ShortIDGenerator
) -> EnhancedCLIIntegration:
    """Create and initialize EnhancedCLIIntegration."""
    return EnhancedCLIIntegration(
        tmux_manager=tmux_manager,
        agent_launcher=agent_launcher,
        redis_bridge=redis_bridge,
        health_monitor=health_monitor,
        short_id_generator=short_id_generator
    )
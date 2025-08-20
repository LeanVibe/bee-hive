"""
Enhanced CLI Commands Integration

This module integrates the enhanced command ecosystem with existing CLI commands,
following the consolidation approach rather than rebuilding existing functionality.
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rich_print
import structlog

logger = structlog.get_logger(__name__)
console = Console()


class EnhancedCommandIntegrator:
    """Integrates enhanced features with existing CLI commands."""
    
    def __init__(self):
        self.enhanced_available = False
        self.ecosystem_integration = None
        self._initialize_enhanced_features()
    
    def _initialize_enhanced_features(self):
        """Initialize enhanced features if available."""
        try:
            # Import enhanced systems if available
            from app.core.command_ecosystem_integration import get_ecosystem_integration
            from app.core.unified_quality_gates import QualityGateValidator
            
            self.enhanced_available = True
            logger.info("Enhanced command features available")
            
        except ImportError as e:
            self.enhanced_available = False
            logger.info("Enhanced command features not available, using standard mode")
    
    async def get_ecosystem_integration(self):
        """Get ecosystem integration if available."""
        if not self.enhanced_available:
            return None
            
        if self.ecosystem_integration is None:
            try:
                from app.core.command_ecosystem_integration import get_ecosystem_integration
                self.ecosystem_integration = await get_ecosystem_integration()
            except Exception as e:
                logger.warning(f"Could not initialize ecosystem integration: {e}")
                self.enhanced_available = False
        
        return self.ecosystem_integration
    
    async def execute_enhanced_command(
        self, 
        command: str, 
        mobile: bool = False,
        quality_gates: bool = True
    ) -> Dict[str, Any]:
        """Execute command with enhanced features if available."""
        
        if not self.enhanced_available:
            return {
                "enhanced": False,
                "message": "Enhanced features not available, using standard command execution",
                "fallback": True
            }
        
        try:
            ecosystem = await self.get_ecosystem_integration()
            if ecosystem:
                result = await ecosystem.execute_enhanced_command(
                    command=command,
                    mobile_optimized=mobile,
                    use_quality_gates=quality_gates
                )
                return {
                    "enhanced": True,
                    "result": result,
                    "mobile_optimized": mobile,
                    "quality_gates_applied": quality_gates
                }
            else:
                return {"enhanced": False, "message": "Ecosystem integration unavailable"}
                
        except Exception as e:
            logger.error(f"Enhanced command execution failed: {e}")
            return {
                "enhanced": False,
                "error": str(e),
                "fallback": True
            }
    
    def format_enhanced_output(
        self, 
        data: Dict[str, Any], 
        mobile: bool = False,
        format_type: str = "table"
    ) -> None:
        """Format and display enhanced command output."""
        
        if not data.get("enhanced", False):
            # Fall back to standard formatting
            self._format_standard_output(data, format_type)
            return
        
        result = data.get("result", {})
        
        if format_type == "json":
            rich_print(json.dumps(result, indent=2))
            return
        
        # Enhanced table format with AI insights
        if format_type == "table":
            self._format_enhanced_table(result, mobile)
        
        # Show enhancement metadata
        if data.get("mobile_optimized"):
            console.print("📱 Mobile-optimized output", style="blue")
        
        if data.get("quality_gates_applied"):
            console.print("✅ Quality gates validated", style="green")
    
    def _format_enhanced_table(self, result: Dict[str, Any], mobile: bool):
        """Format enhanced table with AI insights."""
        
        # Main data table
        if "agents" in result:
            agents = result["agents"]
            table = Table(
                title="🤖 Agent Status (Enhanced)",
                show_header=True,
                header_style="bold magenta"
            )
            
            # Mobile-optimized columns
            if mobile:
                table.add_column("ID", style="cyan", width=12)
                table.add_column("Status", style="green")
                table.add_column("Score", style="yellow", width=8)
            else:
                table.add_column("Agent ID", style="cyan", width=20)
                table.add_column("Name", style="white")
                table.add_column("Type", style="blue")
                table.add_column("Status", style="green")
                table.add_column("Performance", style="yellow")
                table.add_column("AI Insights", style="magenta")
            
            for agent in agents:
                if mobile:
                    table.add_row(
                        agent.get("id", "")[:12],
                        agent.get("status", "unknown"),
                        str(agent.get("performance_score", 0))
                    )
                else:
                    insights = agent.get("ai_insights", {})
                    performance = agent.get("performance_score", "N/A")
                    ai_text = insights.get("recommendation", "No insights") if insights else "Standard mode"
                    
                    table.add_row(
                        agent.get("id", ""),
                        agent.get("name", ""),
                        agent.get("type", ""),
                        agent.get("status", ""),
                        str(performance),
                        ai_text[:30] + "..." if len(str(ai_text)) > 30 else str(ai_text)
                    )
            
            console.print(table)
        
        # Enhanced insights panel
        if "enhanced_insights" in result:
            insights = result["enhanced_insights"]
            
            insights_content = ""
            if "system_health_score" in insights:
                insights_content += f"🏥 System Health: {insights['system_health_score']}/100\n"
            
            if "optimization_opportunities" in insights:
                insights_content += f"⚡ Optimizations: {insights['optimization_opportunities']} available\n"
            
            if "ai_recommendations" in insights:
                recommendations = insights["ai_recommendations"]
                if recommendations:
                    insights_content += "\n🧠 AI Recommendations:\n"
                    for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                        insights_content += f"  {i}. {rec}\n"
            
            if insights_content:
                panel = Panel(
                    insights_content.strip(),
                    title="✨ Enhanced Insights",
                    border_style="bright_blue"
                )
                console.print(panel)
    
    def _format_standard_output(self, data: Dict[str, Any], format_type: str):
        """Format standard output without enhancements."""
        if format_type == "json":
            rich_print(json.dumps(data, indent=2))
        else:
            # Simple table for fallback
            console.print("📋 Standard Output (Enhanced features unavailable)", style="yellow")
            console.print(data)


# Global integrator instance
enhanced_integrator = EnhancedCommandIntegrator()


def enhance_command(func):
    """Decorator to add enhanced capabilities to existing CLI commands."""
    
    def wrapper(*args, **kwargs):
        # Extract enhanced options if present
        enhanced = kwargs.pop('enhanced', False)
        mobile = kwargs.pop('mobile', False)
        
        if enhanced:
            # Run with enhanced features
            return asyncio.run(_run_enhanced_command(func, enhanced, mobile, *args, **kwargs))
        else:
            # Run standard command
            return func(*args, **kwargs)
    
    return wrapper


async def _run_enhanced_command(func, enhanced: bool, mobile: bool, *args, **kwargs):
    """Run command with enhanced features."""
    
    # Determine command name from function
    command_name = func.__name__.replace('_', ':')
    if not command_name.startswith('/'):
        command_name = f"/hive:{command_name}"
    
    # Execute with enhancements
    result = await enhanced_integrator.execute_enhanced_command(
        command=command_name,
        mobile=mobile,
        quality_gates=True
    )
    
    # Format and display enhanced output
    enhanced_integrator.format_enhanced_output(
        result, 
        mobile=mobile, 
        format_type=kwargs.get('format', 'table')
    )
    
    return result


# Enhanced CLI command extensions
@click.group()
def enhanced():
    """Enhanced CLI commands with AI-powered features."""
    pass


@enhanced.command('status')
@click.option('--enhanced', is_flag=True, help='Enable enhanced AI features')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def enhanced_status(enhanced: bool, mobile: bool, format: str):
    """Enhanced system status with AI insights."""
    
    @enhance_command
    def status_command():
        # Fallback to standard status if enhancements unavailable
        return {
            "status": "operational", 
            "agents": [
                {"id": "agent-1", "name": "demo-agent", "status": "active", "type": "general"}
            ],
            "message": "Standard status output"
        }
    
    return status_command(enhanced=enhanced, mobile=mobile, format=format)


@enhanced.command('agents')
@click.option('--enhanced', is_flag=True, help='Enable enhanced AI features')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
@click.argument('action', required=False, default='list')
def enhanced_agents(enhanced: bool, mobile: bool, format: str, action: str):
    """Enhanced agent management with AI insights."""
    
    @enhance_command
    def agents_command():
        # Fallback to standard agent listing if enhancements unavailable
        return {
            "agents": [
                {
                    "id": "agent-demo-1", 
                    "name": "Demo Agent",
                    "type": "backend-engineer",
                    "status": "active",
                    "performance_score": 85
                }
            ],
            "total": 1,
            "message": f"Standard {action} output"
        }
    
    return agents_command(enhanced=enhanced, mobile=mobile, format=format)


@enhanced.command('insights')
@click.option('--mobile', is_flag=True, help='Mobile-optimized output')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def enhanced_insights(mobile: bool, format: str):
    """Display AI-powered system insights (requires enhanced features)."""
    
    async def get_insights():
        result = await enhanced_integrator.execute_enhanced_command(
            command="/hive:insights",
            mobile=mobile,
            quality_gates=True
        )
        return result
    
    if not enhanced_integrator.enhanced_available:
        console.print("❌ Enhanced insights require AI features to be available", style="red")
        console.print("💡 Try: pip install -e .[enhanced] to enable AI features", style="blue")
        return
    
    result = asyncio.run(get_insights())
    enhanced_integrator.format_enhanced_output(result, mobile=mobile, format_type=format)


if __name__ == "__main__":
    # Demo enhanced commands
    console.print("🚀 Enhanced CLI Commands Integration Demo", style="bold green")
    console.print("="*60)
    
    # Test enhanced status
    console.print("📊 Testing enhanced status command...")
    enhanced_status(enhanced=True, mobile=False, format='table')
    
    console.print("\n📱 Testing mobile-optimized output...")  
    enhanced_status(enhanced=True, mobile=True, format='table')
    
    console.print("\n🤖 Testing enhanced agents command...")
    enhanced_agents(enhanced=True, mobile=False, format='table', action='list')
    
    console.print("\n✨ Enhanced CLI Integration Complete!")
    console.print("   • Existing commands preserved")
    console.print("   • Enhanced features integrated")
    console.print("   • Mobile optimization available")
    console.print("   • Graceful fallback for standard mode")
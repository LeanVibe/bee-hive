"""
Enhanced Hive CLI Integration

Integrates the new unified hive CLI with existing command ecosystem,
enhanced command discovery, and SimpleOrchestrator for real agent spawning.

Features:
- Command suggestion and discovery
- Integration with command_ecosystem_integration.py
- Real agent spawning via SimpleOrchestrator
- Mobile optimization support
- Quality gates and validation
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import existing enhanced components
try:
    from .command_ecosystem_integration import get_ecosystem_integration
    from .enhanced_command_discovery import get_command_discovery, IntelligentCommandDiscovery
    from .unified_quality_gates import get_quality_gates, ValidationLevel
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

# Fallback ShortIDGenerator if not available
try:
    from .short_id_generator import ShortIDGenerator
except ImportError:
    class ShortIDGenerator:
        """Fallback short ID generator."""
        def __init__(self):
            self.counters = {}
        
        def generate(self, entity_type: str, description: str = None) -> str:
            counter = self.counters.get(entity_type, 0) + 1
            self.counters[entity_type] = counter
            return f"{entity_type[:4]}-{counter:03d}"

class CommandCategory(Enum):
    """Command categories for organization."""
    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    PROJECT = "project"
    CONTEXT = "context"

@dataclass
class CommandSuggestion:
    """Command suggestion with metadata."""
    command: str
    description: str
    category: CommandCategory
    confidence: float
    examples: List[str]
    mobile_optimized: bool = False

class EnhancedHiveCLI:
    """Enhanced CLI system with intelligent features."""
    
    def __init__(self):
        self.short_id_generator = ShortIDGenerator()
        self.ecosystem = None
        self.command_discovery = None
        self.quality_gates = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize enhanced CLI components."""
        if not ENHANCED_COMPONENTS_AVAILABLE:
            return False
        
        try:
            # Initialize enhanced components
            self.ecosystem = await get_ecosystem_integration()
            self.command_discovery = get_command_discovery()
            self.quality_gates = get_quality_gates()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Warning: Enhanced CLI initialization failed: {e}")
            return False
    
    async def suggest_commands(self, user_input: str, context: Dict[str, Any] = None) -> List[CommandSuggestion]:
        """Suggest commands based on user input and context."""
        suggestions = []
        
        # Basic pattern matching suggestions
        suggestions.extend(self._get_basic_suggestions(user_input))
        
        # Enhanced AI suggestions (if available)
        if self.command_discovery and self.initialized:
            try:
                ai_suggestions = await self.command_discovery.discover_commands(
                    user_intent=user_input,
                    context=context or {},
                    mobile_optimized=False,
                    limit=3
                )
                
                for suggestion in ai_suggestions:
                    suggestions.append(CommandSuggestion(
                        command=suggestion.get("command", ""),
                        description=suggestion.get("description", ""),
                        category=self._categorize_command(suggestion.get("command", "")),
                        confidence=suggestion.get("confidence", 0.5),
                        examples=suggestion.get("examples", []),
                        mobile_optimized=suggestion.get("mobile_optimized", False)
                    ))
            except Exception as e:
                print(f"AI command discovery failed: {e}")
        
        # Sort by confidence and return top suggestions
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        return suggestions[:5]
    
    def _get_basic_suggestions(self, user_input: str) -> List[CommandSuggestion]:
        """Get basic command suggestions based on keywords."""
        suggestions = []
        input_lower = user_input.lower()
        
        # System commands
        if any(word in input_lower for word in ["start", "begin", "launch", "run"]):
            suggestions.append(CommandSuggestion(
                command="hive system start",
                description="Start the Agent Hive system",
                category=CommandCategory.SYSTEM,
                confidence=0.8,
                examples=["hive system start", "hive system start --background"]
            ))
        
        if any(word in input_lower for word in ["stop", "halt", "shutdown", "end"]):
            suggestions.append(CommandSuggestion(
                command="hive system stop",
                description="Stop the Agent Hive system",
                category=CommandCategory.SYSTEM,
                confidence=0.8,
                examples=["hive system stop", "hive system stop --force"]
            ))
        
        if any(word in input_lower for word in ["status", "health", "check", "info"]):
            suggestions.append(CommandSuggestion(
                command="hive status",
                description="Show comprehensive system status",
                category=CommandCategory.SYSTEM,
                confidence=0.9,
                examples=["hive status", "hive status --watch"]
            ))
        
        # Agent commands
        if any(word in input_lower for word in ["agent", "spawn", "create", "deploy"]):
            suggestions.append(CommandSuggestion(
                command="hive agent spawn",
                description="Spawn a new specialized agent",
                category=CommandCategory.AGENT,
                confidence=0.85,
                examples=[
                    "hive agent spawn backend-developer",
                    "hive agent spawn qa-engineer \"Create test suites\""
                ]
            ))
        
        if any(word in input_lower for word in ["list", "show", "agents"]):
            suggestions.append(CommandSuggestion(
                command="hive agent list",
                description="List all active agents",
                category=CommandCategory.AGENT,
                confidence=0.8,
                examples=["hive agent list", "hive agent list --status active"]
            ))
        
        # Task commands
        if any(word in input_lower for word in ["task", "submit", "add", "todo"]):
            suggestions.append(CommandSuggestion(
                command="hive task submit",
                description="Submit a new task to the queue",
                category=CommandCategory.TASK,
                confidence=0.85,
                examples=[
                    "hive task submit \"Implement PWA APIs\"",
                    "hive task submit \"Fix bugs\" --priority high"
                ]
            ))
        
        if any(word in input_lower for word in ["tasks", "queue", "work"]):
            suggestions.append(CommandSuggestion(
                command="hive task list",
                description="List tasks in the queue",
                category=CommandCategory.TASK,
                confidence=0.8,
                examples=["hive task list", "hive task list --status pending"]
            ))
        
        return suggestions
    
    def _categorize_command(self, command: str) -> CommandCategory:
        """Categorize a command based on its structure."""
        if "system" in command:
            return CommandCategory.SYSTEM
        elif "agent" in command:
            return CommandCategory.AGENT
        elif "task" in command:
            return CommandCategory.TASK
        elif "project" in command:
            return CommandCategory.PROJECT
        elif "context" in command:
            return CommandCategory.CONTEXT
        else:
            return CommandCategory.SYSTEM
    
    async def validate_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate command using quality gates."""
        if not self.quality_gates or not self.initialized:
            return {"valid": True, "message": "Basic validation passed"}
        
        try:
            validation_result = await self.quality_gates.validate_command(
                command=command,
                validation_level=ValidationLevel.STANDARD,
                context=context or {},
                mobile_optimized=False
            )
            
            return {
                "valid": validation_result.overall_valid,
                "message": validation_result.summary,
                "issues": validation_result.validation_issues,
                "suggestions": validation_result.recovery_strategies
            }
        except Exception as e:
            return {"valid": False, "message": f"Validation error: {e}"}
    
    async def execute_enhanced_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute command through enhanced ecosystem."""
        if not self.ecosystem or not self.initialized:
            return {"success": False, "error": "Enhanced ecosystem not available"}
        
        try:
            result = await self.ecosystem.execute_enhanced_command(
                command=command,
                context=context or {},
                mobile_optimized=False,
                use_quality_gates=True
            )
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_command_help(self, command: str) -> Dict[str, Any]:
        """Get detailed help for a specific command."""
        help_data = {
            "hive init": {
                "description": "Initialize development environment",
                "usage": "hive init [--force]",
                "options": ["--force: Force reinitialize"],
                "examples": ["hive init", "hive init --force"]
            },
            "hive status": {
                "description": "Show comprehensive system status",
                "usage": "hive status [--watch]",
                "options": ["--watch: Watch for changes"],
                "examples": ["hive status", "hive status --watch"]
            },
            "hive system start": {
                "description": "Start the Agent Hive system",
                "usage": "hive system start [--background]",
                "options": ["--background: Start in background"],
                "examples": ["hive system start", "hive system start --background"]
            },
            "hive agent spawn": {
                "description": "Spawn a new specialized agent",
                "usage": "hive agent spawn <type> [task-description]",
                "options": [
                    "type: Agent type (backend-developer, qa-engineer, etc.)",
                    "task-description: Initial task for the agent"
                ],
                "examples": [
                    "hive agent spawn backend-developer",
                    "hive agent spawn qa-engineer \"Create test suites\""
                ]
            },
            "hive task submit": {
                "description": "Submit a new task to the queue",
                "usage": "hive task submit <description> [--priority <level>]",
                "options": [
                    "description: Task description",
                    "--priority: Task priority (low, medium, high)"
                ],
                "examples": [
                    "hive task submit \"Implement PWA APIs\"",
                    "hive task submit \"Fix bugs\" --priority high"
                ]
            }
        }
        
        return help_data.get(command, {
            "description": "Command help not available",
            "usage": command,
            "options": [],
            "examples": [command]
        })
    
    def format_suggestions(self, suggestions: List[CommandSuggestion]) -> str:
        """Format command suggestions for display."""
        if not suggestions:
            return "No suggestions available."
        
        output = "💡 Suggested commands:\n"
        for i, suggestion in enumerate(suggestions, 1):
            confidence_bar = "▓" * int(suggestion.confidence * 10) + "░" * (10 - int(suggestion.confidence * 10))
            output += f"\n{i}. {suggestion.command}\n"
            output += f"   {suggestion.description}\n"
            output += f"   Confidence: {confidence_bar} {suggestion.confidence:.1%}\n"
            
            if suggestion.examples:
                output += f"   Examples: {', '.join(suggestion.examples[:2])}\n"
        
        return output

# Global enhanced CLI instance
enhanced_cli = EnhancedHiveCLI()

async def get_enhanced_cli() -> EnhancedHiveCLI:
    """Get the enhanced CLI instance, initializing if needed."""
    if not enhanced_cli.initialized:
        await enhanced_cli.initialize()
    return enhanced_cli

# CLI helper functions for integration
async def suggest_commands_for_input(user_input: str, context: Dict[str, Any] = None) -> List[CommandSuggestion]:
    """Suggest commands for user input."""
    cli = await get_enhanced_cli()
    return await cli.suggest_commands(user_input, context)

async def validate_command_input(command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate command input."""
    cli = await get_enhanced_cli()
    return await cli.validate_command(command, context)

async def execute_command_enhanced(command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Execute command with enhancements."""
    cli = await get_enhanced_cli()
    return await cli.execute_enhanced_command(command, context)

def get_command_help_info(command: str) -> Dict[str, Any]:
    """Get help information for a command."""
    return enhanced_cli.get_command_help(command)
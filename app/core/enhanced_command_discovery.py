"""
Enhanced Command Discovery System for LeanVibe Agent Hive 2.0

Provides intelligent command discovery, validation, suggestions, and help
with AI-powered context awareness and mobile optimization.

Features:
- AI-powered command discovery and suggestions
- Smart parameter validation with auto-completion
- Context-aware help system
- Command usage analytics and learning
- Mobile-optimized responses
- Real-time command validation
- Pattern recognition for user intent
- Integration with existing command registry
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from collections import defaultdict, Counter
import structlog

from .hive_slash_commands import get_hive_command_registry, HiveSlashCommand

logger = structlog.get_logger()


class SuggestionType(Enum):
    """Types of command suggestions."""
    COMPLETION = "completion"        # Auto-complete partial command
    CORRECTION = "correction"        # Suggest fix for typo/error
    RELATED = "related"             # Related commands
    CONTEXTUAL = "contextual"       # Context-aware suggestions
    WORKFLOW = "workflow"           # Next logical steps
    SHORTCUT = "shortcut"           # Keyboard/gesture shortcuts


class CommandValidation:
    """Result of command validation."""
    
    def __init__(self):
        self.valid = False
        self.command_name = ""
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.auto_fixes = []
        self.parameter_validation = {}
        self.confidence = 0.0
        self.mobile_compatible = True


class CommandSuggestion:
    """Enhanced command suggestion with context and metadata."""
    
    def __init__(
        self,
        command: str,
        description: str = "",
        suggestion_type: SuggestionType = SuggestionType.COMPLETION,
        confidence: float = 0.5,
        context_relevance: float = 0.5,
        usage_frequency: float = 0.0,
        estimated_execution_time: str = "unknown",
        prerequisites: List[str] = None,
        mobile_optimized: bool = False
    ):
        self.command = command
        self.description = description
        self.suggestion_type = suggestion_type
        self.confidence = confidence
        self.context_relevance = context_relevance
        self.usage_frequency = usage_frequency
        self.estimated_execution_time = estimated_execution_time
        self.prerequisites = prerequisites or []
        self.mobile_optimized = mobile_optimized
        self.relevance_score = self._calculate_relevance_score()
        
    def _calculate_relevance_score(self) -> float:
        """Calculate overall relevance score."""
        return (
            self.confidence * 0.3 +
            self.context_relevance * 0.4 +
            self.usage_frequency * 0.2 +
            (0.1 if self.mobile_optimized else 0.0) * 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "command": self.command,
            "description": self.description,
            "type": self.suggestion_type.value,
            "confidence": round(self.confidence, 2),
            "context_relevance": round(self.context_relevance, 2),
            "usage_frequency": round(self.usage_frequency, 2),
            "estimated_execution_time": self.estimated_execution_time,
            "prerequisites": self.prerequisites,
            "mobile_optimized": self.mobile_optimized,
            "relevance_score": round(self.relevance_score, 2)
        }


class UserCommandProfiler:
    """Tracks user command patterns and builds usage profiles."""
    
    def __init__(self):
        self.command_history = []
        self.command_frequency = Counter()
        self.command_sequences = defaultdict(list)
        self.error_patterns = defaultdict(list)
        self.success_patterns = defaultdict(list)
        self.time_patterns = defaultdict(list)
        self.context_patterns = defaultdict(list)
        
    def record_command_execution(
        self,
        command: str,
        success: bool,
        execution_time: float,
        context: Dict[str, Any] = None
    ):
        """Record a command execution for pattern analysis."""
        try:
            execution_record = {
                "command": command,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.utcnow(),
                "context": context or {}
            }
            
            self.command_history.append(execution_record)
            self.command_frequency[command] += 1
            
            # Track sequences (last 3 commands)
            if len(self.command_history) >= 2:
                previous_command = self.command_history[-2]["command"]
                self.command_sequences[previous_command].append(command)
            
            # Track patterns
            if success:
                self.success_patterns[command].append(execution_record)
            else:
                self.error_patterns[command].append(execution_record)
            
            # Track time patterns
            hour = execution_record["timestamp"].hour
            self.time_patterns[hour].append(command)
            
            # Track context patterns
            if context:
                context_key = self._extract_context_key(context)
                self.context_patterns[context_key].append(command)
            
            # Keep history manageable (last 1000 commands)
            if len(self.command_history) > 1000:
                self.command_history = self.command_history[-1000:]
                
        except Exception as e:
            logger.error("Failed to record command execution", error=str(e))
    
    def get_user_patterns(self) -> Dict[str, Any]:
        """Get analyzed user command patterns."""
        try:
            # Most used commands
            most_used = dict(self.command_frequency.most_common(10))
            
            # Common sequences
            common_sequences = {}
            for cmd, next_cmds in self.command_sequences.items():
                if next_cmds:
                    most_common_next = Counter(next_cmds).most_common(3)
                    common_sequences[cmd] = most_common_next
            
            # Success rates
            success_rates = {}
            for cmd in self.command_frequency:
                total = len(self.success_patterns[cmd]) + len(self.error_patterns[cmd])
                if total > 0:
                    success_rate = len(self.success_patterns[cmd]) / total
                    success_rates[cmd] = success_rate
            
            # Time preferences
            time_preferences = {}
            for hour, commands in self.time_patterns.items():
                time_preferences[hour] = dict(Counter(commands).most_common(5))
            
            return {
                "most_used_commands": most_used,
                "command_sequences": common_sequences,
                "success_rates": success_rates,
                "time_preferences": time_preferences,
                "total_commands": len(self.command_history),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to analyze user patterns", error=str(e))
            return {"error": str(e)}
    
    def _extract_context_key(self, context: Dict[str, Any]) -> str:
        """Extract a key from context for pattern matching."""
        # Simple context key extraction
        mobile = context.get("mobile_optimized", False)
        priority = context.get("priority", "medium")
        return f"mobile:{mobile},priority:{priority}"


class ContextAnalyzer:
    """Analyzes current context to provide relevant command suggestions."""
    
    async def analyze_current_situation(
        self,
        context: Dict[str, Any],
        user_id: str = None
    ) -> Dict[str, Any]:
        """Analyze current situation for context-aware suggestions."""
        try:
            analysis = {
                "timestamp": datetime.utcnow().isoformat(),
                "context_type": self._determine_context_type(context),
                "system_state": await self._analyze_system_state(context),
                "user_context": self._analyze_user_context(context, user_id),
                "situational_factors": self._analyze_situational_factors(context),
                "recommendations": []
            }
            
            # Generate contextual recommendations
            analysis["recommendations"] = await self._generate_contextual_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error("Context analysis failed", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "context_type": "unknown"
            }
    
    def _determine_context_type(self, context: Dict[str, Any]) -> str:
        """Determine the type of current context."""
        # Check for explicit context indicators
        if context.get("mobile_optimized"):
            return "mobile"
        elif context.get("priority") == "critical":
            return "emergency"
        elif "error" in str(context).lower():
            return "troubleshooting"
        elif "develop" in str(context).lower():
            return "development"
        else:
            return "general"
    
    async def _analyze_system_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state."""
        try:
            # Get system status from hive commands
            from .hive_slash_commands import HiveStatusCommand
            status_command = HiveStatusCommand()
            system_status = await status_command.execute(["--mobile"] if context.get("mobile_optimized") else [])
            
            return {
                "agents_active": system_status.get("agent_count", 0),
                "system_ready": system_status.get("system_ready", False),
                "platform_active": system_status.get("platform_active", False),
                "health_status": system_status.get("system_health", "unknown")
            }
            
        except Exception as e:
            logger.error("System state analysis failed", error=str(e))
            return {"error": str(e)}
    
    def _analyze_user_context(self, context: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Analyze user-specific context."""
        return {
            "user_id": user_id,
            "mobile_user": context.get("mobile_optimized", False),
            "priority_level": context.get("priority", "medium"),
            "session_info": context.get("session_info", {}),
            "preferences": context.get("user_preferences", {})
        }
    
    def _analyze_situational_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze situational factors affecting command suggestions."""
        current_time = datetime.utcnow()
        
        return {
            "time_of_day": current_time.hour,
            "day_of_week": current_time.weekday(),
            "is_business_hours": 9 <= current_time.hour <= 17,
            "mobile_context": context.get("mobile_optimized", False),
            "urgency_level": context.get("priority", "medium")
        }
    
    async def _generate_contextual_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate contextual command recommendations."""
        recommendations = []
        
        system_state = analysis.get("system_state", {})
        context_type = analysis.get("context_type", "general")
        
        # System state based recommendations
        if not system_state.get("platform_active"):
            recommendations.append("Start the platform with /hive:start")
        elif system_state.get("agents_active", 0) < 3:
            recommendations.append("Scale team with /hive:spawn <role>")
        
        # Context type based recommendations
        if context_type == "mobile":
            recommendations.extend([
                "Use mobile-optimized commands with --mobile flag",
                "Check mobile performance with /hive:status --mobile"
            ])
        elif context_type == "development":
            recommendations.extend([
                "Start autonomous development with /hive:develop",
                "Monitor progress with /hive:oversight --mobile-info"
            ])
        elif context_type == "troubleshooting":
            recommendations.extend([
                "Run detailed diagnostics with /hive:status --detailed",
                "Get contextual help with /hive:focus"
            ])
        
        return recommendations


class SmartParameterValidator:
    """Advanced parameter validation with smart defaults and suggestions."""
    
    def __init__(self):
        self.parameter_patterns = self._initialize_parameter_patterns()
        self.smart_defaults = SmartDefaultsEngine()
    
    async def validate_command_parameters(
        self,
        command: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Comprehensive parameter validation with suggestions."""
        try:
            # Parse command to get name and args
            command_parts = command.strip().split()
            if not command_parts or not command_parts[0].startswith("/hive:"):
                return {"error": "Invalid command format"}
            
            command_name = command_parts[0].replace("/hive:", "")
            args = command_parts[1:] if len(command_parts) > 1 else []
            
            # Get command definition
            registry = get_hive_command_registry()
            command_obj = registry.get_command(command_name)
            
            if not command_obj:
                return {
                    "valid": False,
                    "error": f"Unknown command: {command_name}",
                    "suggestions": await self._suggest_similar_commands(command_name)
                }
            
            # Validate parameters
            validation_result = {
                "valid": True,
                "command_name": command_name,
                "parameters": parameters,
                "args": args,
                "validation_details": {},
                "suggestions": [],
                "auto_fixes": [],
                "warnings": []
            }
            
            # Check for common parameter patterns
            pattern_validation = await self._validate_parameter_patterns(command_name, args, context)
            validation_result["validation_details"].update(pattern_validation)
            
            # Generate smart suggestions
            suggestions = await self._generate_parameter_suggestions(command_name, args, context)
            validation_result["suggestions"].extend(suggestions)
            
            return validation_result
            
        except Exception as e:
            logger.error("Parameter validation failed", error=str(e))
            return {
                "valid": False,
                "error": str(e),
                "command": command
            }
    
    def _initialize_parameter_patterns(self) -> Dict[str, List[str]]:
        """Initialize known parameter patterns for commands."""
        return {
            "start": [
                r"--quick",
                r"--team-size=\d+",
                r"--timeout=\d+"
            ],
            "spawn": [
                r"(product_manager|architect|backend_developer|frontend_developer|qa_engineer|devops_engineer)",
                r"--capabilities=[\w,]+"
            ],
            "status": [
                r"--detailed",
                r"--agents-only", 
                r"--mobile",
                r"--alerts-only",
                r"--priority=(critical|high|medium|low)"
            ],
            "develop": [
                r'"[^"]+"',  # Quoted project description
                r"--dashboard",
                r"--timeout=\d+"
            ],
            "focus": [
                r"(development|monitoring|performance)",
                r"--mobile",
                r"--priority=(critical|high|medium)"
            ],
            "compact": [
                r"--level=(light|standard|aggressive)",
                r"--target-tokens=\d+",
                r"--preserve-decisions",
                r"--preserve-patterns"
            ]
        }
    
    async def _validate_parameter_patterns(
        self,
        command_name: str,
        args: List[str],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Validate parameters against known patterns."""
        validation = {
            "pattern_matches": [],
            "unrecognized_args": [],
            "suggestions": []
        }
        
        patterns = self.parameter_patterns.get(command_name, [])
        args_str = " ".join(args)
        
        # Check each arg against patterns
        for arg in args:
            matched = False
            for pattern in patterns:
                if re.match(pattern, arg):
                    validation["pattern_matches"].append({
                        "arg": arg,
                        "pattern": pattern,
                        "valid": True
                    })
                    matched = True
                    break
            
            if not matched:
                validation["unrecognized_args"].append(arg)
                # Try to suggest corrections
                suggestion = await self._suggest_parameter_correction(command_name, arg, patterns)
                if suggestion:
                    validation["suggestions"].append(suggestion)
        
        return validation
    
    async def _suggest_similar_commands(self, command_name: str) -> List[str]:
        """Suggest similar command names for typos."""
        registry = get_hive_command_registry()
        available_commands = list(registry.commands.keys())
        
        suggestions = []
        for cmd in available_commands:
            # Simple string distance-based suggestion
            if self._string_similarity(command_name, cmd) > 0.6:
                suggestions.append(f"/hive:{cmd}")
        
        return suggestions
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity (simple implementation)."""
        if not s1 or not s2:
            return 0.0
        
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # Calculate edit distance (simplified)
        matches = sum(1 for a, b in zip(shorter, longer) if a == b)
        return matches / len(longer)
    
    async def _suggest_parameter_correction(
        self,
        command_name: str,
        arg: str,
        patterns: List[str]
    ) -> Optional[str]:
        """Suggest correction for unrecognized parameter."""
        # Try to match against patterns with fuzzy matching
        for pattern in patterns:
            # Simple pattern matching suggestions
            if "--" in arg and "--" in pattern:
                pattern_arg = pattern.split("=")[0] if "=" in pattern else pattern
                if self._string_similarity(arg.split("=")[0], pattern_arg) > 0.7:
                    return f"Did you mean '{pattern_arg}'?"
        
        return None
    
    async def _generate_parameter_suggestions(
        self,
        command_name: str,
        current_args: List[str],
        context: Dict[str, Any] = None
    ) -> List[str]:
        """Generate helpful parameter suggestions."""
        suggestions = []
        
        # Context-aware suggestions
        if context and context.get("mobile_optimized") and "--mobile" not in " ".join(current_args):
            suggestions.append("Consider adding --mobile for mobile optimization")
        
        if context and context.get("priority") == "high" and "--priority" not in " ".join(current_args):
            suggestions.append("Consider adding --priority=high for faster processing")
        
        # Command-specific suggestions
        if command_name == "status" and not current_args:
            suggestions.append("Try --detailed for comprehensive information")
        elif command_name == "start" and "--team-size" not in " ".join(current_args):
            suggestions.append("Consider specifying --team-size=5 for optimal performance")
        elif command_name == "develop" and not any('"' in arg for arg in current_args):
            suggestions.append('Add project description in quotes: "Build user authentication"')
        
        return suggestions


class SmartDefaultsEngine:
    """Engine for generating smart default parameter values."""
    
    async def suggest_default(
        self,
        command_name: str,
        parameter_name: str,
        context: Dict[str, Any]
    ) -> Optional[Any]:
        """Suggest smart default value for parameter."""
        try:
            # Context-based defaults
            if parameter_name == "priority":
                if context.get("mobile_optimized"):
                    return "high"  # Mobile users expect faster responses
                return "medium"
            
            elif parameter_name == "team_size":
                # Suggest team size based on context
                if context.get("urgency_level") == "high":
                    return 7  # Larger team for urgent tasks
                return 5  # Standard team size
            
            elif parameter_name == "compression_level":
                if context.get("mobile_optimized"):
                    return "aggressive"  # More compression for mobile
                return "standard"
            
            elif parameter_name == "timeout":
                if context.get("mobile_optimized"):
                    return 30  # Shorter timeout for mobile
                return 300  # Standard timeout
            
            return None
            
        except Exception as e:
            logger.error("Smart defaults failed", error=str(e))
            return None


class IntelligentCommandDiscovery:
    """Main class for intelligent command discovery and suggestions."""
    
    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.parameter_validator = SmartParameterValidator()
        self.user_profiler = UserCommandProfiler()
        self.suggestion_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def discover_commands(
        self,
        user_intent: str,
        context: Dict[str, Any] = None,
        user_id: str = None,
        limit: int = 5,
        mobile_optimized: bool = False
    ) -> List[CommandSuggestion]:
        """
        Discover relevant commands based on user intent and context.
        
        Args:
            user_intent: Natural language description of what user wants to do
            context: Current context information
            user_id: User identifier for personalization
            limit: Maximum number of suggestions to return
            mobile_optimized: Whether to optimize for mobile interface
            
        Returns:
            List of CommandSuggestion objects ranked by relevance
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(user_intent, context, user_id)
            cached_suggestions = self._get_cached_suggestions(cache_key)
            if cached_suggestions:
                return cached_suggestions[:limit]
            
            # Analyze context
            context = context or {}
            context_analysis = await self.context_analyzer.analyze_current_situation(context, user_id)
            
            # Generate suggestions from multiple sources
            suggestions = []
            
            # Intent-based suggestions
            intent_suggestions = await self._generate_intent_based_suggestions(
                user_intent, context_analysis, mobile_optimized
            )
            suggestions.extend(intent_suggestions)
            
            # Context-based suggestions
            context_suggestions = await self._generate_context_based_suggestions(
                context_analysis, mobile_optimized
            )
            suggestions.extend(context_suggestions)
            
            # Pattern-based suggestions (from user history)
            if user_id:
                pattern_suggestions = await self._generate_pattern_based_suggestions(
                    user_id, context_analysis, mobile_optimized
                )
                suggestions.extend(pattern_suggestions)
            
            # Workflow-based suggestions
            workflow_suggestions = await self._generate_workflow_suggestions(
                context_analysis, mobile_optimized
            )
            suggestions.extend(workflow_suggestions)
            
            # Rank and deduplicate suggestions
            ranked_suggestions = self._rank_and_deduplicate_suggestions(suggestions, limit)
            
            # Apply mobile optimizations
            if mobile_optimized:
                ranked_suggestions = self._apply_mobile_optimizations(ranked_suggestions)
            
            # Cache results
            self._cache_suggestions(cache_key, ranked_suggestions)
            
            return ranked_suggestions
            
        except Exception as e:
            logger.error("Command discovery failed", error=str(e))
            return await self._get_fallback_suggestions(context, mobile_optimized)
    
    async def validate_command(
        self,
        command: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> CommandValidation:
        """Comprehensive command validation with suggestions."""
        validation = CommandValidation()
        
        try:
            # Basic format validation
            if not command or not isinstance(command, str):
                validation.errors.append("Command must be a non-empty string")
                return validation
            
            if not command.startswith("/hive:"):
                validation.errors.append("Command must start with /hive:")
                validation.suggestions = [f"/hive:{command}" if not command.startswith("/") else command.replace("/", "/hive:")]
                return validation
            
            # Extract command name
            command_parts = command.strip().split()
            command_name = command_parts[0].replace("/hive:", "")
            validation.command_name = command_name
            
            # Check command exists
            registry = get_hive_command_registry()
            if not registry.get_command(command_name):
                validation.errors.append(f"Unknown command: {command_name}")
                validation.suggestions = await self.parameter_validator._suggest_similar_commands(command_name)
                return validation
            
            # Validate parameters
            parameter_validation = await self.parameter_validator.validate_command_parameters(
                command, {}, context
            )
            
            if parameter_validation.get("valid", True):
                validation.valid = True
                validation.confidence = 0.9
            else:
                validation.errors.extend(parameter_validation.get("errors", []))
                validation.warnings.extend(parameter_validation.get("warnings", []))
            
            validation.suggestions.extend(parameter_validation.get("suggestions", []))
            validation.auto_fixes.extend(parameter_validation.get("auto_fixes", []))
            
            # Mobile compatibility check
            if mobile_optimized:
                validation.mobile_compatible = self._check_mobile_compatibility(command)
                if not validation.mobile_compatible:
                    validation.warnings.append("Command may not be optimized for mobile")
                    validation.suggestions.append(f"{command} --mobile")
            
            return validation
            
        except Exception as e:
            logger.error("Command validation failed", error=str(e))
            validation.errors.append(f"Validation failed: {e}")
            return validation
    
    async def get_contextual_help(
        self,
        command_name: str,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> Dict[str, Any]:
        """Get contextual help for a specific command."""
        try:
            registry = get_hive_command_registry()
            command_obj = registry.get_command(command_name)
            
            if not command_obj:
                return {
                    "error": f"Command '{command_name}' not found",
                    "suggestions": await self.parameter_validator._suggest_similar_commands(command_name)
                }
            
            # Basic command info
            help_info = {
                "command": command_name,
                "full_command": f"/hive:{command_name}",
                "description": command_obj.description,
                "usage": command_obj.usage,
                "mobile_optimized": mobile_optimized
            }
            
            # Add contextual enhancements
            context_analysis = await self.context_analyzer.analyze_current_situation(context or {})
            
            # Generate contextual examples
            help_info["examples"] = await self._generate_contextual_examples(
                command_name, context_analysis, mobile_optimized
            )
            
            # Add related commands
            help_info["related_commands"] = await self._find_related_commands(command_name)
            
            # Add prerequisites and warnings
            help_info["prerequisites"] = await self._check_command_prerequisites(
                command_name, context_analysis
            )
            
            # Mobile-specific enhancements
            if mobile_optimized:
                help_info["mobile_tips"] = self._get_mobile_specific_tips(command_name)
            
            return help_info
            
        except Exception as e:
            logger.error("Contextual help failed", error=str(e))
            return {
                "error": str(e),
                "command": command_name
            }
    
    def record_command_usage(
        self,
        command: str,
        success: bool,
        execution_time: float,
        context: Dict[str, Any] = None,
        user_id: str = None
    ):
        """Record command usage for pattern learning."""
        try:
            self.user_profiler.record_command_execution(
                command, success, execution_time, context
            )
        except Exception as e:
            logger.error("Failed to record command usage", error=str(e))
    
    # Private methods
    
    def _generate_cache_key(self, user_intent: str, context: Dict[str, Any], user_id: str) -> str:
        """Generate cache key for suggestions."""
        context_hash = hash(json.dumps(context or {}, sort_keys=True))
        return f"suggestions:{hash(user_intent)}:{context_hash}:{user_id}"
    
    def _get_cached_suggestions(self, cache_key: str) -> Optional[List[CommandSuggestion]]:
        """Get cached suggestions if still valid."""
        if cache_key in self.suggestion_cache:
            cached_data = self.suggestion_cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["suggestions"]
        return None
    
    def _cache_suggestions(self, cache_key: str, suggestions: List[CommandSuggestion]):
        """Cache suggestions for future use."""
        self.suggestion_cache[cache_key] = {
            "suggestions": suggestions,
            "timestamp": time.time()
        }
    
    async def _generate_intent_based_suggestions(
        self,
        user_intent: str,
        context_analysis: Dict[str, Any],
        mobile_optimized: bool
    ) -> List[CommandSuggestion]:
        """Generate suggestions based on user intent analysis."""
        suggestions = []
        
        # Intent pattern matching
        intent_lower = user_intent.lower()
        
        if any(word in intent_lower for word in ["start", "begin", "initialize", "launch"]):
            suggestions.append(CommandSuggestion(
                command="/hive:start",
                description="Start the multi-agent platform",
                suggestion_type=SuggestionType.CONTEXTUAL,
                confidence=0.9,
                context_relevance=0.8,
                estimated_execution_time="2-3 minutes"
            ))
        
        if any(word in intent_lower for word in ["status", "check", "health", "state"]):
            command = "/hive:status --mobile" if mobile_optimized else "/hive:status"
            suggestions.append(CommandSuggestion(
                command=command,
                description="Check platform status and health",
                suggestion_type=SuggestionType.CONTEXTUAL,
                confidence=0.9,
                context_relevance=0.9,
                mobile_optimized=mobile_optimized,
                estimated_execution_time="< 30 seconds"
            ))
        
        if any(word in intent_lower for word in ["develop", "build", "create", "code"]):
            suggestions.append(CommandSuggestion(
                command="/hive:develop",
                description="Start autonomous development",
                suggestion_type=SuggestionType.CONTEXTUAL,
                confidence=0.8,
                context_relevance=0.7,
                estimated_execution_time="5-60 minutes",
                prerequisites=["Platform must be running"]
            ))
        
        if any(word in intent_lower for word in ["help", "assist", "guide", "recommend"]):
            command = "/hive:focus --mobile" if mobile_optimized else "/hive:focus"
            suggestions.append(CommandSuggestion(
                command=command,
                description="Get contextual recommendations",
                suggestion_type=SuggestionType.CONTEXTUAL,
                confidence=0.8,
                context_relevance=0.9,
                mobile_optimized=mobile_optimized,
                estimated_execution_time="< 1 minute"
            ))
        
        return suggestions
    
    async def _generate_context_based_suggestions(
        self,
        context_analysis: Dict[str, Any],
        mobile_optimized: bool
    ) -> List[CommandSuggestion]:
        """Generate suggestions based on current context."""
        suggestions = []
        
        system_state = context_analysis.get("system_state", {})
        context_type = context_analysis.get("context_type", "general")
        
        # System state based suggestions
        if not system_state.get("platform_active"):
            suggestions.append(CommandSuggestion(
                command="/hive:start",
                description="Platform is offline - start it now",
                suggestion_type=SuggestionType.CONTEXTUAL,
                confidence=1.0,
                context_relevance=1.0,
                estimated_execution_time="2-3 minutes"
            ))
        
        if system_state.get("agents_active", 0) < 3:
            suggestions.append(CommandSuggestion(
                command="/hive:spawn backend_developer",
                description="Scale team for better performance",
                suggestion_type=SuggestionType.WORKFLOW,
                confidence=0.8,
                context_relevance=0.7,
                estimated_execution_time="1-2 minutes"
            ))
        
        # Context type based suggestions
        if context_type == "mobile":
            suggestions.extend([
                CommandSuggestion(
                    command="/hive:status --mobile --priority=high",
                    description="Quick mobile-optimized status check",
                    suggestion_type=SuggestionType.CONTEXTUAL,
                    confidence=0.9,
                    context_relevance=1.0,
                    mobile_optimized=True,
                    estimated_execution_time="< 5 seconds"
                ),
                CommandSuggestion(
                    command="/hive:focus --mobile",
                    description="Mobile-optimized recommendations",
                    suggestion_type=SuggestionType.CONTEXTUAL,
                    confidence=0.8,
                    context_relevance=0.9,
                    mobile_optimized=True,
                    estimated_execution_time="< 30 seconds"
                )
            ])
        
        return suggestions
    
    async def _generate_pattern_based_suggestions(
        self,
        user_id: str,
        context_analysis: Dict[str, Any],
        mobile_optimized: bool
    ) -> List[CommandSuggestion]:
        """Generate suggestions based on user patterns."""
        suggestions = []
        
        try:
            # Get user patterns
            patterns = self.user_profiler.get_user_patterns()
            most_used = patterns.get("most_used_commands", {})
            sequences = patterns.get("command_sequences", {})
            
            # Suggest frequently used commands
            for command, frequency in list(most_used.items())[:3]:
                suggestions.append(CommandSuggestion(
                    command=command,
                    description=f"Frequently used command ({frequency} times)",
                    suggestion_type=SuggestionType.RELATED,
                    confidence=0.7,
                    context_relevance=0.5,
                    usage_frequency=frequency / max(1, patterns.get("total_commands", 1)),
                    mobile_optimized=mobile_optimized
                ))
            
        except Exception as e:
            logger.error("Pattern-based suggestion generation failed", error=str(e))
        
        return suggestions
    
    async def _generate_workflow_suggestions(
        self,
        context_analysis: Dict[str, Any],
        mobile_optimized: bool
    ) -> List[CommandSuggestion]:
        """Generate workflow-based suggestions."""
        suggestions = []
        
        system_state = context_analysis.get("system_state", {})
        
        # Logical workflow progressions
        if system_state.get("platform_active") and system_state.get("agents_active", 0) >= 3:
            # Platform is ready for development
            suggestions.append(CommandSuggestion(
                command="/hive:develop \"Your project idea\"",
                description="Platform is ready - start development",
                suggestion_type=SuggestionType.WORKFLOW,
                confidence=0.8,
                context_relevance=0.8,
                estimated_execution_time="5-60 minutes"
            ))
            
            # Suggest oversight for active development
            command = "/hive:oversight --mobile-info" if mobile_optimized else "/hive:oversight"
            suggestions.append(CommandSuggestion(
                command=command,
                description="Monitor development progress",
                suggestion_type=SuggestionType.WORKFLOW,
                confidence=0.7,
                context_relevance=0.6,
                mobile_optimized=mobile_optimized,
                estimated_execution_time="< 1 minute"
            ))
        
        return suggestions
    
    def _rank_and_deduplicate_suggestions(
        self,
        suggestions: List[CommandSuggestion],
        limit: int
    ) -> List[CommandSuggestion]:
        """Rank suggestions by relevance and remove duplicates."""
        # Remove duplicates by command
        seen_commands = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            command_base = suggestion.command.split()[0]  # Get base command without args
            if command_base not in seen_commands:
                seen_commands.add(command_base)
                unique_suggestions.append(suggestion)
        
        # Sort by relevance score
        unique_suggestions.sort(key=lambda s: s.relevance_score, reverse=True)
        
        return unique_suggestions[:limit]
    
    def _apply_mobile_optimizations(
        self,
        suggestions: List[CommandSuggestion]
    ) -> List[CommandSuggestion]:
        """Apply mobile-specific optimizations to suggestions."""
        for suggestion in suggestions:
            # Add mobile flags if not present
            if not suggestion.mobile_optimized and "--mobile" not in suggestion.command:
                if any(cmd in suggestion.command for cmd in ["status", "focus", "productivity"]):
                    suggestion.command += " --mobile"
                    suggestion.mobile_optimized = True
            
            # Truncate descriptions for mobile
            if len(suggestion.description) > 60:
                suggestion.description = suggestion.description[:57] + "..."
        
        return suggestions
    
    async def _get_fallback_suggestions(
        self,
        context: Dict[str, Any] = None,
        mobile_optimized: bool = False
    ) -> List[CommandSuggestion]:
        """Get fallback suggestions when discovery fails."""
        basic_suggestions = [
            CommandSuggestion(
                command="/hive:status" + (" --mobile" if mobile_optimized else ""),
                description="Check system status",
                confidence=0.9,
                mobile_optimized=mobile_optimized
            ),
            CommandSuggestion(
                command="/hive:start",
                description="Start the platform",
                confidence=0.8
            ),
            CommandSuggestion(
                command="/hive:focus" + (" --mobile" if mobile_optimized else ""),
                description="Get recommendations",
                confidence=0.7,
                mobile_optimized=mobile_optimized
            )
        ]
        
        return basic_suggestions
    
    def _check_mobile_compatibility(self, command: str) -> bool:
        """Check if command is mobile-compatible."""
        # Most hive commands are mobile-compatible
        # This would be expanded based on actual mobile limitations
        return True
    
    async def _generate_contextual_examples(
        self,
        command_name: str,
        context_analysis: Dict[str, Any],
        mobile_optimized: bool
    ) -> List[str]:
        """Generate contextual examples for a command."""
        examples = []
        
        # Base examples by command
        base_examples = {
            "start": ["/hive:start", "/hive:start --team-size=5", "/hive:start --quick"],
            "status": ["/hive:status", "/hive:status --detailed", "/hive:status --mobile"],
            "develop": ["/hive:develop \"Build a web app\"", "/hive:develop \"API with tests\" --dashboard"],
            "spawn": ["/hive:spawn backend_developer", "/hive:spawn architect --capabilities=system_design"],
            "focus": ["/hive:focus", "/hive:focus development", "/hive:focus --mobile"]
        }
        
        examples.extend(base_examples.get(command_name, [f"/hive:{command_name}"]))
        
        # Add mobile-optimized examples
        if mobile_optimized:
            mobile_examples = [ex + " --mobile" if "--mobile" not in ex else ex for ex in examples[:2]]
            examples.extend(mobile_examples)
        
        return list(set(examples))  # Remove duplicates
    
    async def _find_related_commands(self, command_name: str) -> List[str]:
        """Find commands related to the given command."""
        related_commands = {
            "start": ["status", "spawn", "develop"],
            "status": ["start", "focus", "oversight"],
            "develop": ["start", "oversight", "spawn"],
            "spawn": ["start", "status", "develop"],
            "focus": ["status", "productivity", "develop"],
            "oversight": ["develop", "status", "focus"],
            "productivity": ["focus", "status", "develop"],
            "compact": ["focus", "status"],
            "stop": ["start", "status"]
        }
        
        related = related_commands.get(command_name, [])
        return [f"/hive:{cmd}" for cmd in related]
    
    async def _check_command_prerequisites(
        self,
        command_name: str,
        context_analysis: Dict[str, Any]
    ) -> List[str]:
        """Check prerequisites for a command."""
        prerequisites = []
        system_state = context_analysis.get("system_state", {})
        
        if command_name in ["develop", "spawn", "oversight"] and not system_state.get("platform_active"):
            prerequisites.append("Platform must be running (use /hive:start)")
        
        if command_name == "develop" and system_state.get("agents_active", 0) < 3:
            prerequisites.append("Recommend at least 3 agents for development")
        
        return prerequisites
    
    def _get_mobile_specific_tips(self, command_name: str) -> List[str]:
        """Get mobile-specific tips for a command."""
        mobile_tips = {
            "status": [
                "Use --mobile for faster loading",
                "Add --priority=high for critical information only"
            ],
            "focus": [
                "Mobile interface shows quick actions",
                "Touch commands for instant execution"
            ],
            "develop": [
                "Use --dashboard for mobile oversight",
                "Monitor progress via mobile dashboard"
            ],
            "oversight": [
                "Use --mobile-info for mobile access URL",
                "Dashboard is optimized for mobile viewing"
            ]
        }
        
        return mobile_tips.get(command_name, ["Command supports mobile optimization with --mobile flag"])


# Global instance
_command_discovery: Optional[IntelligentCommandDiscovery] = None


def get_command_discovery() -> IntelligentCommandDiscovery:
    """Get global command discovery instance."""
    global _command_discovery
    if _command_discovery is None:
        _command_discovery = IntelligentCommandDiscovery()
    return _command_discovery
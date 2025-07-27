"""
Modification Generator

LLM-powered code modification generator with safety scoring and context-aware
suggestions. Generates precise, targeted code improvements based on analysis
results and learned patterns from previous modifications.
"""

import json
import re
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import structlog
from anthropic import Anthropic

from .code_analysis_engine import ProjectAnalysis, FileAnalysis, CodePattern

logger = structlog.get_logger()


@dataclass
class ModificationContext:
    """Context for generating modifications."""
    
    # Analysis context
    project_analysis: ProjectAnalysis
    file_analysis: FileAnalysis
    target_patterns: List[CodePattern] = field(default_factory=list)
    
    # Modification goals
    goals: List[str] = field(default_factory=list)
    safety_level: str = "conservative"
    
    # Learning context
    project_conventions: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    anti_patterns: List[str] = field(default_factory=list)
    
    # Technical constraints
    max_lines_changed: int = 50
    preserve_interfaces: bool = True
    require_tests: bool = True


@dataclass
class ModificationSuggestion:
    """A specific code modification suggestion."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    file_path: str = ""
    modification_type: str = ""
    
    # Content changes
    original_content: str = ""
    modified_content: str = ""
    unified_diff: str = ""
    
    # Reasoning and metadata
    reasoning: str = ""
    confidence: float = 0.0
    safety_score: float = 0.0
    complexity_score: float = 0.0
    
    # Impact estimates
    performance_impact: float = 0.0  # Expected percentage improvement
    maintainability_impact: str = "low"  # low, medium, high
    risk_level: str = "low"  # low, medium, high, critical
    
    # Change statistics
    lines_added: int = 0
    lines_removed: int = 0
    functions_modified: Set[str] = field(default_factory=set)
    dependencies_changed: List[str] = field(default_factory=list)
    
    # Validation requirements
    approval_required: bool = False
    suggested_tests: List[str] = field(default_factory=list)
    
    @property
    def net_lines_changed(self) -> int:
        """Calculate net lines changed."""
        return self.lines_added - self.lines_removed


class ModificationGenerator:
    """Generates code modifications using LLM with safety and context awareness."""
    
    def __init__(self, anthropic_client: Optional[Anthropic] = None):
        self.anthropic = anthropic_client or Anthropic()
        self.safety_thresholds = {
            "conservative": {"min_safety": 0.8, "max_complexity": 0.3, "max_lines": 20},
            "moderate": {"min_safety": 0.6, "max_complexity": 0.5, "max_lines": 50}, 
            "aggressive": {"min_safety": 0.4, "max_complexity": 0.8, "max_lines": 100}
        }
        
    def generate_modifications(
        self,
        context: ModificationContext
    ) -> List[ModificationSuggestion]:
        """Generate modification suggestions based on analysis and context."""
        
        logger.info(
            "Generating modifications",
            file_path=context.file_analysis.file_path,
            pattern_count=len(context.target_patterns),
            goals=context.goals
        )
        
        modifications = []
        
        # Group patterns by type and priority
        pattern_groups = self._group_patterns_by_priority(context.target_patterns)
        
        # Generate modifications for each pattern group
        for priority, patterns in pattern_groups.items():
            batch_modifications = self._generate_pattern_modifications(
                patterns, context
            )
            modifications.extend(batch_modifications)
        
        # Apply learning and filtering
        modifications = self._apply_learned_patterns(modifications, context)
        modifications = self._filter_by_safety_level(modifications, context)
        modifications = self._deduplicate_modifications(modifications)
        
        # Sort by confidence and impact
        modifications.sort(
            key=lambda m: (m.confidence, m.performance_impact, -m.complexity_score),
            reverse=True
        )
        
        logger.info(
            "Generated modifications",
            total_generated=len(modifications),
            high_confidence=len([m for m in modifications if m.confidence >= 0.8]),
            requires_approval=len([m for m in modifications if m.approval_required])
        )
        
        return modifications
    
    def _group_patterns_by_priority(
        self, 
        patterns: List[CodePattern]
    ) -> Dict[str, List[CodePattern]]:
        """Group patterns by priority for batch processing."""
        
        groups = {"high": [], "medium": [], "low": []}
        
        for pattern in patterns:
            if pattern.severity in ["critical", "high"]:
                groups["high"].append(pattern)
            elif pattern.severity == "medium":
                groups["medium"].append(pattern)
            else:
                groups["low"].append(pattern)
                
        return groups
    
    def _generate_pattern_modifications(
        self,
        patterns: List[CodePattern],
        context: ModificationContext
    ) -> List[ModificationSuggestion]:
        """Generate modifications for a group of patterns."""
        
        modifications = []
        
        for pattern in patterns:
            try:
                modification = self._generate_single_modification(pattern, context)
                if modification:
                    modifications.append(modification)
            except Exception as e:
                logger.error(
                    "Failed to generate modification for pattern",
                    pattern=pattern.pattern_name,
                    error=str(e)
                )
        
        return modifications
    
    def _generate_single_modification(
        self,
        pattern: CodePattern,
        context: ModificationContext
    ) -> Optional[ModificationSuggestion]:
        """Generate a single modification for a specific pattern."""
        
        # Prepare LLM prompt
        prompt = self._build_modification_prompt(pattern, context)
        
        try:
            # Call LLM for modification suggestion
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.1,  # Low temperature for consistency
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse LLM response
            modification_data = self._parse_llm_response(response.content[0].text)
            
            if not modification_data:
                return None
            
            # Create modification suggestion
            suggestion = ModificationSuggestion(
                file_path=pattern.file_path,
                modification_type=pattern.pattern_type,
                original_content=modification_data.get("original_content", ""),
                modified_content=modification_data.get("modified_content", ""),
                reasoning=modification_data.get("reasoning", ""),
                confidence=modification_data.get("confidence", 0.5),
                performance_impact=modification_data.get("performance_impact", 0.0),
                maintainability_impact=modification_data.get("maintainability_impact", "low"),
                risk_level=modification_data.get("risk_level", "medium")
            )
            
            # Calculate derived properties
            self._calculate_modification_metrics(suggestion, context)
            
            return suggestion
            
        except Exception as e:
            logger.error(
                "LLM modification generation failed",
                pattern=pattern.pattern_name,
                error=str(e)
            )
            return None
    
    def _build_modification_prompt(
        self,
        pattern: CodePattern,
        context: ModificationContext
    ) -> str:
        """Build comprehensive prompt for LLM modification generation."""
        
        # Read current file content around the pattern
        file_content = self._get_file_content_with_context(
            pattern.file_path, pattern.line_number, context_lines=10
        )
        
        # Project conventions and patterns
        conventions_text = self._format_project_conventions(context.project_conventions)
        user_prefs_text = self._format_user_preferences(context.user_preferences)
        
        prompt = f"""You are an expert software engineer tasked with improving code quality and performance. 

ANALYSIS CONTEXT:
- File: {pattern.file_path}
- Pattern Type: {pattern.pattern_type}
- Pattern Name: {pattern.pattern_name}
- Issue: {pattern.description}
- Line: {pattern.line_number}
- Severity: {pattern.severity}
- Suggested Fix: {pattern.suggested_fix or 'Not specified'}

CURRENT CODE CONTEXT:
```python
{file_content}
```

PROJECT CONVENTIONS:
{conventions_text}

USER PREFERENCES:
{user_prefs_text}

MODIFICATION GOALS:
{', '.join(context.goals)}

SAFETY LEVEL: {context.safety_level}

CONSTRAINTS:
- Maximum lines to change: {context.max_lines_changed}
- Preserve public interfaces: {context.preserve_interfaces}
- Require test coverage: {context.require_tests}

ANTI-PATTERNS TO AVOID:
{', '.join(context.anti_patterns) if context.anti_patterns else 'None specified'}

Please provide a specific code modification that addresses the identified pattern. Your response must be in JSON format:

{{
    "modification_type": "performance|bug_fix|refactor|security_fix|style",
    "original_content": "exact code to be replaced",
    "modified_content": "new code to replace it with",
    "reasoning": "detailed explanation of why this change improves the code",
    "confidence": 0.0-1.0,
    "performance_impact": -100.0 to 100.0 (percentage improvement, negative for regression),
    "maintainability_impact": "low|medium|high",
    "risk_level": "low|medium|high|critical",
    "lines_added": number,
    "lines_removed": number,
    "functions_modified": ["function_name1", "function_name2"],
    "dependencies_changed": ["new_dependency1", "removed_dependency2"],
    "suggested_tests": ["test description 1", "test description 2"],
    "approval_required": true/false
}}

IMPORTANT GUIDELINES:
1. Only modify the minimum necessary code to fix the issue
2. Ensure the modification follows project conventions
3. Consider performance implications carefully
4. Maintain backward compatibility unless explicitly needed
5. Provide clear, actionable reasoning
6. Be conservative with confidence scores - only use >0.8 for very certain improvements
7. Flag for approval if the change is complex or risky
8. Suggest appropriate tests for the modification

Focus on the specific pattern identified and provide a targeted, safe improvement."""
        
        return prompt
    
    def _get_file_content_with_context(
        self, 
        file_path: str, 
        line_number: int, 
        context_lines: int = 10
    ) -> str:
        """Get file content with context around the target line."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start_line = max(0, line_number - context_lines - 1)
            end_line = min(len(lines), line_number + context_lines)
            
            context_content = []
            for i in range(start_line, end_line):
                prefix = ">>> " if i == line_number - 1 else "    "
                context_content.append(f"{prefix}{i+1:4d}: {lines[i].rstrip()}")
            
            return "\n".join(context_content)
            
        except Exception as e:
            logger.warning("Failed to read file context", file_path=file_path, error=str(e))
            return f"Error reading file: {str(e)}"
    
    def _format_project_conventions(self, conventions: Dict[str, Any]) -> str:
        """Format project conventions for prompt."""
        if not conventions:
            return "No specific conventions provided."
        
        formatted = []
        for category, rules in conventions.items():
            formatted.append(f"- {category.title()}: {rules}")
        
        return "\n".join(formatted)
    
    def _format_user_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format user preferences for prompt."""
        if not preferences:
            return "No specific preferences provided."
        
        formatted = []
        for pref, value in preferences.items():
            formatted.append(f"- {pref.replace('_', ' ').title()}: {value}")
        
        return "\n".join(formatted)
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response with error handling."""
        
        try:
            # Extract JSON from response (in case there's additional text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                return json.loads(json_text)
            else:
                logger.warning("No JSON found in LLM response", response=response_text[:500])
                return None
                
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON response", error=str(e), response=response_text[:500])
            return None
    
    def _calculate_modification_metrics(
        self,
        suggestion: ModificationSuggestion,
        context: ModificationContext
    ) -> None:
        """Calculate derived metrics for the modification suggestion."""
        
        # Generate unified diff
        if suggestion.original_content and suggestion.modified_content:
            suggestion.unified_diff = "\n".join(unified_diff(
                suggestion.original_content.splitlines(keepends=True),
                suggestion.modified_content.splitlines(keepends=True),
                fromfile=f"a/{suggestion.file_path}",
                tofile=f"b/{suggestion.file_path}",
                lineterm=""
            ))
        
        # Calculate safety score based on various factors
        suggestion.safety_score = self._calculate_safety_score(suggestion, context)
        
        # Calculate complexity score
        suggestion.complexity_score = self._calculate_complexity_score(suggestion)
        
        # Determine if approval is required
        thresholds = self.safety_thresholds[context.safety_level]
        suggestion.approval_required = (
            suggestion.safety_score < thresholds["min_safety"] or
            suggestion.complexity_score > thresholds["max_complexity"] or
            suggestion.net_lines_changed > thresholds["max_lines"] or
            suggestion.risk_level in ["high", "critical"]
        )
    
    def _calculate_safety_score(
        self,
        suggestion: ModificationSuggestion,
        context: ModificationContext
    ) -> float:
        """Calculate safety score based on modification characteristics."""
        
        score = 1.0
        
        # Reduce score for risky modifications
        risk_penalties = {"low": 0.0, "medium": 0.1, "high": 0.3, "critical": 0.5}
        score -= risk_penalties.get(suggestion.risk_level, 0.2)
        
        # Reduce score for large changes
        if suggestion.net_lines_changed > 20:
            score -= 0.2
        elif suggestion.net_lines_changed > 50:
            score -= 0.4
        
        # Reduce score for complex modifications
        if len(suggestion.functions_modified) > 3:
            score -= 0.1
        
        # Reduce score for dependency changes
        if suggestion.dependencies_changed:
            score -= 0.1 * len(suggestion.dependencies_changed)
        
        # Boost score for well-tested modifications
        if suggestion.suggested_tests:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity_score(self, suggestion: ModificationSuggestion) -> float:
        """Calculate complexity score for the modification."""
        
        complexity = 0.0
        
        # Base complexity from lines changed
        complexity += min(0.5, suggestion.net_lines_changed / 100.0)
        
        # Add complexity for multiple functions
        complexity += len(suggestion.functions_modified) * 0.1
        
        # Add complexity for dependency changes
        complexity += len(suggestion.dependencies_changed) * 0.15
        
        # Add complexity based on risk level
        risk_complexity = {"low": 0.0, "medium": 0.2, "high": 0.4, "critical": 0.6}
        complexity += risk_complexity.get(suggestion.risk_level, 0.2)
        
        return min(1.0, complexity)
    
    def _apply_learned_patterns(
        self,
        modifications: List[ModificationSuggestion],
        context: ModificationContext
    ) -> List[ModificationSuggestion]:
        """Apply learned patterns and user preferences to filter modifications."""
        
        filtered = []
        
        for modification in modifications:
            # Skip if matches anti-patterns
            if self._matches_anti_patterns(modification, context.anti_patterns):
                logger.debug(
                    "Skipping modification that matches anti-pattern",
                    file_path=modification.file_path,
                    type=modification.modification_type
                )
                continue
            
            # Apply user preferences
            if self._matches_user_preferences(modification, context.user_preferences):
                modification.confidence += 0.1  # Boost confidence for preferred patterns
            
            filtered.append(modification)
        
        return filtered
    
    def _matches_anti_patterns(
        self,
        modification: ModificationSuggestion,
        anti_patterns: List[str]
    ) -> bool:
        """Check if modification matches known anti-patterns."""
        
        for anti_pattern in anti_patterns:
            if anti_pattern.lower() in modification.reasoning.lower():
                return True
            if anti_pattern.lower() in modification.modified_content.lower():
                return True
        
        return False
    
    def _matches_user_preferences(
        self,
        modification: ModificationSuggestion,
        preferences: Dict[str, Any]
    ) -> bool:
        """Check if modification aligns with user preferences."""
        
        # Check preferred modification types
        preferred_types = preferences.get("preferred_modification_types", [])
        if preferred_types and modification.modification_type in preferred_types:
            return True
        
        # Check preferred coding patterns
        preferred_patterns = preferences.get("preferred_patterns", [])
        for pattern in preferred_patterns:
            if pattern.lower() in modification.modified_content.lower():
                return True
        
        return False
    
    def _filter_by_safety_level(
        self,
        modifications: List[ModificationSuggestion],
        context: ModificationContext
    ) -> List[ModificationSuggestion]:
        """Filter modifications based on safety level requirements."""
        
        thresholds = self.safety_thresholds[context.safety_level]
        
        filtered = []
        for modification in modifications:
            if (modification.safety_score >= thresholds["min_safety"] and
                modification.complexity_score <= thresholds["max_complexity"] and
                abs(modification.net_lines_changed) <= thresholds["max_lines"]):
                filtered.append(modification)
            else:
                logger.debug(
                    "Filtering modification due to safety constraints",
                    file_path=modification.file_path,
                    safety_score=modification.safety_score,
                    complexity_score=modification.complexity_score,
                    lines_changed=modification.net_lines_changed
                )
        
        return filtered
    
    def _deduplicate_modifications(
        self,
        modifications: List[ModificationSuggestion]
    ) -> List[ModificationSuggestion]:
        """Remove duplicate or overlapping modifications."""
        
        # Group by file path and line ranges
        by_file = {}
        for mod in modifications:
            file_path = mod.file_path
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append(mod)
        
        deduplicated = []
        for file_path, file_modifications in by_file.items():
            # Sort by confidence and keep highest confidence for overlapping areas
            file_modifications.sort(key=lambda m: m.confidence, reverse=True)
            
            selected = []
            for mod in file_modifications:
                # Check for overlaps with already selected modifications
                overlaps = False
                for selected_mod in selected:
                    if self._modifications_overlap(mod, selected_mod):
                        overlaps = True
                        break
                
                if not overlaps:
                    selected.append(mod)
            
            deduplicated.extend(selected)
        
        return deduplicated
    
    def _modifications_overlap(
        self,
        mod1: ModificationSuggestion,
        mod2: ModificationSuggestion
    ) -> bool:
        """Check if two modifications overlap in their changes."""
        
        # Simple heuristic: if they modify similar content or same functions
        if mod1.functions_modified & mod2.functions_modified:
            return True
        
        # Check for content similarity (simplified)
        if (mod1.original_content and mod2.original_content and
            mod1.original_content.strip() == mod2.original_content.strip()):
            return True
        
        return False


# Export main class
__all__ = ["ModificationGenerator", "ModificationContext", "ModificationSuggestion"]
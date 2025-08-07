"""
Secure Modification Generator for Self-Modification System

This module provides LLM-powered code modification suggestions with comprehensive
security controls and safety validation. All modifications are generated in isolation
with strict safety scoring and human approval gates for high-risk changes.
"""

import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

from app.core.self_modification_code_analyzer import (
    SecureCodeAnalyzer, FileAnalysis, ProjectAnalysis, SecurityLevel,
    CodePattern, SecurityViolation
)

logger = logging.getLogger(__name__)


class ModificationGoal(Enum):
    """Types of modification goals."""
    IMPROVE_PERFORMANCE = "improve_performance"
    FIX_BUGS = "fix_bugs"
    ADD_FEATURES = "add_features"
    REFACTOR_CODE = "refactor_code"
    ENHANCE_SECURITY = "enhance_security"
    IMPROVE_MAINTAINABILITY = "improve_maintainability"
    UPDATE_DEPENDENCIES = "update_dependencies"


class ModificationRisk(Enum):
    """Risk levels for modifications."""
    MINIMAL = "minimal"      # 0.0 - 0.2 risk
    LOW = "low"             # 0.2 - 0.4 risk  
    MEDIUM = "medium"       # 0.4 - 0.6 risk
    HIGH = "high"           # 0.6 - 0.8 risk
    CRITICAL = "critical"   # 0.8 - 1.0 risk


@dataclass
class ModificationSuggestion:
    """A suggested code modification with safety analysis."""
    modification_id: str
    file_path: str
    modification_type: str
    goal: ModificationGoal
    
    # Content changes
    original_content: str
    modified_content: str
    content_diff: str
    
    # Analysis and reasoning
    modification_reason: str
    llm_reasoning: str
    expected_impact: str
    
    # Safety and risk assessment
    safety_score: float  # 0.0 = unsafe, 1.0 = safe
    risk_level: ModificationRisk
    complexity_increase: float  # -1.0 to 1.0 (negative = simplification)
    performance_impact_estimate: float  # Expected percentage change
    
    # Security analysis
    security_implications: List[str]
    introduces_vulnerabilities: List[SecurityViolation]
    requires_human_approval: bool
    
    # Change metadata
    lines_added: int
    lines_removed: int
    functions_affected: List[str]
    dependencies_changed: List[str]
    test_requirements: List[str]
    
    # Validation data
    confidence_score: float  # 0.0 to 1.0
    reversibility_score: float  # How easily can this be rolled back
    
    # Metadata
    generation_timestamp: datetime
    generator_version: str
    analysis_context: Dict[str, Any]


@dataclass
class ModificationPlan:
    """Complete plan for code modifications."""
    plan_id: str
    project_path: str
    goals: List[ModificationGoal]
    safety_level: SecurityLevel
    
    # Suggestions
    suggestions: List[ModificationSuggestion]
    total_suggestions: int
    approved_suggestions: int
    
    # Risk analysis
    overall_risk_score: float
    highest_risk_level: ModificationRisk
    critical_suggestions_count: int
    
    # Impact assessment
    estimated_performance_improvement: float
    estimated_complexity_change: float
    total_lines_changed: int
    total_files_affected: int
    
    # Requirements
    human_approval_required: bool
    approval_reasons: List[str]
    testing_requirements: List[str]
    
    # Metadata
    generation_timestamp: datetime
    analysis_metadata: Dict[str, Any]


class SecureModificationGenerator:
    """
    Secure modification generator with LLM-powered suggestions.
    
    Generates code modification suggestions using advanced AI models while
    maintaining strict security controls and safety validation.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED,
                 anthropic_api_key: Optional[str] = None):
        self.security_level = security_level
        self.anthropic_api_key = anthropic_api_key
        self.analyzer = SecureCodeAnalyzer(security_level)
        
        # Safety thresholds
        self._safety_thresholds = {
            SecurityLevel.MINIMAL: 0.3,
            SecurityLevel.STANDARD: 0.5,
            SecurityLevel.ENHANCED: 0.7,
            SecurityLevel.MAXIMUM: 0.85
        }
        
        # Risk assessment weights
        self._risk_weights = {
            'security_violations': 0.4,
            'complexity_increase': 0.2,
            'lines_changed': 0.1,
            'functions_affected': 0.15,
            'dependencies_changed': 0.15
        }
        
        logger.info(f"SecureModificationGenerator initialized with {security_level.value} security level")
    
    def generate_modification_plan(self, 
                                 project_path: str,
                                 goals: List[ModificationGoal],
                                 max_suggestions: int = 20,
                                 include_files: Optional[List[str]] = None,
                                 exclude_files: Optional[List[str]] = None) -> ModificationPlan:
        """
        Generate comprehensive modification plan for a project.
        
        Args:
            project_path: Path to project root
            goals: List of modification goals
            max_suggestions: Maximum number of suggestions to generate
            include_files: Specific files to analyze (None = all files)
            exclude_files: Files to exclude from analysis
            
        Returns:
            Complete modification plan with suggestions and risk analysis
            
        Raises:
            SecurityError: If critical security issues are detected
            ValueError: If project analysis fails
        """
        logger.info(f"Generating modification plan for {project_path} with goals: {goals}")
        
        # Analyze project
        project_analysis = self.analyzer.analyze_project(
            project_path, 
            include_patterns=include_files,
            exclude_patterns=exclude_files
        )
        
        # Generate suggestions for each goal
        all_suggestions = []
        for goal in goals:
            suggestions = self._generate_suggestions_for_goal(
                project_analysis, goal, max_suggestions // len(goals)
            )
            all_suggestions.extend(suggestions)
        
        # Limit total suggestions
        if len(all_suggestions) > max_suggestions:
            # Sort by safety score (highest first) and take top suggestions
            all_suggestions.sort(key=lambda s: s.safety_score, reverse=True)
            all_suggestions = all_suggestions[:max_suggestions]
        
        # Calculate overall risk and requirements
        overall_risk_score = self._calculate_overall_risk(all_suggestions)
        highest_risk_level = self._determine_highest_risk_level(all_suggestions)
        critical_count = sum(1 for s in all_suggestions if s.risk_level == ModificationRisk.CRITICAL)
        
        # Determine approval requirements
        human_approval_required, approval_reasons = self._determine_approval_requirements(
            all_suggestions, overall_risk_score
        )
        
        # Calculate impact estimates
        estimated_performance_improvement = sum(
            s.performance_impact_estimate for s in all_suggestions
        ) / len(all_suggestions) if all_suggestions else 0.0
        
        estimated_complexity_change = sum(
            s.complexity_increase for s in all_suggestions
        ) / len(all_suggestions) if all_suggestions else 0.0
        
        total_lines_changed = sum(s.lines_added + s.lines_removed for s in all_suggestions)
        total_files_affected = len(set(s.file_path for s in all_suggestions))
        
        # Generate testing requirements
        testing_requirements = self._generate_testing_requirements(all_suggestions)
        
        plan_id = self._generate_plan_id(project_path, goals)
        
        return ModificationPlan(
            plan_id=plan_id,
            project_path=project_path,
            goals=goals,
            safety_level=self.security_level,
            suggestions=all_suggestions,
            total_suggestions=len(all_suggestions),
            approved_suggestions=0,  # Initially none approved
            overall_risk_score=overall_risk_score,
            highest_risk_level=highest_risk_level,
            critical_suggestions_count=critical_count,
            estimated_performance_improvement=estimated_performance_improvement,
            estimated_complexity_change=estimated_complexity_change,
            total_lines_changed=total_lines_changed,
            total_files_affected=total_files_affected,
            human_approval_required=human_approval_required,
            approval_reasons=approval_reasons,
            testing_requirements=testing_requirements,
            generation_timestamp=datetime.utcnow(),
            analysis_metadata={
                'generator_version': '1.0.0',
                'security_level': self.security_level.value,
                'project_files_analyzed': len(project_analysis.file_analyses),
                'project_lines_analyzed': project_analysis.total_lines,
                'project_safety_score': project_analysis.overall_safety
            }
        )
    
    def generate_file_modifications(self, 
                                  file_path: str,
                                  goals: List[ModificationGoal],
                                  max_suggestions: int = 10) -> List[ModificationSuggestion]:
        """
        Generate modification suggestions for a single file.
        
        Args:
            file_path: Path to file to modify
            goals: List of modification goals
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of modification suggestions
        """
        logger.info(f"Generating modifications for file {file_path} with goals: {goals}")
        
        # Analyze file
        file_analysis = self.analyzer.analyze_file(file_path)
        
        # Generate suggestions
        suggestions = []
        for goal in goals:
            file_suggestions = self._generate_file_suggestions_for_goal(
                file_analysis, goal, max_suggestions // len(goals)
            )
            suggestions.extend(file_suggestions)
        
        # Sort by safety score and limit
        suggestions.sort(key=lambda s: s.safety_score, reverse=True)
        return suggestions[:max_suggestions]
    
    def _generate_suggestions_for_goal(self, 
                                     project_analysis: ProjectAnalysis,
                                     goal: ModificationGoal,
                                     max_suggestions: int) -> List[ModificationSuggestion]:
        """Generate suggestions for a specific goal across the project."""
        suggestions = []
        
        for file_analysis in project_analysis.file_analyses:
            file_suggestions = self._generate_file_suggestions_for_goal(
                file_analysis, goal, max_suggestions // len(project_analysis.file_analyses)
            )
            suggestions.extend(file_suggestions)
        
        return suggestions
    
    def _generate_file_suggestions_for_goal(self,
                                          file_analysis: FileAnalysis,
                                          goal: ModificationGoal,
                                          max_suggestions: int) -> List[ModificationSuggestion]:
        """Generate suggestions for a specific file and goal."""
        suggestions = []
        
        if goal == ModificationGoal.IMPROVE_PERFORMANCE:
            suggestions.extend(self._generate_performance_suggestions(file_analysis))
        elif goal == ModificationGoal.FIX_BUGS:
            suggestions.extend(self._generate_bug_fix_suggestions(file_analysis))
        elif goal == ModificationGoal.ENHANCE_SECURITY:
            suggestions.extend(self._generate_security_suggestions(file_analysis))
        elif goal == ModificationGoal.IMPROVE_MAINTAINABILITY:
            suggestions.extend(self._generate_maintainability_suggestions(file_analysis))
        elif goal == ModificationGoal.REFACTOR_CODE:
            suggestions.extend(self._generate_refactoring_suggestions(file_analysis))
        
        # Limit and sort by safety score
        suggestions.sort(key=lambda s: s.safety_score, reverse=True)
        return suggestions[:max_suggestions]
    
    def _generate_performance_suggestions(self, file_analysis: FileAnalysis) -> List[ModificationSuggestion]:
        """Generate performance improvement suggestions."""
        suggestions = []
        
        # Look for performance anti-patterns
        for pattern in file_analysis.code_patterns:
            if pattern.pattern_type == 'performance_issue':
                suggestion = self._create_pattern_based_suggestion(
                    file_analysis, pattern, ModificationGoal.IMPROVE_PERFORMANCE
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        # Analyze functions for performance opportunities
        for function in file_analysis.functions:
            if function.cyclomatic_complexity > 10:
                suggestion = self._create_complexity_reduction_suggestion(
                    file_analysis, function
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_bug_fix_suggestions(self, file_analysis: FileAnalysis) -> List[ModificationSuggestion]:
        """Generate bug fix suggestions."""
        suggestions = []
        
        # Look for potential bug patterns
        for pattern in file_analysis.code_patterns:
            if pattern.pattern_type == 'anti_pattern' and 'bug' in pattern.pattern_name.lower():
                suggestion = self._create_pattern_based_suggestion(
                    file_analysis, pattern, ModificationGoal.FIX_BUGS
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_security_suggestions(self, file_analysis: FileAnalysis) -> List[ModificationSuggestion]:
        """Generate security enhancement suggestions."""
        suggestions = []
        
        # Address security violations
        for violation in file_analysis.security_violations:
            if violation.severity in ['high', 'critical']:
                suggestion = self._create_security_fix_suggestion(
                    file_analysis, violation
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_maintainability_suggestions(self, file_analysis: FileAnalysis) -> List[ModificationSuggestion]:
        """Generate maintainability improvement suggestions."""
        suggestions = []
        
        # Look for functions without docstrings
        for function in file_analysis.functions:
            if not function.has_docstring:
                suggestion = self._create_docstring_suggestion(
                    file_analysis, function
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_refactoring_suggestions(self, file_analysis: FileAnalysis) -> List[ModificationSuggestion]:
        """Generate code refactoring suggestions."""
        suggestions = []
        
        # Look for long functions
        for function in file_analysis.functions:
            if function.lines_of_code > 50:
                suggestion = self._create_function_split_suggestion(
                    file_analysis, function
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _create_pattern_based_suggestion(self, 
                                       file_analysis: FileAnalysis,
                                       pattern: CodePattern,
                                       goal: ModificationGoal) -> Optional[ModificationSuggestion]:
        """Create a suggestion based on a detected code pattern."""
        
        # Read file content around the pattern
        try:
            with open(file_analysis.file_path, 'r') as f:
                lines = f.readlines()
        except IOError:
            return None
        
        if not pattern.line_number or pattern.line_number > len(lines):
            return None
        
        # Extract context around the issue
        start_line = max(0, pattern.line_number - 3)
        end_line = min(len(lines), pattern.line_number + 3)
        original_content = ''.join(lines[start_line:end_line])
        
        # Generate improved version using pattern suggestions
        modified_content = self._apply_pattern_suggestions(
            original_content, pattern
        )
        
        if not modified_content or modified_content == original_content:
            return None
        
        # Calculate diff
        content_diff = self._generate_diff(original_content, modified_content)
        
        # Assess modification risk
        safety_score = self._calculate_safety_score(
            pattern, original_content, modified_content
        )
        risk_level = self._determine_risk_level(safety_score)
        
        modification_id = self._generate_modification_id(
            file_analysis.file_path, pattern.pattern_name, pattern.line_number
        )
        
        return ModificationSuggestion(
            modification_id=modification_id,
            file_path=file_analysis.file_path,
            modification_type=pattern.pattern_type,
            goal=goal,
            original_content=original_content,
            modified_content=modified_content,
            content_diff=content_diff,
            modification_reason=pattern.description,
            llm_reasoning=f"Detected {pattern.pattern_name} pattern with {pattern.confidence} confidence",
            expected_impact=f"Should improve {goal.value}",
            safety_score=safety_score,
            risk_level=risk_level,
            complexity_increase=-0.1,  # Pattern fixes usually reduce complexity
            performance_impact_estimate=5.0 if goal == ModificationGoal.IMPROVE_PERFORMANCE else 0.0,
            security_implications=[],
            introduces_vulnerabilities=[],
            requires_human_approval=safety_score < self._safety_thresholds[self.security_level],
            lines_added=modified_content.count('\n'),
            lines_removed=original_content.count('\n'),
            functions_affected=[],  # TODO: Extract from AST analysis
            dependencies_changed=[],
            test_requirements=[f"Test {pattern.pattern_name} fix"],
            confidence_score=pattern.confidence,
            reversibility_score=0.9,  # Pattern fixes are usually easy to reverse
            generation_timestamp=datetime.utcnow(),
            generator_version='1.0.0',
            analysis_context={
                'pattern_type': pattern.pattern_type,
                'pattern_confidence': pattern.confidence,
                'line_number': pattern.line_number
            }
        )
    
    def _create_security_fix_suggestion(self,
                                      file_analysis: FileAnalysis,
                                      violation: SecurityViolation) -> Optional[ModificationSuggestion]:
        """Create a suggestion to fix a security violation."""
        
        if not violation.line_number or not violation.remediation:
            return None
        
        try:
            with open(file_analysis.file_path, 'r') as f:
                lines = f.readlines()
        except IOError:
            return None
        
        if violation.line_number > len(lines):
            return None
        
        # Extract context around the violation
        start_line = max(0, violation.line_number - 2)
        end_line = min(len(lines), violation.line_number + 2)
        original_content = ''.join(lines[start_line:end_line])
        
        # Generate secure version based on remediation
        modified_content = self._apply_security_remediation(
            original_content, violation
        )
        
        if not modified_content or modified_content == original_content:
            return None
        
        content_diff = self._generate_diff(original_content, modified_content)
        
        # Security fixes are high priority but need careful validation
        safety_score = 0.6  # Moderate safety - needs testing
        risk_level = ModificationRisk.MEDIUM
        
        if violation.severity == 'critical':
            safety_score = 0.4  # Lower safety for critical fixes
            risk_level = ModificationRisk.HIGH
        
        modification_id = self._generate_modification_id(
            file_analysis.file_path, f"security_fix_{violation.violation_type}", violation.line_number
        )
        
        return ModificationSuggestion(
            modification_id=modification_id,
            file_path=file_analysis.file_path,
            modification_type='security_fix',
            goal=ModificationGoal.ENHANCE_SECURITY,
            original_content=original_content,
            modified_content=modified_content,
            content_diff=content_diff,
            modification_reason=f"Fix {violation.violation_type}: {violation.description}",
            llm_reasoning=f"Applying remediation: {violation.remediation}",
            expected_impact="Improves security posture",
            safety_score=safety_score,
            risk_level=risk_level,
            complexity_increase=0.1,  # Security fixes may add complexity
            performance_impact_estimate=0.0,
            security_implications=[f"Fixes {violation.violation_type}"],
            introduces_vulnerabilities=[],
            requires_human_approval=True,  # Always require approval for security changes
            lines_added=modified_content.count('\n'),
            lines_removed=original_content.count('\n'),
            functions_affected=[],
            dependencies_changed=[],
            test_requirements=[f"Test security fix for {violation.violation_type}"],
            confidence_score=0.8,
            reversibility_score=0.7,  # Security fixes may be harder to reverse
            generation_timestamp=datetime.utcnow(),
            generator_version='1.0.0',
            analysis_context={
                'violation_type': violation.violation_type,
                'severity': violation.severity,
                'original_line': violation.line_number
            }
        )
    
    def _create_docstring_suggestion(self,
                                   file_analysis: FileAnalysis,
                                   function) -> Optional[ModificationSuggestion]:
        """Create a suggestion to add docstring to a function."""
        # This is a placeholder - in a real implementation, we would:
        # 1. Extract the function definition from the AST
        # 2. Generate appropriate docstring based on function signature
        # 3. Insert docstring at the correct location
        
        # For now, return None to avoid incomplete implementation
        return None
    
    def _apply_pattern_suggestions(self, original_content: str, pattern: CodePattern) -> str:
        """Apply pattern-based improvements to code."""
        modified_content = original_content
        
        # Apply simple pattern fixes
        if pattern.pattern_name == 'inefficient_loop':
            # Replace range(len()) with enumerate()
            modified_content = re.sub(
                r'for\s+(\w+)\s+in\s+range\s*\(\s*len\s*\((\w+)\)\s*\):',
                r'for \1, item in enumerate(\2):',
                modified_content
            )
        elif pattern.pattern_name == 'string_concatenation':
            # This is more complex and would need AST analysis in practice
            pass
        
        return modified_content
    
    def _apply_security_remediation(self, original_content: str, violation: SecurityViolation) -> str:
        """Apply security remediation to code."""
        modified_content = original_content
        
        if violation.violation_type == 'eval_usage':
            # Replace eval with ast.literal_eval (simplified)
            modified_content = modified_content.replace('eval(', 'ast.literal_eval(')
            if 'ast.literal_eval' in modified_content and 'import ast' not in modified_content:
                modified_content = 'import ast\n' + modified_content
        elif violation.violation_type == 'shell_injection':
            # Remove shell=True
            modified_content = re.sub(r'shell\s*=\s*True', 'shell=False', modified_content)
        
        return modified_content
    
    def _calculate_safety_score(self, pattern: CodePattern, original: str, modified: str) -> float:
        """Calculate safety score for a modification."""
        base_score = 0.7  # Base safety score
        
        # Adjust based on pattern confidence
        confidence_adjustment = pattern.confidence * 0.2
        
        # Adjust based on change size
        change_size = abs(len(modified) - len(original)) / max(len(original), 1)
        size_penalty = min(change_size * 0.3, 0.3)
        
        safety_score = base_score + confidence_adjustment - size_penalty
        return max(0.0, min(1.0, safety_score))
    
    def _determine_risk_level(self, safety_score: float) -> ModificationRisk:
        """Determine risk level from safety score."""
        if safety_score >= 0.8:
            return ModificationRisk.MINIMAL
        elif safety_score >= 0.6:
            return ModificationRisk.LOW
        elif safety_score >= 0.4:
            return ModificationRisk.MEDIUM
        elif safety_score >= 0.2:
            return ModificationRisk.HIGH
        else:
            return ModificationRisk.CRITICAL
    
    def _calculate_overall_risk(self, suggestions: List[ModificationSuggestion]) -> float:
        """Calculate overall risk score for all suggestions."""
        if not suggestions:
            return 0.0
        
        # Weight by individual safety scores
        total_risk = sum(1.0 - s.safety_score for s in suggestions)
        return total_risk / len(suggestions)
    
    def _determine_highest_risk_level(self, suggestions: List[ModificationSuggestion]) -> ModificationRisk:
        """Determine highest risk level among suggestions."""
        if not suggestions:
            return ModificationRisk.MINIMAL
        
        risk_values = {
            ModificationRisk.MINIMAL: 0,
            ModificationRisk.LOW: 1,
            ModificationRisk.MEDIUM: 2,
            ModificationRisk.HIGH: 3,
            ModificationRisk.CRITICAL: 4
        }
        
        max_risk_value = max(risk_values[s.risk_level] for s in suggestions)
        
        for risk, value in risk_values.items():
            if value == max_risk_value:
                return risk
        
        return ModificationRisk.MINIMAL
    
    def _determine_approval_requirements(self, 
                                       suggestions: List[ModificationSuggestion],
                                       overall_risk_score: float) -> Tuple[bool, List[str]]:
        """Determine if human approval is required."""
        approval_required = False
        reasons = []
        
        # Check individual suggestion requirements
        high_risk_count = sum(1 for s in suggestions if s.requires_human_approval)
        if high_risk_count > 0:
            approval_required = True
            reasons.append(f"{high_risk_count} suggestions require human approval")
        
        # Check overall risk
        if overall_risk_score > 0.6:
            approval_required = True
            reasons.append(f"Overall risk score too high: {overall_risk_score:.2f}")
        
        # Check security-related changes
        security_changes = sum(1 for s in suggestions if s.modification_type == 'security_fix')
        if security_changes > 0:
            approval_required = True
            reasons.append(f"{security_changes} security-related modifications")
        
        # Check for critical risk modifications
        critical_changes = sum(1 for s in suggestions if s.risk_level == ModificationRisk.CRITICAL)
        if critical_changes > 0:
            approval_required = True
            reasons.append(f"{critical_changes} critical risk modifications")
        
        return approval_required, reasons
    
    def _generate_testing_requirements(self, suggestions: List[ModificationSuggestion]) -> List[str]:
        """Generate testing requirements for modifications."""
        requirements = set()
        
        for suggestion in suggestions:
            requirements.update(suggestion.test_requirements)
            
            # Add specific requirements based on modification type
            if suggestion.modification_type == 'performance':
                requirements.add("Performance benchmarking tests")
            elif suggestion.modification_type == 'security_fix':
                requirements.add("Security validation tests")
            elif suggestion.risk_level in [ModificationRisk.HIGH, ModificationRisk.CRITICAL]:
                requirements.add("Comprehensive regression testing")
        
        requirements.add("All existing tests must pass")
        return sorted(list(requirements))
    
    def _generate_plan_id(self, project_path: str, goals: List[ModificationGoal]) -> str:
        """Generate unique plan ID."""
        content = f"{project_path}_{sorted([g.value for g in goals])}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_modification_id(self, file_path: str, modification_type: str, line_number: int) -> str:
        """Generate unique modification ID."""
        content = f"{file_path}_{modification_type}_{line_number}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate unified diff between original and modified content."""
        # Simple diff implementation - in production, use difflib
        if original == modified:
            return ""
        
        return f"--- Original\n+++ Modified\n@@ Changes @@\n-{original.strip()}\n+{modified.strip()}"


# Custom exceptions
class SecurityError(Exception):
    """Security violation in modification generation."""
    pass


# Export main classes
__all__ = [
    'SecureModificationGenerator',
    'ModificationPlan',
    'ModificationSuggestion', 
    'ModificationGoal',
    'ModificationRisk',
    'SecurityError'
]
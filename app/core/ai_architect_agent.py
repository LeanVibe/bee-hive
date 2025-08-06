"""
AI Architect Agent - Advanced Pattern Recognition System

This agent specializes in architectural intelligence, pattern recognition,
and cross-agent knowledge sharing to improve code quality and system design.
Part of the AI Enhancement Team for LeanVibe Agent Hive 2.0.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import hashlib
import re

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .redis import get_message_broker, AgentMessageBroker
from .intelligence_framework import (
    IntelligenceModelInterface, 
    IntelligencePrediction, 
    IntelligenceType,
    DataPoint,
    DataType
)
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus
from ..models.context import Context

logger = structlog.get_logger()


class PatternType(Enum):
    """Types of code/architecture patterns that can be recognized."""
    DESIGN_PATTERN = "design_pattern"
    ARCHITECTURE_PATTERN = "architecture_pattern"
    CODE_PATTERN = "code_pattern"
    ANTI_PATTERN = "anti_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    SECURITY_PATTERN = "security_pattern"
    ERROR_HANDLING_PATTERN = "error_handling_pattern"


class PatternQuality(Enum):
    """Quality assessment for recognized patterns."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    PROBLEMATIC = "problematic"
    ANTI_PATTERN = "anti_pattern"


@dataclass
class CodePattern:
    """Represents a recognized code or architecture pattern."""
    pattern_id: str
    name: str
    pattern_type: PatternType
    quality: PatternQuality
    description: str
    code_snippet: Optional[str]
    usage_context: List[str]
    success_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    related_patterns: List[str]
    detection_confidence: float
    last_updated: datetime
    usage_count: int = 0
    success_rate: float = 0.0
    
    def to_template(self) -> Dict[str, Any]:
        """Convert pattern to reusable template."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'type': self.pattern_type.value,
            'template_code': self.code_snippet,
            'usage_guidelines': self.usage_context,
            'quality_score': self._calculate_quality_score(),
            'recommendations': self.improvement_suggestions,
            'validation_rules': self._generate_validation_rules()
        }
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall quality score for the pattern."""
        quality_weights = {
            PatternQuality.EXCELLENT: 1.0,
            PatternQuality.GOOD: 0.8,
            PatternQuality.ACCEPTABLE: 0.6,
            PatternQuality.PROBLEMATIC: 0.3,
            PatternQuality.ANTI_PATTERN: 0.0
        }
        base_score = quality_weights.get(self.quality, 0.5)
        
        # Adjust based on usage success
        success_adjustment = self.success_rate * 0.3
        confidence_adjustment = self.detection_confidence * 0.2
        
        return min(1.0, base_score + success_adjustment + confidence_adjustment)
    
    def _generate_validation_rules(self) -> List[str]:
        """Generate validation rules for pattern usage."""
        rules = []
        
        if self.quality == PatternQuality.EXCELLENT:
            rules.append("Prioritize this pattern for similar use cases")
        elif self.quality == PatternQuality.ANTI_PATTERN:
            rules.append("Avoid this pattern - suggest alternatives")
            
        if self.success_rate > 0.8:
            rules.append("High success rate - recommend for production use")
        elif self.success_rate < 0.5:
            rules.append("Low success rate - use with caution")
            
        return rules


@dataclass
class ArchitecturalDecision:
    """Represents an architectural decision with intelligence backing."""
    decision_id: str
    context: str
    options: List[Dict[str, Any]]
    recommended_option: str
    reasoning: str
    confidence: float
    risk_assessment: Dict[str, str]
    implementation_guidance: List[str]
    success_probability: float
    patterns_applied: List[str]
    timestamp: datetime


class PatternRecognitionEngine:
    """Advanced pattern recognition engine for code and architecture analysis."""
    
    def __init__(self):
        self.pattern_library: Dict[str, CodePattern] = {}
        self.pattern_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'problematic': 0.3
        }
    
    async def analyze_code(self, code: str, context: Dict[str, Any]) -> List[CodePattern]:
        """Analyze code and identify patterns."""
        patterns = []
        
        # Analyze different pattern types
        design_patterns = await self._detect_design_patterns(code)
        architecture_patterns = await self._detect_architecture_patterns(code, context)
        performance_patterns = await self._detect_performance_patterns(code)
        security_patterns = await self._detect_security_patterns(code)
        anti_patterns = await self._detect_anti_patterns(code)
        
        patterns.extend(design_patterns)
        patterns.extend(architecture_patterns)
        patterns.extend(performance_patterns)
        patterns.extend(security_patterns)
        patterns.extend(anti_patterns)
        
        return patterns
    
    async def _detect_design_patterns(self, code: str) -> List[CodePattern]:
        """Detect common design patterns in code."""
        patterns = []
        
        # Singleton pattern detection
        if self._is_singleton_pattern(code):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="Singleton Pattern",
                pattern_type=PatternType.DESIGN_PATTERN,
                quality=PatternQuality.GOOD,
                description="Ensures a class has only one instance",
                code_snippet=self._extract_singleton_snippet(code),
                usage_context=["Global state management", "Resource management"],
                success_metrics={"maintainability": 0.7, "testability": 0.5},
                improvement_suggestions=["Consider dependency injection", "Ensure thread safety"],
                related_patterns=["Factory", "Registry"],
                detection_confidence=0.8,
                last_updated=datetime.now()
            ))
        
        # Factory pattern detection
        if self._is_factory_pattern(code):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="Factory Pattern",
                pattern_type=PatternType.DESIGN_PATTERN,
                quality=PatternQuality.EXCELLENT,
                description="Creates objects without specifying exact classes",
                code_snippet=self._extract_factory_snippet(code),
                usage_context=["Object creation", "Abstraction"],
                success_metrics={"flexibility": 0.9, "maintainability": 0.8},
                improvement_suggestions=["Add abstract factory for families"],
                related_patterns=["Builder", "Abstract Factory"],
                detection_confidence=0.9,
                last_updated=datetime.now()
            ))
        
        return patterns
    
    async def _detect_architecture_patterns(self, code: str, context: Dict[str, Any]) -> List[CodePattern]:
        """Detect architectural patterns."""
        patterns = []
        
        # Microservices pattern
        if self._is_microservice_architecture(code, context):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="Microservice Architecture",
                pattern_type=PatternType.ARCHITECTURE_PATTERN,
                quality=PatternQuality.EXCELLENT,
                description="Decomposed services communicating via APIs",
                code_snippet=None,
                usage_context=["Scalability", "Independent deployment"],
                success_metrics={"scalability": 0.9, "maintainability": 0.8},
                improvement_suggestions=["Implement circuit breakers", "Add service discovery"],
                related_patterns=["API Gateway", "Event Sourcing"],
                detection_confidence=0.8,
                last_updated=datetime.now()
            ))
        
        return patterns
    
    async def _detect_performance_patterns(self, code: str) -> List[CodePattern]:
        """Detect performance-related patterns."""
        patterns = []
        
        # Caching pattern detection
        if self._has_caching_pattern(code):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="Caching Pattern",
                pattern_type=PatternType.PERFORMANCE_PATTERN,
                quality=PatternQuality.EXCELLENT,
                description="Stores frequently accessed data for faster retrieval",
                code_snippet=self._extract_caching_snippet(code),
                usage_context=["Performance optimization", "Data access"],
                success_metrics={"performance": 0.9, "resource_usage": 0.7},
                improvement_suggestions=["Implement cache invalidation", "Monitor cache hit rates"],
                related_patterns=["Lazy Loading", "Memoization"],
                detection_confidence=0.8,
                last_updated=datetime.now()
            ))
        
        return patterns
    
    async def _detect_security_patterns(self, code: str) -> List[CodePattern]:
        """Detect security-related patterns."""
        patterns = []
        
        # Input validation pattern
        if self._has_input_validation(code):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="Input Validation Pattern",
                pattern_type=PatternType.SECURITY_PATTERN,
                quality=PatternQuality.EXCELLENT,
                description="Validates and sanitizes all user inputs",
                code_snippet=self._extract_validation_snippet(code),
                usage_context=["Security", "Data integrity"],
                success_metrics={"security": 0.9, "reliability": 0.8},
                improvement_suggestions=["Add comprehensive error handling"],
                related_patterns=["Authentication", "Authorization"],
                detection_confidence=0.9,
                last_updated=datetime.now()
            ))
        
        return patterns
    
    async def _detect_anti_patterns(self, code: str) -> List[CodePattern]:
        """Detect anti-patterns that should be avoided."""
        patterns = []
        
        # God class anti-pattern
        if self._is_god_class(code):
            patterns.append(CodePattern(
                pattern_id=str(uuid.uuid4()),
                name="God Class Anti-Pattern",
                pattern_type=PatternType.ANTI_PATTERN,
                quality=PatternQuality.ANTI_PATTERN,
                description="Class with too many responsibilities",
                code_snippet=self._extract_god_class_snippet(code),
                usage_context=["Code smell", "Maintainability issue"],
                success_metrics={"maintainability": 0.1, "testability": 0.2},
                improvement_suggestions=[
                    "Split into smaller, focused classes",
                    "Apply Single Responsibility Principle",
                    "Extract utility methods to separate classes"
                ],
                related_patterns=["Single Responsibility", "Facade"],
                detection_confidence=0.7,
                last_updated=datetime.now()
            ))
        
        return patterns
    
    # Pattern detection helper methods
    def _is_singleton_pattern(self, code: str) -> bool:
        """Check if code implements singleton pattern."""
        singleton_indicators = [
            r'class.*:\s*\n.*_instance\s*=\s*None',
            r'def __new__\(cls[,\)]',
            r'if.*_instance.*is None',
            r'@classmethod\s+def get_instance'
        ]
        return any(re.search(pattern, code, re.IGNORECASE | re.MULTILINE) 
                  for pattern in singleton_indicators)
    
    def _is_factory_pattern(self, code: str) -> bool:
        """Check if code implements factory pattern."""
        factory_indicators = [
            r'def create_\w+\(',
            r'class.*Factory',
            r'@staticmethod\s+def create',
            r'def get_\w+\(.*type'
        ]
        return any(re.search(pattern, code, re.IGNORECASE | re.MULTILINE) 
                  for pattern in factory_indicators)
    
    def _is_microservice_architecture(self, code: str, context: Dict[str, Any]) -> bool:
        """Check if architecture follows microservice pattern."""
        microservice_indicators = [
            'FastAPI' in code,
            'app.route' in code,
            '@app.post' in code or '@app.get' in code,
            'async def' in code,
            context.get('has_api_endpoints', False),
            context.get('has_database_layer', False)
        ]
        return sum(bool(indicator) for indicator in microservice_indicators) >= 3
    
    def _has_caching_pattern(self, code: str) -> bool:
        """Check if code implements caching pattern."""
        cache_indicators = [
            r'@cache',
            r'@lru_cache',
            r'cache\.get\(',
            r'cache\.set\(',
            r'redis.*get',
            r'memcache'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) 
                  for pattern in cache_indicators)
    
    def _has_input_validation(self, code: str) -> bool:
        """Check if code has input validation patterns."""
        validation_indicators = [
            r'validate\(',
            r'@validator',
            r'pydantic',
            r'if.*not.*isinstance',
            r'raise ValueError',
            r'assert.*type'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) 
                  for pattern in validation_indicators)
    
    def _is_god_class(self, code: str) -> bool:
        """Check if class is a god class (too many responsibilities)."""
        lines = code.split('\n')
        class_lines = [line for line in lines if line.strip().startswith('def ') and not line.strip().startswith('def __')]
        method_count = len(class_lines)
        
        # Simple heuristic: more than 15 methods might indicate a god class
        return method_count > 15
    
    # Snippet extraction methods
    def _extract_singleton_snippet(self, code: str) -> str:
        """Extract singleton pattern code snippet."""
        # This is a simplified extraction - in practice, you'd use AST parsing
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if '_instance' in line and 'None' in line:
                return '\n'.join(lines[max(0, i-2):min(len(lines), i+5)])
        return "# Singleton pattern detected"
    
    def _extract_factory_snippet(self, code: str) -> str:
        """Extract factory pattern code snippet."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if 'create_' in line.lower() or 'Factory' in line:
                return '\n'.join(lines[max(0, i-1):min(len(lines), i+4)])
        return "# Factory pattern detected"
    
    def _extract_caching_snippet(self, code: str) -> str:
        """Extract caching pattern code snippet."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if any(cache_term in line.lower() for cache_term in ['cache', 'lru_cache', 'redis']):
                return '\n'.join(lines[max(0, i-1):min(len(lines), i+3)])
        return "# Caching pattern detected"
    
    def _extract_validation_snippet(self, code: str) -> str:
        """Extract input validation code snippet."""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if any(val_term in line.lower() for val_term in ['validate', 'isinstance', 'assert']):
                return '\n'.join(lines[max(0, i-1):min(len(lines), i+3)])
        return "# Input validation pattern detected"
    
    def _extract_god_class_snippet(self, code: str) -> str:
        """Extract god class example."""
        lines = code.split('\n')
        class_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('class '):
                class_start = i
                break
        
        if class_start >= 0:
            return '\n'.join(lines[class_start:min(len(lines), class_start + 10)]) + "\n# ... (many more methods)"
        return "# God class detected - too many responsibilities"


class AIArchitectAgent(IntelligenceModelInterface):
    """
    AI Architect Agent with advanced pattern recognition and architectural intelligence.
    
    This agent specializes in:
    - Recognizing and cataloging code/architecture patterns
    - Providing architectural guidance based on historical success
    - Sharing pattern knowledge across other agents
    - Continuously improving pattern recognition accuracy
    """
    
    def __init__(self, agent_id: str, anthropic_client: Optional[AsyncAnthropic] = None):
        self.agent_id = agent_id
        self.client = anthropic_client
        self.pattern_engine = PatternRecognitionEngine()
        self.decision_history: List[ArchitecturalDecision] = []
        self.performance_metrics: Dict[str, float] = {}
        self.learning_rate = 0.1
        
    async def predict(self, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Make architectural predictions based on code analysis."""
        code = input_data.get('code', '')
        context = input_data.get('context', {})
        request_type = input_data.get('type', 'pattern_analysis')
        
        if request_type == 'pattern_analysis':
            return await self._analyze_patterns(code, context)
        elif request_type == 'architectural_decision':
            return await self._make_architectural_decision(input_data)
        elif request_type == 'code_quality_assessment':
            return await self._assess_code_quality(code, context)
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _analyze_patterns(self, code: str, context: Dict[str, Any]) -> IntelligencePrediction:
        """Analyze code patterns and provide recommendations."""
        patterns = await self.pattern_engine.analyze_code(code, context)
        
        # Calculate overall quality score
        quality_scores = [pattern.quality.value for pattern in patterns]
        avg_quality = sum(self.pattern_engine.quality_thresholds.get(q, 0.5) for q in quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Generate recommendations
        recommendations = []
        for pattern in patterns:
            if pattern.quality == PatternQuality.ANTI_PATTERN:
                recommendations.extend([f"⚠️  {suggestion}" for suggestion in pattern.improvement_suggestions])
            elif pattern.quality == PatternQuality.EXCELLENT:
                recommendations.append(f"✅ Excellent use of {pattern.name}")
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"code": code[:200], "context": context},
            prediction={
                "patterns_detected": [pattern.to_template() for pattern in patterns],
                "quality_score": avg_quality,
                "recommendations": recommendations,
                "pattern_summary": {
                    "excellent": len([p for p in patterns if p.quality == PatternQuality.EXCELLENT]),
                    "good": len([p for p in patterns if p.quality == PatternQuality.GOOD]),
                    "anti_patterns": len([p for p in patterns if p.quality == PatternQuality.ANTI_PATTERN])
                }
            },
            confidence=min(1.0, avg_quality + 0.1),
            explanation=f"Analyzed {len(patterns)} patterns with average quality score of {avg_quality:.2f}",
            timestamp=datetime.now()
        )
    
    async def _make_architectural_decision(self, input_data: Dict[str, Any]) -> IntelligencePrediction:
        """Make architectural decision based on context and patterns."""
        problem = input_data.get('problem', '')
        constraints = input_data.get('constraints', {})
        options = input_data.get('options', [])
        
        # Analyze each option against known patterns
        option_scores = {}
        for option in options:
            score = await self._score_architectural_option(option, constraints)
            option_scores[option['name']] = score
        
        # Select best option
        best_option = max(option_scores.keys(), key=lambda x: option_scores[x]['total_score'])
        
        decision = ArchitecturalDecision(
            decision_id=str(uuid.uuid4()),
            context=problem,
            options=options,
            recommended_option=best_option,
            reasoning=option_scores[best_option]['reasoning'],
            confidence=option_scores[best_option]['confidence'],
            risk_assessment=option_scores[best_option]['risks'],
            implementation_guidance=option_scores[best_option]['guidance'],
            success_probability=option_scores[best_option]['success_probability'],
            patterns_applied=option_scores[best_option]['patterns'],
            timestamp=datetime.now()
        )
        
        self.decision_history.append(decision)
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=decision.decision_id,
            input_data=input_data,
            prediction=asdict(decision),
            confidence=decision.confidence,
            explanation=decision.reasoning,
            timestamp=datetime.now()
        )
    
    async def _score_architectural_option(self, option: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Score an architectural option based on patterns and constraints."""
        name = option.get('name', 'Unknown')
        description = option.get('description', '')
        
        # Base scoring logic (simplified for demonstration)
        scores = {
            'scalability': 0.7,
            'maintainability': 0.8,
            'performance': 0.6,
            'security': 0.7,
            'complexity': 0.5  # Lower is better for complexity
        }
        
        # Adjust scores based on patterns
        if 'microservice' in description.lower():
            scores['scalability'] += 0.2
            scores['complexity'] -= 0.1
        
        if 'monolith' in description.lower():
            scores['complexity'] += 0.2
            scores['maintainability'] -= 0.1
        
        total_score = sum(scores.values()) / len(scores)
        
        return {
            'total_score': total_score,
            'detailed_scores': scores,
            'reasoning': f"{name} scored {total_score:.2f} based on pattern analysis",
            'confidence': 0.8,
            'success_probability': min(0.95, total_score + 0.1),
            'risks': {'complexity': 'medium', 'maintenance': 'low'},
            'guidance': [f"Monitor {name} implementation closely"],
            'patterns': ['architectural_analysis']
        }
    
    async def _assess_code_quality(self, code: str, context: Dict[str, Any]) -> IntelligencePrediction:
        """Assess overall code quality and provide improvement suggestions."""
        patterns = await self.pattern_engine.analyze_code(code, context)
        
        # Calculate quality metrics
        quality_metrics = {
            'pattern_quality': sum(p._calculate_quality_score() for p in patterns) / len(patterns) if patterns else 0.5,
            'anti_pattern_count': len([p for p in patterns if p.quality == PatternQuality.ANTI_PATTERN]),
            'excellent_pattern_count': len([p for p in patterns if p.quality == PatternQuality.EXCELLENT]),
            'overall_score': 0.0
        }
        
        # Calculate overall score
        base_score = quality_metrics['pattern_quality']
        anti_pattern_penalty = quality_metrics['anti_pattern_count'] * 0.1
        excellent_bonus = quality_metrics['excellent_pattern_count'] * 0.05
        
        quality_metrics['overall_score'] = max(0.0, min(1.0, base_score - anti_pattern_penalty + excellent_bonus))
        
        # Generate improvement suggestions
        improvements = []
        for pattern in patterns:
            if pattern.quality in [PatternQuality.PROBLEMATIC, PatternQuality.ANTI_PATTERN]:
                improvements.extend(pattern.improvement_suggestions)
        
        return IntelligencePrediction(
            model_id=self.agent_id,
            prediction_id=str(uuid.uuid4()),
            input_data={"code": code[:200]},
            prediction={
                "quality_metrics": quality_metrics,
                "improvement_suggestions": improvements,
                "grade": self._calculate_grade(quality_metrics['overall_score']),
                "patterns_summary": {
                    pattern.pattern_type.value: len([p for p in patterns if p.pattern_type == pattern.pattern_type])
                    for pattern in patterns
                }
            },
            confidence=0.8,
            explanation=f"Code quality assessment: {quality_metrics['overall_score']:.2f}/1.0",
            timestamp=datetime.now()
        )
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def train(self, training_data: List[DataPoint]) -> bool:
        """Train the AI Architect Agent with new pattern data."""
        try:
            for data_point in training_data:
                if data_point.data_type == DataType.TEXT:
                    # Extract patterns from training code
                    code = data_point.value
                    context = data_point.metadata
                    patterns = await self.pattern_engine.analyze_code(code, context)
                    
                    # Update pattern library with feedback
                    for pattern in patterns:
                        if pattern.pattern_id in self.pattern_engine.pattern_library:
                            existing = self.pattern_engine.pattern_library[pattern.pattern_id]
                            existing.usage_count += 1
                            # Update success rate based on feedback in metadata
                            if 'success_feedback' in data_point.metadata:
                                feedback = data_point.metadata['success_feedback']
                                existing.success_rate = (existing.success_rate + feedback) / 2
                        else:
                            self.pattern_engine.pattern_library[pattern.pattern_id] = pattern
            
            logger.info(f"AI Architect Agent trained on {len(training_data)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    async def evaluate(self, test_data: List[DataPoint]) -> Dict[str, float]:
        """Evaluate the AI Architect Agent performance."""
        correct_predictions = 0
        total_predictions = len(test_data)
        
        for data_point in test_data:
            try:
                prediction = await self.predict({
                    'code': data_point.value,
                    'context': data_point.metadata,
                    'type': 'pattern_analysis'
                })
                
                # Simple evaluation: check if prediction confidence is reasonable
                if prediction.confidence > 0.5:
                    correct_predictions += 1
                    
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_evaluations': total_predictions,
            'successful_predictions': correct_predictions,
            'pattern_library_size': len(self.pattern_engine.pattern_library)
        }
    
    async def get_pattern_templates(self) -> List[Dict[str, Any]]:
        """Get all pattern templates for sharing with other agents."""
        return [pattern.to_template() for pattern in self.pattern_engine.pattern_library.values()]
    
    async def share_architectural_insights(self) -> Dict[str, Any]:
        """Share architectural insights and decision patterns with other agents."""
        return {
            'decision_patterns': [asdict(decision) for decision in self.decision_history[-10:]],
            'successful_patterns': [
                pattern.to_template() 
                for pattern in self.pattern_engine.pattern_library.values()
                if pattern.success_rate > 0.8
            ],
            'anti_patterns_to_avoid': [
                pattern.to_template()
                for pattern in self.pattern_engine.pattern_library.values()
                if pattern.quality == PatternQuality.ANTI_PATTERN
            ],
            'performance_metrics': self.performance_metrics,
            'learning_insights': {
                'total_decisions': len(self.decision_history),
                'pattern_library_size': len(self.pattern_engine.pattern_library),
                'high_confidence_patterns': len([
                    p for p in self.pattern_engine.pattern_library.values()
                    if p.detection_confidence > 0.8
                ])
            }
        }


async def create_ai_architect_agent(agent_id: str) -> AIArchitectAgent:
    """Factory function to create a new AI Architect Agent."""
    anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
    return AIArchitectAgent(agent_id, anthropic_client)
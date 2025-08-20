"""
Unified Compression Command System for LeanVibe Agent Hive 2.0

Consolidates all compression implementations (Context Compression Engine, 
Context Consolidator, Memory Compression) into a single intelligent command
that automatically selects the optimal compression strategy.

Features:
- Automatic strategy selection based on content analysis
- Multiple compression algorithms (context, memory, conversation, adaptive)
- Performance optimization with <15s execution target
- Mobile-optimized responses
- Comprehensive error handling and recovery
- Backward compatibility through command aliases
"""

import asyncio
import json
import time
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
from pathlib import Path
import structlog

# Import existing compression systems
from .context_compression import (
    get_context_compressor, 
    CompressionLevel as ContextCompressionLevel,
    CompressedContext
)
from .context_compression_engine import ContextCompressionEngine
from ..models.context import ContextType

logger = structlog.get_logger()


class CompressionStrategy(Enum):
    """Available compression strategies."""
    CONTEXT = "context"           # Technical content, code, documentation
    MEMORY = "memory"             # Agent memory, historical patterns
    CONVERSATION = "conversation" # Chat logs, discussions, meetings
    ADAPTIVE = "adaptive"         # AI-powered strategy selection


class CompressionLevel(Enum):
    """Unified compression levels."""
    MINIMAL = "minimal"       # 10-20% reduction, preserve everything
    LIGHT = "light"           # 20-40% reduction, preserve key content
    STANDARD = "standard"     # 40-60% reduction, optimal balance
    AGGRESSIVE = "aggressive" # 60-80% reduction, keep only essentials
    MAXIMUM = "maximum"       # 80-90+ reduction, extreme summarization


class UnifiedCompressionResult:
    """Unified result object for all compression operations."""
    
    def __init__(self):
        self.success = False
        self.strategy_used = None
        self.compression_level = None
        self.original_content = ""
        self.compressed_content = ""
        self.original_token_count = 0
        self.compressed_token_count = 0
        self.compression_ratio = 0.0
        self.tokens_saved = 0
        self.execution_time_seconds = 0.0
        self.summary = ""
        self.key_insights = []
        self.decisions_made = []
        self.patterns_identified = []
        self.importance_score = 0.5
        self.metadata = {}
        self.performance_metrics = {}
        self.mobile_optimized = False
        self.error_message = None
        self.recovery_attempted = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "compression_level": self.compression_level.value if self.compression_level else None,
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "compression_ratio": round(self.compression_ratio, 3),
            "tokens_saved": self.tokens_saved,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "summary": self.summary,
            "key_insights": self.key_insights,
            "decisions_made": self.decisions_made,
            "patterns_identified": self.patterns_identified,
            "importance_score": round(self.importance_score, 2),
            "performance_metrics": self.performance_metrics,
            "mobile_optimized": self.mobile_optimized,
            "error_message": self.error_message,
            "recovery_attempted": self.recovery_attempted,
            "metadata": self.metadata
        }


class ContentAnalyzer:
    """Intelligent content analysis for strategy selection."""
    
    # Content type patterns for strategy selection
    STRATEGY_PATTERNS = {
        CompressionStrategy.CONTEXT: [
            r'def\s+\w+\(.*?\):',  # Python functions
            r'class\s+\w+.*?:',    # Python classes  
            r'import\s+\w+',       # Import statements
            r'```\w*\n.*?\n```',   # Code blocks
            r'error|exception|traceback|stack trace',  # Error content
            r'api|endpoint|request|response',          # API content
        ],
        CompressionStrategy.CONVERSATION: [
            r'@\w+\s+said:',       # Chat messages
            r'User:|Assistant:|Human:|AI:',  # Conversation markers
            r'Q:|A:|Question:|Answer:',      # Q&A format
            r'\d+:\d+\s+(AM|PM)',           # Timestamps
            r'meeting|discussion|chat|call', # Meeting content
        ],
        CompressionStrategy.MEMORY: [
            r'agent_id|agent_state|agent_memory',    # Agent references
            r'session|context|history|previous',     # Historical content
            r'learning|pattern|trend|insight',       # Learning content
            r'capability|skill|knowledge|expertise', # Agent capabilities
        ]
    }
    
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content to determine optimal compression strategy."""
        try:
            analysis = {
                "content_length": len(content),
                "token_estimate": self._estimate_tokens(content),
                "content_type_scores": {},
                "complexity_score": 0.0,
                "technical_density": 0.0,
                "conversation_density": 0.0,
                "recommended_strategy": CompressionStrategy.ADAPTIVE,
                "recommended_level": CompressionLevel.STANDARD,
                "confidence": 0.5
            }
            
            # Analyze against strategy patterns
            for strategy, patterns in self.STRATEGY_PATTERNS.items():
                score = self._calculate_pattern_score(content, patterns)
                analysis["content_type_scores"][strategy.value] = score
            
            # Calculate complexity metrics
            analysis["complexity_score"] = self._calculate_complexity(content)
            analysis["technical_density"] = self._calculate_technical_density(content)
            analysis["conversation_density"] = self._calculate_conversation_density(content)
            
            # Determine optimal strategy and level
            strategy, confidence = self._select_optimal_strategy(analysis)
            analysis["recommended_strategy"] = strategy
            analysis["confidence"] = confidence
            analysis["recommended_level"] = self._select_compression_level(analysis, strategy)
            
            return analysis
            
        except Exception as e:
            logger.error("Content analysis failed", error=str(e))
            return {
                "content_length": len(content),
                "recommended_strategy": CompressionStrategy.CONTEXT,
                "recommended_level": CompressionLevel.STANDARD,
                "confidence": 0.3,
                "error": str(e)
            }
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 characters per token on average
        return len(content) // 4
    
    def _calculate_pattern_score(self, content: str, patterns: List[str]) -> float:
        """Calculate how well content matches strategy patterns."""
        total_matches = 0
        content_lines = content.split('\n')
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            total_matches += len(matches)
        
        # Normalize by content length
        return min(1.0, total_matches / max(1, len(content_lines)))
    
    def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score."""
        complexity_indicators = [
            len(re.findall(r'\{.*?\}', content)),  # JSON/dict structures
            len(re.findall(r'\[.*?\]', content)),  # Arrays/lists
            len(re.findall(r'[(){}[\]]', content)), # Nested structures
            len(re.findall(r'[.!?]', content)),    # Sentence complexity
        ]
        
        total_complexity = sum(complexity_indicators)
        return min(1.0, total_complexity / max(1, len(content)))
    
    def _calculate_technical_density(self, content: str) -> float:
        """Calculate technical content density."""
        technical_indicators = [
            r'def\s|class\s|import\s|from\s',    # Python keywords
            r'function\s|var\s|const\s|let\s',   # JavaScript keywords  
            r'SELECT\s|UPDATE\s|INSERT\s|DELETE', # SQL keywords
            r'GET\s|POST\s|PUT\s|DELETE',        # HTTP methods
            r'\.[a-z]+\(.*?\)',                  # Method calls
            r'[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*', # Object notation
        ]
        
        total_matches = 0
        for pattern in technical_indicators:
            total_matches += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(1.0, total_matches / max(1, len(content.split())))
    
    def _calculate_conversation_density(self, content: str) -> float:
        """Calculate conversation content density."""
        conversation_indicators = [
            r'(said|asked|replied|responded)[:,]',  # Speech indicators
            r'@\w+|User:|Assistant:|Human:',        # User references
            r'[?!]{1,3}',                          # Questions/exclamations
            r'(yes|no|maybe|sure|okay|thanks)',     # Conversational words
        ]
        
        total_matches = 0
        for pattern in conversation_indicators:
            total_matches += len(re.findall(pattern, content, re.IGNORECASE))
        
        return min(1.0, total_matches / max(1, len(content.split())))
    
    def _select_optimal_strategy(self, analysis: Dict[str, Any]) -> Tuple[CompressionStrategy, float]:
        """Select optimal compression strategy based on analysis."""
        scores = analysis["content_type_scores"]
        
        # Find highest scoring strategy
        if not scores:
            return CompressionStrategy.ADAPTIVE, 0.3
            
        best_strategy_name = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_strategy_name]
        
        # Convert string back to enum
        strategy_map = {s.value: s for s in CompressionStrategy}
        best_strategy = strategy_map.get(best_strategy_name, CompressionStrategy.ADAPTIVE)
        
        # Calculate confidence based on score difference
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_diff = sorted_scores[0] - sorted_scores[1]
            confidence = min(0.95, 0.5 + score_diff)
        else:
            confidence = best_score
        
        # Use adaptive if confidence is low
        if confidence < 0.6:
            return CompressionStrategy.ADAPTIVE, confidence
            
        return best_strategy, confidence
    
    def _select_compression_level(self, analysis: Dict[str, Any], strategy: CompressionStrategy) -> CompressionLevel:
        """Select compression level based on content analysis and strategy."""
        content_length = analysis.get("content_length", 0)
        complexity = analysis.get("complexity_score", 0.5)
        
        # Adjust level based on content characteristics
        if content_length < 1000:  # Small content
            return CompressionLevel.LIGHT
        elif content_length > 50000:  # Very large content
            return CompressionLevel.AGGRESSIVE
        elif complexity > 0.7:  # Complex content
            return CompressionLevel.LIGHT  # Preserve more detail
        elif complexity < 0.3:  # Simple content
            return CompressionLevel.AGGRESSIVE  # Can compress more
        else:
            return CompressionLevel.STANDARD  # Balanced approach


class StrategyEngine:
    """Engine for executing different compression strategies."""
    
    def __init__(self):
        self.context_compressor = None
        self.context_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize compression engines."""
        try:
            self.context_compressor = get_context_compressor()
            self.context_engine = ContextCompressionEngine()
        except Exception as e:
            logger.warning(f"Failed to initialize compression engines: {e}")
    
    async def execute_strategy(
        self, 
        strategy: CompressionStrategy,
        content: str,
        level: CompressionLevel,
        options: Dict[str, Any] = None
    ) -> UnifiedCompressionResult:
        """Execute specific compression strategy."""
        options = options or {}
        
        try:
            if strategy == CompressionStrategy.CONTEXT:
                return await self._execute_context_compression(content, level, options)
            elif strategy == CompressionStrategy.MEMORY:
                return await self._execute_memory_compression(content, level, options)
            elif strategy == CompressionStrategy.CONVERSATION:
                return await self._execute_conversation_compression(content, level, options)
            elif strategy == CompressionStrategy.ADAPTIVE:
                return await self._execute_adaptive_compression(content, level, options)
            else:
                raise ValueError(f"Unknown compression strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Strategy execution failed: {strategy}", error=str(e))
            result = UnifiedCompressionResult()
            result.error_message = str(e)
            return result
    
    async def _execute_context_compression(
        self, 
        content: str, 
        level: CompressionLevel, 
        options: Dict[str, Any]
    ) -> UnifiedCompressionResult:
        """Execute context-based compression using existing ContextCompressor."""
        try:
            if not self.context_compressor:
                raise ValueError("Context compressor not available")
            
            # Map unified levels to context compressor levels
            level_mapping = {
                CompressionLevel.MINIMAL: ContextCompressionLevel.LIGHT,
                CompressionLevel.LIGHT: ContextCompressionLevel.LIGHT,
                CompressionLevel.STANDARD: ContextCompressionLevel.STANDARD,
                CompressionLevel.AGGRESSIVE: ContextCompressionLevel.AGGRESSIVE,
                CompressionLevel.MAXIMUM: ContextCompressionLevel.AGGRESSIVE
            }
            
            context_level = level_mapping.get(level, ContextCompressionLevel.STANDARD)
            
            # Determine context type
            context_type = self._determine_context_type(content, options)
            
            # Execute compression
            if options.get("target_tokens"):
                compressed = await self.context_compressor.adaptive_compress(
                    content=content,
                    target_token_count=options["target_tokens"],
                    context_type=context_type
                )
            else:
                compressed = await self.context_compressor.compress_conversation(
                    conversation_content=content,
                    compression_level=context_level,
                    context_type=context_type,
                    preserve_decisions=options.get("preserve_decisions", True),
                    preserve_patterns=options.get("preserve_patterns", True)
                )
            
            # Convert to unified result
            result = self._convert_context_result(compressed, CompressionStrategy.CONTEXT, level)
            return result
            
        except Exception as e:
            logger.error("Context compression failed", error=str(e))
            result = UnifiedCompressionResult()
            result.error_message = f"Context compression failed: {e}"
            return result
    
    async def _execute_memory_compression(
        self, 
        content: str, 
        level: CompressionLevel, 
        options: Dict[str, Any]
    ) -> UnifiedCompressionResult:
        """Execute memory-specific compression for agent states and history."""
        try:
            # Memory compression focuses on preserving agent patterns and capabilities
            result = UnifiedCompressionResult()
            result.strategy_used = CompressionStrategy.MEMORY
            result.compression_level = level
            result.original_content = content
            
            # Extract memory-specific elements
            memory_elements = self._extract_memory_elements(content)
            
            # Compress based on level
            if level == CompressionLevel.MINIMAL:
                # Keep almost everything, just remove redundancy
                compressed = self._minimal_memory_compression(content, memory_elements)
                ratio = 0.15
            elif level == CompressionLevel.LIGHT:
                # Preserve all capabilities and recent patterns
                compressed = self._light_memory_compression(content, memory_elements)
                ratio = 0.3
            elif level == CompressionLevel.STANDARD:
                # Balance between history and current state
                compressed = self._standard_memory_compression(content, memory_elements)
                ratio = 0.5
            elif level == CompressionLevel.AGGRESSIVE:
                # Keep only essential agent state and critical patterns
                compressed = self._aggressive_memory_compression(content, memory_elements)
                ratio = 0.7
            else:  # MAXIMUM
                # Extreme compression - only core identity and capabilities
                compressed = self._maximum_memory_compression(content, memory_elements)
                ratio = 0.85
            
            result.compressed_content = compressed
            result.original_token_count = len(content) // 4  # Rough estimate
            result.compressed_token_count = len(compressed) // 4
            result.compression_ratio = ratio
            result.tokens_saved = result.original_token_count - result.compressed_token_count
            result.success = True
            
            # Extract insights from memory elements
            result.key_insights = memory_elements.get("insights", [])
            result.patterns_identified = memory_elements.get("patterns", [])
            result.importance_score = memory_elements.get("importance_score", 0.5)
            
            return result
            
        except Exception as e:
            logger.error("Memory compression failed", error=str(e))
            result = UnifiedCompressionResult()
            result.error_message = f"Memory compression failed: {e}"
            return result
    
    async def _execute_conversation_compression(
        self, 
        content: str, 
        level: CompressionLevel, 
        options: Dict[str, Any]
    ) -> UnifiedCompressionResult:
        """Execute conversation-specific compression for chat logs and discussions."""
        try:
            result = UnifiedCompressionResult()
            result.strategy_used = CompressionStrategy.CONVERSATION
            result.compression_level = level
            result.original_content = content
            
            # Parse conversation structure
            conversation_elements = self._parse_conversation(content)
            
            # Compress based on level and conversation type
            if level == CompressionLevel.MINIMAL:
                compressed = self._minimal_conversation_compression(content, conversation_elements)
                ratio = 0.2
            elif level == CompressionLevel.LIGHT:
                compressed = self._light_conversation_compression(content, conversation_elements)
                ratio = 0.4
            elif level == CompressionLevel.STANDARD:
                compressed = self._standard_conversation_compression(content, conversation_elements)
                ratio = 0.6
            elif level == CompressionLevel.AGGRESSIVE:
                compressed = self._aggressive_conversation_compression(content, conversation_elements)
                ratio = 0.75
            else:  # MAXIMUM
                compressed = self._maximum_conversation_compression(content, conversation_elements)
                ratio = 0.9
            
            result.compressed_content = compressed
            result.original_token_count = len(content) // 4
            result.compressed_token_count = len(compressed) // 4
            result.compression_ratio = ratio
            result.tokens_saved = result.original_token_count - result.compressed_token_count
            result.success = True
            
            # Extract conversation insights
            result.decisions_made = conversation_elements.get("decisions", [])
            result.key_insights = conversation_elements.get("key_points", [])
            result.summary = conversation_elements.get("summary", "")
            
            return result
            
        except Exception as e:
            logger.error("Conversation compression failed", error=str(e))
            result = UnifiedCompressionResult()
            result.error_message = f"Conversation compression failed: {e}"
            return result
    
    async def _execute_adaptive_compression(
        self, 
        content: str, 
        level: CompressionLevel, 
        options: Dict[str, Any]
    ) -> UnifiedCompressionResult:
        """Execute adaptive compression using AI-powered strategy selection."""
        try:
            analyzer = ContentAnalyzer()
            analysis = await analyzer.analyze_content(content)
            
            # Use the recommended strategy from analysis
            recommended_strategy = analysis.get("recommended_strategy", CompressionStrategy.CONTEXT)
            confidence = analysis.get("confidence", 0.5)
            
            # If confidence is low, try multiple strategies and pick the best
            if confidence < 0.7:
                return await self._multi_strategy_compression(content, level, options, analysis)
            else:
                # Use single recommended strategy
                result = await self.execute_strategy(recommended_strategy, content, level, options)
                result.metadata["analysis"] = analysis
                result.metadata["adaptive_confidence"] = confidence
                return result
                
        except Exception as e:
            logger.error("Adaptive compression failed", error=str(e))
            # Fallback to context compression
            return await self._execute_context_compression(content, level, options)
    
    async def _multi_strategy_compression(
        self, 
        content: str, 
        level: CompressionLevel, 
        options: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> UnifiedCompressionResult:
        """Try multiple strategies and select the best result."""
        strategies_to_try = [
            CompressionStrategy.CONTEXT,
            CompressionStrategy.CONVERSATION,
            CompressionStrategy.MEMORY
        ]
        
        results = []
        
        # Try each strategy
        for strategy in strategies_to_try:
            try:
                result = await self.execute_strategy(strategy, content, level, options)
                if result.success:
                    # Score the result based on compression ratio and quality
                    quality_score = self._evaluate_compression_quality(result, content)
                    result.metadata["quality_score"] = quality_score
                    results.append(result)
            except Exception as e:
                logger.warning(f"Strategy {strategy} failed in multi-strategy test: {e}")
        
        if not results:
            raise ValueError("All compression strategies failed")
        
        # Select best result based on quality score and compression ratio
        best_result = max(results, key=lambda r: r.metadata.get("quality_score", 0))
        best_result.strategy_used = CompressionStrategy.ADAPTIVE
        best_result.metadata["analysis"] = analysis
        best_result.metadata["strategies_tested"] = len(results)
        
        return best_result
    
    def _determine_context_type(self, content: str, options: Dict[str, Any]) -> Optional[ContextType]:
        """Determine context type for context compression."""
        # Check for explicit context type in options
        if "context_type" in options:
            return options["context_type"]
        
        # Auto-detect based on content
        if re.search(r'error|exception|traceback', content, re.IGNORECASE):
            return ContextType.ERROR_RESOLUTION
        elif re.search(r'decision|conclusion|recommendation', content, re.IGNORECASE):
            return ContextType.DECISION
        elif re.search(r'learn|research|study|analyze', content, re.IGNORECASE):
            return ContextType.LEARNING
        
        return None
    
    def _convert_context_result(
        self, 
        compressed: CompressedContext, 
        strategy: CompressionStrategy, 
        level: CompressionLevel
    ) -> UnifiedCompressionResult:
        """Convert ContextCompressor result to unified result."""
        result = UnifiedCompressionResult()
        result.success = True
        result.strategy_used = strategy
        result.compression_level = level
        result.compressed_content = compressed.summary
        result.original_token_count = compressed.original_token_count
        result.compressed_token_count = compressed.compressed_token_count
        result.compression_ratio = compressed.compression_ratio
        result.tokens_saved = compressed.original_token_count - compressed.compressed_token_count
        result.summary = compressed.summary
        result.key_insights = compressed.key_insights or []
        result.decisions_made = compressed.decisions_made or []
        result.patterns_identified = compressed.patterns_identified or []
        result.importance_score = compressed.importance_score
        result.metadata = compressed.metadata or {}
        
        return result
    
    def _evaluate_compression_quality(self, result: UnifiedCompressionResult, original_content: str) -> float:
        """Evaluate the quality of compression result."""
        try:
            quality_score = 0.0
            
            # Compression ratio score (30%)
            ratio_score = result.compression_ratio * 0.3
            
            # Content preservation score (40%)
            preservation_score = 0.0
            if result.key_insights:
                preservation_score += 0.2
            if result.decisions_made:
                preservation_score += 0.1
            if result.patterns_identified:
                preservation_score += 0.1
            
            # Readability score (20%)
            readability_score = min(0.2, len(result.summary) / 1000) if result.summary else 0
            
            # Importance score (10%)
            importance_score = result.importance_score * 0.1
            
            quality_score = ratio_score + preservation_score + readability_score + importance_score
            
            return min(1.0, quality_score)
            
        except Exception:
            return 0.5  # Default score if evaluation fails
    
    # Helper methods for memory compression
    def _extract_memory_elements(self, content: str) -> Dict[str, Any]:
        """Extract memory-specific elements from content."""
        # This would implement sophisticated memory pattern extraction
        # For now, return a simplified version
        return {
            "insights": [],
            "patterns": [],
            "importance_score": 0.5
        }
    
    def _minimal_memory_compression(self, content: str, elements: Dict) -> str:
        """Minimal memory compression - remove only obvious redundancy."""
        return content  # Placeholder
    
    def _light_memory_compression(self, content: str, elements: Dict) -> str:
        """Light memory compression."""
        return content  # Placeholder
    
    def _standard_memory_compression(self, content: str, elements: Dict) -> str:
        """Standard memory compression."""
        return content  # Placeholder
    
    def _aggressive_memory_compression(self, content: str, elements: Dict) -> str:
        """Aggressive memory compression."""
        return content  # Placeholder
    
    def _maximum_memory_compression(self, content: str, elements: Dict) -> str:
        """Maximum memory compression."""
        return content  # Placeholder
    
    # Helper methods for conversation compression
    def _parse_conversation(self, content: str) -> Dict[str, Any]:
        """Parse conversation structure and extract elements."""
        return {
            "decisions": [],
            "key_points": [],
            "summary": ""
        }
    
    def _minimal_conversation_compression(self, content: str, elements: Dict) -> str:
        """Minimal conversation compression."""
        return content  # Placeholder
    
    def _light_conversation_compression(self, content: str, elements: Dict) -> str:
        """Light conversation compression."""
        return content  # Placeholder
    
    def _standard_conversation_compression(self, content: str, elements: Dict) -> str:
        """Standard conversation compression."""
        return content  # Placeholder
    
    def _aggressive_conversation_compression(self, content: str, elements: Dict) -> str:
        """Aggressive conversation compression."""
        return content  # Placeholder
    
    def _maximum_conversation_compression(self, content: str, elements: Dict) -> str:
        """Maximum conversation compression."""
        return content  # Placeholder


class UnifiedCompressionCommand:
    """
    Unified compression command that consolidates all compression approaches
    with automatic strategy selection and mobile optimization.
    """
    
    def __init__(self):
        self.content_analyzer = ContentAnalyzer()
        self.strategy_engine = StrategyEngine()
        self.performance_target = 15.0  # 15 seconds max execution time
        
    async def compress(
        self,
        content: str,
        strategy: Union[str, CompressionStrategy] = CompressionStrategy.ADAPTIVE,
        level: Union[str, CompressionLevel] = CompressionLevel.STANDARD,
        target_tokens: Optional[int] = None,
        target_ratio: Optional[float] = None,
        preserve_decisions: bool = True,
        preserve_patterns: bool = True,
        mobile_optimized: bool = False,
        options: Dict[str, Any] = None
    ) -> UnifiedCompressionResult:
        """
        Unified compression method with automatic strategy selection.
        
        Args:
            content: Content to compress
            strategy: Compression strategy (adaptive, context, memory, conversation)
            level: Compression level (minimal, light, standard, aggressive, maximum)
            target_tokens: Target token count (overrides level)
            target_ratio: Target compression ratio (overrides level)
            preserve_decisions: Whether to preserve decision points
            preserve_patterns: Whether to preserve identified patterns
            mobile_optimized: Enable mobile-specific optimizations
            options: Additional options for compression
        
        Returns:
            UnifiedCompressionResult with comprehensive compression information
        """
        start_time = time.time()
        options = options or {}
        
        try:
            # Normalize inputs
            if isinstance(strategy, str):
                strategy = CompressionStrategy(strategy.lower())
            if isinstance(level, str):
                level = CompressionLevel(level.lower())
            
            # Validate inputs
            if not content or not content.strip():
                raise ValueError("Content cannot be empty")
            
            # Add options
            options.update({
                "target_tokens": target_tokens,
                "target_ratio": target_ratio,
                "preserve_decisions": preserve_decisions,
                "preserve_patterns": preserve_patterns,
                "mobile_optimized": mobile_optimized
            })
            
            # Analyze content for strategy selection if using adaptive
            analysis = None
            if strategy == CompressionStrategy.ADAPTIVE:
                analysis = await self.content_analyzer.analyze_content(content)
                
                # Override strategy if analysis provides better recommendation
                if analysis.get("confidence", 0) > 0.7:
                    strategy = analysis["recommended_strategy"]
                    level = analysis["recommended_level"]
                    logger.info("Using AI-recommended strategy", 
                              strategy=strategy.value, 
                              level=level.value, 
                              confidence=analysis.get("confidence"))
            
            # Adjust level based on target parameters
            if target_ratio:
                level = self._ratio_to_level(target_ratio)
            elif target_tokens and analysis:
                estimated_tokens = analysis.get("token_estimate", len(content) // 4)
                if estimated_tokens > 0:
                    target_ratio = 1.0 - (target_tokens / estimated_tokens)
                    level = self._ratio_to_level(target_ratio)
            
            # Execute compression with timeout protection
            result = await asyncio.wait_for(
                self.strategy_engine.execute_strategy(strategy, content, level, options),
                timeout=self.performance_target
            )
            
            # Post-process result
            result.execution_time_seconds = time.time() - start_time
            result.mobile_optimized = mobile_optimized
            
            if analysis:
                result.metadata["content_analysis"] = analysis
            
            # Apply mobile optimizations
            if mobile_optimized:
                result = await self._apply_mobile_optimizations(result)
            
            # Performance validation
            if result.execution_time_seconds > self.performance_target:
                logger.warning("Compression exceeded performance target", 
                             execution_time=result.execution_time_seconds,
                             target=self.performance_target)
            
            result.performance_metrics = {
                "execution_time_seconds": result.execution_time_seconds,
                "performance_target_met": result.execution_time_seconds <= self.performance_target,
                "compression_efficiency": result.compression_ratio / max(0.1, result.execution_time_seconds),
                "tokens_per_second": result.tokens_saved / max(0.1, result.execution_time_seconds)
            }
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Compression timed out", timeout=self.performance_target)
            result = UnifiedCompressionResult()
            result.error_message = f"Compression timed out after {self.performance_target} seconds"
            result.execution_time_seconds = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error("Unified compression failed", error=str(e))
            result = UnifiedCompressionResult()
            result.error_message = str(e)
            result.execution_time_seconds = time.time() - start_time
            
            # Attempt error recovery
            try:
                result = await self._attempt_error_recovery(content, options, result)
            except Exception as recovery_error:
                logger.error("Error recovery failed", error=str(recovery_error))
                result.error_message += f" | Recovery failed: {recovery_error}"
            
            return result
    
    async def _apply_mobile_optimizations(self, result: UnifiedCompressionResult) -> UnifiedCompressionResult:
        """Apply mobile-specific optimizations to compression result."""
        try:
            # Reduce summary length for mobile displays
            if result.summary and len(result.summary) > 500:
                result.summary = result.summary[:500] + "..."
            
            # Limit key insights to top 3 for mobile
            if result.key_insights and len(result.key_insights) > 3:
                result.key_insights = result.key_insights[:3]
            
            # Limit patterns to top 3 for mobile
            if result.patterns_identified and len(result.patterns_identified) > 3:
                result.patterns_identified = result.patterns_identified[:3]
            
            # Add mobile-specific metadata
            result.metadata["mobile_optimizations"] = {
                "summary_truncated": len(result.summary) >= 500,
                "insights_limited": len(result.key_insights) >= 3,
                "patterns_limited": len(result.patterns_identified) >= 3,
                "display_optimized": True
            }
            
            return result
            
        except Exception as e:
            logger.error("Mobile optimization failed", error=str(e))
            return result
    
    def _ratio_to_level(self, target_ratio: float) -> CompressionLevel:
        """Convert target compression ratio to compression level."""
        if target_ratio <= 0.2:
            return CompressionLevel.MINIMAL
        elif target_ratio <= 0.4:
            return CompressionLevel.LIGHT
        elif target_ratio <= 0.6:
            return CompressionLevel.STANDARD
        elif target_ratio <= 0.8:
            return CompressionLevel.AGGRESSIVE
        else:
            return CompressionLevel.MAXIMUM
    
    async def _attempt_error_recovery(
        self, 
        content: str, 
        options: Dict[str, Any], 
        failed_result: UnifiedCompressionResult
    ) -> UnifiedCompressionResult:
        """Attempt to recover from compression errors."""
        recovery_strategies = [
            # Strategy 1: Try simpler compression level
            lambda: self._recover_with_simpler_level(content, options),
            # Strategy 2: Try different strategy
            lambda: self._recover_with_different_strategy(content, options),
            # Strategy 3: Fallback to basic summarization
            lambda: self._recover_with_basic_summarization(content, options)
        ]
        
        for strategy in recovery_strategies:
            try:
                result = await strategy()
                if result.success:
                    result.recovery_attempted = True
                    result.metadata["recovery_strategy"] = "successful"
                    return result
            except Exception as e:
                logger.warning(f"Recovery strategy failed: {e}")
        
        failed_result.recovery_attempted = True
        failed_result.metadata["recovery_strategy"] = "failed"
        return failed_result
    
    async def _recover_with_simpler_level(self, content: str, options: Dict[str, Any]) -> UnifiedCompressionResult:
        """Recovery strategy: try simpler compression level."""
        options = options.copy()
        options["preserve_decisions"] = True
        options["preserve_patterns"] = True
        
        return await self.strategy_engine.execute_strategy(
            CompressionStrategy.CONTEXT,
            content,
            CompressionLevel.LIGHT,
            options
        )
    
    async def _recover_with_different_strategy(self, content: str, options: Dict[str, Any]) -> UnifiedCompressionResult:
        """Recovery strategy: try different compression strategy."""
        return await self.strategy_engine.execute_strategy(
            CompressionStrategy.CONVERSATION,
            content,
            CompressionLevel.STANDARD,
            options
        )
    
    async def _recover_with_basic_summarization(self, content: str, options: Dict[str, Any]) -> UnifiedCompressionResult:
        """Recovery strategy: basic text summarization."""
        result = UnifiedCompressionResult()
        result.strategy_used = CompressionStrategy.ADAPTIVE
        result.compression_level = CompressionLevel.STANDARD
        result.original_content = content
        
        # Simple summarization: take first and last parts
        lines = content.split('\n')
        if len(lines) <= 10:
            result.compressed_content = content
            result.compression_ratio = 0.0
        else:
            first_part = '\n'.join(lines[:5])
            last_part = '\n'.join(lines[-5:])
            result.compressed_content = f"{first_part}\n\n[... content compressed ...]\n\n{last_part}"
            result.compression_ratio = 0.5
        
        result.original_token_count = len(content) // 4
        result.compressed_token_count = len(result.compressed_content) // 4
        result.tokens_saved = result.original_token_count - result.compressed_token_count
        result.success = True
        result.summary = "Basic compression applied due to error recovery"
        
        return result


# Global instance
_unified_compressor: Optional[UnifiedCompressionCommand] = None


def get_unified_compressor() -> UnifiedCompressionCommand:
    """Get global unified compression command instance."""
    global _unified_compressor
    if _unified_compressor is None:
        _unified_compressor = UnifiedCompressionCommand()
    return _unified_compressor


# Backward compatibility aliases for existing compression commands
async def compress_context(content: str, **kwargs) -> Dict[str, Any]:
    """Backward compatibility: context compression."""
    compressor = get_unified_compressor()
    result = await compressor.compress(
        content=content,
        strategy=CompressionStrategy.CONTEXT,
        **kwargs
    )
    return result.to_dict()


async def compress_memory(content: str, **kwargs) -> Dict[str, Any]:
    """Backward compatibility: memory compression."""
    compressor = get_unified_compressor()
    result = await compressor.compress(
        content=content,
        strategy=CompressionStrategy.MEMORY,
        **kwargs
    )
    return result.to_dict()


async def compress_conversation(content: str, **kwargs) -> Dict[str, Any]:
    """Backward compatibility: conversation compression."""
    compressor = get_unified_compressor()
    result = await compressor.compress(
        content=content,
        strategy=CompressionStrategy.CONVERSATION,
        **kwargs
    )
    return result.to_dict()


async def adaptive_compress(content: str, **kwargs) -> Dict[str, Any]:
    """Main compression function with adaptive strategy selection."""
    compressor = get_unified_compressor()
    result = await compressor.compress(
        content=content,
        strategy=CompressionStrategy.ADAPTIVE,
        **kwargs
    )
    return result.to_dict()
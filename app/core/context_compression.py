"""
Context Compression Service for intelligent conversation summarization.

Provides LLM-based compression that preserves key insights, decisions, and patterns
while reducing token count by 60-80% for optimal context management.
"""

import time
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging

from anthropic import AsyncAnthropic
import tiktoken

from ..core.config import get_settings
from ..models.context import Context, ContextType


logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level options."""
    LIGHT = "light"          # 10-30% reduction
    STANDARD = "standard"    # 40-60% reduction  
    AGGRESSIVE = "aggressive" # 70-80% reduction


class CompressedContext:
    """Represents a compressed context with metadata."""
    
    def __init__(
        self,
        original_id: Optional[str] = None,
        summary: str = "",
        key_insights: List[str] = None,
        decisions_made: List[str] = None,
        patterns_identified: List[str] = None,
        importance_score: float = 0.5,
        compression_ratio: float = 0.0,
        original_token_count: int = 0,
        compressed_token_count: int = 0,
        metadata: Dict[str, Any] = None
    ):
        self.original_id = original_id
        self.summary = summary
        self.key_insights = key_insights or []
        self.decisions_made = decisions_made or []
        self.patterns_identified = patterns_identified or []
        self.importance_score = importance_score
        self.compression_ratio = compression_ratio
        self.original_token_count = original_token_count
        self.compressed_token_count = compressed_token_count
        self.metadata = metadata or {}
        self.compressed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_id": self.original_id,
            "summary": self.summary,
            "key_insights": self.key_insights,
            "decisions_made": self.decisions_made,
            "patterns_identified": self.patterns_identified,
            "importance_score": self.importance_score,
            "compression_ratio": self.compression_ratio,
            "original_token_count": self.original_token_count,
            "compressed_token_count": self.compressed_token_count,
            "metadata": self.metadata,
            "compressed_at": self.compressed_at.isoformat()
        }


class ContextCompressor:
    """
    Intelligent context compression service using Claude.
    
    Features:
    - Multi-level compression strategies
    - Preservation of critical information
    - Adaptive compression based on content type
    - Performance monitoring and optimization
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        llm_client: Optional[AsyncAnthropic] = None,
        model_name: str = "claude-3-haiku-20240307"
    ):
        """
        Initialize context compressor.
        
        Args:
            llm_client: Anthropic client (optional, will create if not provided)
            model_name: Claude model to use for compression
        """
        self.settings = get_settings()
        self.model_name = model_name
        
        # Initialize Anthropic client
        self.llm_client = llm_client or AsyncAnthropic(
            api_key=self.settings.ANTHROPIC_API_KEY
        )
        
        # Token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # Close approximation
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Performance metrics
        self._compression_count = 0
        self._total_compression_time = 0.0
        self._total_tokens_saved = 0
        self._compression_ratios = []
    
    async def compress_conversation(
        self,
        conversation_content: str,
        compression_level: CompressionLevel = CompressionLevel.STANDARD,
        context_type: Optional[ContextType] = None,
        preserve_decisions: bool = True,
        preserve_patterns: bool = True
    ) -> CompressedContext:
        """
        Compress a conversation while preserving critical information.
        
        Args:
            conversation_content: Original conversation text
            compression_level: Desired compression level
            context_type: Type of context for tailored compression
            preserve_decisions: Whether to extract and preserve decisions
            preserve_patterns: Whether to extract and preserve patterns
            
        Returns:
            CompressedContext with summary and metadata
        """
        start_time = time.time()
        
        try:
            # Count original tokens
            original_tokens = len(self.tokenizer.encode(conversation_content))
            
            if original_tokens < 100:
                # Don't compress very short content
                return CompressedContext(
                    summary=conversation_content,
                    compression_ratio=0.0,
                    original_token_count=original_tokens,
                    compressed_token_count=original_tokens
                )
            
            # Build compression prompt based on level and type
            prompt = self._build_compression_prompt(
                content=conversation_content,
                compression_level=compression_level,
                context_type=context_type,
                preserve_decisions=preserve_decisions,
                preserve_patterns=preserve_patterns
            )
            
            # Call Claude for compression
            logger.debug(f"Compressing {original_tokens} tokens at {compression_level.value} level")
            response = await self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=min(4000, int(original_tokens * 0.8)),  # Adaptive max tokens
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            compressed_data = self._parse_compression_response(response.content[0].text)
            
            # Count compressed tokens
            compressed_tokens = len(self.tokenizer.encode(compressed_data["summary"]))
            compression_ratio = 1 - (compressed_tokens / original_tokens)
            
            # Create compressed context
            compressed_context = CompressedContext(
                summary=compressed_data["summary"],
                key_insights=compressed_data.get("key_insights", []),
                decisions_made=compressed_data.get("decisions_made", []),
                patterns_identified=compressed_data.get("patterns_identified", []),
                importance_score=compressed_data.get("importance_score", 0.5),
                compression_ratio=compression_ratio,
                original_token_count=original_tokens,
                compressed_token_count=compressed_tokens,
                metadata={
                    "compression_level": compression_level.value,
                    "context_type": context_type.value if context_type else None,
                    "model_used": self.model_name
                }
            )
            
            # Update metrics
            compression_time = time.time() - start_time
            self._compression_count += 1
            self._total_compression_time += compression_time
            self._total_tokens_saved += (original_tokens - compressed_tokens)
            self._compression_ratios.append(compression_ratio)
            
            logger.info(
                f"Compressed {original_tokens} → {compressed_tokens} tokens "
                f"({compression_ratio:.1%} reduction) in {compression_time:.2f}s"
            )
            
            return compressed_context
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return original content as fallback
            return CompressedContext(
                summary=conversation_content,
                compression_ratio=0.0,
                original_token_count=len(self.tokenizer.encode(conversation_content)),
                compressed_token_count=len(self.tokenizer.encode(conversation_content)),
                metadata={"error": str(e)}
            )
    
    async def compress_context_batch(
        self,
        contexts: List[Context],
        compression_level: CompressionLevel = CompressionLevel.STANDARD
    ) -> List[CompressedContext]:
        """
        Compress multiple contexts in batch for efficiency.
        
        Args:
            contexts: List of contexts to compress
            compression_level: Compression level to apply
            
        Returns:
            List of compressed contexts
        """
        results = []
        
        # Process in smaller batches to avoid API limits
        batch_size = 5
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]
            
            # Process batch concurrently
            import asyncio
            tasks = [
                self.compress_conversation(
                    conversation_content=context.content,
                    compression_level=compression_level,
                    context_type=context.context_type
                )
                for context in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for context, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to compress context {context.id}: {result}")
                    # Create fallback result
                    result = CompressedContext(
                        original_id=str(context.id),
                        summary=context.content,
                        compression_ratio=0.0,
                        metadata={"error": str(result)}
                    )
                else:
                    result.original_id = str(context.id)
                
                results.append(result)
        
        return results
    
    async def adaptive_compress(
        self,
        content: str,
        target_token_count: int,
        context_type: Optional[ContextType] = None
    ) -> CompressedContext:
        """
        Adaptively compress content to target token count.
        
        Args:
            content: Content to compress
            target_token_count: Target number of tokens
            context_type: Type of context for tailored compression
            
        Returns:
            CompressedContext achieving target size
        """
        original_tokens = len(self.tokenizer.encode(content))
        
        if original_tokens <= target_token_count:
            return CompressedContext(
                summary=content,
                compression_ratio=0.0,
                original_token_count=original_tokens,
                compressed_token_count=original_tokens
            )
        
        # Calculate required compression ratio
        required_ratio = 1 - (target_token_count / original_tokens)
        
        # Choose compression level based on required ratio
        if required_ratio < 0.3:
            level = CompressionLevel.LIGHT
        elif required_ratio < 0.6:
            level = CompressionLevel.STANDARD
        else:
            level = CompressionLevel.AGGRESSIVE
        
        # Compress with chosen level
        result = await self.compress_conversation(
            conversation_content=content,
            compression_level=level,
            context_type=context_type
        )
        
        # If still too long, try more aggressive compression
        if result.compressed_token_count > target_token_count and level != CompressionLevel.AGGRESSIVE:
            logger.debug(f"First compression insufficient, trying aggressive compression")
            result = await self.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.AGGRESSIVE,
                context_type=context_type
            )
        
        return result
    
    def _build_compression_prompt(
        self,
        content: str,
        compression_level: CompressionLevel,
        context_type: Optional[ContextType],
        preserve_decisions: bool,
        preserve_patterns: bool
    ) -> str:
        """Build compression prompt tailored to requirements."""
        
        # Base compression instruction
        if compression_level == CompressionLevel.LIGHT:
            compression_instruction = "Lightly summarize this content, reducing length by 20-30% while preserving most details."
        elif compression_level == CompressionLevel.STANDARD:
            compression_instruction = "Compress this content by 50-60%, keeping only the most important information."
        else:  # AGGRESSIVE
            compression_instruction = "Aggressively compress this content by 70-80%, extracting only critical insights."
        
        # Context-specific instructions
        context_specific = ""
        if context_type == ContextType.DECISION:
            context_specific = "\nFocus on preserving the decision made, rationale, and outcomes."
        elif context_type == ContextType.ERROR_RESOLUTION:
            context_specific = "\nEmphasize the error encountered, root cause, and solution implemented."
        elif context_type == ContextType.LEARNING:
            context_specific = "\nHighlight key learnings, insights, and applicable patterns."
        elif context_type == ContextType.CODE_SNIPPET:
            context_specific = "\nPreserve the core functionality and any important comments or explanations."
        
        # Preservation instructions
        preserve_instructions = []
        if preserve_decisions:
            preserve_instructions.append("- Any decisions made and their rationale")
        if preserve_patterns:
            preserve_instructions.append("- Patterns, best practices, or reusable insights")
        
        preserve_text = ""
        if preserve_instructions:
            preserve_text = f"\n\nALWAYS preserve:\n" + "\n".join(preserve_instructions)
        
        # Format response instruction
        format_instruction = """

Format your response as JSON with these fields:
{
  "summary": "The compressed content",
  "key_insights": ["insight1", "insight2", ...],
  "decisions_made": ["decision1", "decision2", ...],
  "patterns_identified": ["pattern1", "pattern2", ...],
  "importance_score": 0.8
}

Importance score should be 0.0-1.0 based on the criticality of the information."""
        
        return f"""
{compression_instruction}{context_specific}{preserve_text}

Content to compress:
{content}

{format_instruction}
"""
    
    def _parse_compression_response(self, response: str) -> Dict[str, Any]:
        """Parse Claude's compression response."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: treat entire response as summary
                return {
                    "summary": response.strip(),
                    "key_insights": [],
                    "decisions_made": [],
                    "patterns_identified": [],
                    "importance_score": 0.5
                }
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "summary": response.strip(),
                "key_insights": [],
                "decisions_made": [],
                "patterns_identified": [],
                "importance_score": 0.5
            }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def estimate_compression_time(self, token_count: int) -> float:
        """Estimate compression time based on token count."""
        # Based on observed performance metrics
        base_time = 2.0  # Base time in seconds
        token_factor = token_count / 1000  # Additional time per 1k tokens
        return base_time + (token_factor * 0.5)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get compression performance metrics."""
        avg_compression_time = self._total_compression_time / max(1, self._compression_count)
        avg_compression_ratio = sum(self._compression_ratios) / max(1, len(self._compression_ratios))
        avg_tokens_saved = self._total_tokens_saved / max(1, self._compression_count)
        
        return {
            "total_compressions": self._compression_count,
            "average_compression_time_s": avg_compression_time,
            "average_compression_ratio": avg_compression_ratio,
            "total_tokens_saved": self._total_tokens_saved,
            "average_tokens_saved_per_compression": avg_tokens_saved,
            "model_used": self.model_name
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on compression service."""
        try:
            # Test with simple compression
            test_content = "This is a test message for health checking the compression service."
            result = await self.compress_conversation(
                test_content,
                CompressionLevel.LIGHT
            )
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "test_compression_ratio": result.compression_ratio,
                "performance": self.get_performance_metrics()
            }
        except Exception as e:
            logger.error(f"Compression health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name
            }


# Singleton instance for application use
_compressor: Optional[ContextCompressor] = None


def get_context_compressor() -> ContextCompressor:
    """
    Get singleton context compressor instance.
    
    Returns:
        ContextCompressor instance
    """
    global _compressor
    
    if _compressor is None:
        _compressor = ContextCompressor()
    
    return _compressor


async def cleanup_compressor() -> None:
    """Cleanup compressor resources."""
    global _compressor
    _compressor = None
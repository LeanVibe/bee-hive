#!/usr/bin/env python3
"""
Auto-Compact Check Hook for Claude Code

Monitors conversation length and complexity to suggest intelligent 
context compression opportunities.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import re


def count_conversation_metrics(transcript_path: str) -> Dict[str, Any]:
    """Analyze conversation transcript for compression metrics."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        message_count = len(lines)
        total_chars = sum(len(line) for line in lines)
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = total_chars // 4
        
        # Count tool usage patterns
        tool_usage_count = sum(1 for line in lines if '"tool_name"' in line)
        
        # Identify decision points (heuristic: lines with "decision", "choose", "implement")
        decision_indicators = ['decision', 'choose', 'implement', 'solution', 'approach']
        decision_count = 0
        for line in lines:
            if any(indicator in line.lower() for indicator in decision_indicators):
                decision_count += 1
        
        return {
            'message_count': message_count,
            'estimated_tokens': estimated_tokens,
            'tool_usage_count': tool_usage_count,
            'decision_count': decision_count,
            'total_chars': total_chars
        }
    
    except Exception as e:
        return {
            'message_count': 0,
            'estimated_tokens': 0,
            'tool_usage_count': 0,
            'decision_count': 0,
            'total_chars': 0,
            'error': str(e)
        }


def should_suggest_compression(metrics: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Determine if compression should be suggested based on conversation metrics.
    
    Returns:
        (should_suggest, compression_level, reason)
    """
    message_count = metrics['message_count']
    estimated_tokens = metrics['estimated_tokens']
    tool_usage = metrics['tool_usage_count']
    decisions = metrics['decision_count']
    
    # Compression thresholds
    if estimated_tokens > 80000:  # Approaching context limits
        return True, "aggressive", f"High token count ({estimated_tokens:,} estimated tokens)"
    
    elif estimated_tokens > 50000 and decisions > 10:
        return True, "standard", f"Substantial content with {decisions} decisions"
    
    elif message_count > 100 and tool_usage > 20:
        return True, "standard", f"Extended session with {tool_usage} tool operations"
    
    elif estimated_tokens > 30000 and message_count > 60:
        return True, "light", f"Growing conversation ({message_count} messages)"
    
    return False, "none", "Conversation size within optimal range"


def get_compression_recommendations(metrics: Dict[str, Any], level: str) -> List[str]:
    """Generate specific compression recommendations based on conversation patterns."""
    recommendations = []
    
    if metrics['tool_usage_count'] > 15:
        recommendations.append("Focus on preserving successful tool usage patterns")
    
    if metrics['decision_count'] > 8:
        recommendations.append("Emphasize architectural and implementation decisions")
    
    if metrics['message_count'] > 80:
        recommendations.append("Consider using 'standard' level for balanced compression")
    
    if level == "aggressive":
        recommendations.append("Use 'code' focus to maintain implementation context")
    
    return recommendations


def format_suggestion_context(should_suggest: bool, level: str, reason: str, 
                            recommendations: List[str], metrics: Dict[str, Any]) -> str:
    """Format the compression suggestion for Claude Code context."""
    
    if not should_suggest:
        return ""
    
    context = f"""

🔄 **Context Optimization Opportunity**

**Reason**: {reason}
**Recommended Level**: `/smart-compact {level}`
**Current Stats**: {metrics['message_count']} messages, ~{metrics['estimated_tokens']:,} tokens

**Optimization Tips**:"""
    
    for rec in recommendations:
        context += f"\n• {rec}"
    
    context += f"""

**Quick Actions**:
• `/smart-compact {level}` - Apply recommended compression
• `/smart-compact adaptive` - Let Claude choose optimal level
• `/universal-compact claude-code {level}` - Claude Code optimized compression

*This suggestion appears automatically when conversations grow large.*
"""
    
    return context


def main():
    """Main hook execution."""
    try:
        # Load hook input
        input_data = json.load(sys.stdin)
        
        # Only process UserPromptSubmit events
        if input_data.get("hook_event_name") != "UserPromptSubmit":
            sys.exit(0)
        
        transcript_path = input_data.get("transcript_path", "")
        if not transcript_path or not os.path.exists(transcript_path):
            sys.exit(0)
        
        # Analyze conversation metrics
        metrics = count_conversation_metrics(transcript_path)
        
        # Check if compression should be suggested
        should_suggest, level, reason = should_suggest_compression(metrics)
        
        if should_suggest:
            # Generate recommendations
            recommendations = get_compression_recommendations(metrics, level)
            
            # Format suggestion context
            suggestion = format_suggestion_context(
                should_suggest, level, reason, recommendations, metrics
            )
            
            # Output suggestion as additional context
            if suggestion:
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": suggestion
                    }
                }
                print(json.dumps(output))
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Hook execution error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Always exit successfully
    sys.exit(0)


if __name__ == "__main__":
    main()
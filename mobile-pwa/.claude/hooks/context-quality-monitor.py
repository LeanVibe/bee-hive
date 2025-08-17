#!/usr/bin/env python3
"""
Context Quality Monitor Hook for Claude Code

Monitors conversation quality and provides intelligent compression 
recommendations based on context efficiency and information density.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from datetime import datetime


class ContextQualityAnalyzer:
    """Analyzes conversation context for compression opportunities."""
    
    def __init__(self):
        self.quality_thresholds = {
            'high_redundancy': 0.7,      # 70% similar content suggests compression
            'low_information_density': 0.3,  # 30% or less new info per message
            'high_tool_failure_rate': 0.4,   # 40% tool failures suggest noise
            'decision_fatigue': 15,          # 15+ decisions without implementation
        }
    
    def analyze_transcript(self, transcript_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of conversation transcript."""
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            analysis = {
                'message_count': len(lines),
                'redundancy_score': self._calculate_redundancy(lines),
                'information_density': self._calculate_information_density(lines),
                'tool_success_rate': self._calculate_tool_success_rate(lines),
                'decision_implementation_ratio': self._calculate_decision_ratio(lines),
                'context_fragmentation': self._calculate_fragmentation(lines),
                'compression_opportunity_score': 0.0,
                'recommendations': []
            }
            
            # Calculate overall compression opportunity score
            analysis['compression_opportunity_score'] = self._calculate_opportunity_score(analysis)
            
            # Generate specific recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'compression_opportunity_score': 0.0}
    
    def _calculate_redundancy(self, lines: List[str]) -> float:
        """Calculate content redundancy in conversation."""
        if len(lines) < 2:
            return 0.0
        
        # Simple redundancy heuristic: repeated phrases and concepts
        content_lines = [line for line in lines if len(line.strip()) > 20]
        if len(content_lines) < 2:
            return 0.0
        
        # Count repeated patterns (simplified)
        repeated_patterns = 0
        total_comparisons = 0
        
        for i, line1 in enumerate(content_lines):
            for j, line2 in enumerate(content_lines[i+1:], i+1):
                total_comparisons += 1
                # Simple similarity check
                words1 = set(line1.lower().split())
                words2 = set(line2.lower().split())
                if len(words1) > 5 and len(words2) > 5:
                    overlap = len(words1.intersection(words2))
                    similarity = overlap / max(len(words1), len(words2))
                    if similarity > 0.5:  # 50% word overlap
                        repeated_patterns += 1
        
        return repeated_patterns / max(total_comparisons, 1) if total_comparisons > 0 else 0.0
    
    def _calculate_information_density(self, lines: List[str]) -> float:
        """Calculate information density per message."""
        if len(lines) == 0:
            return 0.0
        
        # Heuristic: messages with code, decisions, or substantial content
        high_info_messages = 0
        
        for line in lines:
            content = line.lower()
            # High information indicators
            if any(indicator in content for indicator in [
                'function', 'class', 'def ', 'const ', 'let ', 'var ',
                'decision', 'implement', 'solution', 'approach',
                'bug', 'error', 'fix', 'problem',
                '```', 'file:', 'path:'
            ]):
                high_info_messages += 1
        
        return high_info_messages / len(lines)
    
    def _calculate_tool_success_rate(self, lines: List[str]) -> float:
        """Calculate tool usage success rate."""
        tool_calls = 0
        tool_errors = 0
        
        for line in lines:
            if '"tool_name"' in line:
                tool_calls += 1
            if 'error' in line.lower() and 'tool' in line.lower():
                tool_errors += 1
        
        if tool_calls == 0:
            return 1.0  # No tools used, perfect success rate
        
        return 1.0 - (tool_errors / tool_calls)
    
    def _calculate_decision_ratio(self, lines: List[str]) -> float:
        """Calculate ratio of decisions to implementations."""
        decisions = 0
        implementations = 0
        
        for line in lines:
            content = line.lower()
            if any(word in content for word in ['decide', 'choice', 'option', 'consider']):
                decisions += 1
            if any(word in content for word in ['implement', 'create', 'build', 'write']):
                implementations += 1
        
        if decisions == 0:
            return 1.0  # No decisions, perfect ratio
        
        return implementations / decisions
    
    def _calculate_fragmentation(self, lines: List[str]) -> float:
        """Calculate context fragmentation (topic jumping)."""
        if len(lines) < 10:
            return 0.0
        
        # Simple topic change detection
        topic_changes = 0
        prev_topics = set()
        
        for line in lines:
            # Extract key terms as topics
            words = line.lower().split()
            current_topics = set(word for word in words if len(word) > 4)
            
            if prev_topics and current_topics:
                overlap = len(prev_topics.intersection(current_topics))
                if overlap / max(len(prev_topics), len(current_topics)) < 0.2:
                    topic_changes += 1
            
            prev_topics = current_topics
        
        return topic_changes / max(len(lines) - 1, 1)
    
    def _calculate_opportunity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall compression opportunity score (0-1)."""
        score = 0.0
        
        # High redundancy increases opportunity
        score += analysis.get('redundancy_score', 0) * 0.3
        
        # Low information density increases opportunity
        density = analysis.get('information_density', 1)
        if density < self.quality_thresholds['low_information_density']:
            score += (1 - density) * 0.25
        
        # High fragmentation increases opportunity
        score += analysis.get('context_fragmentation', 0) * 0.2
        
        # Low tool success rate increases opportunity
        tool_success = analysis.get('tool_success_rate', 1)
        if tool_success < (1 - self.quality_thresholds['high_tool_failure_rate']):
            score += (1 - tool_success) * 0.15
        
        # Many messages increase opportunity
        message_count = analysis.get('message_count', 0)
        if message_count > 50:
            score += min((message_count - 50) / 100, 0.1)
        
        return min(score, 1.0)
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate specific compression recommendations."""
        recommendations = []
        
        opportunity_score = analysis.get('compression_opportunity_score', 0)
        
        if opportunity_score > 0.7:
            recommendations.append("High compression opportunity - consider aggressive level")
        elif opportunity_score > 0.4:
            recommendations.append("Moderate compression opportunity - standard level recommended")
        elif opportunity_score > 0.2:
            recommendations.append("Light compression could improve context efficiency")
        
        # Specific recommendations based on analysis
        if analysis.get('redundancy_score', 0) > self.quality_thresholds['high_redundancy']:
            recommendations.append("High content redundancy detected - focus on consolidation")
        
        if analysis.get('information_density', 1) < self.quality_thresholds['low_information_density']:
            recommendations.append("Low information density - preserve only key decisions")
        
        if analysis.get('context_fragmentation', 0) > 0.5:
            recommendations.append("Context fragmentation detected - reorganize by topic")
        
        tool_success = analysis.get('tool_success_rate', 1)
        if tool_success < (1 - self.quality_thresholds['high_tool_failure_rate']):
            recommendations.append("Tool failures present - clean up error context")
        
        return recommendations


def main():
    """Main hook execution for context quality monitoring."""
    try:
        # Load hook input
        input_data = json.load(sys.stdin)
        
        # Monitor PostToolUse events for quality assessment
        hook_event = input_data.get("hook_event_name", "")
        if hook_event not in ["PostToolUse", "Stop"]:
            sys.exit(0)
        
        transcript_path = input_data.get("transcript_path", "")
        if not transcript_path or not os.path.exists(transcript_path):
            sys.exit(0)
        
        # Analyze context quality
        analyzer = ContextQualityAnalyzer()
        analysis = analyzer.analyze_transcript(transcript_path)
        
        # Check if quality monitoring should provide feedback
        opportunity_score = analysis.get('compression_opportunity_score', 0)
        
        if opportunity_score > 0.4:  # Significant compression opportunity
            recommendations = analysis.get('recommendations', [])
            
            feedback = f"""

📊 **Context Quality Assessment**

**Compression Opportunity**: {opportunity_score:.1%}
**Current Stats**: {analysis.get('message_count', 0)} messages, {analysis.get('information_density', 0):.1%} info density

**Quality Insights**:"""
            
            for rec in recommendations[:3]:  # Limit to top 3 recommendations
                feedback += f"\n• {rec}"
            
            if opportunity_score > 0.7:
                feedback += f"""

**Recommended Action**: `/smart-compact aggressive` for maximum efficiency
**Alternative**: `/universal-compact claude-code standard` for balanced compression"""
            
            elif opportunity_score > 0.4:
                feedback += f"""

**Recommended Action**: `/smart-compact standard` for balanced optimization"""
            
            feedback += "\n\n*Quality monitoring active - optimize when beneficial.*"
            
            # Output as blocking feedback to Claude
            output = {
                "decision": "block",
                "reason": feedback
            }
            print(json.dumps(output))
    
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Quality monitor error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Allow normal flow to continue
    sys.exit(0)


if __name__ == "__main__":
    main()
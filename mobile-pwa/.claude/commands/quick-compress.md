---
description: Quick context compression with smart defaults
argument-hint: [optional-focus]
---

# Quick Context Compression

Perform immediate context compression using intelligent defaults:

## Smart Analysis & Compression

**Automatic Level Selection**: Analyze conversation and choose optimal compression level
- Short conversations (< 30 messages): Light compression
- Medium conversations (30-80 messages): Standard compression  
- Long conversations (80+ messages): Aggressive compression
- Very long conversations (120+ messages): Maximum compression with focus preservation

## Focus Area: $ARGUMENTS

**Available Focus Options:**
- `code` - Preserve code patterns, implementations, and technical solutions
- `decisions` - Emphasize architectural and design decisions
- `debugging` - Maintain error analysis and troubleshooting context
- `planning` - Keep strategic planning and roadmap information
- (no argument) - Balanced preservation across all areas

## Compression Strategy

**Always Preserve:**
- Current active task and progress
- Recent successful solutions and patterns
- Key decisions with rationale
- Critical file locations and structures

**Smart Consolidation:**
- Group related discussions
- Eliminate redundant explanations
- Compress verbose tool outputs
- Summarize repeated patterns

**Context Optimization:**
- Target 60-70% token reduction
- Maintain conversation continuity
- Enable seamless task continuation
- Preserve essential development context

Execute compression and provide summary of what was preserved and what was optimized.
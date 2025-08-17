---
description: Intelligent context compression with preservation options
argument-hint: [level] [focus-area]
allowed-tools: Read, Glob, Grep
---

# Smart Context Compression

Perform intelligent context compression with the following approach:

## Compression Analysis

First, analyze the current conversation:
- **Message count**: How many exchanges have occurred
- **Context complexity**: Technical depth and branching topics
- **Decision points**: Key architectural or implementation decisions made
- **Active task status**: Current work in progress

## Compression Level: $ARGUMENTS

**Available levels:**
- `light` (30-50% reduction): Preserve most details, light summarization
- `standard` (50-70% reduction): Balanced compression with key insight preservation  
- `aggressive` (70-85% reduction): Maximum compression, core decisions only
- `adaptive` (auto-select): Choose optimal level based on context analysis

## Preservation Strategy

**Always preserve:**
1. **Current Task Context**: Active implementation, debugging, or planning
2. **Architectural Decisions**: Design choices and their rationale
3. **Implementation Patterns**: Successful approaches and techniques used
4. **Bug Fixes & Solutions**: Root causes identified and fixes applied
5. **File Structure Knowledge**: Important code locations and relationships
6. **Failed Attempts**: What didn't work and why (prevents repetition)

**Focus Areas** (if specified in arguments):
- `code`: Prioritize code structure, patterns, and implementation details
- `architecture`: Emphasize system design and component relationships
- `debugging`: Preserve error analysis and troubleshooting approaches
- `planning`: Maintain strategic decisions and roadmap information

## Compression Execution

Create a comprehensive summary that:

1. **Consolidates Related Topics**: Group similar discussions and decisions
2. **Preserves Decision Timeline**: Maintain chronology of key choices
3. **Retains Code Context**: Keep essential code references and patterns
4. **Maintains Task Continuity**: Enable seamless work continuation
5. **Optimizes Token Efficiency**: Achieve target compression ratio

**Output Format:**
- Clear section headers for different topic areas
- Bullet points for key decisions and insights
- Code snippets for important patterns or solutions
- Reference list of critical files or components

The result should enable seamless conversation continuation while achieving optimal token reduction for extended development sessions.
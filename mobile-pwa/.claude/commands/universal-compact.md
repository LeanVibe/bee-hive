---
description: Universal context compression for any coding agent
argument-hint: [agent-type] [compression-level]
allowed-tools: Read, Glob, Grep
---

# Universal Context Compression

Perform context compression optimized for **$ARGUMENTS** coding environment:

## Agent-Specific Optimizations

### Claude Code
- **Tool Usage Patterns**: Preserve Read, Edit, Write, Bash command sequences
- **File References**: Maintain @file.ext references and directory structures  
- **Slash Commands**: Keep custom command usage and workflows
- **Hook Integration**: Preserve automation and workflow context

### Cursor
- **Code Context**: Maintain cursor positions and selection ranges
- **AI Suggestions**: Preserve accepted/rejected suggestion patterns
- **File Navigation**: Keep recently edited file relationships
- **Inline Chat**: Maintain conversation context within code

### Gemini / Google AI
- **Code Analysis**: Preserve analytical insights and code quality assessments
- **Suggestion Patterns**: Maintain optimization and refactoring recommendations
- **Multi-file Context**: Keep cross-file relationship understanding
- **Performance Focus**: Preserve efficiency and optimization discussions

### GitHub Copilot / OpenAI Codex
- **Completion Context**: Maintain code completion patterns and preferences
- **Function Signatures**: Preserve API usage and parameter patterns
- **Code Style**: Keep formatting and convention decisions
- **Repository Context**: Maintain understanding of codebase structure

### Windsurf / Bolt
- **Project Context**: Preserve full-stack application understanding
- **Component Relationships**: Maintain frontend/backend integration patterns
- **Deployment Context**: Keep build and deployment configuration knowledge
- **Framework Patterns**: Preserve framework-specific implementations

## Universal Preservation Framework

**Core Context Elements** (preserved across all agents):
1. **Current Implementation State**: What's being built, current progress
2. **Technical Decisions**: Architecture choices, technology selections
3. **Problem-Solution Mapping**: Issues encountered and resolutions
4. **Code Quality Insights**: Performance, security, maintainability notes
5. **Development Workflow**: Successful patterns and team processes

**Cross-Agent Compatibility**:
- **Language-agnostic patterns**: Design principles that apply universally
- **Framework-neutral insights**: Architectural concepts that transfer
- **Tool-independent knowledge**: Core programming and system design wisdom
- **Collaboration context**: Team decisions and shared understanding

## Compression Strategy

### Level Selection
- **light** (30-50%): Preserve agent-specific nuances and detailed context
- **standard** (50-70%): Balance between compression and agent compatibility
- **aggressive** (70-85%): Focus on universal patterns and core decisions
- **adaptive**: Auto-select based on conversation complexity and agent type

### Context Handoff Preparation
Format compressed context for easy transition between coding agents:

```markdown
## Context Summary
- **Primary Task**: [Current development objective]
- **Architecture**: [Key system design decisions]
- **Progress**: [Completed work and next steps]

## Technical Context
- **Languages/Frameworks**: [Tech stack in use]
- **Key Files**: [Critical codebase locations]
- **Recent Changes**: [Important modifications made]

## Agent-Agnostic Insights
- **Patterns**: [Successful approaches identified]
- **Constraints**: [Limitations and requirements]
- **Decisions**: [Choices made with rationale]
```

## Multi-Agent Workflow Support

Enable seamless context transfer between different coding environments:

1. **Standardized Formats**: Use common markdown patterns all agents understand
2. **File Reference Consistency**: Maintain clear file paths and structure references
3. **Decision Documentation**: Record rationale in agent-neutral language
4. **Pattern Libraries**: Build reusable solution templates
5. **Context Checkpoints**: Create transfer-friendly summary points

The result enables productive coding sessions across different AI assistants while maintaining context continuity and avoiding repeated explanations.
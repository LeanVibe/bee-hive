---
description: LeanVibe Agent Hive custom command reference and workflow guidance
---

# LeanVibe Agent Hive - Custom Command Reference

Comprehensive guide to custom development commands for enhanced workflow efficiency:

## 🏗️ **System Management Commands**

### `/health`
**Purpose**: Comprehensive system health check and diagnostics
**Usage**: `/health`
**Features**: Database, Redis, Python environment, CLI validation, performance metrics

### `/startup`  
**Purpose**: Intelligent system startup and service initialization
**Usage**: `/startup`
**Features**: Service validation, environment setup, automatic recovery, readiness assessment

## 📋 **Session Management Commands**

### `/consolidate`
**Purpose**: Intelligent session consolidation with context preservation
**Usage**: `/consolidate`
**Features**: Session state capture, quality validation, memory optimization, auto-commit

### `/optimize`
**Purpose**: Optimize session context and memory usage for extended development
**Usage**: `/optimize` 
**Features**: Context analysis, memory consolidation, continuity preservation, efficiency optimization

## 🤖 **Agent Coordination Commands**

### `/deploy`
**Purpose**: Deploy specialized agent for complex task execution
**Usage**: `/deploy [agent-type] [task-description]`
**Agents**: backend-engineer, frontend-builder, qa-test-guardian, devops-deployer, project-orchestrator

## ✅ **Quality Assurance Commands**

### `/validate`
**Purpose**: Execute comprehensive quality gate validation before commits
**Usage**: `/validate`
**Features**: Code validation, test execution, system health, commit readiness assessment

## 🚀 **Development Flow Commands**

### `/commit`
**Purpose**: Intelligent commit with quality validation and enterprise standards
**Usage**: `/commit`
**Features**: Pre-commit validation, intelligent commit messages, quality gates, post-commit actions

### `/epic`
**Purpose**: Epic-level development coordination and task management  
**Usage**: `/epic [1|2|3|4] [task-focus]`
**Features**: Strategic alignment, agent deployment, progress tracking, milestone management

## 📖 **Command Categories Overview**

### **System Commands** (`system/`)
- Health monitoring and diagnostics
- Service initialization and recovery
- Environment validation and setup

### **Session Commands** (`session/`)  
- Context management and optimization
- Memory consolidation and preservation
- Session state transitions

### **Agent Commands** (`agent/`)
- Specialized agent deployment
- Task coordination and delegation
- Multi-agent workflow management

### **Quality Commands** (`quality/`)
- Pre-commit validation gates  
- Code quality assessment
- Enterprise standard compliance

### **Development Commands** (`dev/`)
- Intelligent commit workflows
- Epic-level coordination
- Strategic development alignment

## 🎯 **Workflow Best Practices**

### **Daily Development Flow**:
1. `/startup` - Initialize system and validate services
2. `/health` - Check system readiness before major work
3. `/epic [n]` - Focus on strategic epic development
4. `/deploy [agent]` - Use specialized agents for complex tasks
5. `/validate` - Quality check before commits
6. `/commit` - Intelligent commit with enterprise standards
7. `/consolidate` - End session with proper state preservation

### **Session Optimization**:
- Use `/optimize` when context grows large (>75% usage)
- Use `/consolidate` before sleep/wake cycles
- Use `/health` to diagnose any system issues

### **Quality Assurance**:  
- Always `/validate` before commits
- Use `/commit` for enterprise-standard commits
- Deploy `qa-test-guardian` agent for comprehensive testing

## ⚡ **Quick Reference**

**System Health**: `/health`
**Start Services**: `/startup` 
**Deploy Agent**: `/deploy backend-engineer [task]`
**Quality Check**: `/validate`
**Smart Commit**: `/commit`
**Epic Work**: `/epic 1 [focus]`
**Optimize Context**: `/optimize`
**End Session**: `/consolidate`

These commands enhance the LeanVibe Agent Hive development workflow with enterprise-grade automation, quality assurance, and intelligent coordination patterns.
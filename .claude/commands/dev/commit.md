---
description: Intelligent commit with quality validation and enterprise standards
allowed-tools: Bash(git*), TodoWrite(*), Task(qa-test-guardian)
---

# Intelligent Development Commit

Execute comprehensive commit workflow with quality validation:

## 📋 **Pre-Commit Quality Gate**

### System Validation
!`python3 -c "import sys; sys.path.insert(0, '.'); import app; print('✅ App module OK')" 2>/dev/null || echo "❌ Import issues - commit blocked"`

### Git Status Assessment
!`git status --porcelain`
!`git diff --cached --name-only 2>/dev/null || echo "No staged changes"`
!`git diff --name-only HEAD`

## 🔍 **Change Analysis**

Analyze current changes for:
1. **Impact Assessment**: Critical system modifications
2. **Test Requirements**: Need for additional test coverage  
3. **Documentation**: Updates required for changes
4. **Breaking Changes**: API or schema modifications

## ✅ **Quality Validation Process**

Before commit, ensure:
- **Code Compiles**: All Python files have valid syntax
- **Imports Work**: No broken import paths or dependencies  
- **Tests Pass**: Critical functionality validated (where available)
- **Services Stable**: Database and Redis connections operational

## 📝 **Intelligent Commit Message Generation**

Generate commit message following format:
```
<type>: <description>

<detailed explanation of changes>
<impact assessment>
<testing notes>

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## 🚀 **Commit Execution**

Execute commit workflow:
1. **Stage appropriate changes** (exclude temp files, logs)
2. **Generate descriptive commit message** with context
3. **Execute commit** with enterprise signature
4. **Validate commit success** and update session state

## 📊 **Post-Commit Actions**

After successful commit:
- Update session progress tracking
- Mark completed tasks in todo system
- Assess next development priorities
- Validate system remains stable

**Executing intelligent commit workflow...**
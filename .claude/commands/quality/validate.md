---
description: Execute comprehensive quality gate validation before commits
allowed-tools: Bash(*), Read(*), TodoWrite(*)
---

# Quality Gate Validation

Execute comprehensive pre-commit quality validation following enterprise standards:

## 🔍 **Code Quality Validation**

### Python Code Validation
!`python3 -m py_compile app/**/*.py 2>/dev/null && echo "✅ Python syntax valid" || echo "❌ Python syntax errors detected"`

### Import Path Validation  
!`python3 -c "import sys; sys.path.insert(0, '.'); import app; print('✅ App imports OK')" 2>/dev/null || echo "❌ Import issues detected"`

## 🧪 **Test Validation**
!`find . -name "*test*.py" -type f | wc -l | awk '{print "📊 Test files found: " $1}'`

### Test Execution (if available)
!`python3 -m pytest --tb=short -v 2>/dev/null | head -20 || echo "⚠️ Pytest not available or tests failed"`

## 🔧 **System Health Check**
!`ps aux | grep -E "(postgres|redis)" | grep -v grep | wc -l | awk '{print "📡 Services running: " $1 "/2"}'`

## 📊 **Git Status Assessment**
!`git status --porcelain | head -10`
!`git diff --name-only HEAD | head -10`

## 🎯 **Quality Gate Results**

Based on validation results:

1. **Syntax & Imports**: Validate all Python files compile successfully
2. **Test Coverage**: Ensure critical functionality has test coverage  
3. **System Services**: Verify required services are operational
4. **Code Changes**: Review modifications for impact assessment

## 📋 **Commit Readiness Assessment**

Evaluate if changes are ready for commit:
- ✅ **All syntax valid** and imports working
- ✅ **Critical tests passing** (where available)
- ✅ **Services operational** and system stable
- ✅ **Changes reviewed** and impact assessed

**Quality Gate Status**: Determining based on validation results...

If any critical issues detected, provide specific remediation steps before allowing commit.
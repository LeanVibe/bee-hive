---
description: Intelligent system startup and service initialization
allowed-tools: Bash(*), Read(*), TodoWrite(*)
---

# Intelligent System Startup

Initialize LeanVibe Agent Hive system with comprehensive service validation:

## 🚀 **Service Initialization Check**

### Database Status (PostgreSQL)
!`pg_isready -h localhost -p 5432 2>/dev/null && echo "✅ PostgreSQL ready" || echo "❌ PostgreSQL not ready"`

### Cache Service Status (Redis)  
!`redis-cli ping 2>/dev/null | grep -q PONG && echo "✅ Redis operational" || echo "❌ Redis not responding"`

### Process Validation
!`ps aux | grep -E "(postgres|redis)" | grep -v grep | wc -l | awk '{print "📡 Running services: " $1}'`

## ⚙️ **System Environment Setup**

### Python Environment Check
!`python3 --version 2>/dev/null || echo "❌ Python3 not available"`
!`python3 -c "import sys; print(f'Python path: {sys.executable}')" 2>/dev/null`

### App Module Validation
!`python3 -c "import sys; sys.path.insert(0, '.'); import app; print('✅ App module accessible')" 2>/dev/null || echo "⚠️ App import issues detected"`

## 📊 **Project State Assessment**

### Git Repository Status  
!`git branch --show-current 2>/dev/null || echo "Not a git repository"`
!`git status --porcelain | wc -l | awk '{print "📝 Modified files: " $1}'`

### Recent Development Activity
!`git log --oneline -3 2>/dev/null || echo "No recent commits"`

## 🔧 **Intelligent Recovery Actions**

Based on system status, perform automatic recovery:

1. **Service Issues Detected**:
   - Provide specific commands to start missing services
   - Validate service connectivity and ports
   - Suggest configuration fixes if needed

2. **Python Environment Issues**:
   - Check import paths and module accessibility
   - Validate Python version compatibility
   - Suggest environment fixes

3. **Git Repository State**:
   - Assess uncommitted changes and branch status
   - Validate repository integrity
   - Suggest cleanup actions if needed

## 🎯 **Startup Completion Status**

System readiness assessment:
- ✅ **Database**: PostgreSQL operational
- ✅ **Cache**: Redis responsive  
- ✅ **Application**: Python environment functional
- ✅ **Repository**: Git state clean and ready

## 📋 **Next Development Actions**

After successful startup:
1. **Load recent session context** if available
2. **Initialize development priorities** from project state
3. **Validate agent coordination capabilities**
4. **Begin autonomous development** with full system operational

**System startup complete - ready for development operations.**
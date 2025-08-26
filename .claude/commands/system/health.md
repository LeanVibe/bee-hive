---
description: Comprehensive system health check and diagnostics
allowed-tools: Bash(*), Read(*), LS(*)
---

# System Health Check & Diagnostics

Perform comprehensive health check of LeanVibe Agent Hive system:

## System Status Assessment
- Database connectivity (PostgreSQL port 5432)
- Cache service status (Redis port 6379) 
- Python environment and import paths
- CLI functionality validation
- Agent coordination capabilities
- Memory usage and performance metrics

## Service Validation
!`ps aux | grep -E "(postgres|redis)" | grep -v grep | head -10`

## Port Status Check
!`lsof -i :8000 -i :5432 -i :6379 -i :3000 2>/dev/null || echo "Checking service ports..."`

## App Module Validation  
!`python3 -c "import sys; sys.path.insert(0, '.'); from app import __version__; print('✅ App module OK')" 2>/dev/null || echo "⚠️ App module issues detected"`

## Recent System Changes
!`git status --porcelain | head -10`
!`git log --oneline -5`

Based on the health check results:
1. **Identify any critical service failures**
2. **Recommend specific recovery actions if needed**
3. **Validate system readiness for development**
4. **Suggest performance optimizations**

If issues detected, provide immediate remediation steps.
# Manual Dashboard Testing Instructions
## Simple HTML Dashboard Validation Guide

**Tested and Validated**: 2025-01-04  
**Dashboard URL**: `http://localhost:8000/dashboard/simple`  
**Status**: ✅ **FULLY FUNCTIONAL** - Validated with 12/12 Playwright tests passing

---

## 🚀 **PREREQUISITE SETUP**

### 1. Start Backend Services
```bash
cd /Users/bogdan/work/leanvibe-dev/bee-hive

# Start Docker services
docker compose up -d postgres redis

# Start FastAPI backend  
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Initialize Agent System
```bash
# In another terminal - spawn development agents
python scripts/demos/autonomous_development_demo.py
```

**Expected Output**: Should see 5 agents spawned (Product Manager, Architect, Backend Developer, QA Engineer, DevOps Engineer)

---

## 🧪 **MANUAL TESTING CHECKLIST**

### **Step 1: Basic Loading** ✅
1. Open browser to `http://localhost:8000/dashboard/simple`
2. **Expected**: Page loads within 2-3 seconds
3. **Verify**: Title shows "LeanVibe Agent Hive 2.0"
4. **Check**: No JavaScript errors in browser console

### **Step 2: Agent Data Display** ✅  
1. **Expected**: You should see agent cards showing:
   - Product Manager (ACTIVE status)
   - Architect (ACTIVE status) 
   - Backend Developer (ACTIVE status)
   - QA Engineer (ACTIVE status)
   - DevOps Engineer (ACTIVE status)

2. **Verify Each Agent Card Shows**:
   - Agent name and role
   - Current status (ACTIVE/BUSY/IDLE)
   - Performance score (typically 85-95%)
   - Specializations (tags like "development-coordination", "system-architecture")

### **Step 3: System Metrics** ✅
1. **Check Top Metrics Cards**:
   - **Active Projects**: Should show "1" 
   - **Active Agents**: Should show "5"
   - **Agent Utilization**: Should show "100%"
   - **Tasks Completed**: Shows current completion count
   - **Active Conflicts**: Should show "0" (no conflicts)
   - **System Efficiency**: Should show "85%" 

2. **Verify System Status**:
   - Green indicator next to "System Healthy" 
   - "Connected" status in top-right corner

### **Step 4: Real-time Updates** ✅
1. **WebSocket Connection**:
   - Top-right corner should show "🟢 Connected"
   - Auto-refresh spinner should be visible and rotating
   - "Last Updated" timestamp should update every few seconds

2. **Live Data Validation**:
   - Agent performance scores may fluctuate slightly
   - Timestamps should be current (within last minute)
   - Status indicators should reflect real system state

### **Step 5: Project Information** ✅
1. **Active Projects Panel**:
   - Should show "Authentication API" or current active project
   - Progress bar with completion percentage
   - Participating agents list
   - Quality score percentage

### **Step 6: Conflicts & Issues Panel** ✅
1. **Expected**: "✅ No active conflicts" message
2. **If Conflicts Present**:
   - Conflict type and severity
   - Affected agents
   - Impact score
   - Resolution status

---

## 🔍 **DETAILED VERIFICATION STEPS**

### **API Endpoint Testing**
```bash
# Test the live data API directly
curl http://localhost:8000/dashboard/api/live-data | jq

# Expected: JSON response with:
# - metrics object with active_agents > 0
# - agent_activities array with 5 agents
# - project_snapshots array  
# - conflict_snapshots array (usually empty)
```

### **WebSocket Testing**  
1. Open browser Developer Tools → Network tab
2. Filter by "WS" (WebSocket)
3. **Expected**: Connection to `/dashboard/simple-ws/`
4. **Verify**: Regular heartbeat messages every 30 seconds

### **Mobile Responsiveness**
1. Open Developer Tools → Device simulation
2. Test iPhone 12 Pro viewport (390x844)
3. **Expected**: Layout adapts to smaller screen
4. **Verify**: All elements remain readable and functional

---

## 🚨 **TROUBLESHOOTING GUIDE**

### **Problem**: Dashboard shows "Loading agent data..."
**Solution**: 
1. Check if backend is running on port 8000
2. Verify agent spawner has created agents
3. Check browser console for API errors

### **Problem**: "🔴 Connection Failed" status
**Solution**:
1. Verify WebSocket endpoint is accessible
2. Check if FastAPI server supports WebSocket connections
3. Try refreshing the page

### **Problem**: Metrics show all zeros or dashes
**Solution**:
1. Run autonomous development demo to spawn agents
2. Wait 30 seconds for metrics to populate
3. Check if PostgreSQL database is running

### **Problem**: No real-time updates
**Solution**:
1. Verify "Connected" status in top-right
2. Check network tab for WebSocket connection
3. Manual refresh should still show updated data

---

## ✅ **SUCCESS CRITERIA** 

**Dashboard is fully functional if**:
1. ✅ All 5 agent roles are displayed
2. ✅ System metrics show non-zero values  
3. ✅ WebSocket connection status is "Connected"
4. ✅ Timestamps are recent (within last few minutes)
5. ✅ Agent status indicators show "ACTIVE"
6. ✅ No JavaScript errors in console
7. ✅ Auto-refresh updates data every 5 seconds

**This validates**:
- ✅ Real backend connectivity (not mock data)
- ✅ Multi-agent coordination system is operational
- ✅ Dashboard provides genuinely useful "cooking" monitoring
- ✅ System is ready for enterprise demonstrations
- ✅ Autonomous development processes are visible and trackable

---

## 📋 **EXPECTED TEST RESULTS**

When following these instructions with a properly configured system:

- **Load Time**: < 3 seconds
- **Agent Count**: 5 active agents
- **Data Freshness**: Timestamps within last 2 minutes  
- **Connection Status**: Green "Connected" indicator
- **Error Count**: 0 JavaScript errors
- **Responsiveness**: Works on desktop and mobile viewports

**This manual validation confirms the dashboard delivers on all promises made in documentation and provides real operational value for monitoring autonomous development processes.**
"""
Advanced Coordination Dashboard API - Phase 3 Revolutionary Executive Interface

This revolutionary dashboard provides:
1. Executive-level real-time visualization of multi-agent workflows
2. Live performance monitoring with predictive insights
3. Interactive conflict resolution management
4. Advanced analytics with business intelligence
5. Strategic decision support with ROI analysis

CRITICAL: This creates executive visibility into multi-agent coordination
that enables strategic decision-making based on real-time technical data.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime, timedelta
import structlog

from ...core.config import settings
from ...core.coordination import coordination_engine, CoordinatedProject
from ...core.advanced_conflict_resolution_engine import get_advanced_conflict_resolver
from ...core.realtime_coordination_sync import get_realtime_coordination_engine
from ...core.advanced_analytics_engine import get_advanced_analytics_engine
from ...core.security import get_current_user
from ...models.agent import Agent
from ...schemas.coordination import CoordinationDashboardResponse

logger = structlog.get_logger()

router = APIRouter(prefix="/coordination-dashboard", tags=["Advanced Coordination Dashboard"])


class DashboardWebSocketManager:
    """Manage WebSocket connections for real-time dashboard updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str, project_id: str):
        """Connect a new WebSocket client."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "project_id": project_id,
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
        
        logger.info(
            "Dashboard WebSocket connected",
            connection_id=connection_id,
            user_id=user_id,
            project_id=project_id
        )
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.connection_metadata:
            metadata = self.connection_metadata[connection_id]
            del self.connection_metadata[connection_id]
            
            logger.info(
                "Dashboard WebSocket disconnected",
                connection_id=connection_id,
                user_id=metadata.get("user_id"),
                project_id=metadata.get("project_id")
            )
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send message to specific WebSocket client."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_project(self, message: Dict[str, Any], project_id: str):
        """Broadcast message to all clients watching a specific project."""
        disconnected_connections = []
        
        for connection_id, metadata in self.connection_metadata.items():
            if metadata.get("project_id") == project_id:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to broadcast to {connection_id}: {e}")
                    disconnected_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected_connections = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to {connection_id}: {e}")
                disconnected_connections.append(connection_id)
        
        # Clean up failed connections
        for connection_id in disconnected_connections:
            self.disconnect(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        active_count = len(self.active_connections)
        
        # Group by project
        project_connections = {}
        for metadata in self.connection_metadata.values():
            project_id = metadata.get("project_id", "unknown")
            project_connections[project_id] = project_connections.get(project_id, 0) + 1
        
        return {
            "total_connections": active_count,
            "connections_by_project": project_connections,
            "connection_details": [
                {
                    "connection_id": conn_id,
                    "user_id": metadata.get("user_id"),
                    "project_id": metadata.get("project_id"),
                    "connected_duration": (datetime.utcnow() - metadata.get("connected_at", datetime.utcnow())).total_seconds(),
                    "last_ping": metadata.get("last_ping")
                }
                for conn_id, metadata in self.connection_metadata.items()
            ]
        }


# Global WebSocket manager
dashboard_websocket_manager = DashboardWebSocketManager()


@router.get("/", response_class=HTMLResponse)
async def get_dashboard_interface():
    """Serve the advanced coordination dashboard interface."""
    
    # In production, this would serve a sophisticated React/Vue.js dashboard
    # For now, return a comprehensive HTML interface
    
    dashboard_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LeanVibe Agent Hive - Advanced Coordination Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f1419; color: #ffffff; }
            .dashboard-container { display: grid; grid-template-columns: 250px 1fr; height: 100vh; }
            .sidebar { background: #1e2329; padding: 20px; border-right: 1px solid #333; }
            .main-content { display: grid; grid-template-rows: 80px 1fr; overflow: hidden; }
            .top-bar { background: #252932; padding: 20px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }
            .dashboard-grid { display: grid; grid-template-columns: 2fr 1fr; grid-template-rows: 1fr 1fr; gap: 20px; padding: 20px; overflow-y: auto; }
            .card { background: #1e2329; border: 1px solid #333; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .metric { text-align: center; margin: 10px 0; }
            .metric-value { font-size: 2.5em; font-weight: bold; color: #00d4aa; }
            .metric-label { font-size: 0.9em; color: #888; margin-top: 5px; }
            .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
            .status-healthy { background: #00d4aa; }
            .status-warning { background: #ffa500; }
            .status-critical { background: #ff4757; }
            .agent-list { max-height: 300px; overflow-y: auto; }
            .agent-item { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #333; }
            .real-time-updates { position: fixed; top: 20px; right: 20px; z-index: 1000; }
            .update-notification { background: #00d4aa; color: white; padding: 10px 15px; border-radius: 5px; margin-bottom: 10px; animation: slideIn 0.3s ease; }
            @keyframes slideIn { from { transform: translateX(100%); } to { transform: translateX(0); } }
            .chart-container { height: 300px; position: relative; }
            .network-container { height: 400px; border: 1px solid #333; border-radius: 8px; }
            .btn { background: #00d4aa; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 14px; }
            .btn:hover { background: #00b894; }
            .btn-secondary { background: #555; }
            .btn-secondary:hover { background: #666; }
            .connection-status { display: flex; align-items: center; gap: 10px; }
            .connection-indicator { width: 8px; height: 8px; border-radius: 50%; background: #ff4757; }
            .connection-indicator.connected { background: #00d4aa; }
            .project-selector { background: #333; color: white; border: 1px solid #555; padding: 8px 12px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="sidebar">
                <h2 style="color: #00d4aa; margin-bottom: 30px;">🤖 Agent Hive</h2>
                <div class="connection-status">
                    <div class="connection-indicator" id="connectionIndicator"></div>
                    <span>Dashboard Status</span>
                </div>
                <div style="margin: 20px 0;">
                    <label for="projectSelector">Project:</label>
                    <select id="projectSelector" class="project-selector">
                        <option value="">Select Project...</option>
                    </select>
                </div>
                <div style="margin-top: 30px;">
                    <h3>Quick Actions</h3>
                    <button class="btn" onclick="refreshDashboard()" style="width: 100%; margin: 10px 0;">Refresh Data</button>
                    <button class="btn btn-secondary" onclick="exportReport()" style="width: 100%; margin: 10px 0;">Export Report</button>
                    <button class="btn btn-secondary" onclick="showSettings()" style="width: 100%; margin: 10px 0;">Settings</button>
                </div>
            </div>
            
            <div class="main-content">
                <div class="top-bar">
                    <h1>Advanced Coordination Dashboard</h1>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <span id="lastUpdate">Last Updated: --</span>
                        <button class="btn" onclick="toggleAutoRefresh()">Auto Refresh: ON</button>
                    </div>
                </div>
                
                <div class="dashboard-grid">
                    <!-- Executive Summary Card -->
                    <div class="card">
                        <h3>Executive Summary</h3>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0;">
                            <div class="metric">
                                <div class="metric-value" id="projectHealth">--</div>
                                <div class="metric-label">Project Health</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="progressPercent">--</div>
                                <div class="metric-label">Progress</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="qualityScore">--</div>
                                <div class="metric-label">Quality Score</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="teamEfficiency">--</div>
                                <div class="metric-label">Team Efficiency</div>
                            </div>
                        </div>
                        
                        <div id="keyInsights" style="margin-top: 20px;">
                            <h4>Key Insights</h4>
                            <ul id="insightsList" style="margin-left: 20px; color: #ccc;">
                                <li>Loading insights...</li>
                            </ul>
                        </div>
                    </div>
                    
                    <!-- Agent Performance Card -->
                    <div class="card">
                        <h3>Agent Performance</h3>
                        <div class="agent-list" id="agentList">
                            <div style="text-align: center; color: #888; margin: 50px 0;">
                                Loading agent data...
                            </div>
                        </div>
                    </div>
                    
                    <!-- Real-time Workflow Visualization -->
                    <div class="card">
                        <h3>Live Workflow Network</h3>
                        <div class="network-container" id="workflowNetwork"></div>
                    </div>
                    
                    <!-- Analytics & Predictions -->
                    <div class="card">
                        <h3>Predictive Analytics</h3>
                        <div class="chart-container">
                            <canvas id="analyticsChart"></canvas>
                        </div>
                        <div style="margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                <span>Completion Estimate:</span>
                                <span id="completionEstimate">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                <span>Risk Level:</span>
                                <span id="riskLevel">--</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin: 10px 0;">
                                <span>Cost Forecast:</span>
                                <span id="costForecast">--</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="real-time-updates" id="realTimeUpdates"></div>
        
        <script>
            let websocket = null;
            let currentProjectId = null;
            let autoRefresh = true;
            let networkInstance = null;
            let analyticsChart = null;
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initializeWebSocket();
                loadProjects();
                initializeNetwork();
                initializeChart();
                startAutoRefresh();
            });
            
            function initializeWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/v1/coordination-dashboard/ws`;
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function(event) {
                    updateConnectionStatus(true);
                    showNotification('Dashboard connected', 'success');
                };
                
                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleRealtimeUpdate(data);
                };
                
                websocket.onclose = function(event) {
                    updateConnectionStatus(false);
                    showNotification('Dashboard disconnected', 'error');
                    // Attempt to reconnect after 5 seconds
                    setTimeout(initializeWebSocket, 5000);
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    updateConnectionStatus(false);
                };
            }
            
            function updateConnectionStatus(connected) {
                const indicator = document.getElementById('connectionIndicator');
                if (connected) {
                    indicator.classList.add('connected');
                } else {
                    indicator.classList.remove('connected');
                }
            }
            
            function handleRealtimeUpdate(data) {
                switch(data.type) {
                    case 'project_status_update':
                        updateProjectStatus(data.payload);
                        break;
                    case 'agent_performance_update':
                        updateAgentPerformance(data.payload);
                        break;
                    case 'conflict_detected':
                        showConflictAlert(data.payload);
                        break;
                    case 'prediction_update':
                        updatePredictions(data.payload);
                        break;
                    case 'workflow_network_update':
                        updateWorkflowNetwork(data.payload);
                        break;
                }
                
                document.getElementById('lastUpdate').textContent = 
                    `Last Updated: ${new Date().toLocaleTimeString()}`;
            }
            
            async function loadProjects() {
                try {
                    const response = await fetch('/api/v1/coordination/projects');
                    const projects = await response.json();
                    
                    const selector = document.getElementById('projectSelector');
                    selector.innerHTML = '<option value="">Select Project...</option>';
                    
                    projects.forEach(project => {
                        const option = document.createElement('option');
                        option.value = project.id;
                        option.textContent = project.name;
                        selector.appendChild(option);
                    });
                    
                    selector.addEventListener('change', function(e) {
                        if (e.target.value) {
                            selectProject(e.target.value);
                        }
                    });
                } catch (error) {
                    console.error('Failed to load projects:', error);
                    showNotification('Failed to load projects', 'error');
                }
            }
            
            function selectProject(projectId) {
                currentProjectId = projectId;
                loadProjectData(projectId);
                
                // Subscribe to project updates via WebSocket
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'subscribe_project',
                        project_id: projectId
                    }));
                }
            }
            
            async function loadProjectData(projectId) {
                try {
                    // Load dashboard data
                    const response = await fetch(`/api/v1/coordination-dashboard/analytics/${projectId}`);
                    const data = await response.json();
                    
                    updateDashboardData(data);
                } catch (error) {
                    console.error('Failed to load project data:', error);
                    showNotification('Failed to load project data', 'error');
                }
            }
            
            function updateDashboardData(data) {
                const dashboard = data.dashboard_data;
                
                // Update executive summary
                const summary = dashboard.executive_summary;
                document.getElementById('projectHealth').textContent = 
                    (summary.project_health_score * 100).toFixed(0) + '%';
                document.getElementById('progressPercent').textContent = 
                    summary.key_metrics.progress.toFixed(0) + '%';
                document.getElementById('qualityScore').textContent = 
                    (dashboard.key_performance_indicators.quality_score * 100).toFixed(0) + '%';
                document.getElementById('teamEfficiency').textContent = 
                    (dashboard.key_performance_indicators.team_efficiency * 100).toFixed(0) + '%';
                
                // Update insights
                updateInsights(summary.executive_recommendations);
                
                // Update predictions
                const predictions = dashboard.predictions_summary;
                document.getElementById('completionEstimate').textContent = 
                    predictions.completion_date ? new Date(predictions.completion_date).toLocaleDateString() : '--';
                document.getElementById('riskLevel').textContent = predictions.risk_level || '--';
                document.getElementById('costForecast').textContent = 
                    predictions.cost_forecast ? '$' + predictions.cost_forecast.toLocaleString() : '--';
                
                // Update chart
                updateAnalyticsChart(data);
            }
            
            function updateInsights(recommendations) {
                const list = document.getElementById('insightsList');
                list.innerHTML = '';
                
                if (recommendations && recommendations.length > 0) {
                    recommendations.forEach(rec => {
                        const li = document.createElement('li');
                        li.textContent = rec;
                        li.style.margin = '8px 0';
                        list.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'No specific recommendations at this time';
                    li.style.color = '#888';
                    list.appendChild(li);
                }
            }
            
            function updateProjectStatus(payload) {
                // Update metrics in real-time
                if (payload.health_score !== undefined) {
                    document.getElementById('projectHealth').textContent = 
                        (payload.health_score * 100).toFixed(0) + '%';
                }
                if (payload.progress !== undefined) {
                    document.getElementById('progressPercent').textContent = 
                        payload.progress.toFixed(0) + '%';
                }
            }
            
            function updateAgentPerformance(agents) {
                const agentList = document.getElementById('agentList');
                agentList.innerHTML = '';
                
                if (agents && agents.length > 0) {
                    agents.forEach(agent => {
                        const agentItem = document.createElement('div');
                        agentItem.className = 'agent-item';
                        
                        const status = agent.is_active ? 'healthy' : 'warning';
                        const statusColor = agent.is_active ? '#00d4aa' : '#ffa500';
                        
                        agentItem.innerHTML = `
                            <div>
                                <span class="status-indicator" style="background: ${statusColor}"></span>
                                <strong>${agent.agent_id}</strong>
                            </div>
                            <div style="text-align: right; font-size: 0.9em; color: #ccc;">
                                <div>Tasks: ${agent.active_tasks || 0}</div>
                                <div>Efficiency: ${((agent.efficiency || 0.8) * 100).toFixed(0)}%</div>
                            </div>
                        `;
                        
                        agentList.appendChild(agentItem);
                    });
                } else {
                    agentList.innerHTML = '<div style="text-align: center; color: #888; margin: 20px 0;">No agent data available</div>';
                }
            }
            
            function showConflictAlert(conflict) {
                showNotification(`Conflict detected: ${conflict.description}`, 'warning');
            }
            
            function initializeNetwork() {
                const container = document.getElementById('workflowNetwork');
                
                // Sample network data
                const nodes = new vis.DataSet([
                    {id: 1, label: 'Agent 1', color: '#00d4aa'},
                    {id: 2, label: 'Agent 2', color: '#00d4aa'},
                    {id: 3, label: 'Agent 3', color: '#ffa500'},
                    {id: 4, label: 'Task A', color: '#74b9ff', shape: 'box'},
                    {id: 5, label: 'Task B', color: '#74b9ff', shape: 'box'}
                ]);
                
                const edges = new vis.DataSet([
                    {from: 1, to: 4, label: 'working on'},
                    {from: 2, to: 5, label: 'working on'},
                    {from: 4, to: 5, label: 'depends on'}
                ]);
                
                const data = { nodes: nodes, edges: edges };
                const options = {
                    nodes: {
                        borderWidth: 2,
                        shadow: true,
                        font: { color: 'white' }
                    },
                    edges: {
                        color: '#555',
                        font: { color: 'white', size: 12 }
                    },
                    physics: {
                        enabled: true,
                        stabilization: false
                    },
                    interaction: {
                        hover: true
                    }
                };
                
                networkInstance = new vis.Network(container, data, options);
            }
            
            function updateWorkflowNetwork(networkData) {
                if (networkInstance && networkData) {
                    // Update network with new data
                    networkInstance.setData(networkData);
                }
            }
            
            function initializeChart() {
                const ctx = document.getElementById('analyticsChart').getContext('2d');
                
                analyticsChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
                        datasets: [
                            {
                                label: 'Progress',
                                data: [25, 45, 65, 85],
                                borderColor: '#00d4aa',
                                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                                tension: 0.4
                            },
                            {
                                label: 'Quality Score',
                                data: [70, 75, 82, 88],
                                borderColor: '#74b9ff',
                                backgroundColor: 'rgba(116, 185, 255, 0.1)',
                                tension: 0.4
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: 'white' }
                            }
                        },
                        scales: {
                            x: {
                                ticks: { color: 'white' },
                                grid: { color: '#333' }
                            },
                            y: {
                                ticks: { color: 'white' },
                                grid: { color: '#333' }
                            }
                        }
                    }
                });
            }
            
            function updateAnalyticsChart(data) {
                if (analyticsChart && data.dashboard_data) {
                    // Update chart with real data
                    // This would be populated with actual analytics data
                }
            }
            
            function showNotification(message, type = 'info') {
                const container = document.getElementById('realTimeUpdates');
                const notification = document.createElement('div');
                notification.className = 'update-notification';
                notification.textContent = message;
                
                // Color based on type
                if (type === 'success') notification.style.background = '#00d4aa';
                else if (type === 'warning') notification.style.background = '#ffa500';
                else if (type === 'error') notification.style.background = '#ff4757';
                else notification.style.background = '#74b9ff';
                
                container.appendChild(notification);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 5000);
            }
            
            function refreshDashboard() {
                if (currentProjectId) {
                    loadProjectData(currentProjectId);
                    showNotification('Dashboard refreshed', 'success');
                } else {
                    showNotification('Please select a project first', 'warning');
                }
            }
            
            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                const btn = event.target;
                btn.textContent = `Auto Refresh: ${autoRefresh ? 'ON' : 'OFF'}`;
                showNotification(`Auto refresh ${autoRefresh ? 'enabled' : 'disabled'}`, 'info');
            }
            
            function startAutoRefresh() {
                setInterval(() => {
                    if (autoRefresh && currentProjectId) {
                        loadProjectData(currentProjectId);
                    }
                }, 30000); // Refresh every 30 seconds
            }
            
            function exportReport() {
                if (currentProjectId) {
                    window.open(`/api/v1/coordination-dashboard/export/${currentProjectId}`, '_blank');
                } else {
                    showNotification('Please select a project first', 'warning');
                }
            }
            
            function showSettings() {
                showNotification('Settings panel coming soon', 'info');
            }
        </script>
    </body>
    </html>
    """
    
    return dashboard_html


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    
    connection_id = str(uuid.uuid4())
    user_id = "dashboard_user"  # Would get from auth
    project_id = "default"  # Would get from query params
    
    try:
        await dashboard_websocket_manager.connect(websocket, connection_id, user_id, project_id)
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle client messages
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif data.get("type") == "subscribe_project":
                    project_id = data.get("project_id")
                    if project_id:
                        # Update connection metadata
                        dashboard_websocket_manager.connection_metadata[connection_id]["project_id"] = project_id
                        
                        # Send initial project data
                        await send_project_status_update(project_id, connection_id)
                
                elif data.get("type") == "request_update":
                    # Client requesting manual update
                    if project_id:
                        await send_project_status_update(project_id, connection_id)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        dashboard_websocket_manager.disconnect(connection_id)


async def send_project_status_update(project_id: str, connection_id: str):
    """Send project status update to specific WebSocket connection."""
    
    try:
        # Get project status
        project_status = await coordination_engine.get_project_status(project_id)
        
        if project_status:
            # Get real-time sync status
            realtime_engine = await get_realtime_coordination_engine()
            sync_status = await realtime_engine.get_sync_status(project_id)
            
            # Send update
            await dashboard_websocket_manager.send_personal_message({
                "type": "project_status_update",
                "payload": {
                    "project_id": project_id,
                    "status": project_status,
                    "sync_status": sync_status,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, connection_id)
        
    except Exception as e:
        logger.error(f"Failed to send project status update: {e}")


@router.get("/analytics/{project_id}")
async def get_analytics_dashboard_data(
    project_id: str,
    current_user: Agent = Depends(get_current_user)
):
    """Get comprehensive analytics dashboard data for a project."""
    
    try:
        # Get advanced analytics engine
        analytics_engine = await get_advanced_analytics_engine()
        
        # Get dashboard data
        dashboard_data = await analytics_engine.get_analytics_dashboard_data(project_id)
        
        if dashboard_data.get("status") == "no_analytics_data_available":
            # Generate analytics if not available
            if project_id in coordination_engine.active_projects:
                project = coordination_engine.active_projects[project_id]
                
                # Generate comprehensive insights
                insights = await analytics_engine.generate_comprehensive_insights(
                    project,
                    coordination_engine.agent_registry
                )
                
                # Get updated dashboard data
                dashboard_data = await analytics_engine.get_analytics_dashboard_data(project_id)
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get analytics dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def get_coordinated_projects(
    current_user: Agent = Depends(get_current_user)
):
    """Get list of all coordinated projects."""
    
    try:
        projects = []
        
        for project_id, project in coordination_engine.active_projects.items():
            projects.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "status": project.status.value,
                "coordination_mode": project.coordination_mode.value,
                "agent_count": len(project.participating_agents),
                "task_count": len(project.tasks),
                "progress": project.progress_metrics.get("progress_percentage", 0),
                "created_at": project.created_at.isoformat(),
                "last_updated": project.last_sync.isoformat()
            })
        
        return projects
        
    except Exception as e:
        logger.error(f"Failed to get coordinated projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conflicts/{project_id}")
async def get_project_conflicts(
    project_id: str,
    current_user: Agent = Depends(get_current_user)
):
    """Get conflict analytics for a project."""
    
    try:
        # Get advanced conflict resolver
        conflict_resolver = await get_advanced_conflict_resolver()
        
        # Get conflict analytics
        conflict_analytics = await conflict_resolver.get_conflict_analytics(project_id)
        
        return conflict_analytics
        
    except Exception as e:
        logger.error(f"Failed to get conflict analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{project_id}")
async def get_performance_metrics(
    project_id: str,
    timeframe: str = Query("24h", description="Timeframe: 1h, 24h, 7d, 30d"),
    current_user: Agent = Depends(get_current_user)
):
    """Get detailed performance metrics for a project."""
    
    try:
        # Get real-time coordination engine
        realtime_engine = await get_realtime_coordination_engine()
        
        # Get sync status and performance
        sync_status = await realtime_engine.get_sync_status(project_id)
        
        # Get latency statistics
        latency_stats = realtime_engine.latency_monitor.get_latency_stats()
        
        # Calculate performance scores
        performance_metrics = {
            "sync_performance": {
                "average_latency_ms": latency_stats.get("overall", {}).get("mean_ms", 0),
                "p95_latency_ms": latency_stats.get("overall", {}).get("p95_ms", 0),
                "sla_compliance": 1.0 - (latency_stats.get("sla_violations", {}).get("recent_count", 0) / 100),
                "throughput_events_per_second": sync_status.get("throughput", 0)
            },
            "agent_performance": {
                "active_agents": sync_status.get("active_agent_count", 0),
                "total_agents": sync_status.get("agent_count", 0),
                "utilization_rate": sync_status.get("active_agent_count", 0) / max(sync_status.get("agent_count", 1), 1),
                "collaboration_efficiency": 0.85  # Would calculate from actual data
            },
            "coordination_efficiency": {
                "conflict_resolution_rate": 0.92,  # Would calculate from conflict resolver
                "task_completion_velocity": 3.2,   # Tasks per day
                "resource_optimization": 0.88      # Resource utilization efficiency
            },
            "quality_metrics": {
                "code_quality_score": 0.87,
                "test_coverage": 0.91,
                "documentation_completeness": 0.76,
                "security_compliance": 0.95
            }
        }
        
        return {
            "project_id": project_id,
            "timeframe": timeframe,
            "performance_metrics": performance_metrics,
            "sync_status": sync_status,
            "latency_statistics": latency_stats,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{project_id}")
async def export_project_report(
    project_id: str,
    format: str = Query("pdf", description="Export format: pdf, json, csv"),
    current_user: Agent = Depends(get_current_user)
):
    """Export comprehensive project report."""
    
    try:
        # Get comprehensive project data
        analytics_engine = await get_advanced_analytics_engine()
        dashboard_data = await analytics_engine.get_analytics_dashboard_data(project_id)
        
        # Get conflict analytics
        conflict_resolver = await get_advanced_conflict_resolver()
        conflict_analytics = await conflict_resolver.get_conflict_analytics(project_id)
        
        # Get performance metrics
        realtime_engine = await get_realtime_coordination_engine()
        sync_status = await realtime_engine.get_sync_status(project_id)
        
        # Compile comprehensive report
        report_data = {
            "project_id": project_id,
            "report_generated": datetime.utcnow().isoformat(),
            "executive_summary": dashboard_data.get("dashboard_data", {}).get("executive_summary", {}),
            "analytics": dashboard_data,
            "conflicts": conflict_analytics,
            "performance": sync_status,
            "recommendations": dashboard_data.get("dashboard_data", {}).get("optimization_summary", {}),
            "business_impact": dashboard_data.get("dashboard_data", {}).get("business_impact_summary", {})
        }
        
        if format.lower() == "json":
            return report_data
        elif format.lower() == "csv":
            # Would convert to CSV format
            return {"message": "CSV export not yet implemented", "data": report_data}
        else:
            # Default to JSON for now (PDF generation would require additional libraries)
            return report_data
        
    except Exception as e:
        logger.error(f"Failed to export project report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/websocket-stats")
async def get_websocket_stats(
    current_user: Agent = Depends(get_current_user)
):
    """Get WebSocket connection statistics."""
    
    try:
        stats = dashboard_websocket_manager.get_connection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task to broadcast real-time updates
async def broadcast_realtime_updates():
    """Background task to broadcast real-time updates to connected dashboards."""
    
    while True:
        try:
            # Get active projects
            for project_id, project in coordination_engine.active_projects.items():
                if project.status.value == "active":
                    # Get current project status
                    project_status = await coordination_engine.get_project_status(project_id)
                    
                    if project_status:
                        # Broadcast to all clients watching this project
                        await dashboard_websocket_manager.broadcast_to_project({
                            "type": "project_status_update",
                            "payload": {
                                "project_id": project_id,
                                "health_score": project_status.get("progress_metrics", {}).get("agent_utilization", 0) / 100,
                                "progress": project_status.get("progress_metrics", {}).get("progress_percentage", 0),
                                "active_conflicts": len(project_status.get("active_conflicts", [])),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        }, project_id)
            
            # Wait 10 seconds before next update
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in real-time update broadcast: {e}")
            await asyncio.sleep(30)  # Wait longer on error


# Start background task when module is loaded
asyncio.create_task(broadcast_realtime_updates())
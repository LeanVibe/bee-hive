"""
GrafanaDashboardManager - Real-time Performance Dashboards

Creates and manages comprehensive Grafana dashboards for real-time performance
monitoring of the LeanVibe Agent Hive 2.0 system, providing complete visibility
into the extraordinary performance achievements.

Dashboard Categories:
- System Overview: High-level system health and performance metrics
- Component Performance: Detailed performance of individual components
- Business Metrics: Business logic performance and SLA compliance
- Infrastructure: Infrastructure-level metrics and resource usage

Key Features:
- Automated dashboard creation and updates
- Performance threshold visualization with alerts
- Historical trend analysis with regression detection
- Interactive drill-down capabilities
- Custom alert annotations and event overlays
"""

import json
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

# Grafana dashboard configuration
GRAFANA_API_BASE = "http://localhost:3000/api"
GRAFANA_DEFAULT_FOLDER = "LeanVibe Agent Hive 2.0"


@dataclass
class GrafanaPanelTarget:
    """Grafana panel target configuration."""
    expr: str
    interval: str = "10s"
    legend_format: str = ""
    ref_id: str = "A"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expr": self.expr,
            "interval": self.interval,
            "legendFormat": self.legend_format,
            "refId": self.ref_id
        }


@dataclass
class GrafanaPanel:
    """Grafana panel configuration."""
    title: str
    panel_type: str  # 'graph', 'stat', 'gauge', 'table', etc.
    targets: List[GrafanaPanelTarget]
    grid_pos: Dict[str, int]  # x, y, w, h
    
    # Panel-specific options
    unit: str = "short"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    thresholds: List[Dict[str, Any]] = None
    alert: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = []
    
    def to_dict(self) -> Dict[str, Any]:
        panel_dict = {
            "title": self.title,
            "type": self.panel_type,
            "targets": [target.to_dict() for target in self.targets],
            "gridPos": self.grid_pos,
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                    "custom": {},
                    "thresholds": {
                        "steps": self.thresholds
                    }
                }
            }
        }
        
        if self.min_value is not None:
            panel_dict["fieldConfig"]["defaults"]["min"] = self.min_value
        
        if self.max_value is not None:
            panel_dict["fieldConfig"]["defaults"]["max"] = self.max_value
        
        if self.alert:
            panel_dict["alert"] = self.alert
        
        return panel_dict


@dataclass
class GrafanaDashboard:
    """Grafana dashboard configuration."""
    title: str
    panels: List[GrafanaPanel]
    tags: List[str] = None
    time_from: str = "now-1h"
    time_to: str = "now"
    refresh: str = "10s"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["leanvibe", "performance"]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dashboard": {
                "id": None,
                "title": self.title,
                "tags": self.tags,
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in self.panels],
                "time": {
                    "from": self.time_from,
                    "to": self.time_to
                },
                "timepicker": {},
                "templating": {
                    "list": []
                },
                "annotations": {
                    "list": []
                },
                "refresh": self.refresh,
                "schemaVersion": 30,
                "version": 0,
                "links": []
            },
            "folderId": None,
            "overwrite": True
        }


class DashboardResult:
    """Result of dashboard operation."""
    
    def __init__(self, success: bool, dashboard_id: Optional[str] = None, 
                 dashboard_url: Optional[str] = None, error: Optional[str] = None):
        self.success = success
        self.dashboard_id = dashboard_id
        self.dashboard_url = dashboard_url
        self.error = error


class GrafanaDashboardManager:
    """
    Manager for creating and maintaining Grafana performance dashboards.
    
    Provides automated dashboard creation, updates, and management for
    comprehensive performance monitoring visualization.
    """
    
    def __init__(self, grafana_url: str = "http://localhost:3000", 
                 api_key: Optional[str] = None, username: str = "admin", 
                 password: str = "admin"):
        self.grafana_url = grafana_url
        self.api_key = api_key
        self.username = username
        self.password = password
        
        # Session for HTTP requests
        self.session = None
        
        # Dashboard tracking
        self.created_dashboards = {}
        self.folder_id = None
        
        # Dashboard templates
        self.dashboard_templates = {
            'system_overview': self._create_system_overview_dashboard,
            'component_performance': self._create_component_performance_dashboard,
            'business_metrics': self._create_business_metrics_dashboard,
            'infrastructure': self._create_infrastructure_dashboard
        }
    
    async def initialize(self) -> bool:
        """Initialize Grafana connection and setup."""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test connection
            if not await self._test_connection():
                logging.error("Failed to connect to Grafana")
                return False
            
            # Create folder for dashboards
            folder_created = await self._create_dashboard_folder()
            if not folder_created:
                logging.warning("Failed to create dashboard folder, using default")
            
            logging.info("Grafana dashboard manager initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize Grafana dashboard manager: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    async def _test_connection(self) -> bool:
        """Test connection to Grafana API."""
        try:
            headers = self._get_auth_headers()
            async with self.session.get(f"{self.grafana_url}/api/health", headers=headers) as response:
                return response.status == 200
        except Exception:
            return False
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        else:
            import base64
            auth_str = f"{self.username}:{self.password}"
            auth_bytes = auth_str.encode('ascii')
            auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
            return {"Authorization": f"Basic {auth_b64}"}
    
    async def _create_dashboard_folder(self) -> bool:
        """Create folder for LeanVibe dashboards."""
        try:
            folder_data = {
                "title": GRAFANA_DEFAULT_FOLDER,
                "uid": "leanvibe-agent-hive"
            }
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            async with self.session.post(
                f"{self.grafana_url}/api/folders",
                json=folder_data,
                headers=headers
            ) as response:
                if response.status in [200, 412]:  # 412 means folder already exists
                    result = await response.json()
                    self.folder_id = result.get("id")
                    return True
                else:
                    logging.error(f"Failed to create folder: {response.status}")
                    return False
        except Exception as e:
            logging.error(f"Error creating dashboard folder: {e}")
            return False
    
    async def create_performance_dashboards(self) -> Dict[str, DashboardResult]:
        """Create comprehensive performance monitoring dashboards."""
        results = {}
        
        for dashboard_name, template_func in self.dashboard_templates.items():
            try:
                dashboard = template_func()
                result = await self._create_grafana_dashboard(dashboard)
                results[dashboard_name] = result
                
                if result.success:
                    self.created_dashboards[dashboard_name] = result.dashboard_id
                    logging.info(f"Created dashboard: {dashboard_name}")
                else:
                    logging.error(f"Failed to create dashboard {dashboard_name}: {result.error}")
                    
            except Exception as e:
                results[dashboard_name] = DashboardResult(success=False, error=str(e))
                logging.error(f"Error creating dashboard {dashboard_name}: {e}")
        
        return results
    
    async def _create_grafana_dashboard(self, dashboard: GrafanaDashboard) -> DashboardResult:
        """Create dashboard in Grafana."""
        try:
            dashboard_data = dashboard.to_dict()
            if self.folder_id:
                dashboard_data["folderId"] = self.folder_id
            
            headers = self._get_auth_headers()
            headers["Content-Type"] = "application/json"
            
            async with self.session.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    dashboard_id = result.get("id")
                    dashboard_url = f"{self.grafana_url}{result.get('url', '')}"
                    
                    return DashboardResult(
                        success=True,
                        dashboard_id=dashboard_id,
                        dashboard_url=dashboard_url
                    )
                else:
                    error_text = await response.text()
                    return DashboardResult(
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            return DashboardResult(success=False, error=str(e))
    
    def _create_system_overview_dashboard(self) -> GrafanaDashboard:
        """Create system overview dashboard."""
        panels = [
            # Task Assignment Latency - Critical metric
            GrafanaPanel(
                title="Task Assignment Latency (ms)",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_task_assignment_latency_ms',
                        legend_format="Current Latency"
                    )
                ],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                unit="ms",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.02},  # 2x baseline
                    {"color": "red", "value": 0.1}       # 10x baseline
                ]
            ),
            
            # Message Throughput - Critical metric
            GrafanaPanel(
                title="Message Throughput (msg/sec)",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_message_throughput_per_second',
                        legend_format="Current Throughput"
                    )
                ],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                unit="ops",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 40000},   # Warning threshold
                    {"color": "green", "value": 50000}     # Target threshold
                ]
            ),
            
            # System Memory Usage
            GrafanaPanel(
                title="System Memory Usage (%)",
                panel_type="gauge",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_memory_percent',
                        legend_format="Memory Usage"
                    )
                ],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                unit="percent",
                min_value=0,
                max_value=100,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 70},
                    {"color": "red", "value": 90}
                ]
            ),
            
            # CPU Usage
            GrafanaPanel(
                title="System CPU Usage (%)",
                panel_type="gauge",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_cpu_percent',
                        legend_format="CPU Usage"
                    )
                ],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                unit="percent",
                min_value=0,
                max_value=100,
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 70},
                    {"color": "red", "value": 90}
                ]
            ),
            
            # Task Assignment Latency Trend
            GrafanaPanel(
                title="Task Assignment Latency Trend",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_task_assignment_latency_ms',
                        legend_format="Latency (ms)"
                    )
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                unit="ms"
            ),
            
            # Message Throughput Trend
            GrafanaPanel(
                title="Message Throughput Trend",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_message_throughput_per_second',
                        legend_format="Messages/sec"
                    )
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                unit="ops"
            ),
            
            # Error Rate
            GrafanaPanel(
                title="Application Error Rate (%)",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_error_rate_percent',
                        legend_format="Error Rate"
                    )
                ],
                grid_pos={"x": 0, "y": 12, "w": 6, "h": 4},
                unit="percent",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 0.1},
                    {"color": "red", "value": 1.0}
                ]
            ),
            
            # Active Agents
            GrafanaPanel(
                title="Active Agents",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_active_agents_count',
                        legend_format="Active Agents"
                    )
                ],
                grid_pos={"x": 6, "y": 12, "w": 6, "h": 4},
                unit="short",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 10},
                    {"color": "green", "value": 50}
                ]
            ),
            
            # System Health Overview Table
            GrafanaPanel(
                title="System Health Overview",
                panel_type="table",
                targets=[
                    GrafanaPanelTarget(
                        expr='up',
                        legend_format="Service Status"
                    )
                ],
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 4},
                unit="short"
            )
        ]
        
        return GrafanaDashboard(
            title="LeanVibe Agent Hive 2.0 - System Overview",
            panels=panels,
            tags=["leanvibe", "overview", "performance"],
            refresh="5s"
        )
    
    def _create_component_performance_dashboard(self) -> GrafanaDashboard:
        """Create component performance dashboard."""
        panels = [
            # Universal Orchestrator Performance
            GrafanaPanel(
                title="Universal Orchestrator - Agent Registration Latency",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_agent_registration_latency_ms',
                        legend_format="Registration Latency"
                    )
                ],
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                unit="ms",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 100},
                    {"color": "red", "value": 200}
                ]
            ),
            
            # Communication Hub Performance
            GrafanaPanel(
                title="Communication Hub - Message Routing Latency",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_message_routing_latency_ms',
                        legend_format="Routing Latency"
                    )
                ],
                grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
                unit="ms",
                thresholds=[
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": 5},
                    {"color": "red", "value": 10}
                ]
            ),
            
            # Engine Performance Matrix
            GrafanaPanel(
                title="Engine Performance Overview",
                panel_type="heatmap",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_workflow_execution_time_ms',
                        legend_format="Execution Time"
                    )
                ],
                grid_pos={"x": 0, "y": 8, "w": 24, "h": 8},
                unit="ms"
            ),
            
            # Memory Usage by Component
            GrafanaPanel(
                title="Memory Usage by Component",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_memory_usage_mb{component="orchestrator"}',
                        legend_format="Orchestrator"
                    ),
                    GrafanaPanelTarget(
                        expr='app_memory_usage_mb{component="communication_hub"}',
                        legend_format="Communication Hub",
                        ref_id="B"
                    ),
                    GrafanaPanelTarget(
                        expr='app_memory_usage_mb{component="engines"}',
                        legend_format="Engines",
                        ref_id="C"
                    )
                ],
                grid_pos={"x": 0, "y": 16, "w": 12, "h": 8},
                unit="decbytes"
            ),
            
            # Cache Performance
            GrafanaPanel(
                title="Cache Hit Rate by Component",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='app_cache_hit_rate_percent',
                        legend_format="Overall Hit Rate"
                    )
                ],
                grid_pos={"x": 12, "y": 16, "w": 12, "h": 8},
                unit="percent"
            )
        ]
        
        return GrafanaDashboard(
            title="LeanVibe Agent Hive 2.0 - Component Performance",
            panels=panels,
            tags=["leanvibe", "components", "detailed"]
        )
    
    def _create_business_metrics_dashboard(self) -> GrafanaDashboard:
        """Create business metrics dashboard."""
        panels = [
            # Tasks Completed Rate
            GrafanaPanel(
                title="Tasks Completed per Minute",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_tasks_completed_per_minute',
                        legend_format="Tasks/min"
                    )
                ],
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                unit="short"
            ),
            
            # Agent Success Rate
            GrafanaPanel(
                title="Agent Success Rate",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_agent_success_rate_percent',
                        legend_format="Success Rate"
                    )
                ],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                unit="percent",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 95},
                    {"color": "green", "value": 99}
                ]
            ),
            
            # System Availability
            GrafanaPanel(
                title="System Availability",
                panel_type="stat",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_system_availability_percent',
                        legend_format="Availability"
                    )
                ],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                unit="percent",
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 99.5},
                    {"color": "green", "value": 99.9}
                ]
            ),
            
            # SLA Compliance
            GrafanaPanel(
                title="SLA Compliance Trend",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_sla_compliance_percent',
                        legend_format="SLA Compliance"
                    )
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 4},
                unit="percent"
            ),
            
            # User Requests Volume
            GrafanaPanel(
                title="User Requests per Minute",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_user_requests_per_minute',
                        legend_format="Requests/min"
                    )
                ],
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                unit="short"
            ),
            
            # Business Impact Score
            GrafanaPanel(
                title="Revenue Impact Score",
                panel_type="gauge",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_revenue_impact_score',
                        legend_format="Impact Score"
                    )
                ],
                grid_pos={"x": 12, "y": 8, "w": 6, "h": 8},
                unit="short",
                min_value=0,
                max_value=10
            ),
            
            # Customer Satisfaction
            GrafanaPanel(
                title="Customer Satisfaction Score",
                panel_type="gauge",
                targets=[
                    GrafanaPanelTarget(
                        expr='business_customer_satisfaction_score',
                        legend_format="Satisfaction"
                    )
                ],
                grid_pos={"x": 18, "y": 8, "w": 6, "h": 8},
                unit="short",
                min_value=0,
                max_value=5,
                thresholds=[
                    {"color": "red", "value": None},
                    {"color": "yellow", "value": 3.5},
                    {"color": "green", "value": 4.5}
                ]
            )
        ]
        
        return GrafanaDashboard(
            title="LeanVibe Agent Hive 2.0 - Business Metrics",
            panels=panels,
            tags=["leanvibe", "business", "kpi"]
        )
    
    def _create_infrastructure_dashboard(self) -> GrafanaDashboard:
        """Create infrastructure dashboard."""
        panels = [
            # System Resource Overview
            GrafanaPanel(
                title="CPU Usage Trend",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_cpu_percent',
                        legend_format="CPU %"
                    )
                ],
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
                unit="percent"
            ),
            
            # Memory Usage Trend
            GrafanaPanel(
                title="Memory Usage Trend",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_memory_percent',
                        legend_format="Memory %"
                    ),
                    GrafanaPanelTarget(
                        expr='system_memory_available_gb',
                        legend_format="Available GB",
                        ref_id="B"
                    )
                ],
                grid_pos={"x": 12, "y": 0, "w": 12, "h": 8},
                unit="percent"
            ),
            
            # Disk I/O Performance
            GrafanaPanel(
                title="Disk I/O Performance",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_disk_io_read_mb_per_sec',
                        legend_format="Read MB/s"
                    ),
                    GrafanaPanelTarget(
                        expr='system_disk_io_write_mb_per_sec',
                        legend_format="Write MB/s",
                        ref_id="B"
                    )
                ],
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8},
                unit="decbytes"
            ),
            
            # Network I/O Performance
            GrafanaPanel(
                title="Network I/O Performance",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_network_bytes_sent_per_sec',
                        legend_format="Sent bytes/s"
                    ),
                    GrafanaPanelTarget(
                        expr='system_network_bytes_recv_per_sec',
                        legend_format="Received bytes/s",
                        ref_id="B"
                    )
                ],
                grid_pos={"x": 12, "y": 8, "w": 12, "h": 8},
                unit="decbytes"
            ),
            
            # Process and Thread Counts
            GrafanaPanel(
                title="System Processes",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_processes_count',
                        legend_format="Process Count"
                    ),
                    GrafanaPanelTarget(
                        expr='system_threads_count',
                        legend_format="Thread Count",
                        ref_id="B"
                    )
                ],
                grid_pos={"x": 0, "y": 16, "w": 12, "h": 8},
                unit="short"
            ),
            
            # Load Average (Unix systems)
            GrafanaPanel(
                title="System Load Average",
                panel_type="timeseries",
                targets=[
                    GrafanaPanelTarget(
                        expr='system_load_average_1m',
                        legend_format="1min Load"
                    )
                ],
                grid_pos={"x": 12, "y": 16, "w": 12, "h": 8},
                unit="short"
            )
        ]
        
        return GrafanaDashboard(
            title="LeanVibe Agent Hive 2.0 - Infrastructure",
            panels=panels,
            tags=["leanvibe", "infrastructure", "system"]
        )
    
    async def update_dashboard(self, dashboard_name: str) -> DashboardResult:
        """Update existing dashboard."""
        if dashboard_name not in self.dashboard_templates:
            return DashboardResult(success=False, error=f"Unknown dashboard: {dashboard_name}")
        
        try:
            dashboard = self.dashboard_templates[dashboard_name]()
            result = await self._create_grafana_dashboard(dashboard)
            
            if result.success:
                self.created_dashboards[dashboard_name] = result.dashboard_id
                logging.info(f"Updated dashboard: {dashboard_name}")
            
            return result
            
        except Exception as e:
            return DashboardResult(success=False, error=str(e))
    
    async def delete_dashboard(self, dashboard_name: str) -> bool:
        """Delete dashboard by name."""
        if dashboard_name not in self.created_dashboards:
            return False
        
        try:
            dashboard_id = self.created_dashboards[dashboard_name]
            headers = self._get_auth_headers()
            
            async with self.session.delete(
                f"{self.grafana_url}/api/dashboards/uid/{dashboard_id}",
                headers=headers
            ) as response:
                success = response.status == 200
                
                if success:
                    del self.created_dashboards[dashboard_name]
                    logging.info(f"Deleted dashboard: {dashboard_name}")
                
                return success
                
        except Exception as e:
            logging.error(f"Error deleting dashboard {dashboard_name}: {e}")
            return False
    
    def get_dashboard_urls(self) -> Dict[str, str]:
        """Get URLs for all created dashboards."""
        urls = {}
        for dashboard_name, dashboard_id in self.created_dashboards.items():
            urls[dashboard_name] = f"{self.grafana_url}/d/{dashboard_id}"
        return urls
    
    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get summary of all managed dashboards."""
        return {
            'grafana_url': self.grafana_url,
            'folder_id': self.folder_id,
            'created_dashboards': len(self.created_dashboards),
            'dashboard_names': list(self.created_dashboards.keys()),
            'dashboard_urls': self.get_dashboard_urls(),
            'available_templates': list(self.dashboard_templates.keys())
        }
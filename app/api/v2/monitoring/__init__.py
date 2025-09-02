"""
SystemMonitoringAPI v2 - Unified Monitoring and Observability

Consolidated monitoring module implementing Phase 2 of Epic 4 API Architecture Consolidation.
Unifies 9 monitoring modules into a single, high-performance monitoring API.

Consolidated Modules:
- dashboard_monitoring.py -> Core monitoring capabilities
- observability.py -> Real-time observability and event processing
- performance_intelligence.py -> Performance analytics and intelligence  
- monitoring_reporting.py -> Monitoring reports and analytics
- business_analytics.py -> Business intelligence and metrics
- dashboard_prometheus.py -> Prometheus metrics endpoints
- strategic_monitoring.py -> Strategic market analytics
- mobile_monitoring.py -> Mobile-responsive monitoring
- observability_hooks.py -> Real-time event capture and streaming

Key Features:
- <200ms response times with intelligent caching
- Real-time WebSocket streaming with <50ms latency
- Prometheus-compatible metrics export
- Mobile-responsive interfaces with QR code access
- Strategic business intelligence and market analytics
- Comprehensive observability hooks and event processing
- OAuth2 + RBAC security integration
- Full backwards compatibility with v1 endpoints
"""

from .core import router as monitoring_router
from .models import *
from .middleware import *
from .utils import *

__all__ = [
    "monitoring_router",
    "MonitoringResponse", 
    "MetricsResponse",
    "PerformanceStats",
    "BusinessMetrics",
    "StrategicIntelligence",
    "ObservabilityEvent"
]
"""
Example Plugins for LeanVibe Plugin SDK.

Demonstrates SDK capabilities and best practices through working examples.

This module provides comprehensive examples showcasing different plugin types:

1. DataPipelinePlugin (workflow_example.py):
   - Advanced data processing pipeline with multi-step operations
   - Demonstrates filtering, transformation, aggregation, validation, enrichment, deduplication
   - Epic 1 performance optimizations with batch processing and lazy loading
   - Comprehensive error handling and metrics tracking

2. SystemMonitorPlugin (monitoring_example.py):
   - Real-time system metrics collection and monitoring
   - Configurable alerting thresholds and automated alert management
   - Performance trend analysis and resource usage optimization
   - Epic 1 optimized metrics collection (<5ms per sample)

3. SecurityScannerPlugin (security_example.py):
   - Security vulnerability scanning for code, dependencies, and secrets
   - Pattern-based detection with comprehensive remediation suggestions
   - Efficient scanning algorithms with incremental capabilities
   - Security compliance reporting and threat pattern detection

4. WebhookIntegrationPlugin (integration_example.py):
   - External service integration with webhook delivery and API communication
   - Rate limiting, circuit breaker patterns, and retry mechanisms
   - HTTP connection pooling and async delivery optimization
   - Event transformation and integration monitoring

Each example demonstrates Epic 1 compliance (<50ms response, <80MB memory) and
showcases SDK best practices for different use cases.
"""

from .workflow_example import DataPipelinePlugin
from .monitoring_example import SystemMonitorPlugin
from .security_example import SecurityScannerPlugin
from .integration_example import WebhookIntegrationPlugin

__all__ = [
    "DataPipelinePlugin",
    "SystemMonitorPlugin", 
    "SecurityScannerPlugin",
    "WebhookIntegrationPlugin"
]

# Example configurations for quick start
EXAMPLE_CONFIGS = {
    "data_pipeline": {
        "name": "DataPipelineExample",
        "version": "1.0.0",
        "description": "Example data processing pipeline",
        "parameters": {
            "batch_size": 1000,
            "max_retries": 3,
            "output_format": "json",
            "enable_validation": True,
            "pipeline_steps": [
                {
                    "name": "filter_step",
                    "type": "filter",
                    "parameters": {
                        "field": "status",
                        "operation": "equals",
                        "value": "active"
                    }
                },
                {
                    "name": "transform_step", 
                    "type": "transform",
                    "parameters": {
                        "transformations": [
                            {"field": "name", "operation": "uppercase"},
                            {"field": "timestamp", "operation": "timestamp"}
                        ]
                    }
                }
            ]
        }
    },
    
    "system_monitor": {
        "name": "SystemMonitorExample",
        "version": "1.0.0", 
        "description": "Example system monitoring plugin",
        "parameters": {
            "collection_interval": 10,
            "retention_hours": 24,
            "enable_alerts": True,
            "thresholds": [
                {
                    "metric_name": "cpu_percent",
                    "warning_threshold": 70.0,
                    "critical_threshold": 90.0
                },
                {
                    "metric_name": "memory_percent",
                    "warning_threshold": 80.0, 
                    "critical_threshold": 95.0
                }
            ]
        }
    },
    
    "security_scanner": {
        "name": "SecurityScannerExample",
        "version": "1.0.0",
        "description": "Example security scanning plugin", 
        "parameters": {
            "dependency_check_enabled": True,
            "max_file_size_mb": 10,
            "excluded_paths": [".git", "node_modules", "__pycache__"]
        }
    },
    
    "webhook_integration": {
        "name": "WebhookIntegrationExample",
        "version": "1.0.0",
        "description": "Example webhook integration plugin",
        "parameters": {
            "max_concurrent_deliveries": 10,
            "delivery_timeout_seconds": 30,
            "webhooks": [
                {
                    "webhook_id": "example_webhook",
                    "url": "https://example.com/webhook",
                    "timeout_seconds": 30,
                    "retry_attempts": 3
                }
            ],
            "api_endpoints": [
                {
                    "endpoint_id": "example_api",
                    "base_url": "https://api.example.com",
                    "auth_type": "api_key",
                    "auth_config": {"api_key": "your_api_key"}
                }
            ]
        }
    }
}
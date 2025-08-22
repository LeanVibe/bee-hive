# LeanVibe Plugin SDK Integration Tutorials

**Version:** 2.3.0  
**Epic 2 Phase 2.3: Developer SDK & Documentation**

Comprehensive step-by-step tutorials for integrating different types of plugins using the LeanVibe Plugin SDK.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Processing Pipeline Tutorial](#data-processing-pipeline-tutorial)
3. [System Monitoring Integration Tutorial](#system-monitoring-integration-tutorial)
4. [Security Scanner Integration Tutorial](#security-scanner-integration-tutorial)
5. [Webhook Integration Tutorial](#webhook-integration-tutorial)
6. [Custom Plugin Development Tutorial](#custom-plugin-development-tutorial)
7. [Advanced Integration Patterns](#advanced-integration-patterns)
8. [Troubleshooting and Common Issues](#troubleshooting-and-common-issues)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- LeanVibe Agent Hive 2.0 environment
- Basic understanding of async/await programming
- Access to LeanVibe Plugin SDK

### SDK Installation and Setup

1. **Install the SDK components:**

```python
# Import the core SDK components
from leanvibe.plugin_sdk import (
    PluginBase, WorkflowPlugin, MonitoringPlugin, SecurityPlugin,
    PluginConfig, TaskInterface, TaskResult,
    plugin_method, performance_tracked, error_handled, cached_result,
    PluginGenerator, PluginTestFramework
)
```

2. **Set up your development environment:**

```python
from leanvibe.plugin_sdk.tools import PluginGenerator

# Initialize the plugin generator
generator = PluginGenerator()

# Create a development workspace
workspace_path = generator.create_development_workspace(
    workspace_name="my_plugins",
    output_dir="./development"
)

print(f"Development workspace created at: {workspace_path}")
```

3. **Validate SDK installation:**

```python
from leanvibe.plugin_sdk.testing import PluginTestFramework

# Create test framework
test_framework = PluginTestFramework()

# Run SDK validation
validation_result = await test_framework.validate_sdk_installation()
print(f"SDK validation: {validation_result}")
```

## Data Processing Pipeline Tutorial

### Scenario: Customer Data Processing Pipeline

This tutorial demonstrates building a data processing pipeline for customer data analysis.

#### Step 1: Create the Pipeline Plugin

```python
from leanvibe.plugin_sdk import WorkflowPlugin, PluginConfig, TaskInterface, TaskResult
from leanvibe.plugin_sdk.decorators import plugin_method, performance_tracked
from typing import Dict, List, Any
import json

class CustomerDataPipelinePlugin(WorkflowPlugin):
    """
    Customer data processing pipeline plugin.
    
    Processes customer data through multiple transformation steps:
    1. Data validation and cleansing
    2. Customer segmentation
    3. Enrichment with external data
    4. Analytics and reporting
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Configuration
        self.batch_size = config.parameters.get("batch_size", 1000)
        self.enable_validation = config.parameters.get("enable_validation", True)
        self.segmentation_rules = config.parameters.get("segmentation_rules", {})
        
    async def _on_initialize(self) -> None:
        """Initialize the customer data pipeline."""
        await self.log_info("Initializing CustomerDataPipelinePlugin")
        
        # Validate configuration
        if not self.segmentation_rules:
            self.segmentation_rules = {
                "high_value": {"min_purchase_amount": 1000, "min_frequency": 10},
                "medium_value": {"min_purchase_amount": 500, "min_frequency": 5},
                "low_value": {"min_purchase_amount": 0, "min_frequency": 1}
            }
        
        await self.log_info(f"Initialized with batch_size={self.batch_size}")
    
    @performance_tracked(alert_threshold_ms=2000, memory_limit_mb=50)
    @plugin_method(timeout_seconds=300, max_retries=2)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle customer data processing tasks."""
        task_type = task.task_type
        
        if task_type == "process_customers":
            return await self._process_customer_data(task)
        elif task_type == "segment_customers":
            return await self._segment_customers(task)
        elif task_type == "generate_report":
            return await self._generate_customer_report(task)
        else:
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=f"Unknown task type: {task_type}",
                error_code="INVALID_TASK_TYPE"
            )
    
    async def _process_customer_data(self, task: TaskInterface) -> TaskResult:
        """Process customer data through the pipeline."""
        try:
            customer_data = task.parameters.get("customer_data", [])
            
            await task.update_status("running", progress=0.1)
            
            # Step 1: Validate and cleanse data
            validated_data = await self._validate_customer_data(customer_data)
            await task.update_status("running", progress=0.3)
            
            # Step 2: Enrich with external data
            enriched_data = await self._enrich_customer_data(validated_data)
            await task.update_status("running", progress=0.6)
            
            # Step 3: Apply business rules
            processed_data = await self._apply_business_rules(enriched_data)
            await task.update_status("running", progress=0.9)
            
            # Step 4: Generate output
            output_path = await self._save_processed_data(processed_data, task.task_id)
            
            await task.update_status("completed", progress=1.0)
            
            return TaskResult(
                success=True,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                data={
                    "processed_customers": len(processed_data),
                    "output_path": output_path,
                    "processing_summary": {
                        "input_records": len(customer_data),
                        "validated_records": len(validated_data),
                        "enriched_records": len(enriched_data),
                        "final_records": len(processed_data)
                    }
                }
            )
            
        except Exception as e:
            await self.log_error(f"Customer data processing failed: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="PROCESSING_FAILED"
            )
    
    async def _validate_customer_data(self, customer_data: List[Dict]) -> List[Dict]:
        """Validate and cleanse customer data."""
        validated_data = []
        
        for customer in customer_data:
            # Check required fields
            if all(field in customer for field in ["customer_id", "email", "name"]):
                # Cleanse data
                customer["email"] = customer["email"].lower().strip()
                customer["name"] = customer["name"].strip()
                
                # Validate email format
                if "@" in customer["email"] and "." in customer["email"]:
                    validated_data.append(customer)
        
        return validated_data
    
    async def _enrich_customer_data(self, customer_data: List[Dict]) -> List[Dict]:
        """Enrich customer data with additional information."""
        enriched_data = []
        
        for customer in customer_data:
            # Add calculated fields
            purchase_history = customer.get("purchases", [])
            
            customer["total_purchases"] = len(purchase_history)
            customer["total_amount"] = sum(p.get("amount", 0) for p in purchase_history)
            customer["average_purchase"] = (
                customer["total_amount"] / customer["total_purchases"] 
                if customer["total_purchases"] > 0 else 0
            )
            
            # Add lifecycle stage
            if customer["total_purchases"] == 0:
                customer["lifecycle_stage"] = "prospect"
            elif customer["total_purchases"] < 3:
                customer["lifecycle_stage"] = "new_customer"
            elif customer["total_amount"] > 1000:
                customer["lifecycle_stage"] = "vip"
            else:
                customer["lifecycle_stage"] = "regular"
            
            enriched_data.append(customer)
        
        return enriched_data
    
    async def _apply_business_rules(self, customer_data: List[Dict]) -> List[Dict]:
        """Apply business rules and segmentation."""
        processed_data = []
        
        for customer in customer_data:
            # Apply segmentation
            customer_segment = self._determine_customer_segment(customer)
            customer["segment"] = customer_segment
            
            # Calculate customer score
            customer["customer_score"] = self._calculate_customer_score(customer)
            
            # Add recommendations
            customer["recommendations"] = self._generate_recommendations(customer)
            
            processed_data.append(customer)
        
        return processed_data
    
    def _determine_customer_segment(self, customer: Dict) -> str:
        """Determine customer segment based on rules."""
        total_amount = customer.get("total_amount", 0)
        total_purchases = customer.get("total_purchases", 0)
        
        for segment, rules in self.segmentation_rules.items():
            if (total_amount >= rules["min_purchase_amount"] and 
                total_purchases >= rules["min_frequency"]):
                return segment
        
        return "prospect"
    
    def _calculate_customer_score(self, customer: Dict) -> float:
        """Calculate customer value score."""
        score = 0.0
        
        # Purchase amount contribution (40%)
        score += min(customer.get("total_amount", 0) / 1000, 1.0) * 40
        
        # Purchase frequency contribution (30%)
        score += min(customer.get("total_purchases", 0) / 20, 1.0) * 30
        
        # Recency contribution (20%)
        # Simplified - in real implementation, would use actual dates
        score += 20
        
        # Engagement contribution (10%)
        score += 10
        
        return round(score, 2)
    
    def _generate_recommendations(self, customer: Dict) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        segment = customer.get("segment", "prospect")
        
        if segment == "high_value":
            recommendations.extend([
                "Offer VIP program benefits",
                "Personal account manager assignment",
                "Exclusive product previews"
            ])
        elif segment == "medium_value":
            recommendations.extend([
                "Loyalty program enrollment",
                "Cross-selling opportunities",
                "Seasonal promotions"
            ])
        elif segment == "low_value":
            recommendations.extend([
                "Welcome series emails",
                "First purchase incentives",
                "Product education content"
            ])
        
        return recommendations
    
    async def _save_processed_data(self, data: List[Dict], task_id: str) -> str:
        """Save processed data to output file."""
        output_path = f"/tmp/customer_pipeline_output_{task_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
```

#### Step 2: Configure and Use the Pipeline

```python
# Create plugin configuration
config = PluginConfig(
    name="CustomerDataPipeline",
    version="1.0.0",
    description="Customer data processing pipeline",
    parameters={
        "batch_size": 500,
        "enable_validation": True,
        "segmentation_rules": {
            "vip": {"min_purchase_amount": 2000, "min_frequency": 15},
            "high_value": {"min_purchase_amount": 1000, "min_frequency": 10},
            "medium_value": {"min_purchase_amount": 500, "min_frequency": 5},
            "low_value": {"min_purchase_amount": 0, "min_frequency": 1}
        }
    }
)

# Initialize plugin
pipeline_plugin = CustomerDataPipelinePlugin(config)
await pipeline_plugin.initialize()

# Sample customer data
customer_data = [
    {
        "customer_id": "CUST001",
        "name": "John Doe",
        "email": "JOHN.DOE@EXAMPLE.COM",
        "purchases": [
            {"amount": 150, "date": "2023-01-15"},
            {"amount": 300, "date": "2023-02-20"},
            {"amount": 450, "date": "2023-03-10"}
        ]
    },
    {
        "customer_id": "CUST002", 
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "purchases": [
            {"amount": 1200, "date": "2023-01-10"},
            {"amount": 800, "date": "2023-01-25"},
            {"amount": 1500, "date": "2023-02-15"}
        ]
    }
]

# Create processing task
task = TaskInterface(
    task_id="process_001",
    task_type="process_customers",
    parameters={"customer_data": customer_data}
)

# Execute pipeline
result = await pipeline_plugin.handle_task(task)

print(f"Processing result: {result.success}")
print(f"Processed customers: {result.data['processed_customers']}")
print(f"Output saved to: {result.data['output_path']}")
```

#### Step 3: Test the Pipeline

```python
from leanvibe.plugin_sdk.testing import PluginTestFramework

# Create test framework
test_framework = PluginTestFramework()

# Register the plugin for testing
test_framework.register_plugin("customer_pipeline", CustomerDataPipelinePlugin)

# Run comprehensive tests
async def test_customer_pipeline():
    # Test data validation
    validation_result = await test_framework.test_plugin_task(
        plugin_id="customer_pipeline",
        task_type="process_customers",
        parameters={"customer_data": customer_data},
        expected_success=True
    )
    
    # Test performance compliance
    performance_result = await test_framework.test_plugin_performance(
        plugin_id="customer_pipeline",
        task_type="process_customers", 
        parameters={"customer_data": customer_data * 100},  # Larger dataset
        max_execution_time_ms=5000,
        max_memory_mb=80
    )
    
    # Test error handling
    error_result = await test_framework.test_plugin_error_handling(
        plugin_id="customer_pipeline",
        task_type="process_customers",
        parameters={"customer_data": "invalid_data"},  # Invalid input
        expected_error_code="PROCESSING_FAILED"
    )
    
    print(f"Validation test: {validation_result.passed}")
    print(f"Performance test: {performance_result.passed}")
    print(f"Error handling test: {error_result.passed}")

# Run the tests
await test_customer_pipeline()
```

## System Monitoring Integration Tutorial

### Scenario: Application Performance Monitoring

This tutorial shows how to integrate system monitoring capabilities for application performance tracking.

#### Step 1: Create the Monitoring Plugin

```python
from leanvibe.plugin_sdk import MonitoringPlugin, PluginConfig
from leanvibe.plugin_sdk.examples import SystemMonitorPlugin

class ApplicationMonitorPlugin(SystemMonitorPlugin):
    """
    Extended system monitoring plugin for application-specific metrics.
    
    Adds application-level monitoring on top of system metrics:
    - Application response times
    - Database connection pool status
    - Cache hit rates
    - Custom business metrics
    """
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        
        # Application-specific configuration
        self.app_endpoints = config.parameters.get("app_endpoints", [])
        self.db_connections = config.parameters.get("db_connections", [])
        self.cache_instances = config.parameters.get("cache_instances", [])
        
    async def _collect_system_metrics(self):
        """Override to add application metrics."""
        # Get base system metrics
        base_metrics = await super()._collect_system_metrics()
        
        # Add application metrics
        app_metrics = await self._collect_application_metrics()
        
        # Combine metrics
        combined_metrics = base_metrics.to_dict()
        combined_metrics.update(app_metrics)
        
        return combined_metrics
    
    async def _collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        metrics = {}
        
        # Collect endpoint response times
        if self.app_endpoints:
            endpoint_metrics = await self._collect_endpoint_metrics()
            metrics["endpoint_metrics"] = endpoint_metrics
        
        # Collect database metrics
        if self.db_connections:
            db_metrics = await self._collect_database_metrics()
            metrics["database_metrics"] = db_metrics
        
        # Collect cache metrics
        if self.cache_instances:
            cache_metrics = await self._collect_cache_metrics()
            metrics["cache_metrics"] = cache_metrics
        
        return metrics
    
    async def _collect_endpoint_metrics(self) -> Dict[str, Any]:
        """Collect endpoint performance metrics."""
        endpoint_metrics = {}
        
        for endpoint in self.app_endpoints:
            endpoint_id = endpoint["endpoint_id"]
            url = endpoint["url"]
            
            try:
                # Measure response time
                start_time = time.time()
                
                async with self._http_session.get(url, timeout=5) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    endpoint_metrics[endpoint_id] = {
                        "response_time_ms": response_time,
                        "status_code": response.status,
                        "available": response.status < 400
                    }
                    
            except Exception as e:
                endpoint_metrics[endpoint_id] = {
                    "response_time_ms": None,
                    "status_code": None,
                    "available": False,
                    "error": str(e)
                }
        
        return endpoint_metrics
    
    async def _collect_database_metrics(self) -> Dict[str, Any]:
        """Collect database performance metrics."""
        db_metrics = {}
        
        for db_config in self.db_connections:
            db_id = db_config["db_id"]
            
            # Mock database metrics - in real implementation,
            # would connect to actual database and collect metrics
            db_metrics[db_id] = {
                "connection_pool_size": 10,
                "active_connections": 3,
                "idle_connections": 7,
                "query_latency_ms": 45.2,
                "queries_per_second": 150,
                "connection_errors": 0
            }
        
        return db_metrics
    
    async def _collect_cache_metrics(self) -> Dict[str, Any]:
        """Collect cache performance metrics."""
        cache_metrics = {}
        
        for cache_config in self.cache_instances:
            cache_id = cache_config["cache_id"]
            
            # Mock cache metrics - in real implementation,
            # would connect to actual cache and collect metrics
            cache_metrics[cache_id] = {
                "hit_rate": 0.85,
                "miss_rate": 0.15,
                "keys_count": 15420,
                "memory_usage_mb": 256.7,
                "evictions": 125,
                "operations_per_second": 8500
            }
        
        return cache_metrics
```

#### Step 2: Configure Application Monitoring

```python
# Configure application monitoring
monitoring_config = PluginConfig(
    name="ApplicationMonitor",
    version="1.0.0", 
    description="Application performance monitoring",
    parameters={
        "collection_interval": 30,
        "retention_hours": 48,
        "enable_alerts": True,
        "app_endpoints": [
            {
                "endpoint_id": "api_health",
                "url": "http://localhost:8000/health"
            },
            {
                "endpoint_id": "api_metrics", 
                "url": "http://localhost:8000/metrics"
            }
        ],
        "db_connections": [
            {
                "db_id": "primary_db",
                "connection_string": "postgresql://user:pass@localhost:5432/app"
            }
        ],
        "cache_instances": [
            {
                "cache_id": "redis_cache",
                "connection_string": "redis://localhost:6379/0"
            }
        ],
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
            },
            {
                "metric_name": "endpoint_response_time",
                "warning_threshold": 1000.0,  # 1 second
                "critical_threshold": 3000.0   # 3 seconds
            }
        ]
    }
)

# Initialize monitoring plugin
app_monitor = ApplicationMonitorPlugin(monitoring_config)
await app_monitor.initialize()
```

#### Step 3: Start Monitoring and Handle Alerts

```python
# Start monitoring
start_task = TaskInterface(
    task_id="start_monitoring_001",
    task_type="start_monitoring",
    parameters={}
)

monitoring_result = await app_monitor.handle_task(start_task)
print(f"Monitoring started: {monitoring_result.success}")

# Set up alert handling
async def handle_monitoring_alerts():
    """Handle monitoring alerts as they occur."""
    while True:
        try:
            # Check for recent alerts
            stats_task = TaskInterface(
                task_id="get_stats_001",
                task_type="get_status",
                parameters={}
            )
            
            stats_result = await app_monitor.handle_task(stats_task)
            
            if stats_result.success:
                latest_metrics = stats_result.data.get("latest_metrics")
                
                if latest_metrics:
                    # Check application-specific thresholds
                    await check_application_thresholds(latest_metrics)
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Alert handling error: {e}")
            await asyncio.sleep(60)

async def check_application_thresholds(metrics: Dict[str, Any]):
    """Check application-specific alert thresholds."""
    
    # Check endpoint response times
    endpoint_metrics = metrics.get("endpoint_metrics", {})
    for endpoint_id, endpoint_data in endpoint_metrics.items():
        response_time = endpoint_data.get("response_time_ms")
        available = endpoint_data.get("available", False)
        
        if not available:
            print(f"ALERT: Endpoint {endpoint_id} is not available!")
        elif response_time and response_time > 3000:
            print(f"ALERT: Endpoint {endpoint_id} response time is {response_time}ms (critical)")
        elif response_time and response_time > 1000:
            print(f"WARNING: Endpoint {endpoint_id} response time is {response_time}ms")
    
    # Check database metrics
    db_metrics = metrics.get("database_metrics", {})
    for db_id, db_data in db_metrics.items():
        connection_errors = db_data.get("connection_errors", 0)
        query_latency = db_data.get("query_latency_ms", 0)
        
        if connection_errors > 0:
            print(f"ALERT: Database {db_id} has {connection_errors} connection errors!")
        elif query_latency > 100:
            print(f"WARNING: Database {db_id} query latency is {query_latency}ms")
    
    # Check cache metrics
    cache_metrics = metrics.get("cache_metrics", {})
    for cache_id, cache_data in cache_metrics.items():
        hit_rate = cache_data.get("hit_rate", 1.0)
        
        if hit_rate < 0.5:
            print(f"ALERT: Cache {cache_id} hit rate is {hit_rate:.2%} (critical)")
        elif hit_rate < 0.8:
            print(f"WARNING: Cache {cache_id} hit rate is {hit_rate:.2%}")

# Start alert handling in background
asyncio.create_task(handle_monitoring_alerts())
```

## Security Scanner Integration Tutorial

### Scenario: Automated Security Scanning in CI/CD Pipeline

This tutorial demonstrates integrating security scanning into a development workflow.

#### Step 1: Configure Security Scanner

```python
from leanvibe.plugin_sdk.examples import SecurityScannerPlugin

# Configure security scanner
security_config = PluginConfig(
    name="CIPipelineSecurityScanner",
    version="1.0.0",
    description="Security scanner for CI/CD pipeline",
    parameters={
        "dependency_check_enabled": True,
        "max_file_size_mb": 50,
        "excluded_paths": [
            ".git", ".venv", "node_modules", "__pycache__",
            "*.min.js", "*.min.css", "dist/", "build/"
        ],
        "scan_schedule": {
            "code_scan": "on_commit",
            "dependency_scan": "daily",
            "secret_scan": "on_commit"
        },
        "severity_thresholds": {
            "block_critical": True,
            "block_high": True,
            "warn_medium": True,
            "ignore_low": False
        }
    }
)

# Initialize security scanner
security_scanner = SecurityScannerPlugin(security_config)
await security_scanner.initialize()
```

#### Step 2: Implement CI/CD Integration

```python
import subprocess
import sys

class CIPipelineIntegration:
    """CI/CD pipeline integration for security scanning."""
    
    def __init__(self, security_scanner: SecurityScannerPlugin):
        self.security_scanner = security_scanner
        self.scan_results = {}
    
    async def run_security_checks(self, project_path: str) -> Dict[str, Any]:
        """Run comprehensive security checks for CI/CD pipeline."""
        
        print("🔒 Starting security scans...")
        
        # 1. Code vulnerability scan
        print("📝 Scanning code for vulnerabilities...")
        code_scan_result = await self._run_code_scan(project_path)
        
        # 2. Dependency scan
        print("📦 Scanning dependencies...")
        dependency_scan_result = await self._run_dependency_scan(project_path)
        
        # 3. Secret scan
        print("🔐 Scanning for hardcoded secrets...")
        secret_scan_result = await self._run_secret_scan(project_path)
        
        # 4. Configuration scan
        print("⚙️ Scanning configuration files...")
        config_scan_result = await self._run_config_scan(project_path)
        
        # Aggregate results
        aggregated_results = {
            "code_scan": code_scan_result,
            "dependency_scan": dependency_scan_result,
            "secret_scan": secret_scan_result,
            "config_scan": config_scan_result,
            "overall_status": self._determine_overall_status([
                code_scan_result, dependency_scan_result, 
                secret_scan_result, config_scan_result
            ])
        }
        
        # Generate security report
        report = await self._generate_security_report(aggregated_results)
        
        return {
            "scan_results": aggregated_results,
            "security_report": report,
            "pipeline_decision": self._make_pipeline_decision(aggregated_results)
        }
    
    async def _run_code_scan(self, project_path: str) -> Dict[str, Any]:
        """Run code vulnerability scan."""
        task = TaskInterface(
            task_id=f"code_scan_{uuid.uuid4().hex[:8]}",
            task_type="scan_code",
            parameters={
                "target_path": project_path,
                "file_extensions": [".py", ".js", ".php", ".java", ".rb", ".go"]
            }
        )
        
        result = await self.security_scanner.handle_task(task)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    async def _run_dependency_scan(self, project_path: str) -> Dict[str, Any]:
        """Run dependency vulnerability scan."""
        task = TaskInterface(
            task_id=f"dep_scan_{uuid.uuid4().hex[:8]}",
            task_type="scan_dependencies", 
            parameters={
                "project_path": project_path,
                "package_files": [
                    "requirements.txt", "package.json", "Pipfile",
                    "pom.xml", "Gemfile", "go.mod"
                ]
            }
        )
        
        result = await self.security_scanner.handle_task(task)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    async def _run_secret_scan(self, project_path: str) -> Dict[str, Any]:
        """Run secret detection scan."""
        task = TaskInterface(
            task_id=f"secret_scan_{uuid.uuid4().hex[:8]}",
            task_type="scan_secrets",
            parameters={
                "target_path": project_path,
                "file_extensions": [
                    ".py", ".js", ".php", ".java", ".rb", ".go",
                    ".env", ".yml", ".yaml", ".json", ".xml", ".config"
                ]
            }
        )
        
        result = await self.security_scanner.handle_task(task)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    async def _run_config_scan(self, project_path: str) -> Dict[str, Any]:
        """Run configuration security scan."""
        task = TaskInterface(
            task_id=f"config_scan_{uuid.uuid4().hex[:8]}",
            task_type="scan_config",
            parameters={"project_path": project_path}
        )
        
        result = await self.security_scanner.handle_task(task)
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error
        }
    
    def _determine_overall_status(self, scan_results: List[Dict]) -> str:
        """Determine overall security status."""
        failed_scans = [r for r in scan_results if not r["success"]]
        
        if failed_scans:
            return "failed"
        
        # Check for critical/high severity vulnerabilities
        for scan in scan_results:
            if scan.get("data", {}).get("vulnerabilities_found", 0) > 0:
                vulnerabilities = scan["data"].get("vulnerabilities", [])
                critical_count = len([v for v in vulnerabilities if v.get("severity") == "critical"])
                high_count = len([v for v in vulnerabilities if v.get("severity") == "high"])
                
                if critical_count > 0:
                    return "critical_vulnerabilities"
                elif high_count > 0:
                    return "high_vulnerabilities"
        
        return "passed"
    
    async def _generate_security_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        report_task = TaskInterface(
            task_id=f"report_{uuid.uuid4().hex[:8]}",
            task_type="get_vulnerability_report",
            parameters={}
        )
        
        report_result = await self.security_scanner.handle_task(report_task)
        
        if report_result.success:
            base_report = report_result.data["vulnerability_report"]
        else:
            base_report = {"error": "Failed to generate base report"}
        
        # Add CI/CD specific information
        ci_report = {
            "ci_scan_summary": {
                "scan_timestamp": datetime.utcnow().isoformat(),
                "scans_performed": len([r for r in results.values() if isinstance(r, dict)]),
                "total_vulnerabilities": sum(
                    r.get("data", {}).get("vulnerabilities_found", 0) 
                    for r in results.values() 
                    if isinstance(r, dict) and r.get("data")
                ),
                "overall_status": results["overall_status"]
            },
            "scan_breakdown": {
                "code_vulnerabilities": results["code_scan"].get("data", {}).get("vulnerabilities_found", 0),
                "dependency_vulnerabilities": results["dependency_scan"].get("data", {}).get("vulnerabilities_found", 0),
                "secrets_found": results["secret_scan"].get("data", {}).get("secrets_found", 0),
                "config_issues": results["config_scan"].get("data", {}).get("security_issues_found", 0)
            },
            "base_report": base_report
        }
        
        return ci_report
    
    def _make_pipeline_decision(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Make CI/CD pipeline decision based on scan results."""
        overall_status = results["overall_status"]
        
        # Apply severity thresholds from configuration
        decision = {
            "allow_deployment": True,
            "decision_reason": "No security issues found",
            "required_actions": []
        }
        
        if overall_status == "critical_vulnerabilities":
            decision["allow_deployment"] = False
            decision["decision_reason"] = "Critical vulnerabilities found - blocking deployment"
            decision["required_actions"].append("Fix all critical vulnerabilities")
        
        elif overall_status == "high_vulnerabilities":
            # Check configuration for high severity blocking
            if self.security_scanner.config.parameters.get("severity_thresholds", {}).get("block_high", False):
                decision["allow_deployment"] = False
                decision["decision_reason"] = "High severity vulnerabilities found - blocking deployment"
                decision["required_actions"].append("Fix all high severity vulnerabilities")
            else:
                decision["decision_reason"] = "High severity vulnerabilities found - deployment allowed with warning"
                decision["required_actions"].append("Schedule remediation for high severity vulnerabilities")
        
        elif overall_status == "failed":
            decision["allow_deployment"] = False
            decision["decision_reason"] = "Security scans failed - unable to verify security status"
            decision["required_actions"].append("Fix scan failures and re-run security checks")
        
        return decision

# Example CI/CD pipeline integration
async def run_ci_security_pipeline(project_path: str = "."):
    """Example CI/CD security pipeline."""
    
    # Initialize CI pipeline integration
    ci_integration = CIPipelineIntegration(security_scanner)
    
    # Run security checks
    print("🚀 Running CI/CD security pipeline...")
    pipeline_results = await ci_integration.run_security_checks(project_path)
    
    # Print results
    decision = pipeline_results["pipeline_decision"]
    
    print(f"\n{'='*60}")
    print("🔒 SECURITY SCAN RESULTS")
    print(f"{'='*60}")
    
    print(f"Overall Status: {pipeline_results['scan_results']['overall_status']}")
    print(f"Deployment Decision: {'✅ ALLOWED' if decision['allow_deployment'] else '❌ BLOCKED'}")
    print(f"Reason: {decision['decision_reason']}")
    
    if decision["required_actions"]:
        print("\nRequired Actions:")
        for action in decision["required_actions"]:
            print(f"  • {action}")
    
    # Print detailed results
    scan_results = pipeline_results["scan_results"]
    
    print(f"\n📊 Scan Breakdown:")
    print(f"  Code Vulnerabilities: {scan_results['code_scan'].get('data', {}).get('vulnerabilities_found', 0)}")
    print(f"  Dependency Issues: {scan_results['dependency_scan'].get('data', {}).get('vulnerabilities_found', 0)}")
    print(f"  Secrets Found: {scan_results['secret_scan'].get('data', {}).get('secrets_found', 0)}")
    print(f"  Config Issues: {scan_results['config_scan'].get('data', {}).get('security_issues_found', 0)}")
    
    # Exit with appropriate code for CI/CD
    exit_code = 0 if decision["allow_deployment"] else 1
    print(f"\n🎯 Pipeline Exit Code: {exit_code}")
    
    return exit_code

# Run the CI/CD security pipeline
if __name__ == "__main__":
    exit_code = asyncio.run(run_ci_security_pipeline())
    sys.exit(exit_code)
```

## Webhook Integration Tutorial

### Scenario: Event-Driven Integration System

This tutorial shows how to build an event-driven system using webhooks for external service integration.

#### Step 1: Configure Webhook Integration

```python
from leanvibe.plugin_sdk.examples import WebhookIntegrationPlugin

# Configure webhook integration
webhook_config = PluginConfig(
    name="EventDrivenIntegration",
    version="1.0.0",
    description="Event-driven webhook integration system",
    parameters={
        "max_concurrent_deliveries": 20,
        "delivery_timeout_seconds": 30,
        "enable_delivery_history": True,
        "max_history_entries": 5000,
        "webhooks": [
            {
                "webhook_id": "order_notifications",
                "url": "https://api.crm.example.com/webhooks/orders",
                "secret": "webhook_secret_key_123",
                "headers": {"X-Source": "LeanVibe-System"},
                "timeout_seconds": 15,
                "retry_attempts": 3,
                "retry_delay_seconds": 5
            },
            {
                "webhook_id": "user_events",
                "url": "https://analytics.example.com/events",
                "headers": {"Authorization": "Bearer analytics_token_456"},
                "timeout_seconds": 10,
                "retry_attempts": 2
            },
            {
                "webhook_id": "slack_alerts",
                "url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                "timeout_seconds": 10,
                "retry_attempts": 1
            }
        ],
        "api_endpoints": [
            {
                "endpoint_id": "payment_gateway",
                "base_url": "https://api.payment.example.com",
                "auth_type": "api_key",
                "auth_config": {
                    "api_key": "payment_api_key_789",
                    "header_name": "X-API-Key"
                },
                "timeout_seconds": 30,
                "rate_limit_requests": 100,
                "rate_limit_window_seconds": 60
            },
            {
                "endpoint_id": "inventory_service",
                "base_url": "https://inventory.example.com/api/v1",
                "auth_type": "bearer",
                "auth_config": {"token": "inventory_bearer_token_abc"},
                "timeout_seconds": 20
            }
        ]
    }
)

# Initialize webhook integration
webhook_integration = WebhookIntegrationPlugin(webhook_config)
await webhook_integration.initialize()
```

#### Step 2: Build Event Processing System

```python
class EventProcessor:
    """Event processing system using webhook integration."""
    
    def __init__(self, webhook_integration: WebhookIntegrationPlugin):
        self.webhook_integration = webhook_integration
        self.event_handlers = {}
        self.event_transformers = {}
        
    def register_event_handler(self, event_type: str, handler_func):
        """Register an event handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler_func)
    
    def register_event_transformer(self, event_type: str, transformer_func):
        """Register an event transformer for payload modification."""
        self.event_transformers[event_type] = transformer_func
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an event through the integration system."""
        
        print(f"📨 Processing event: {event_type}")
        
        # Transform event data if transformer is registered
        if event_type in self.event_transformers:
            event_data = await self.event_transformers[event_type](event_data)
        
        # Process event through registered handlers
        results = []
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    result = await handler(event_data)
                    results.append(result)
                except Exception as e:
                    print(f"❌ Handler error for {event_type}: {e}")
                    results.append({"success": False, "error": str(e)})
        
        return {
            "event_type": event_type,
            "processed_handlers": len(results),
            "results": results,
            "success": all(r.get("success", False) for r in results)
        }

# Initialize event processor
event_processor = EventProcessor(webhook_integration)

# Define event handlers
async def handle_order_created(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle order created event."""
    order_id = event_data.get("order_id")
    customer_email = event_data.get("customer_email")
    
    # Send order notification webhook
    webhook_task = TaskInterface(
        task_id=f"order_webhook_{order_id}",
        task_type="send_webhook",
        parameters={
            "webhook_id": "order_notifications",
            "payload": {
                "event_type": "order_created",
                "order_id": order_id,
                "customer_email": customer_email,
                "timestamp": datetime.utcnow().isoformat(),
                "order_details": event_data.get("order_details", {})
            }
        }
    )
    
    webhook_result = await webhook_integration.handle_task(webhook_task)
    
    # Update inventory via API call
    inventory_task = TaskInterface(
        task_id=f"inventory_update_{order_id}",
        task_type="call_api",
        parameters={
            "endpoint_id": "inventory_service",
            "method": "POST",
            "path": "/orders/reserve",
            "payload": {
                "order_id": order_id,
                "items": event_data.get("order_details", {}).get("items", [])
            }
        }
    )
    
    inventory_result = await webhook_integration.handle_task(inventory_task)
    
    return {
        "success": webhook_result.success and inventory_result.success,
        "webhook_delivery": webhook_result.data if webhook_result.success else webhook_result.error,
        "inventory_update": inventory_result.data if inventory_result.success else inventory_result.error
    }

async def handle_user_signup(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle user signup event."""
    user_id = event_data.get("user_id")
    user_email = event_data.get("user_email")
    
    # Send analytics event
    analytics_task = TaskInterface(
        task_id=f"analytics_{user_id}",
        task_type="send_webhook",
        parameters={
            "webhook_id": "user_events",
            "payload": {
                "event": "user_signup",
                "user_id": user_id,
                "email": user_email,
                "timestamp": datetime.utcnow().isoformat(),
                "properties": event_data.get("user_properties", {})
            }
        }
    )
    
    analytics_result = await webhook_integration.handle_task(analytics_task)
    
    return {
        "success": analytics_result.success,
        "analytics_event": analytics_result.data if analytics_result.success else analytics_result.error
    }

async def handle_payment_failed(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle payment failure event."""
    order_id = event_data.get("order_id")
    error_message = event_data.get("error_message")
    
    # Send Slack alert
    slack_task = TaskInterface(
        task_id=f"slack_alert_{order_id}",
        task_type="send_webhook",
        parameters={
            "webhook_id": "slack_alerts",
            "payload": {
                "text": f"🚨 Payment Failed",
                "attachments": [
                    {
                        "color": "danger",
                        "fields": [
                            {"title": "Order ID", "value": order_id, "short": True},
                            {"title": "Error", "value": error_message, "short": False},
                            {"title": "Time", "value": datetime.utcnow().isoformat(), "short": True}
                        ]
                    }
                ]
            }
        }
    )
    
    slack_result = await webhook_integration.handle_task(slack_task)
    
    # Call payment gateway for retry logic
    payment_task = TaskInterface(
        task_id=f"payment_retry_{order_id}",
        task_type="call_api",
        parameters={
            "endpoint_id": "payment_gateway",
            "method": "POST",
            "path": "/payments/retry",
            "payload": {
                "order_id": order_id,
                "retry_reason": "automatic_retry_after_failure"
            }
        }
    )
    
    payment_result = await webhook_integration.handle_task(payment_task)
    
    return {
        "success": slack_result.success and payment_result.success,
        "slack_notification": slack_result.data if slack_result.success else slack_result.error,
        "payment_retry": payment_result.data if payment_result.success else payment_result.error
    }

# Register event handlers
event_processor.register_event_handler("order_created", handle_order_created)
event_processor.register_event_handler("user_signup", handle_user_signup)
event_processor.register_event_handler("payment_failed", handle_payment_failed)

# Define event transformers
async def transform_order_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform order event data."""
    # Add computed fields
    event_data["total_amount"] = sum(
        item.get("price", 0) * item.get("quantity", 0)
        for item in event_data.get("order_details", {}).get("items", [])
    )
    
    # Add event metadata
    event_data["processed_at"] = datetime.utcnow().isoformat()
    event_data["source_system"] = "leanvibe_agent_hive"
    
    return event_data

event_processor.register_event_transformer("order_created", transform_order_event)
```

#### Step 3: Implement Event-Driven Workflow

```python
class EventDrivenWorkflow:
    """Event-driven workflow orchestration."""
    
    def __init__(self, event_processor: EventProcessor):
        self.event_processor = event_processor
        self.running = False
        self.event_queue = asyncio.Queue()
        self.workflow_stats = {
            "events_processed": 0,
            "events_failed": 0,
            "average_processing_time_ms": 0.0
        }
        self._processing_times = []
    
    async def start_workflow(self):
        """Start the event-driven workflow."""
        self.running = True
        print("🚀 Starting event-driven workflow...")
        
        # Start event processing workers
        workers = [
            asyncio.create_task(self._event_worker(f"worker_{i}"))
            for i in range(3)  # 3 concurrent workers
        ]
        
        await asyncio.gather(*workers)
    
    async def stop_workflow(self):
        """Stop the event-driven workflow."""
        self.running = False
        print("🛑 Stopping event-driven workflow...")
    
    async def submit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Submit an event for processing."""
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "event_data": event_data,
            "submitted_at": datetime.utcnow()
        }
        
        await self.event_queue.put(event)
        print(f"📤 Event submitted: {event_type} ({event['event_id']})")
    
    async def _event_worker(self, worker_id: str):
        """Event processing worker."""
        print(f"👷 Event worker {worker_id} started")
        
        while self.running:
            try:
                # Get next event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                start_time = time.time()
                
                result = await self.event_processor.process_event(
                    event["event_type"],
                    event["event_data"]
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Update statistics
                self.workflow_stats["events_processed"] += 1
                self._processing_times.append(processing_time)
                
                if self._processing_times:
                    self.workflow_stats["average_processing_time_ms"] = (
                        sum(self._processing_times) / len(self._processing_times)
                    )
                    # Keep only recent times
                    if len(self._processing_times) > 100:
                        self._processing_times = self._processing_times[-100:]
                
                if result["success"]:
                    print(f"✅ Event processed: {event['event_type']} in {processing_time:.1f}ms")
                else:
                    print(f"❌ Event failed: {event['event_type']}")
                    self.workflow_stats["events_failed"] += 1
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                self.workflow_stats["events_failed"] += 1
        
        print(f"👷 Event worker {worker_id} stopped")

# Initialize and start workflow
workflow = EventDrivenWorkflow(event_processor)

# Example event processing
async def simulate_events():
    """Simulate various events for demonstration."""
    
    # Start workflow
    workflow_task = asyncio.create_task(workflow.start_workflow())
    
    # Give workflow time to start
    await asyncio.sleep(1)
    
    # Submit test events
    events = [
        {
            "event_type": "order_created",
            "event_data": {
                "order_id": "ORD001",
                "customer_email": "customer@example.com",
                "order_details": {
                    "items": [
                        {"product_id": "PROD001", "price": 29.99, "quantity": 2},
                        {"product_id": "PROD002", "price": 49.99, "quantity": 1}
                    ]
                }
            }
        },
        {
            "event_type": "user_signup",
            "event_data": {
                "user_id": "USER001",
                "user_email": "newuser@example.com",
                "user_properties": {
                    "signup_source": "website",
                    "plan": "free"
                }
            }
        },
        {
            "event_type": "payment_failed",
            "event_data": {
                "order_id": "ORD002",
                "error_message": "Insufficient funds",
                "payment_method": "credit_card"
            }
        }
    ]
    
    # Submit events with delay
    for event in events:
        await workflow.submit_event(event["event_type"], event["event_data"])
        await asyncio.sleep(2)  # 2 second delay between events
    
    # Wait for processing to complete
    await workflow.event_queue.join()
    
    # Print statistics
    print(f"\n📊 Workflow Statistics:")
    print(f"  Events Processed: {workflow.workflow_stats['events_processed']}")
    print(f"  Events Failed: {workflow.workflow_stats['events_failed']}")
    print(f"  Average Processing Time: {workflow.workflow_stats['average_processing_time_ms']:.1f}ms")
    
    # Stop workflow
    await workflow.stop_workflow()

# Run the simulation
await simulate_events()
```

## Troubleshooting and Common Issues

### Performance Issues

**Issue:** Plugin response times exceed Epic 1 requirements (<50ms)

**Solutions:**
1. Use lazy loading for expensive resources
2. Implement caching with `@cached_result` decorator
3. Process data in batches to manage memory
4. Use async operations and avoid blocking calls

```python
# Example: Optimizing slow plugin
@cached_result(ttl_seconds=300)  # Cache results for 5 minutes
@performance_tracked(alert_threshold_ms=45)  # Alert if approaching limit
async def handle_task(self, task: TaskInterface) -> TaskResult:
    # Use lazy loading
    if self._expensive_resource is None:
        self._expensive_resource = await self._load_resource()
    
    # Process in batches
    data = task.parameters.get("data", [])
    batch_size = 100
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        await self._process_batch(batch)
        
        # Yield control to prevent blocking
        await asyncio.sleep(0.001)
```

**Issue:** Memory usage exceeds Epic 1 limits (<80MB)

**Solutions:**
1. Clear large data structures after use
2. Use generators instead of lists for large datasets
3. Implement memory monitoring

```python
import psutil
import gc

class MemoryOptimizedPlugin(PluginBase):
    def __init__(self, config):
        super().__init__(config)
        self._memory_threshold_mb = 70  # Alert before hitting 80MB limit
    
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        # Monitor memory usage
        initial_memory = self._get_memory_usage()
        
        try:
            # Process task
            result = await self._process_task(task)
            
            # Check memory usage
            current_memory = self._get_memory_usage()
            if current_memory > self._memory_threshold_mb:
                await self.log_warning(f"High memory usage: {current_memory}MB")
                gc.collect()  # Force garbage collection
            
            return result
            
        finally:
            # Clean up large objects
            self._cleanup_memory()
    
    def _get_memory_usage(self) -> float:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB
    
    def _cleanup_memory(self):
        # Clear any large data structures
        if hasattr(self, '_large_cache'):
            self._large_cache.clear()
```

### Integration Issues

**Issue:** External API calls failing or timing out

**Solutions:**
1. Implement retry logic with exponential backoff
2. Use circuit breaker pattern
3. Add proper error handling and logging

```python
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

class ResilientAPIClient:
    def __init__(self):
        self._session = None
        self._circuit_breaker = CircuitBreaker()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def make_api_call(self, url: str, payload: Dict) -> Dict:
        if self._circuit_breaker.is_open():
            raise Exception("Circuit breaker is open")
        
        try:
            async with self._session.post(url, json=payload) as response:
                if response.status >= 400:
                    self._circuit_breaker.record_failure()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                self._circuit_breaker.record_success()
                return await response.json()
                
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout_seconds=60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half_open"
                return False
            return True
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
```

### Configuration Issues

**Issue:** Plugin configuration validation errors

**Solutions:**
1. Implement comprehensive validation in `_on_initialize`
2. Provide clear error messages with suggestions
3. Use configuration schemas

```python
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError

class PluginConfigSchema(BaseModel):
    batch_size: int = 1000
    timeout_seconds: int = 30
    enable_retries: bool = True
    max_retries: int = 3
    
    class Config:
        extra = "forbid"  # Reject unknown fields

class WellConfiguredPlugin(PluginBase):
    async def _on_initialize(self) -> None:
        try:
            # Validate configuration using schema
            config_schema = PluginConfigSchema(**self.config.parameters)
            
            # Extract validated values
            self.batch_size = config_schema.batch_size
            self.timeout_seconds = config_schema.timeout_seconds
            
        except ValidationError as e:
            # Provide helpful error message
            error_details = []
            for error in e.errors():
                field = error["loc"][0]
                message = error["msg"]
                error_details.append(f"{field}: {message}")
            
            raise PluginConfigurationError(
                f"Configuration validation failed: {'; '.join(error_details)}",
                config_key="parameters",
                plugin_id=self.plugin_id,
                suggestions=[
                    "Check parameter names and types",
                    "Ensure all required parameters are provided",
                    "Remove any unknown configuration parameters"
                ]
            )
```

### Testing Issues

**Issue:** Plugin tests failing or incomplete coverage

**Solutions:**
1. Use the comprehensive test framework
2. Mock external dependencies
3. Test error conditions

```python
from leanvibe.plugin_sdk.testing import PluginTestFramework

async def comprehensive_plugin_test():
    test_framework = PluginTestFramework()
    
    # Register plugin
    test_framework.register_plugin("my_plugin", MyPlugin)
    
    # Test successful operations
    success_test = await test_framework.test_plugin_task(
        plugin_id="my_plugin",
        task_type="process_data",
        parameters={"data": [{"id": 1, "value": "test"}]},
        expected_success=True
    )
    
    # Test error handling
    error_test = await test_framework.test_plugin_error_handling(
        plugin_id="my_plugin",
        task_type="process_data",
        parameters={"data": "invalid"},  # Invalid input
        expected_error_code="INVALID_INPUT"
    )
    
    # Test performance
    performance_test = await test_framework.test_plugin_performance(
        plugin_id="my_plugin",
        task_type="process_data",
        parameters={"data": [{"id": i, "value": f"test_{i}"} for i in range(1000)]},
        max_execution_time_ms=2000,
        max_memory_mb=50
    )
    
    # Test concurrent operations
    concurrency_test = await test_framework.test_plugin_concurrency(
        plugin_id="my_plugin",
        task_type="process_data",
        parameters={"data": [{"id": 1, "value": "test"}]},
        concurrent_tasks=10
    )
    
    # Generate test report
    report = test_framework.generate_test_report()
    
    print("Test Results:")
    print(f"  Success Test: {success_test.passed}")
    print(f"  Error Test: {error_test.passed}")
    print(f"  Performance Test: {performance_test.passed}")
    print(f"  Concurrency Test: {concurrency_test.passed}")
    print(f"  Overall Score: {report.overall_score}/100")

await comprehensive_plugin_test()
```

---

This comprehensive tutorial covers the main integration scenarios for the LeanVibe Plugin SDK. Each example demonstrates real-world usage patterns while maintaining Epic 1 performance standards and showcasing best practices for plugin development.
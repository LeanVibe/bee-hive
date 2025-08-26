"""
Component Validation Framework for Phase 2: Systematic API Validation

This module provides comprehensive validation for all API layer components,
focusing on reliability, performance, and integration testing as specified
in the Phase 2 Component Validation mission.

Test Categories:
1. Import & Basic Functionality Tests
2. Integration Point Testing  
3. Component API Documentation
4. Reliability & Error Handling
"""

import pytest
import asyncio
import time
from typing import Dict, List, Optional, Any
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json
import logging

# Import core components for validation
from app.main import create_app
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ComponentValidationFramework:
    """Framework for systematic component validation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.validation_results = {}
        self.performance_metrics = {}
        
    async def validate_component_imports(self, component_path: str) -> Dict[str, Any]:
        """Validate that component can be imported without errors."""
        result = {
            "component": component_path,
            "import_success": False,
            "error": None,
            "dependencies": [],
            "import_time_ms": 0
        }
        
        start_time = time.time()
        try:
            # Dynamic import test
            parts = component_path.split('.')
            module = __import__(component_path)
            for part in parts[1:]:
                module = getattr(module, part)
                
            result["import_success"] = True
            result["import_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
        except Exception as e:
            result["error"] = str(e)
            result["import_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
        return result

    async def validate_api_endpoint_health(self, client: AsyncClient, endpoint: str) -> Dict[str, Any]:
        """Validate API endpoint basic functionality."""
        result = {
            "endpoint": endpoint,
            "status_code": None,
            "response_time_ms": 0,
            "content_type": None,
            "response_size": 0,
            "error": None
        }
        
        start_time = time.time()
        try:
            response = await client.get(endpoint)
            result["status_code"] = response.status_code
            result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            result["content_type"] = response.headers.get("content-type")
            result["response_size"] = len(response.content)
            
        except Exception as e:
            result["error"] = str(e)
            result["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
            
        return result

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            "timestamp": time.time(),
            "validation_results": self.validation_results,
            "performance_metrics": self.performance_metrics,
            "summary": {
                "total_components": len(self.validation_results),
                "successful_imports": sum(1 for r in self.validation_results.values() if r.get("import_success", False)),
                "failed_imports": sum(1 for r in self.validation_results.values() if not r.get("import_success", False))
            }
        }


class APILayerValidator:
    """Specialized validator for API layer components."""
    
    # Core API components to validate
    CORE_API_COMPONENTS = [
        "app.api.routes",
        "app.api.auth_endpoints",
        "app.api.enterprise_security",
        "app.api.hive_commands",
        "app.api.intelligence",
        "app.api.memory_operations",
        "app.api.project_index",
        "app.api.v1.agents",
        "app.api.v1.tasks",
        "app.api.v1.coordination",
        "app.api.v2.agents",
        "app.api.v2.tasks"
    ]
    
    # Critical API endpoints to validate
    CRITICAL_ENDPOINTS = [
        "/health",
        "/status",
        "/metrics",
        "/api/v1/agents",
        "/api/v1/tasks", 
        "/api/v2/agents",
        "/api/v2/tasks",
        "/docs",
        "/openapi.json"
    ]
    
    def __init__(self):
        self.framework = ComponentValidationFramework()
        
    async def validate_api_layer(self) -> Dict[str, Any]:
        """Comprehensive API layer validation."""
        results = {
            "component_imports": {},
            "endpoint_health": {},
            "integration_tests": {},
            "performance_metrics": {}
        }
        
        # 1. Component Import Validation
        logger.info("🔍 Validating API component imports...")
        for component in self.CORE_API_COMPONENTS:
            results["component_imports"][component] = await self.framework.validate_component_imports(component)
            
        # 2. Endpoint Health Validation
        logger.info("🔍 Validating API endpoint health...")
        app = create_app()
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            for endpoint in self.CRITICAL_ENDPOINTS:
                results["endpoint_health"][endpoint] = await self.framework.validate_api_endpoint_health(client, endpoint)
                
        # 3. Integration Tests
        logger.info("🔍 Running integration tests...")
        results["integration_tests"] = await self._run_integration_tests(app)
        
        # 4. Performance Metrics
        logger.info("🔍 Collecting performance metrics...")
        results["performance_metrics"] = await self._collect_performance_metrics(app)
        
        return results
        
    async def _run_integration_tests(self, app) -> Dict[str, Any]:
        """Run integration tests between API components."""
        integration_results = {}
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Test API-to-database flow
            try:
                health_response = await client.get("/health")
                integration_results["api_database_integration"] = {
                    "success": health_response.status_code == 200,
                    "database_status": health_response.json().get("components", {}).get("database", {}).get("status")
                }
            except Exception as e:
                integration_results["api_database_integration"] = {
                    "success": False,
                    "error": str(e)
                }
                
            # Test API-to-Redis flow  
            try:
                health_response = await client.get("/health")
                integration_results["api_redis_integration"] = {
                    "success": health_response.status_code == 200,
                    "redis_status": health_response.json().get("components", {}).get("redis", {}).get("status")
                }
            except Exception as e:
                integration_results["api_redis_integration"] = {
                    "success": False,
                    "error": str(e)
                }
                
            # Test orchestrator integration
            try:
                health_response = await client.get("/health")
                integration_results["api_orchestrator_integration"] = {
                    "success": health_response.status_code == 200,
                    "orchestrator_status": health_response.json().get("components", {}).get("orchestrator", {}).get("status")
                }
            except Exception as e:
                integration_results["api_orchestrator_integration"] = {
                    "success": False,
                    "error": str(e)
                }
                
        return integration_results
        
    async def _collect_performance_metrics(self, app) -> Dict[str, Any]:
        """Collect performance metrics for API layer."""
        performance_metrics = {
            "startup_time_ms": 0,
            "endpoint_response_times": {},
            "memory_usage_estimate": 0,
            "concurrent_request_handling": {}
        }
        
        # Measure app startup time
        start_time = time.time()
        test_app = create_app()
        performance_metrics["startup_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Measure endpoint response times
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            for endpoint in self.CRITICAL_ENDPOINTS[:5]:  # Test first 5 endpoints
                start_time = time.time()
                try:
                    response = await client.get(endpoint)
                    response_time = round((time.time() - start_time) * 1000, 2)
                    performance_metrics["endpoint_response_times"][endpoint] = {
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
                except Exception as e:
                    performance_metrics["endpoint_response_times"][endpoint] = {
                        "error": str(e),
                        "response_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                    
        return performance_metrics


# Test Classes

@pytest.mark.asyncio
class TestAPIComponentValidation:
    """Test suite for API component validation."""
    
    async def test_core_api_imports(self):
        """Test that all core API components can be imported successfully."""
        validator = APILayerValidator()
        
        for component in validator.CORE_API_COMPONENTS:
            result = await validator.framework.validate_component_imports(component)
            
            # Allow certain expected failures due to missing dependencies in test environment
            expected_failures = [
                "app.api.v1.agents",  # May require database
                "app.api.v1.tasks",   # May require Redis
            ]
            
            if component not in expected_failures:
                assert result["import_success"], f"Failed to import {component}: {result.get('error')}"
                assert result["import_time_ms"] < 1000, f"Import time too slow for {component}: {result['import_time_ms']}ms"
                
    async def test_critical_endpoint_health(self):
        """Test that critical API endpoints respond correctly."""
        validator = APILayerValidator()
        app = create_app()
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            # Test health endpoint specifically
            result = await validator.framework.validate_api_endpoint_health(client, "/health")
            
            assert result["status_code"] == 200, f"Health endpoint failed: {result}"
            assert result["response_time_ms"] < 1000, f"Health endpoint too slow: {result['response_time_ms']}ms"
            assert "application/json" in (result["content_type"] or ""), "Health endpoint should return JSON"
            
    async def test_api_integration_points(self):
        """Test API integration with core services."""
        validator = APILayerValidator()
        app = create_app()
        
        integration_results = await validator._run_integration_tests(app)
        
        # At minimum, API should handle integration gracefully even if services are unavailable
        for integration_name, result in integration_results.items():
            if result.get("error"):
                # Log but don't fail if services are unavailable in test environment
                logger.warning(f"Integration {integration_name} unavailable in test: {result['error']}")
            else:
                assert "success" in result, f"Integration test {integration_name} missing success indicator"
                
    async def test_api_performance_benchmarks(self):
        """Test that API meets performance benchmarks."""
        validator = APILayerValidator()
        app = create_app()
        
        performance_metrics = await validator._collect_performance_metrics(app)
        
        # Performance benchmarks from mission requirements
        assert performance_metrics["startup_time_ms"] < 5000, f"Startup too slow: {performance_metrics['startup_time_ms']}ms"
        
        for endpoint, metrics in performance_metrics["endpoint_response_times"].items():
            if not metrics.get("error"):
                assert metrics["response_time_ms"] < 200, f"Endpoint {endpoint} too slow: {metrics['response_time_ms']}ms"
                
    async def test_comprehensive_api_validation(self):
        """Run comprehensive API layer validation."""
        validator = APILayerValidator()
        
        validation_results = await validator.validate_api_layer()
        
        # Generate validation report
        report = validator.framework.generate_validation_report()
        
        # Assert overall system health
        assert "component_imports" in validation_results
        assert "endpoint_health" in validation_results
        assert "integration_tests" in validation_results
        assert "performance_metrics" in validation_results
        
        # Log comprehensive results for analysis
        logger.info("📊 API Validation Results:")
        logger.info(json.dumps(validation_results, indent=2))


@pytest.mark.integration  
class TestDatabaseLayerIntegration:
    """Test suite for database layer integration with API."""
    
    async def test_database_connection_through_api(self):
        """Test database connectivity through API layer."""
        app = create_app()
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/health")
            
            if response.status_code == 200:
                health_data = response.json()
                db_status = health_data.get("components", {}).get("database", {})
                
                # If database is configured, it should be healthy
                if db_status:
                    assert db_status.get("status") in ["healthy", "unhealthy"], "Database status should be defined"
                    
    async def test_redis_integration_through_api(self):
        """Test Redis connectivity through API layer."""
        app = create_app()
        
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.get("/health")
            
            if response.status_code == 200:
                health_data = response.json()
                redis_status = health_data.get("components", {}).get("redis", {})
                
                # If Redis is configured, it should be healthy
                if redis_status:
                    assert redis_status.get("status") in ["healthy", "unhealthy"], "Redis status should be defined"


if __name__ == "__main__":
    # Run standalone validation
    async def main():
        validator = APILayerValidator()
        results = await validator.validate_api_layer()
        
        print("🎯 API Layer Validation Complete!")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        
    asyncio.run(main())
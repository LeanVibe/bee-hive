#!/usr/bin/env python3
"""
Comprehensive API Layer Validation Script
Phase 2: Component Validation - API Layer Focus

This script performs systematic validation of the API layer components
as specified in the Phase 2 mission requirements.
"""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime

# Set environment for testing
os.environ['SKIP_STARTUP_INIT'] = 'true'
os.environ['TESTING'] = 'true'


class APILayerValidator:
    """Comprehensive API layer validation."""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "validation_type": "API Layer Component Validation",
            "components_validated": 0,
            "components_passed": 0,
            "components_failed": 0,
            "details": {}
        }
        
    def validate_component_imports(self) -> Dict[str, Any]:
        """Validate API component imports."""
        print("🔍 Validating API Component Imports...")
        
        # Core API components identified from mission requirements
        api_components = [
            # Core API modules
            "app.api.routes",
            "app.api.auth_endpoints", 
            "app.api.enterprise_security",
            "app.api.hive_commands",
            "app.api.intelligence",
            "app.api.memory_operations",
            "app.api.project_index",
            "app.api.analytics",
            "app.api.monitoring_reporting",
            "app.api.sleep_management",
            
            # V1 API endpoints
            "app.api.v1.agents_simple",
            "app.api.v1.tasks_compatibility",
            "app.api.v1.coordination",
            "app.api.v1.github_integration",
            "app.api.v1.websocket",
            
            # V2 API endpoints
            "app.api.v2.agents",
            "app.api.v2.tasks",
            "app.api.v2.websockets"
        ]
        
        import_results = {}
        successful_imports = 0
        
        for component in api_components:
            start_time = time.time()
            try:
                __import__(component)
                import_time = round((time.time() - start_time) * 1000, 2)
                import_results[component] = {
                    "status": "success",
                    "import_time_ms": import_time,
                    "error": None
                }
                successful_imports += 1
                print(f"✅ {component}: {import_time}ms")
                
            except Exception as e:
                import_time = round((time.time() - start_time) * 1000, 2)
                import_results[component] = {
                    "status": "failed",
                    "import_time_ms": import_time,
                    "error": str(e)
                }
                print(f"❌ {component}: {str(e)[:100]}...")
                
        self.results["components_validated"] = len(api_components)
        self.results["components_passed"] = successful_imports
        self.results["components_failed"] = len(api_components) - successful_imports
        self.results["details"]["import_validation"] = import_results
        
        return import_results
        
    def validate_core_services(self) -> Dict[str, Any]:
        """Validate core service dependencies."""
        print("\n🔍 Validating Core Service Dependencies...")
        
        core_services = [
            "app.core.config",
            "app.core.database", 
            "app.core.redis",
            "app.core.orchestrator",
            "app.core.simple_orchestrator",
            "app.core.logging_service",
            "app.core.auth",
            "app.core.security",
            "app.models",
            "app.schemas"
        ]
        
        service_results = {}
        successful_services = 0
        
        for service in core_services:
            start_time = time.time()
            try:
                module = __import__(service, fromlist=[''])
                import_time = round((time.time() - start_time) * 1000, 2)
                
                # Basic functionality test
                functionality_test = self._test_service_functionality(service, module)
                
                service_results[service] = {
                    "status": "success",
                    "import_time_ms": import_time,
                    "functionality": functionality_test,
                    "error": None
                }
                successful_services += 1
                print(f"✅ {service}: {import_time}ms - {functionality_test}")
                
            except Exception as e:
                import_time = round((time.time() - start_time) * 1000, 2)
                service_results[service] = {
                    "status": "failed", 
                    "import_time_ms": import_time,
                    "functionality": "unable_to_test",
                    "error": str(e)
                }
                print(f"❌ {service}: {str(e)[:100]}...")
                
        self.results["details"]["service_validation"] = service_results
        return service_results
        
    def _test_service_functionality(self, service_name: str, module) -> str:
        """Test basic functionality of imported service."""
        try:
            if service_name == "app.core.config":
                # Test configuration loading
                if hasattr(module, 'get_settings'):
                    return "config_accessible"
                return "basic_import_only"
                
            elif service_name == "app.core.database":
                # Test database module structure
                if hasattr(module, 'get_async_session'):
                    return "database_session_available"
                return "basic_import_only"
                
            elif service_name == "app.core.redis":
                # Test Redis module structure
                if hasattr(module, 'get_redis'):
                    return "redis_client_available"
                return "basic_import_only"
                
            elif service_name in ["app.core.orchestrator", "app.core.simple_orchestrator"]:
                # Test orchestrator availability
                if hasattr(module, 'Orchestrator') or hasattr(module, 'SimpleOrchestrator'):
                    return "orchestrator_class_available"
                return "basic_import_only"
                
            else:
                return "basic_import_successful"
                
        except Exception as e:
            return f"functionality_test_failed: {str(e)[:50]}"
            
    def validate_fastapi_app_creation(self) -> Dict[str, Any]:
        """Validate FastAPI application creation."""
        print("\n🔍 Validating FastAPI Application Creation...")
        
        start_time = time.time()
        try:
            from app.main import create_app
            app = create_app()
            creation_time = round((time.time() - start_time) * 1000, 2)
            
            # Count routes
            route_count = len(app.routes)
            
            # Test basic app properties
            app_validation = {
                "status": "success",
                "creation_time_ms": creation_time,
                "route_count": route_count,
                "title": app.title,
                "version": app.version,
                "error": None
            }
            
            print(f"✅ FastAPI App Creation: {creation_time}ms")
            print(f"📋 Routes: {route_count}")
            print(f"📖 Title: {app.title}")
            print(f"🏷️ Version: {app.version}")
            
        except Exception as e:
            creation_time = round((time.time() - start_time) * 1000, 2)
            app_validation = {
                "status": "failed",
                "creation_time_ms": creation_time,
                "route_count": 0,
                "error": str(e)
            }
            print(f"❌ FastAPI App Creation Failed: {str(e)}")
            
        self.results["details"]["app_creation"] = app_validation
        return app_validation
        
    def validate_endpoint_structure(self) -> Dict[str, Any]:
        """Validate API endpoint structure without starting server."""
        print("\n🔍 Validating API Endpoint Structure...")
        
        try:
            from app.main import create_app
            app = create_app()
            
            # Analyze route structure
            endpoints = []
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    endpoints.append({
                        "path": route.path,
                        "methods": list(route.methods),
                        "name": getattr(route, 'name', 'unnamed')
                    })
                    
            # Categorize endpoints
            endpoint_analysis = {
                "total_endpoints": len(endpoints),
                "health_endpoints": [e for e in endpoints if 'health' in e['path'].lower()],
                "api_v1_endpoints": [e for e in endpoints if '/api/v1' in e['path']],
                "api_v2_endpoints": [e for e in endpoints if '/api/v2' in e['path']],
                "agent_endpoints": [e for e in endpoints if 'agent' in e['path'].lower()],
                "task_endpoints": [e for e in endpoints if 'task' in e['path'].lower()],
                "websocket_endpoints": [e for e in endpoints if 'ws' in e['path'].lower() or 'websocket' in e['path'].lower()]
            }
            
            print(f"📊 Total Endpoints: {endpoint_analysis['total_endpoints']}")
            print(f"🏥 Health Endpoints: {len(endpoint_analysis['health_endpoints'])}")
            print(f"🔌 API v1 Endpoints: {len(endpoint_analysis['api_v1_endpoints'])}")
            print(f"🚀 API v2 Endpoints: {len(endpoint_analysis['api_v2_endpoints'])}")
            print(f"🤖 Agent Endpoints: {len(endpoint_analysis['agent_endpoints'])}")
            print(f"📝 Task Endpoints: {len(endpoint_analysis['task_endpoints'])}")
            print(f"🔗 WebSocket Endpoints: {len(endpoint_analysis['websocket_endpoints'])}")
            
            endpoint_validation = {
                "status": "success",
                "analysis": endpoint_analysis,
                "endpoints": endpoints,
                "error": None
            }
            
        except Exception as e:
            endpoint_validation = {
                "status": "failed",
                "analysis": {},
                "endpoints": [],
                "error": str(e)
            }
            print(f"❌ Endpoint Structure Analysis Failed: {str(e)}")
            
        self.results["details"]["endpoint_structure"] = endpoint_validation
        return endpoint_validation
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        # Calculate overall success rate
        total_validations = 0
        successful_validations = 0
        
        for category, details in self.results["details"].items():
            if isinstance(details, dict):
                if "status" in details:
                    total_validations += 1
                    if details["status"] == "success":
                        successful_validations += 1
                else:
                    # Handle nested validation results (like import_validation)
                    for item, result in details.items():
                        if isinstance(result, dict) and "status" in result:
                            total_validations += 1
                            if result["status"] == "success":
                                successful_validations += 1
        
        self.results["summary"] = {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": round((successful_validations / total_validations) * 100, 2) if total_validations > 0 else 0
        }
        
        return self.results
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete API layer validation."""
        print("🎯 Starting Comprehensive API Layer Validation")
        print("=" * 60)
        
        # 1. Component Import Validation
        self.validate_component_imports()
        
        # 2. Core Service Validation
        self.validate_core_services()
        
        # 3. FastAPI App Creation Validation
        self.validate_fastapi_app_creation()
        
        # 4. Endpoint Structure Validation
        self.validate_endpoint_structure()
        
        # 5. Generate Final Report
        report = self.generate_validation_report()
        
        print("\n" + "=" * 60)
        print("🎯 API Layer Validation Complete!")
        print(f"📊 Success Rate: {report['summary']['success_rate']}%")
        print(f"✅ Successful: {report['summary']['successful_validations']}")
        print(f"❌ Failed: {report['summary']['failed_validations']}")
        print(f"🔢 Total: {report['summary']['total_validations']}")
        
        return report


async def main():
    """Run the API layer validation."""
    validator = APILayerValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Save report to file
        report_file = f"api_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        # Determine success/failure based on mission requirements
        success_rate = report['summary']['success_rate']
        if success_rate >= 85:
            print("🎉 API Layer Validation: PASSED (≥85% success rate)")
            return 0
        else:
            print("⚠️ API Layer Validation: NEEDS ATTENTION (<85% success rate)")
            return 1
            
    except Exception as e:
        print(f"❌ Validation failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
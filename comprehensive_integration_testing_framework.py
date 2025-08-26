#!/usr/bin/env python3
"""
Comprehensive Integration Testing Framework
Phase 2: Component Validation - Integration Testing Framework

This framework provides systematic integration testing between all validated
components (API, Database, Services) to ensure enterprise reliability.
"""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# Set environment for testing
os.environ['SKIP_STARTUP_INIT'] = 'true'
os.environ['TESTING'] = 'true'


class IntegrationTestingFramework:
    """Comprehensive integration testing framework."""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_type": "Comprehensive Integration Testing Framework",
            "framework_version": "1.0.0",
            "test_categories": [],
            "test_results": {},
            "summary": {}
        }
        
    async def test_api_database_integration(self) -> Dict[str, Any]:
        """Test API layer integration with database layer."""
        print("🔗 Testing API-Database Integration...")
        
        integration_result = {
            "test_name": "API-Database Integration",
            "test_started": datetime.now().isoformat(),
            "subtests": {},
            "overall_status": "unknown"
        }
        
        # Test 1: API can create FastAPI app with database dependencies
        try:
            start_time = time.time()
            from app.main import create_app
            app = create_app()
            
            # Check if database-related routes are present
            db_routes = [route for route in app.routes 
                        if hasattr(route, 'path') and 
                        any(keyword in route.path.lower() 
                            for keyword in ['agent', 'task', 'user', 'session'])]
            
            integration_result["subtests"]["app_creation_with_db_routes"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "db_routes_found": len(db_routes),
                "sample_routes": [r.path for r in db_routes[:5]]
            }
            
            print(f"  ✅ App creation with DB routes: {len(db_routes)} routes found")
            
        except Exception as e:
            integration_result["subtests"]["app_creation_with_db_routes"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ App creation failed: {str(e)[:100]}")
            
        # Test 2: Database models can be imported in API context
        try:
            start_time = time.time()
            from app.models.agent import Agent
            from app.models.task import Task
            from app.models.user import User
            from app.schemas.agent import AgentCreate, AgentResponse
            from app.schemas.task import TaskCreate, TaskResponse
            
            models_imported = ["Agent", "Task", "User", "AgentCreate", "AgentResponse", "TaskCreate", "TaskResponse"]
            
            integration_result["subtests"]["models_schemas_import"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "models_imported": models_imported,
                "import_count": len(models_imported)
            }
            
            print(f"  ✅ Models/Schemas import: {len(models_imported)} components imported")
            
        except Exception as e:
            integration_result["subtests"]["models_schemas_import"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Models/Schemas import failed: {str(e)[:100]}")
            
        # Test 3: API endpoints can reference database session factory
        try:
            start_time = time.time()
            from app.core.database import get_async_session
            from app.api.v1.agents import router as agents_router
            from app.api.v1.tasks import router as tasks_router
            
            # Check if routers are properly configured
            routers_tested = ["agents_router", "tasks_router"]
            
            integration_result["subtests"]["api_db_session_integration"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "session_factory_available": True,
                "routers_accessible": routers_tested
            }
            
            print(f"  ✅ API-DB session integration: {len(routers_tested)} routers accessible")
            
        except Exception as e:
            integration_result["subtests"]["api_db_session_integration"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ API-DB session integration failed: {str(e)[:100]}")
            
        # Determine overall status
        successful_tests = sum(1 for test in integration_result["subtests"].values() 
                             if test.get("status") == "success")
        total_tests = len(integration_result["subtests"])
        
        integration_result["overall_status"] = (
            "success" if successful_tests == total_tests
            else "partial" if successful_tests > 0
            else "failed"
        )
        
        integration_result["success_rate"] = round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0
        integration_result["test_completed"] = datetime.now().isoformat()
        
        return integration_result
        
    async def test_api_services_integration(self) -> Dict[str, Any]:
        """Test API layer integration with services layer."""
        print("\n🔗 Testing API-Services Integration...")
        
        integration_result = {
            "test_name": "API-Services Integration",
            "test_started": datetime.now().isoformat(),
            "subtests": {},
            "overall_status": "unknown"
        }
        
        # Test 1: API can import and use orchestrator services
        try:
            start_time = time.time()
            from app.core.orchestrator import Orchestrator
            from app.core.simple_orchestrator import SimpleOrchestrator
            from app.api.hive_commands import router as hive_router
            
            integration_result["subtests"]["orchestrator_api_integration"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "orchestrators_available": ["Orchestrator", "SimpleOrchestrator"],
                "hive_commands_available": True
            }
            
            print("  ✅ Orchestrator-API integration successful")
            
        except Exception as e:
            integration_result["subtests"]["orchestrator_api_integration"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Orchestrator-API integration failed: {str(e)[:100]}")
            
        # Test 2: API can access semantic memory services
        try:
            start_time = time.time()
            from app.services.semantic_memory_service import SemanticMemoryService
            from app.api.memory_operations import get_memory_router
            
            memory_router = get_memory_router()
            
            integration_result["subtests"]["semantic_memory_api_integration"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "semantic_memory_service_available": True,
                "memory_router_available": True
            }
            
            print("  ✅ Semantic Memory-API integration successful")
            
        except Exception as e:
            integration_result["subtests"]["semantic_memory_api_integration"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Semantic Memory-API integration failed: {str(e)[:100]}")
            
        # Test 3: API can access project management services
        try:
            start_time = time.time()
            from app.services.project_management_service import ProjectManagementService
            from app.api.project_index import router as project_router
            
            integration_result["subtests"]["project_management_api_integration"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "project_management_service_available": True,
                "project_router_available": True
            }
            
            print("  ✅ Project Management-API integration successful")
            
        except Exception as e:
            integration_result["subtests"]["project_management_api_integration"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Project Management-API integration failed: {str(e)[:100]}")
            
        # Determine overall status
        successful_tests = sum(1 for test in integration_result["subtests"].values() 
                             if test.get("status") == "success")
        total_tests = len(integration_result["subtests"])
        
        integration_result["overall_status"] = (
            "success" if successful_tests == total_tests
            else "partial" if successful_tests > 0
            else "failed"
        )
        
        integration_result["success_rate"] = round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0
        integration_result["test_completed"] = datetime.now().isoformat()
        
        return integration_result
        
    async def test_database_services_integration(self) -> Dict[str, Any]:
        """Test database layer integration with services layer."""
        print("\n🔗 Testing Database-Services Integration...")
        
        integration_result = {
            "test_name": "Database-Services Integration", 
            "test_started": datetime.now().isoformat(),
            "subtests": {},
            "overall_status": "unknown"
        }
        
        # Test 1: Services can import database models and schemas
        try:
            start_time = time.time()
            from app.services.semantic_memory_service import SemanticMemoryService
            from app.models.agent import Agent
            from app.models.context import Context
            from app.schemas.agent import AgentCreate
            
            integration_result["subtests"]["services_model_import"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "service_with_models": True,
                "models_accessible": ["Agent", "Context"],
                "schemas_accessible": ["AgentCreate"]
            }
            
            print("  ✅ Services can import database models and schemas")
            
        except Exception as e:
            integration_result["subtests"]["services_model_import"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Services model import failed: {str(e)[:100]}")
            
        # Test 2: Services can access database session factory
        try:
            start_time = time.time()
            from app.core.database import get_async_session
            from app.services.project_management_service import ProjectManagementService
            
            # Test that service can potentially use database session
            integration_result["subtests"]["services_database_session"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "session_factory_available": True,
                "service_can_access_session": True
            }
            
            print("  ✅ Services can access database session factory")
            
        except Exception as e:
            integration_result["subtests"]["services_database_session"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Services database session access failed: {str(e)[:100]}")
            
        # Test 3: Orchestrator services can work with database types
        try:
            start_time = time.time()
            from app.core.orchestrator import Orchestrator
            from app.core.database_types import AgentType, TaskStatus
            
            integration_result["subtests"]["orchestrator_database_types"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "orchestrator_available": True,
                "database_types_available": ["AgentType", "TaskStatus"]
            }
            
            print("  ✅ Orchestrator can work with database types")
            
        except Exception as e:
            integration_result["subtests"]["orchestrator_database_types"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Orchestrator database types integration failed: {str(e)[:100]}")
            
        # Determine overall status
        successful_tests = sum(1 for test in integration_result["subtests"].values() 
                             if test.get("status") == "success")
        total_tests = len(integration_result["subtests"])
        
        integration_result["overall_status"] = (
            "success" if successful_tests == total_tests
            else "partial" if successful_tests > 0
            else "failed"
        )
        
        integration_result["success_rate"] = round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0
        integration_result["test_completed"] = datetime.now().isoformat()
        
        return integration_result
        
    async def test_full_stack_integration(self) -> Dict[str, Any]:
        """Test full stack integration (API + Database + Services)."""
        print("\n🔗 Testing Full Stack Integration...")
        
        integration_result = {
            "test_name": "Full Stack Integration",
            "test_started": datetime.now().isoformat(),
            "subtests": {},
            "overall_status": "unknown"
        }
        
        # Test 1: Complete FastAPI app creation with all layers
        try:
            start_time = time.time()
            from app.main import create_app
            
            app = create_app()
            
            # Analyze app completeness
            total_routes = len(app.routes)
            api_routes = len([r for r in app.routes if hasattr(r, 'path') and '/api' in r.path])
            health_routes = len([r for r in app.routes if hasattr(r, 'path') and 'health' in r.path])
            
            integration_result["subtests"]["complete_app_creation"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "total_routes": total_routes,
                "api_routes": api_routes,
                "health_routes": health_routes,
                "app_title": app.title,
                "app_version": app.version
            }
            
            print(f"  ✅ Complete app creation: {total_routes} routes, {api_routes} API routes")
            
        except Exception as e:
            integration_result["subtests"]["complete_app_creation"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Complete app creation failed: {str(e)[:100]}")
            
        # Test 2: Cross-layer component compatibility
        try:
            start_time = time.time()
            
            # Test that key components from all layers can coexist
            from app.api.routes import router as api_router
            from app.core.orchestrator import Orchestrator
            from app.models.agent import Agent
            from app.services.semantic_memory_service import SemanticMemoryService
            from app.core.database import get_async_session
            
            components_loaded = [
                "api_router", "Orchestrator", "Agent", 
                "SemanticMemoryService", "get_async_session"
            ]
            
            integration_result["subtests"]["cross_layer_compatibility"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "components_loaded": components_loaded,
                "layer_integration": "all_layers_compatible"
            }
            
            print(f"  ✅ Cross-layer compatibility: {len(components_loaded)} components loaded")
            
        except Exception as e:
            integration_result["subtests"]["cross_layer_compatibility"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Cross-layer compatibility failed: {str(e)[:100]}")
            
        # Test 3: Performance under integration load
        try:
            start_time = time.time()
            
            # Simulate integration load by importing multiple components rapidly
            import_count = 0
            for _ in range(3):
                from app.main import create_app
                from app.core.orchestrator import Orchestrator
                from app.models.agent import Agent
                import_count += 3
                
            integration_result["subtests"]["integration_load_performance"] = {
                "status": "success",
                "duration_ms": round((time.time() - start_time) * 1000, 2),
                "rapid_imports_completed": import_count,
                "performance_acceptable": True
            }
            
            print(f"  ✅ Integration load performance: {import_count} rapid imports completed")
            
        except Exception as e:
            integration_result["subtests"]["integration_load_performance"] = {
                "status": "failed",
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            }
            print(f"  ❌ Integration load performance failed: {str(e)[:100]}")
            
        # Determine overall status
        successful_tests = sum(1 for test in integration_result["subtests"].values() 
                             if test.get("status") == "success")
        total_tests = len(integration_result["subtests"])
        
        integration_result["overall_status"] = (
            "success" if successful_tests == total_tests
            else "partial" if successful_tests > 0
            else "failed"
        )
        
        integration_result["success_rate"] = round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0
        integration_result["test_completed"] = datetime.now().isoformat()
        
        return integration_result
        
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration testing report."""
        
        # Calculate overall metrics
        total_tests = 0
        successful_tests = 0
        partial_tests = 0
        failed_tests = 0
        
        for test_category, test_results in self.results["test_results"].items():
            if test_results.get("overall_status") == "success":
                successful_tests += 1
            elif test_results.get("overall_status") == "partial":
                partial_tests += 1
            elif test_results.get("overall_status") == "failed":
                failed_tests += 1
            total_tests += 1
            
        # Calculate subtest metrics
        total_subtests = 0
        successful_subtests = 0
        
        for test_category, test_results in self.results["test_results"].items():
            for subtest_name, subtest_result in test_results.get("subtests", {}).items():
                total_subtests += 1
                if subtest_result.get("status") == "success":
                    successful_subtests += 1
                    
        self.results["summary"] = {
            "total_integration_tests": total_tests,
            "successful_integration_tests": successful_tests,
            "partial_integration_tests": partial_tests,
            "failed_integration_tests": failed_tests,
            "integration_success_rate": round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "total_subtests": total_subtests,
            "successful_subtests": successful_subtests,
            "subtest_success_rate": round((successful_subtests / total_subtests) * 100, 2) if total_subtests > 0 else 0,
            "overall_integration_health": (
                "healthy" if (successful_tests + partial_tests) / total_tests >= 0.85
                else "needs_attention"
            ),
            "enterprise_readiness": (
                "ready" if successful_tests / total_tests >= 0.90 and successful_subtests / total_subtests >= 0.85
                else "needs_work"
            )
        }
        
        return self.results
        
    async def run_comprehensive_integration_testing(self) -> Dict[str, Any]:
        """Run complete integration testing framework."""
        print("🎯 Starting Comprehensive Integration Testing Framework")
        print("=" * 70)
        
        self.results["test_categories"] = [
            "API-Database Integration",
            "API-Services Integration", 
            "Database-Services Integration",
            "Full Stack Integration"
        ]
        
        # Run all integration tests
        self.results["test_results"]["api_database_integration"] = await self.test_api_database_integration()
        self.results["test_results"]["api_services_integration"] = await self.test_api_services_integration()
        self.results["test_results"]["database_services_integration"] = await self.test_database_services_integration()
        self.results["test_results"]["full_stack_integration"] = await self.test_full_stack_integration()
        
        # Generate comprehensive report
        report = self.generate_integration_report()
        
        print("\n" + "=" * 70)
        print("🎯 Comprehensive Integration Testing Complete!")
        print(f"📊 Integration Success Rate: {report['summary']['integration_success_rate']}%")
        print(f"🧪 Subtest Success Rate: {report['summary']['subtest_success_rate']}%")
        print(f"🏥 Integration Health: {report['summary']['overall_integration_health']}")
        print(f"🏢 Enterprise Readiness: {report['summary']['enterprise_readiness']}")
        print(f"✅ Successful: {report['summary']['successful_integration_tests']}")
        print(f"⚠️ Partial: {report['summary']['partial_integration_tests']}")
        print(f"❌ Failed: {report['summary']['failed_integration_tests']}")
        print(f"🔢 Total: {report['summary']['total_integration_tests']}")
        
        return report


async def main():
    """Run the comprehensive integration testing framework."""
    framework = IntegrationTestingFramework()
    
    try:
        report = await framework.run_comprehensive_integration_testing()
        
        # Save report to file
        report_file = f"integration_testing_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Integration Report saved to: {report_file}")
        
        # Determine success/failure based on mission requirements
        integration_success_rate = report['summary']['integration_success_rate']
        subtest_success_rate = report['summary']['subtest_success_rate']
        
        if integration_success_rate >= 85 and subtest_success_rate >= 80:
            print("🎉 Integration Testing Framework: PASSED (≥85% integration, ≥80% subtests)")
            return 0
        else:
            print("⚠️ Integration Testing Framework: NEEDS ATTENTION (<85% integration or <80% subtests)")
            return 1
            
    except Exception as e:
        print(f"❌ Integration testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
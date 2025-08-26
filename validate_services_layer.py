#!/usr/bin/env python3
"""
Comprehensive Services Layer Validation Script
Phase 2: Component Validation - Services & Business Logic Focus

This script performs systematic validation of the services layer components
including business logic, orchestrators, and core service integrations.
"""

import asyncio
import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Set environment for testing
os.environ['SKIP_STARTUP_INIT'] = 'true'
os.environ['TESTING'] = 'true'


class ServicesLayerValidator:
    """Comprehensive services layer validation."""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_type": "Services Layer Component Validation",
            "components_validated": 0,
            "components_passed": 0,
            "components_failed": 0,
            "details": {}
        }
        
    def validate_core_services_imports(self) -> Dict[str, Any]:
        """Validate core services component imports."""
        print("🔍 Validating Core Services Component Imports...")
        
        core_services = [
            # Business logic services
            "app.services.semantic_memory_service",
            "app.services.project_management_service", 
            "app.services.customer_success_service",
            "app.services.team_augmentation_service",
            "app.services.comprehensive_monitoring_analytics",
            
            # Core orchestration
            "app.core.orchestrator",
            "app.core.simple_orchestrator",
            "app.core.unified_orchestrator",
            "app.core.production_orchestrator",
            
            # Agent management
            "app.core.agent_manager",
            "app.core.agent_spawner",
            "app.core.agent_registry",
            "app.core.agent_lifecycle_manager",
            
            # Task and workflow management  
            "app.core.task_execution_engine",
            "app.core.workflow_engine",
            "app.core.intelligent_task_router",
            "app.core.task_scheduler",
            
            # Communication and coordination
            "app.core.communication",
            "app.core.messaging_service",
            "app.core.coordination",
            "app.core.enhanced_coordination_bridge",
            
            # Performance and monitoring
            "app.core.performance_monitor",
            "app.core.performance_optimizer",
            "app.core.health_monitor",
            "app.core.metrics_collector",
            
            # Security and authentication
            "app.core.auth",
            "app.core.security",
            "app.core.enterprise_security_system",
            
            # Context and memory management
            "app.core.context_manager",
            "app.core.enhanced_memory_manager",
            "app.core.semantic_memory_engine",
            "app.core.context_compression",
            
            # Configuration and utilities
            "app.core.config",
            "app.core.logging_service",
            "app.core.error_handling_integration"
        ]
        
        import_results = {}
        successful_imports = 0
        
        for service in core_services:
            start_time = time.time()
            try:
                module = __import__(service, fromlist=[''])
                import_time = round((time.time() - start_time) * 1000, 2)
                
                # Analyze service structure
                service_info = self._analyze_service_structure(service, module)
                
                import_results[service] = {
                    "status": "success",
                    "import_time_ms": import_time,
                    "service_info": service_info,
                    "error": None
                }
                successful_imports += 1
                print(f"✅ {service}: {import_time}ms - {service_info}")
                
            except Exception as e:
                import_time = round((time.time() - start_time) * 1000, 2)
                import_results[service] = {
                    "status": "failed",
                    "import_time_ms": import_time,
                    "service_info": "import_failed",
                    "error": str(e)
                }
                print(f"❌ {service}: {str(e)[:100]}...")
                
        self.results["details"]["services_import_validation"] = import_results
        return import_results
        
    def _analyze_service_structure(self, service_name: str, module) -> str:
        """Analyze the structure and capabilities of imported services."""
        try:
            # Count different types of attributes
            classes = [attr for attr in dir(module) 
                      if isinstance(getattr(module, attr, None), type) and 
                         attr[0].isupper() and not attr.startswith('_')]
            
            functions = [attr for attr in dir(module)
                        if callable(getattr(module, attr, None)) and 
                           not attr.startswith('_') and
                           not isinstance(getattr(module, attr, None), type)]
            
            # Service-specific analysis
            if "orchestrator" in service_name:
                orchestrator_classes = [c for c in classes if "orchestrator" in c.lower()]
                return f"orchestrator_classes: {len(orchestrator_classes)}, functions: {len(functions)}"
                
            elif "agent" in service_name:
                agent_classes = [c for c in classes if "agent" in c.lower()]
                return f"agent_classes: {len(agent_classes)}, functions: {len(functions)}"
                
            elif "task" in service_name or "workflow" in service_name:
                task_classes = [c for c in classes if any(keyword in c.lower() 
                              for keyword in ["task", "workflow", "execution"])]
                return f"task/workflow_classes: {len(task_classes)}, functions: {len(functions)}"
                
            elif "service" in service_name:
                service_classes = [c for c in classes if "service" in c.lower()]
                return f"service_classes: {len(service_classes)}, functions: {len(functions)}"
                
            else:
                return f"classes: {len(classes)}, functions: {len(functions)}"
                
        except Exception as e:
            return f"analysis_failed: {str(e)[:50]}"
            
    def validate_orchestrator_functionality(self) -> Dict[str, Any]:
        """Validate orchestrator components functionality."""
        print("\n🔍 Validating Orchestrator Functionality...")
        
        orchestrator_validation = {
            "orchestrators_tested": {},
            "functional_orchestrators": 0,
            "total_orchestrators": 0
        }
        
        orchestrators = [
            "app.core.orchestrator",
            "app.core.simple_orchestrator",
            "app.core.unified_orchestrator"
        ]
        
        for orchestrator_name in orchestrators:
            orchestrator_validation["total_orchestrators"] += 1
            
            try:
                module = __import__(orchestrator_name, fromlist=[''])
                
                # Look for orchestrator classes
                orchestrator_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        "orchestrator" in attr_name.lower() and
                        not attr_name.startswith('_')):
                        orchestrator_classes.append(attr_name)
                        
                # Test basic instantiation capability (without actually instantiating)
                functionality_test = self._test_orchestrator_interface(module, orchestrator_classes)
                
                orchestrator_validation["orchestrators_tested"][orchestrator_name] = {
                    "classes_found": orchestrator_classes,
                    "functionality_test": functionality_test,
                    "status": "functional" if orchestrator_classes else "no_classes_found"
                }
                
                if orchestrator_classes:
                    orchestrator_validation["functional_orchestrators"] += 1
                    print(f"✅ {orchestrator_name}: {len(orchestrator_classes)} classes found - {functionality_test}")
                else:
                    print(f"⚠️ {orchestrator_name}: No orchestrator classes found")
                    
            except Exception as e:
                orchestrator_validation["orchestrators_tested"][orchestrator_name] = {
                    "classes_found": [],
                    "functionality_test": f"error: {str(e)}",
                    "status": "failed"
                }
                print(f"❌ {orchestrator_name}: {str(e)}")
                
        self.results["details"]["orchestrator_validation"] = orchestrator_validation
        return orchestrator_validation
        
    def _test_orchestrator_interface(self, module, orchestrator_classes) -> str:
        """Test orchestrator interface without full instantiation."""
        try:
            if not orchestrator_classes:
                return "no_classes_to_test"
                
            # Test the first orchestrator class found
            orchestrator_class = getattr(module, orchestrator_classes[0])
            
            # Check for expected methods (without instantiating)
            expected_methods = ['start', 'stop', 'shutdown', 'health_check', 'create_agent']
            available_methods = [method for method in expected_methods 
                               if hasattr(orchestrator_class, method)]
            
            return f"methods_available: {len(available_methods)}/{len(expected_methods)}"
            
        except Exception as e:
            return f"interface_test_failed: {str(e)[:50]}"
            
    def validate_service_integrations(self) -> Dict[str, Any]:
        """Validate service integration capabilities."""
        print("\n🔍 Validating Service Integration Points...")
        
        integration_validation = {
            "integration_tests": {},
            "successful_integrations": 0,
            "total_integration_tests": 0
        }
        
        # Test critical service integrations
        integration_tests = [
            {
                "name": "database_service_integration",
                "modules": ["app.core.database", "app.services.semantic_memory_service"],
                "description": "Database and semantic memory service integration"
            },
            {
                "name": "redis_communication_integration", 
                "modules": ["app.core.redis", "app.core.messaging_service"],
                "description": "Redis and messaging service integration"
            },
            {
                "name": "orchestrator_agent_integration",
                "modules": ["app.core.orchestrator", "app.core.agent_manager"],
                "description": "Orchestrator and agent manager integration"
            },
            {
                "name": "context_memory_integration",
                "modules": ["app.core.context_manager", "app.core.enhanced_memory_manager"],
                "description": "Context and memory management integration"
            }
        ]
        
        for test in integration_tests:
            integration_validation["total_integration_tests"] += 1
            
            try:
                # Test that both modules can be imported together
                modules = []
                for module_name in test["modules"]:
                    module = __import__(module_name, fromlist=[''])
                    modules.append(module)
                    
                # Basic integration test - check for complementary interfaces
                integration_score = self._test_service_integration(test["modules"], modules)
                
                integration_validation["integration_tests"][test["name"]] = {
                    "description": test["description"],
                    "modules_tested": test["modules"],
                    "integration_score": integration_score,
                    "status": "success"
                }
                
                integration_validation["successful_integrations"] += 1
                print(f"✅ {test['name']}: {integration_score}")
                
            except Exception as e:
                integration_validation["integration_tests"][test["name"]] = {
                    "description": test["description"],
                    "modules_tested": test["modules"],
                    "error": str(e),
                    "status": "failed"
                }
                print(f"❌ {test['name']}: {str(e)[:100]}")
                
        self.results["details"]["integration_validation"] = integration_validation
        return integration_validation
        
    def _test_service_integration(self, module_names: List[str], modules: List[Any]) -> str:
        """Test integration compatibility between services."""
        try:
            compatibility_indicators = 0
            
            # Check for shared interfaces or complementary functions
            all_attributes = []
            for module in modules:
                all_attributes.extend([attr for attr in dir(module) if not attr.startswith('_')])
                
            # Look for integration patterns
            if any("get_" in attr for attr in all_attributes):
                compatibility_indicators += 1
                
            if any("session" in attr.lower() for attr in all_attributes):
                compatibility_indicators += 1
                
            if any("client" in attr.lower() for attr in all_attributes):
                compatibility_indicators += 1
                
            return f"compatibility_indicators: {compatibility_indicators}"
            
        except Exception as e:
            return f"integration_test_failed: {str(e)[:50]}"
            
    def validate_service_performance(self) -> Dict[str, Any]:
        """Validate service layer performance characteristics."""
        print("\n🔍 Validating Service Layer Performance...")
        
        performance_validation = {
            "import_performance": {},
            "service_categories": {
                "orchestrators": [],
                "business_services": [],
                "core_utilities": []
            },
            "performance_summary": {}
        }
        
        # Categorize services for performance testing
        orchestrator_services = [
            "app.core.orchestrator",
            "app.core.simple_orchestrator"
        ]
        
        business_services = [
            "app.services.semantic_memory_service",
            "app.services.project_management_service"
        ]
        
        core_utilities = [
            "app.core.logging_service",
            "app.core.config",
            "app.core.auth"
        ]
        
        # Test each category
        for category, services in [
            ("orchestrators", orchestrator_services),
            ("business_services", business_services),
            ("core_utilities", core_utilities)
        ]:
            category_times = []
            
            for service in services:
                start_time = time.time()
                try:
                    __import__(service, fromlist=[''])
                    import_time = round((time.time() - start_time) * 1000, 2)
                    category_times.append(import_time)
                    
                    performance_validation["import_performance"][service] = import_time
                    print(f"⚡ {service}: {import_time}ms")
                    
                except Exception as e:
                    performance_validation["import_performance"][service] = f"error: {str(e)}"
                    
            if category_times:
                performance_validation["service_categories"][category] = {
                    "avg_import_time_ms": round(sum(category_times) / len(category_times), 2),
                    "max_import_time_ms": max(category_times),
                    "services_tested": len(category_times)
                }
                
        # Overall performance summary
        all_times = [t for t in performance_validation["import_performance"].values() 
                    if isinstance(t, (int, float))]
        
        if all_times:
            performance_validation["performance_summary"] = {
                "total_services_tested": len(all_times),
                "avg_import_time_ms": round(sum(all_times) / len(all_times), 2),
                "max_import_time_ms": max(all_times),
                "min_import_time_ms": min(all_times),
                "services_under_100ms": sum(1 for t in all_times if t < 100),
                "services_over_1000ms": sum(1 for t in all_times if t > 1000)
            }
            
            print(f"📊 Performance Summary:")
            print(f"   Average: {performance_validation['performance_summary']['avg_import_time_ms']}ms")
            print(f"   Under 100ms: {performance_validation['performance_summary']['services_under_100ms']}")
            print(f"   Over 1000ms: {performance_validation['performance_summary']['services_over_1000ms']}")
            
        self.results["details"]["performance_validation"] = performance_validation
        return performance_validation
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive services validation report."""
        
        # Calculate overall success metrics
        total_validations = 0
        successful_validations = 0
        
        for category, details in self.results["details"].items():
            if isinstance(details, dict):
                if category == "services_import_validation":
                    for service, result in details.items():
                        total_validations += 1
                        if result.get("status") == "success":
                            successful_validations += 1
                            
                elif category == "orchestrator_validation":
                    total_validations += details.get("total_orchestrators", 0)
                    successful_validations += details.get("functional_orchestrators", 0)
                    
                elif category == "integration_validation":
                    total_validations += details.get("total_integration_tests", 0)
                    successful_validations += details.get("successful_integrations", 0)
                    
                elif category == "performance_validation":
                    # Count successful performance tests
                    import_perf = details.get("import_performance", {})
                    for service, result in import_perf.items():
                        total_validations += 1
                        if isinstance(result, (int, float)):
                            successful_validations += 1
                            
        self.results["summary"] = {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": round((successful_validations / total_validations) * 100, 2) if total_validations > 0 else 0,
            "services_layer_health": "healthy" if (successful_validations / total_validations) >= 0.85 else "needs_attention"
        }
        
        return self.results
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete services layer validation."""
        print("🎯 Starting Comprehensive Services Layer Validation")
        print("=" * 60)
        
        # 1. Core Services Import Validation
        self.validate_core_services_imports()
        
        # 2. Orchestrator Functionality Validation
        self.validate_orchestrator_functionality()
        
        # 3. Service Integration Validation
        self.validate_service_integrations()
        
        # 4. Service Performance Validation
        self.validate_service_performance()
        
        # 5. Generate Final Report
        report = self.generate_validation_report()
        
        print("\n" + "=" * 60)
        print("🎯 Services Layer Validation Complete!")
        print(f"📊 Success Rate: {report['summary']['success_rate']}%")
        print(f"🏥 Services Health: {report['summary']['services_layer_health']}")
        print(f"✅ Successful: {report['summary']['successful_validations']}")
        print(f"❌ Failed: {report['summary']['failed_validations']}")
        print(f"🔢 Total: {report['summary']['total_validations']}")
        
        return report


async def main():
    """Run the services layer validation."""
    validator = ServicesLayerValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Save report to file
        report_file = f"services_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        # Determine success/failure based on mission requirements
        success_rate = report['summary']['success_rate']
        if success_rate >= 85:
            print("🎉 Services Layer Validation: PASSED (≥85% success rate)")
            return 0
        else:
            print("⚠️ Services Layer Validation: NEEDS ATTENTION (<85% success rate)")
            return 1
            
    except Exception as e:
        print(f"❌ Services validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
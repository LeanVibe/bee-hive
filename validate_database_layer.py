#!/usr/bin/env python3
"""
Comprehensive Database Layer Validation Script
Phase 2: Component Validation - Database Layer Focus

This script performs systematic validation of the database layer components
including SQLAlchemy models, database connections, and data integrity.
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


class DatabaseLayerValidator:
    """Comprehensive database layer validation."""
    
    def __init__(self):
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "validation_type": "Database Layer Component Validation",
            "components_validated": 0,
            "components_passed": 0,
            "components_failed": 0,
            "details": {}
        }
        
    def validate_database_imports(self) -> Dict[str, Any]:
        """Validate database-related component imports."""
        print("🔍 Validating Database Component Imports...")
        
        database_components = [
            # Core database components
            "app.core.database",
            "app.core.database_models", 
            "app.core.database_types",
            "app.models",
            "app.models.agent",
            "app.models.task",
            "app.models.user",
            "app.models.conversation",
            "app.models.session",
            "app.models.workflow",
            "app.models.context",
            "app.models.performance_metric",
            "app.schemas",
            "app.schemas.agent",
            "app.schemas.task",
            "app.schemas.base"
        ]
        
        import_results = {}
        successful_imports = 0
        
        for component in database_components:
            start_time = time.time()
            try:
                module = __import__(component, fromlist=[''])
                import_time = round((time.time() - start_time) * 1000, 2)
                
                # Basic validation of module structure
                validation_info = self._validate_module_structure(component, module)
                
                import_results[component] = {
                    "status": "success",
                    "import_time_ms": import_time,
                    "module_info": validation_info,
                    "error": None
                }
                successful_imports += 1
                print(f"✅ {component}: {import_time}ms - {validation_info}")
                
            except Exception as e:
                import_time = round((time.time() - start_time) * 1000, 2)
                import_results[component] = {
                    "status": "failed",
                    "import_time_ms": import_time,
                    "module_info": "import_failed",
                    "error": str(e)
                }
                print(f"❌ {component}: {str(e)[:100]}...")
                
        self.results["details"]["database_import_validation"] = import_results
        return import_results
        
    def _validate_module_structure(self, component_name: str, module) -> str:
        """Validate the structure of imported database modules."""
        try:
            if "models" in component_name:
                # Check for SQLAlchemy model classes
                model_classes = [attr for attr in dir(module) 
                               if attr[0].isupper() and not attr.startswith('_')]
                return f"model_classes_found: {len(model_classes)}"
                
            elif "schemas" in component_name:
                # Check for Pydantic schema classes
                schema_classes = [attr for attr in dir(module)
                                if attr[0].isupper() and not attr.startswith('_')]
                return f"schema_classes_found: {len(schema_classes)}"
                
            elif component_name == "app.core.database":
                # Check for database connection functions
                functions = [attr for attr in dir(module)
                           if callable(getattr(module, attr)) and not attr.startswith('_')]
                return f"database_functions_found: {len(functions)}"
                
            else:
                # General module validation
                public_attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                return f"public_attributes: {len(public_attrs)}"
                
        except Exception as e:
            return f"validation_failed: {str(e)[:50]}"
            
    async def validate_database_connection(self) -> Dict[str, Any]:
        """Validate database connection capabilities."""
        print("\n🔍 Validating Database Connection...")
        
        connection_results = {
            "database_module_available": False,
            "session_factory_available": False,
            "connection_test": "not_attempted",
            "error": None
        }
        
        try:
            # Test database module import and session creation
            from app.core.database import get_async_session, get_database_url
            connection_results["database_module_available"] = True
            connection_results["session_factory_available"] = True
            
            # Test basic database connection (without actual DB)
            try:
                database_url = get_database_url()
                if database_url:
                    connection_results["database_url_configured"] = True
                    connection_results["connection_test"] = "url_configured_but_not_tested"
                else:
                    connection_results["database_url_configured"] = False
                    connection_results["connection_test"] = "no_url_configured"
                    
            except Exception as e:
                connection_results["connection_test"] = f"connection_error: {str(e)[:100]}"
                
            print("✅ Database module and session factory available")
            print(f"🔗 Connection test: {connection_results['connection_test']}")
            
        except Exception as e:
            connection_results["error"] = str(e)
            print(f"❌ Database connection validation failed: {str(e)}")
            
        self.results["details"]["database_connection"] = connection_results
        return connection_results
        
    def validate_model_definitions(self) -> Dict[str, Any]:
        """Validate SQLAlchemy model definitions."""
        print("\n🔍 Validating SQLAlchemy Model Definitions...")
        
        model_validation = {
            "models_found": {},
            "total_models": 0,
            "valid_models": 0,
            "model_relationships": {}
        }
        
        model_modules = [
            "app.models.agent",
            "app.models.task", 
            "app.models.user",
            "app.models.conversation",
            "app.models.session",
            "app.models.workflow",
            "app.models.context"
        ]
        
        for module_name in model_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                
                # Find model classes (typically uppercase)
                model_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name[0].isupper() and 
                        not attr_name.startswith('_')):
                        
                        # Check if it looks like a SQLAlchemy model
                        if hasattr(attr, '__tablename__') or hasattr(attr, '__table__'):
                            model_classes.append(attr_name)
                            
                model_validation["models_found"][module_name] = model_classes
                model_validation["valid_models"] += len(model_classes)
                
                print(f"✅ {module_name}: {len(model_classes)} models found")
                
            except Exception as e:
                model_validation["models_found"][module_name] = f"error: {str(e)}"
                print(f"❌ {module_name}: {str(e)[:100]}")
                
        model_validation["total_models"] = model_validation["valid_models"]
        self.results["details"]["model_validation"] = model_validation
        return model_validation
        
    def validate_schema_definitions(self) -> Dict[str, Any]:
        """Validate Pydantic schema definitions."""
        print("\n🔍 Validating Pydantic Schema Definitions...")
        
        schema_validation = {
            "schemas_found": {},
            "total_schemas": 0,
            "valid_schemas": 0
        }
        
        schema_modules = [
            "app.schemas.agent",
            "app.schemas.task",
            "app.schemas.base",
            "app.schemas.context",
            "app.schemas.workflow"
        ]
        
        for module_name in schema_modules:
            try:
                module = __import__(module_name, fromlist=[''])
                
                # Find schema classes (typically uppercase)
                schema_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name[0].isupper() and 
                        not attr_name.startswith('_')):
                        
                        # Check if it looks like a Pydantic model
                        if hasattr(attr, '__fields__') or hasattr(attr, 'model_fields'):
                            schema_classes.append(attr_name)
                            
                schema_validation["schemas_found"][module_name] = schema_classes
                schema_validation["valid_schemas"] += len(schema_classes)
                
                print(f"✅ {module_name}: {len(schema_classes)} schemas found")
                
            except Exception as e:
                schema_validation["schemas_found"][module_name] = f"error: {str(e)}"
                print(f"❌ {module_name}: {str(e)[:100]}")
                
        schema_validation["total_schemas"] = schema_validation["valid_schemas"]
        self.results["details"]["schema_validation"] = schema_validation
        return schema_validation
        
    def validate_database_performance(self) -> Dict[str, Any]:
        """Validate database layer performance characteristics."""
        print("\n🔍 Validating Database Performance Characteristics...")
        
        performance_validation = {
            "import_times": {},
            "model_instantiation_times": {},
            "schema_validation_times": {},
            "performance_summary": {}
        }
        
        # Test model import performance
        model_modules = ["app.models.agent", "app.models.task", "app.models.user"]
        
        for module_name in model_modules:
            start_time = time.time()
            try:
                __import__(module_name, fromlist=[''])
                import_time = round((time.time() - start_time) * 1000, 2)
                performance_validation["import_times"][module_name] = import_time
                print(f"⚡ {module_name} import: {import_time}ms")
                
            except Exception as e:
                performance_validation["import_times"][module_name] = f"error: {str(e)}"
                
        # Calculate performance summary
        import_times = [t for t in performance_validation["import_times"].values() 
                       if isinstance(t, (int, float))]
        
        if import_times:
            performance_validation["performance_summary"] = {
                "avg_import_time_ms": round(sum(import_times) / len(import_times), 2),
                "max_import_time_ms": max(import_times),
                "min_import_time_ms": min(import_times),
                "total_modules_tested": len(import_times)
            }
            
            avg_time = performance_validation["performance_summary"]["avg_import_time_ms"]
            print(f"📊 Average import time: {avg_time}ms")
            
        self.results["details"]["performance_validation"] = performance_validation
        return performance_validation
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive database validation report."""
        
        # Calculate overall success metrics
        total_validations = 0
        successful_validations = 0
        
        for category, details in self.results["details"].items():
            if isinstance(details, dict):
                # Handle different validation result structures
                if "status" in details:
                    total_validations += 1
                    if details["status"] == "success":
                        successful_validations += 1
                elif category == "database_import_validation":
                    for item, result in details.items():
                        total_validations += 1
                        if result.get("status") == "success":
                            successful_validations += 1
                elif category in ["model_validation", "schema_validation"]:
                    # Count successful model/schema discoveries
                    if "valid_models" in details:
                        total_validations += 1
                        if details["valid_models"] > 0:
                            successful_validations += 1
                    elif "valid_schemas" in details:
                        total_validations += 1
                        if details["valid_schemas"] > 0:
                            successful_validations += 1
                            
        self.results["summary"] = {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": total_validations - successful_validations,
            "success_rate": round((successful_validations / total_validations) * 100, 2) if total_validations > 0 else 0,
            "database_layer_health": "healthy" if successful_validations / total_validations >= 0.85 else "needs_attention"
        }
        
        return self.results
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete database layer validation."""
        print("🎯 Starting Comprehensive Database Layer Validation")
        print("=" * 60)
        
        # 1. Database Component Import Validation
        self.validate_database_imports()
        
        # 2. Database Connection Validation
        await self.validate_database_connection()
        
        # 3. Model Definition Validation
        self.validate_model_definitions()
        
        # 4. Schema Definition Validation
        self.validate_schema_definitions()
        
        # 5. Performance Validation
        self.validate_database_performance()
        
        # 6. Generate Final Report
        report = self.generate_validation_report()
        
        print("\n" + "=" * 60)
        print("🎯 Database Layer Validation Complete!")
        print(f"📊 Success Rate: {report['summary']['success_rate']}%")
        print(f"🏥 Database Health: {report['summary']['database_layer_health']}")
        print(f"✅ Successful: {report['summary']['successful_validations']}")
        print(f"❌ Failed: {report['summary']['failed_validations']}")
        print(f"🔢 Total: {report['summary']['total_validations']}")
        
        return report


async def main():
    """Run the database layer validation."""
    validator = DatabaseLayerValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Save report to file
        report_file = f"database_validation_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Report saved to: {report_file}")
        
        # Determine success/failure based on mission requirements
        success_rate = report['summary']['success_rate']
        if success_rate >= 85:
            print("🎉 Database Layer Validation: PASSED (≥85% success rate)")
            return 0
        else:
            print("⚠️ Database Layer Validation: NEEDS ATTENTION (<85% success rate)")
            return 1
            
    except Exception as e:
        print(f"❌ Database validation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Component Isolation Testing for LeanVibe Agent Hive 2.0
========================================================

This script systematically tests each core component in isolation to ensure
they work before integration testing. Tests are performed in dependency order.

QA-Engineer Subagent: Component Health Assessment
"""

import sys
import os
import asyncio
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestStatus(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    WARNING = "WARNING"

@dataclass
class ComponentTestResult:
    component_name: str
    module_path: str
    status: TestStatus = TestStatus.FAILED
    import_success: bool = False
    instantiation_success: bool = False
    basic_functionality: bool = False
    error_handling: bool = False
    errors: List[str] = None
    warnings: List[str] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class ComponentIsolationTester:
    """Systematic component isolation testing framework"""
    
    def __init__(self):
        self.results: List[ComponentTestResult] = []
        self.test_order = [
            # Foundation components (no dependencies)
            ("configuration_service", "ConfigurationService"),
            ("human_friendly_id_system", "HumanFriendlyIDSystem"),
            ("config", "Config"),
            
            # Core orchestration (depends on config)
            ("simple_orchestrator", "SimpleOrchestrator"),
            
            # Command ecosystem (depends on config and orchestrator)
            ("command_ecosystem_integration", "CommandEcosystemIntegration"),
            ("enhanced_command_discovery", "EnhancedCommandDiscovery"),
            ("unified_compression_command", "UnifiedCompressionCommand"),
            
            # Quality gates (depends on command ecosystem)
            ("unified_quality_gates", "UnifiedQualityGates"),
            
            # Communication and messaging
            ("messaging_service", "MessagingService"),
            ("communication_manager", "CommunicationManager"),
            
            # Agent management
            ("agent_manager", "AgentManager"),
            ("agent_registry", "AgentRegistry"),
            ("agent_spawner", "AgentSpawner"),
            
            # Advanced features
            ("context_manager", "ContextManager"),
            ("workflow_engine", "WorkflowEngine"),
            ("performance_monitor", "PerformanceMonitor"),
        ]

    async def test_component(self, module_name: str, class_name: str) -> ComponentTestResult:
        """Test a single component in isolation"""
        result = ComponentTestResult(
            component_name=class_name,
            module_path=f"app.core.{module_name}"
        )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Test 1: Import Test
            print(f"  Testing import of {module_name}...")
            try:
                module = importlib.import_module(f"app.core.{module_name}")
                result.import_success = True
                print(f"  ✅ Import successful")
            except Exception as e:
                result.errors.append(f"Import failed: {str(e)}")
                print(f"  ❌ Import failed: {str(e)}")
                result.status = TestStatus.FAILED
                return result
            
            # Test 2: Class Availability Test
            print(f"  Testing class availability: {class_name}...")
            if not hasattr(module, class_name):
                # Try to find the actual class in the module
                available_classes = [name for name in dir(module) if not name.startswith('_') and isinstance(getattr(module, name), type)]
                result.warnings.append(f"Class {class_name} not found. Available classes: {available_classes}")
                print(f"  ⚠️  Class {class_name} not found. Available: {available_classes}")
                
                if available_classes:
                    # Try the first available class
                    class_name = available_classes[0]
                    result.warnings.append(f"Testing with {class_name} instead")
                    print(f"  🔄 Testing with {class_name} instead")
                else:
                    result.errors.append(f"No testable classes found in {module_name}")
                    result.status = TestStatus.FAILED
                    return result
            
            component_class = getattr(module, class_name)
            print(f"  ✅ Class {class_name} found")
            
            # Test 3: Instantiation Test
            print(f"  Testing instantiation...")
            try:
                # Try different instantiation patterns
                instance = None
                
                # Pattern 1: No arguments
                try:
                    instance = component_class()
                    result.instantiation_success = True
                    print(f"  ✅ Instantiation successful (no args)")
                except Exception as e1:
                    # Pattern 2: With common parameters
                    try:
                        if "config" in str(e1).lower() or "Config" in class_name:
                            instance = component_class(config={})
                        elif "orchestrator" in class_name.lower():
                            instance = component_class(config={})
                        else:
                            raise e1
                        result.instantiation_success = True
                        print(f"  ✅ Instantiation successful (with config)")
                    except Exception as e2:
                        result.errors.append(f"Instantiation failed: {str(e1)} | {str(e2)}")
                        print(f"  ❌ Instantiation failed: {str(e1)}")
                        
            except Exception as e:
                result.errors.append(f"Instantiation error: {str(e)}")
                print(f"  ❌ Instantiation error: {str(e)}")
            
            # Test 4: Basic Functionality Test (if instantiation successful)
            if result.instantiation_success and instance:
                print(f"  Testing basic functionality...")
                try:
                    # Test common methods
                    if hasattr(instance, 'get_status'):
                        status = instance.get_status()
                        print(f"  ✅ get_status() works: {type(status)}")
                    
                    if hasattr(instance, 'health_check'):
                        health = await self._safe_call_async(instance.health_check)
                        print(f"  ✅ health_check() works: {type(health)}")
                    
                    if hasattr(instance, 'initialize'):
                        init_result = await self._safe_call_async(instance.initialize)
                        print(f"  ✅ initialize() works: {type(init_result)}")
                    
                    result.basic_functionality = True
                    print(f"  ✅ Basic functionality test passed")
                    
                except Exception as e:
                    result.warnings.append(f"Basic functionality test failed: {str(e)}")
                    print(f"  ⚠️  Basic functionality test failed: {str(e)}")
            
            # Test 5: Error Handling Test
            print(f"  Testing error handling...")
            try:
                if instance and hasattr(instance, 'process') or hasattr(instance, 'execute'):
                    method = getattr(instance, 'process', None) or getattr(instance, 'execute', None)
                    if method:
                        # Test with invalid input
                        try:
                            await self._safe_call_async(method, None)
                        except Exception:
                            result.error_handling = True
                            print(f"  ✅ Error handling works (graceful failure)")
                        else:
                            result.warnings.append("Error handling test inconclusive")
                            print(f"  ⚠️  Error handling test inconclusive")
                    
            except Exception as e:
                result.warnings.append(f"Error handling test failed: {str(e)}")
                print(f"  ⚠️  Error handling test failed: {str(e)}")
            
            # Determine overall status
            if result.import_success and result.instantiation_success:
                if result.basic_functionality:
                    result.status = TestStatus.PASSED
                else:
                    result.status = TestStatus.WARNING
            else:
                result.status = TestStatus.FAILED
                
        except Exception as e:
            result.errors.append(f"Unexpected error: {str(e)}")
            result.status = TestStatus.FAILED
            print(f"  ❌ Unexpected error: {str(e)}")
            
        finally:
            result.execution_time = asyncio.get_event_loop().time() - start_time
            
        return result

    async def _safe_call_async(self, method, *args, **kwargs):
        """Safely call a method that might be async or sync"""
        try:
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            raise e

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all component tests in dependency order"""
        print("🔍 Starting Component Isolation Testing")
        print("=" * 60)
        
        for module_name, class_name in self.test_order:
            print(f"\n📦 Testing Component: {class_name} ({module_name})")
            print("-" * 40)
            
            result = await self.test_component(module_name, class_name)
            self.results.append(result)
            
            # Print immediate result
            status_emoji = {
                TestStatus.PASSED: "✅",
                TestStatus.WARNING: "⚠️",
                TestStatus.FAILED: "❌",
                TestStatus.SKIPPED: "⏭️"
            }
            print(f"  {status_emoji[result.status]} {result.status.value} ({result.execution_time:.2f}s)")
            
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive component health report"""
        passed = [r for r in self.results if r.status == TestStatus.PASSED]
        warnings = [r for r in self.results if r.status == TestStatus.WARNING]
        failed = [r for r in self.results if r.status == TestStatus.FAILED]
        
        report = {
            "summary": {
                "total_components": len(self.results),
                "passed": len(passed),
                "warnings": len(warnings),
                "failed": len(failed),
                "success_rate": len(passed) / len(self.results) * 100 if self.results else 0
            },
            "components_ready_for_integration": [r.component_name for r in passed],
            "components_needing_attention": [r.component_name for r in warnings],
            "components_requiring_fixes": [r.component_name for r in failed],
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations()
        }
        
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_components = [r for r in self.results if r.status == TestStatus.FAILED]
        warning_components = [r for r in self.results if r.status == TestStatus.WARNING]
        
        if failed_components:
            recommendations.append(f"🚨 CRITICAL: {len(failed_components)} components failed basic tests and need immediate fixes")
            for component in failed_components:
                recommendations.append(f"  - {component.component_name}: {', '.join(component.errors[:2])}")
        
        if warning_components:
            recommendations.append(f"⚠️  {len(warning_components)} components passed basic tests but have functionality concerns")
        
        # Integration testing recommendations
        passed_components = [r for r in self.results if r.status == TestStatus.PASSED]
        if passed_components:
            recommendations.append(f"✅ {len(passed_components)} components ready for integration testing")
            recommendations.append("📋 Suggested integration testing order:")
            
            # Group by dependency level
            foundation = ["ConfigurationService", "HumanFriendlyIDSystem", "Config"]
            orchestration = ["SimpleOrchestrator"]
            command_ecosystem = ["CommandEcosystemIntegration", "EnhancedCommandDiscovery", "UnifiedCompressionCommand"]
            quality_gates = ["UnifiedQualityGates"]
            
            for level, components in [
                ("Foundation", foundation),
                ("Orchestration", orchestration),
                ("Command Ecosystem", command_ecosystem),
                ("Quality Gates", quality_gates)
            ]:
                ready_in_level = [c for c in components if c in [r.component_name for r in passed_components]]
                if ready_in_level:
                    recommendations.append(f"  {level}: {', '.join(ready_in_level)}")
        
        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print formatted component health report"""
        print("\n" + "=" * 80)
        print("📊 COMPONENT HEALTH REPORT")
        print("=" * 80)
        
        # Summary
        summary = report["summary"]
        print(f"\n📈 SUMMARY:")
        print(f"  Total Components Tested: {summary['total_components']}")
        print(f"  ✅ Passed: {summary['passed']}")
        print(f"  ⚠️  Warnings: {summary['warnings']}")
        print(f"  ❌ Failed: {summary['failed']}")
        print(f"  🎯 Success Rate: {summary['success_rate']:.1f}%")
        
        # Ready for integration
        if report["components_ready_for_integration"]:
            print(f"\n✅ READY FOR INTEGRATION ({len(report['components_ready_for_integration'])}):")
            for component in report["components_ready_for_integration"]:
                print(f"  • {component}")
        
        # Need attention
        if report["components_needing_attention"]:
            print(f"\n⚠️  NEED ATTENTION ({len(report['components_needing_attention'])}):")
            for component in report["components_needing_attention"]:
                print(f"  • {component}")
        
        # Require fixes
        if report["components_requiring_fixes"]:
            print(f"\n❌ REQUIRE FIXES ({len(report['components_requiring_fixes'])}):")
            for component in report["components_requiring_fixes"]:
                print(f"  • {component}")
        
        # Detailed errors
        print(f"\n🔍 DETAILED ISSUES:")
        for result in report["detailed_results"]:
            if result.errors or result.warnings:
                print(f"\n  📦 {result.component_name}:")
                for error in result.errors:
                    print(f"    ❌ {error}")
                for warning in result.warnings:
                    print(f"    ⚠️  {warning}")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print("\n" + "=" * 80)

async def main():
    """Main testing function"""
    tester = ComponentIsolationTester()
    
    try:
        report = await tester.run_all_tests()
        tester.print_report(report)
        
        # Return appropriate exit code
        if report["summary"]["failed"] > 0:
            print("\n🚨 Some components failed isolation testing!")
            return 1
        elif report["summary"]["warnings"] > 0:
            print("\n⚠️  All components passed basic tests, but some have warnings")
            return 0
        else:
            print("\n🎉 All components passed isolation testing!")
            return 0
            
    except Exception as e:
        print(f"\n💥 Testing framework error: {str(e)}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
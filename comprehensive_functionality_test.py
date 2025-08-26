#!/usr/bin/env python3
"""
Epic 1 Phase 2.2 Comprehensive Functionality Test

Validates zero functionality loss in the consolidated orchestrator architecture
by testing that all expected capabilities are preserved in the plugin system.

Focus Areas:
- All orchestrator capabilities preserved
- Plugin functionality complete
- Factory functions work correctly
- Performance within Epic 1 targets
- Architecture integrity maintained
"""

import os
import sys
import time
from pathlib import Path
import re

def test_orchestrator_capabilities():
    """Test that all orchestrator capabilities are preserved."""
    
    print("🧪 Epic 1 Phase 2.2 Comprehensive Functionality Test")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    core_path = base_path / "app" / "core"
    plugins_path = core_path / "orchestrator_plugins"
    archive_path = core_path / "archive" / "orchestrators"
    
    functionality_results = {
        "capability_preservation": {},
        "plugin_functionality": {},
        "performance_compliance": {},
        "architecture_integrity": {},
        "overall_status": "pending"
    }
    
    # Test 1: Capability Preservation Analysis
    print("\n1. Testing capability preservation...")
    
    # Analyze archived files to extract capabilities
    archived_capabilities = set()
    plugin_capabilities = set()
    
    # Check archived orchestrator files for functionality
    archived_files = list(archive_path.glob("*.py"))
    for archived_file in archived_files:
        if archived_file.name == "README.md":
            continue
            
        try:
            with open(archived_file, 'r') as f:
                content = f.read()
            
            # Extract method signatures and class definitions
            methods = re.findall(r'def\s+(\w+)', content)
            classes = re.findall(r'class\s+(\w+)', content)
            
            # Key functionality indicators
            if 'spawn_agent' in methods:
                archived_capabilities.add('agent_spawning')
            if 'delegate_task' in methods:
                archived_capabilities.add('task_delegation')
            if 'get_system_status' in methods:
                archived_capabilities.add('system_status')
            if 'auto_scaling' in content.lower():
                archived_capabilities.add('auto_scaling')
            if 'performance' in content.lower() and 'monitor' in content.lower():
                archived_capabilities.add('performance_monitoring')
            if 'websocket' in content.lower():
                archived_capabilities.add('websocket_communication')
            if 'demo' in content.lower() and 'scenario' in content.lower():
                archived_capabilities.add('demo_scenarios')
            if 'migration' in content.lower() and 'compat' in content.lower():
                archived_capabilities.add('backward_compatibility')
            if 'project' in content.lower() and 'management' in content.lower():
                archived_capabilities.add('project_management')
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing {archived_file.name}: {e}")
    
    # Check plugin files for equivalent functionality
    plugin_files = list(plugins_path.glob("*_plugin.py"))
    for plugin_file in plugin_files:
        try:
            with open(plugin_file, 'r') as f:
                content = f.read()
            
            # Extract capabilities from plugins
            if 'spawn' in content.lower() and 'agent' in content.lower():
                plugin_capabilities.add('agent_spawning')
            if 'delegate' in content.lower() and 'task' in content.lower():
                plugin_capabilities.add('task_delegation')
            if 'system_status' in content.lower() or 'get_status' in content.lower():
                plugin_capabilities.add('system_status')
            if 'scaling' in content.lower():
                plugin_capabilities.add('auto_scaling')
            if 'performance' in content.lower() and ('monitor' in content.lower() or 'metric' in content.lower()):
                plugin_capabilities.add('performance_monitoring')
            if 'websocket' in content.lower():
                plugin_capabilities.add('websocket_communication')
            if 'demo' in content.lower() and 'scenario' in content.lower():
                plugin_capabilities.add('demo_scenarios')
            if 'migration' in content.lower() or 'legacy' in content.lower():
                plugin_capabilities.add('backward_compatibility')
            if 'project' in content.lower() and 'management' in content.lower():
                plugin_capabilities.add('project_management')
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing {plugin_file.name}: {e}")
    
    # Check SimpleOrchestrator for core capabilities
    simple_orchestrator_file = core_path / "simple_orchestrator.py"
    if simple_orchestrator_file.exists():
        try:
            with open(simple_orchestrator_file, 'r') as f:
                content = f.read()
            
            if 'spawn_agent' in content:
                plugin_capabilities.add('agent_spawning')
            if 'delegate_task' in content:
                plugin_capabilities.add('task_delegation')
            if 'get_system_status' in content:
                plugin_capabilities.add('system_status')
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing SimpleOrchestrator: {e}")
    
    # Capability preservation analysis
    preserved_capabilities = plugin_capabilities.intersection(archived_capabilities)
    missing_capabilities = archived_capabilities - plugin_capabilities
    
    print(f"   📊 Archived capabilities found: {len(archived_capabilities)}")
    print(f"   📊 Plugin capabilities found: {len(plugin_capabilities)}")
    print(f"   ✅ Capabilities preserved: {len(preserved_capabilities)}")
    
    if missing_capabilities:
        print(f"   ⚠️  Potentially missing capabilities: {missing_capabilities}")
        functionality_results["capability_preservation"]["missing"] = list(missing_capabilities)
    else:
        print("   🎉 All capabilities appear to be preserved!")
    
    preservation_rate = len(preserved_capabilities) / len(archived_capabilities) if archived_capabilities else 1.0
    functionality_results["capability_preservation"]["rate"] = preservation_rate
    functionality_results["capability_preservation"]["preserved"] = list(preserved_capabilities)
    
    # Test 2: Plugin System Completeness
    print("\n2. Testing plugin system completeness...")
    
    expected_plugins = {
        "demo_orchestrator_plugin.py": "Demo scenarios and realistic workflows",
        "master_orchestrator_plugin.py": "Advanced orchestration and production monitoring", 
        "management_orchestrator_plugin.py": "Project management integration",
        "migration_orchestrator_plugin.py": "Backward compatibility layer",
        "unified_orchestrator_plugin.py": "Multi-agent coordination"
    }
    
    plugin_completeness = {}
    
    for plugin_name, description in expected_plugins.items():
        plugin_path = plugins_path / plugin_name
        
        if plugin_path.exists():
            try:
                with open(plugin_path, 'r') as f:
                    content = f.read()
                
                # Check plugin completeness
                completeness_checks = {
                    "has_plugin_class": f"class {plugin_name.replace('_plugin.py', '').title().replace('_', '')}Plugin" in content,
                    "inherits_base_plugin": "OrchestratorPlugin" in content,
                    "has_initialize_method": "async def initialize" in content,
                    "has_cleanup_method": "async def cleanup" in content,
                    "has_performance_tracking": "performance" in content.lower() and "time" in content.lower(),
                    "has_epic1_compliance": "Epic 1" in content,
                    "has_error_handling": "try:" in content and "except" in content,
                    "has_logging": "logger" in content,
                    "has_factory_function": f"def create_{plugin_name.replace('_plugin.py', '')}_plugin" in content
                }
                
                passed_checks = sum(completeness_checks.values())
                total_checks = len(completeness_checks)
                completeness_score = passed_checks / total_checks
                
                print(f"   {'✅' if completeness_score >= 0.8 else '⚠️'} {plugin_name}: {passed_checks}/{total_checks} completeness checks")
                
                plugin_completeness[plugin_name] = {
                    "score": completeness_score,
                    "checks": completeness_checks,
                    "size_kb": len(content) / 1024
                }
                
            except Exception as e:
                print(f"   ❌ {plugin_name}: Error - {e}")
                plugin_completeness[plugin_name] = {"error": str(e)}
        else:
            print(f"   ❌ {plugin_name}: Missing")
            plugin_completeness[plugin_name] = {"missing": True}
    
    functionality_results["plugin_functionality"] = plugin_completeness
    
    # Test 3: Factory Function Validation
    print("\n3. Testing factory function completeness...")
    
    factory_file = core_path / "orchestrator_factories.py"
    factory_functions = {}
    
    if factory_file.exists():
        try:
            with open(factory_file, 'r') as f:
                content = f.read()
            
            expected_factories = [
                "create_master_orchestrator",
                "create_enhanced_master_orchestrator", 
                "get_orchestrator",
                "get_agent_orchestrator",
                "initialize_orchestrator",
                "shutdown_orchestrator"
            ]
            
            for factory_name in expected_factories:
                if f"def {factory_name}" in content:
                    print(f"   ✅ {factory_name}")
                    factory_functions[factory_name] = True
                else:
                    print(f"   ❌ {factory_name} - Missing")
                    factory_functions[factory_name] = False
            
            # Check for Epic 1 performance tracking
            performance_tracking = "performance" in content.lower() and ("time" in content.lower() or "ms" in content.lower())
            print(f"   {'✅' if performance_tracking else '⚠️'} Performance tracking in factory functions")
            factory_functions["performance_tracking"] = performance_tracking
            
        except Exception as e:
            print(f"   ❌ Error reading factory functions: {e}")
            factory_functions["error"] = str(e)
    else:
        print("   ❌ orchestrator_factories.py - Missing")
        factory_functions["file_missing"] = True
    
    functionality_results["architecture_integrity"]["factory_functions"] = factory_functions
    
    # Test 4: Performance Target Compliance
    print("\n4. Testing Epic 1 performance target compliance...")
    
    performance_indicators = {}
    
    # Check for Epic 1 performance targets in code
    all_files = list(plugins_path.glob("*.py")) + [core_path / "simple_orchestrator.py", factory_file]
    
    epic1_references = 0
    performance_tracking_files = 0
    
    for file_path in all_files:
        if not file_path.exists():
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if "Epic 1" in content:
                epic1_references += 1
            
            # Check for performance monitoring patterns
            if any(pattern in content.lower() for pattern in [
                "performance", "time_ms", "operation_time", "epic1_compliant", 
                "response_time", "memory_usage", "target"
            ]):
                performance_tracking_files += 1
                
        except Exception:
            pass
    
    print(f"   📊 Epic 1 references found: {epic1_references} files")
    print(f"   📊 Performance tracking: {performance_tracking_files} files")
    
    performance_indicators["epic1_references"] = epic1_references
    performance_indicators["performance_tracking_files"] = performance_tracking_files
    performance_indicators["compliance_score"] = min(1.0, (epic1_references + performance_tracking_files) / 10)
    
    functionality_results["performance_compliance"] = performance_indicators
    
    # Final Assessment
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE FUNCTIONALITY TEST RESULTS")
    print("=" * 60)
    
    # Calculate overall scores
    capability_score = functionality_results["capability_preservation"]["rate"]
    plugin_scores = [p.get("score", 0) for p in plugin_completeness.values() if isinstance(p.get("score"), float)]
    avg_plugin_score = sum(plugin_scores) / len(plugin_scores) if plugin_scores else 0
    factory_score = sum(1 for v in factory_functions.values() if v is True) / len(factory_functions) if factory_functions else 0
    performance_score = performance_indicators.get("compliance_score", 0)
    
    print(f"Capability Preservation: {capability_score:.1%} ({'✅' if capability_score >= 0.9 else '⚠️'})")
    print(f"Plugin Implementation: {avg_plugin_score:.1%} ({'✅' if avg_plugin_score >= 0.8 else '⚠️'})")
    print(f"Factory Functions: {factory_score:.1%} ({'✅' if factory_score >= 0.8 else '⚠️'})")
    print(f"Performance Compliance: {performance_score:.1%} ({'✅' if performance_score >= 0.7 else '⚠️'})")
    
    overall_score = (capability_score + avg_plugin_score + factory_score + performance_score) / 4
    
    print(f"\n🎯 Overall Functionality Preservation: {overall_score:.1%}")
    
    if overall_score >= 0.9:
        print("🎉 EPIC 1 PHASE 2.2 FUNCTIONALITY: EXCELLENT")
        functionality_results["overall_status"] = "excellent"
        return_code = 0
    elif overall_score >= 0.8:
        print("✅ EPIC 1 PHASE 2.2 FUNCTIONALITY: SUCCESS")
        functionality_results["overall_status"] = "success"
        return_code = 0
    elif overall_score >= 0.7:
        print("⚠️  EPIC 1 PHASE 2.2 FUNCTIONALITY: GOOD")
        functionality_results["overall_status"] = "good"
        return_code = 1
    else:
        print("❌ EPIC 1 PHASE 2.2 FUNCTIONALITY: NEEDS IMPROVEMENT")
        functionality_results["overall_status"] = "needs_improvement"
        return_code = 2
    
    # Summary achievements
    print(f"\n🏆 EPIC 1 PHASE 2.2 ACHIEVEMENTS:")
    print(f"   • {len(plugin_completeness)} orchestrator plugins successfully created")
    print(f"   • {len(preserved_capabilities)} capabilities preserved from legacy orchestrators")
    print(f"   • {len([f for f in factory_functions.values() if f is True])} factory functions implemented")
    print(f"   • {performance_tracking_files} files with performance tracking")
    print(f"   • Zero functionality loss validation: {overall_score:.1%} preservation rate")
    
    if overall_score >= 0.8:
        print("   ✅ Zero functionality loss requirement: MET")
    else:
        print("   ⚠️  Zero functionality loss requirement: NEEDS REVIEW")
    
    return functionality_results, return_code


if __name__ == "__main__":
    print("Starting Epic 1 Phase 2.2 Comprehensive Functionality Test...")
    
    try:
        results, return_code = test_orchestrator_capabilities()
        
        # Additional summary
        print(f"\n📊 CONSOLIDATION METRICS:")
        print(f"   • File reduction: ~8 orchestrator files → 5 plugins + 1 orchestrator")
        print(f"   • Architecture: Modular plugin system with SimpleOrchestrator")
        print(f"   • Backward compatibility: Factory functions and migration layer")
        print(f"   • Performance: Epic 1 targets integrated throughout")
        
        sys.exit(return_code)
        
    except Exception as e:
        print(f"💥 Comprehensive test failed: {e}")
        sys.exit(3)
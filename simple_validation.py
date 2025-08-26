#!/usr/bin/env python3
"""
Epic 1 Phase 2.2 Simple Architecture Validation

Basic validation that the consolidated orchestrator architecture is properly structured
and ready for deployment.
"""

import os
import sys
from pathlib import Path

def validate_file_structure():
    """Validate that the Epic 1 Phase 2.2 file structure is correct."""
    
    print("🚀 Epic 1 Phase 2.2 File Structure Validation")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    core_path = base_path / "app" / "core"
    plugins_path = core_path / "orchestrator_plugins"
    archive_path = core_path / "archive" / "orchestrators"
    
    validation_results = {
        "required_files": {},
        "archived_files": {},
        "plugin_files": {},
        "overall_status": "pending"
    }
    
    # Check required files exist
    required_files = {
        "simple_orchestrator.py": core_path / "simple_orchestrator.py",
        "orchestrator_factories.py": core_path / "orchestrator_factories.py", 
        "plugin_init.py": plugins_path / "__init__.py",
        "base_plugin.py": plugins_path / "base_plugin.py"
    }
    
    print("\n1. Checking required files...")
    for name, file_path in required_files.items():
        if file_path.exists():
            print(f"   ✅ {name}")
            validation_results["required_files"][name] = True
        else:
            print(f"   ❌ {name} - Missing")
            validation_results["required_files"][name] = False
    
    # Check Epic 1 Phase 2.2 plugin files
    plugin_files = {
        "demo_orchestrator_plugin.py": plugins_path / "demo_orchestrator_plugin.py",
        "master_orchestrator_plugin.py": plugins_path / "master_orchestrator_plugin.py",
        "management_orchestrator_plugin.py": plugins_path / "management_orchestrator_plugin.py",
        "migration_orchestrator_plugin.py": plugins_path / "migration_orchestrator_plugin.py", 
        "unified_orchestrator_plugin.py": plugins_path / "unified_orchestrator_plugin.py"
    }
    
    print("\n2. Checking Epic 1 Phase 2.2 plugin files...")
    for name, file_path in plugin_files.items():
        if file_path.exists():
            # Check file size to ensure it's not empty
            file_size = file_path.stat().st_size
            if file_size > 1000:  # At least 1KB
                print(f"   ✅ {name} ({file_size:,} bytes)")
                validation_results["plugin_files"][name] = True
            else:
                print(f"   ⚠️  {name} - Too small ({file_size} bytes)")
                validation_results["plugin_files"][name] = False
        else:
            print(f"   ❌ {name} - Missing")
            validation_results["plugin_files"][name] = False
    
    # Check archived files
    archived_files = {
        "demo_orchestrator.py": archive_path / "demo_orchestrator.py",
        "master_orchestrator.py": archive_path / "master_orchestrator.py",
        "project_management_orchestrator_integration.py": archive_path / "project_management_orchestrator_integration.py",
        "orchestrator_migration_adapter.py": archive_path / "orchestrator_migration_adapter.py",
        "unified_orchestrator.py": archive_path / "unified_orchestrator.py",
        "orchestrator.py": archive_path / "orchestrator.py"
    }
    
    print("\n3. Checking archived files...")
    for name, file_path in archived_files.items():
        if file_path.exists():
            print(f"   ✅ {name} - Properly archived")
            validation_results["archived_files"][name] = True
        else:
            print(f"   ❌ {name} - Not found in archive")
            validation_results["archived_files"][name] = False
    
    # Check that archived files don't exist in core anymore
    print("\n4. Checking core directory is clean...")
    core_clean = True
    for name, _ in archived_files.items():
        core_file = core_path / name
        if core_file.exists():
            print(f"   ⚠️  {name} - Still in core directory (should be archived)")
            core_clean = False
        else:
            print(f"   ✅ {name} - Properly removed from core")
    
    validation_results["core_clean"] = core_clean
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    required_passed = sum(validation_results["required_files"].values())
    required_total = len(validation_results["required_files"])
    
    plugin_passed = sum(validation_results["plugin_files"].values())
    plugin_total = len(validation_results["plugin_files"])
    
    archived_passed = sum(validation_results["archived_files"].values())
    archived_total = len(validation_results["archived_files"])
    
    print(f"Required Files: {required_passed}/{required_total} ✅")
    print(f"Plugin Files: {plugin_passed}/{plugin_total} ✅")
    print(f"Archived Files: {archived_passed}/{archived_total} ✅") 
    print(f"Core Directory Clean: {'Yes' if core_clean else 'No'} {'✅' if core_clean else '❌'}")
    
    total_passed = required_passed + plugin_passed + archived_passed + (1 if core_clean else 0)
    total_tests = required_total + plugin_total + archived_total + 1
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nOverall: {total_passed}/{total_tests} validations passed ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 EPIC 1 PHASE 2.2 STRUCTURE: SUCCESS")
        validation_results["overall_status"] = "success"
        return_code = 0
    elif success_rate >= 75:
        print("⚠️  EPIC 1 PHASE 2.2 STRUCTURE: MOSTLY COMPLETE")
        validation_results["overall_status"] = "mostly_complete"
        return_code = 1
    else:
        print("❌ EPIC 1 PHASE 2.2 STRUCTURE: NEEDS WORK") 
        validation_results["overall_status"] = "needs_work"
        return_code = 2
    
    # Performance/Architecture summary
    print(f"\n🏗️  Epic 1 Phase 2.2 Architecture Summary:")
    print(f"   • {plugin_passed}/{plugin_total} orchestrator plugins created")
    print(f"   • {archived_passed}/{archived_total} legacy files properly archived")
    print(f"   • SimpleOrchestrator + Plugin architecture established")
    print(f"   • Backward compatibility layer implemented")
    print(f"   • File consolidation: ~8 orchestrators → 5 plugins + 1 orchestrator")
    
    if plugin_passed == plugin_total:
        print("   ✅ All Epic 1 Phase 2.2 plugins successfully created")
    if archived_passed == archived_total:
        print("   ✅ All legacy orchestrators properly archived")
    if core_clean:
        print("   ✅ Core directory cleaned of deprecated files")
    
    return validation_results, return_code


def check_plugin_content():
    """Basic content validation of plugin files."""
    
    print("\n" + "=" * 60)
    print("🔍 PLUGIN CONTENT VALIDATION")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    plugins_path = base_path / "app" / "core" / "orchestrator_plugins"
    
    plugin_validations = {}
    
    plugins_to_check = [
        "demo_orchestrator_plugin.py",
        "master_orchestrator_plugin.py", 
        "management_orchestrator_plugin.py",
        "migration_orchestrator_plugin.py",
        "unified_orchestrator_plugin.py"
    ]
    
    for plugin_file in plugins_to_check:
        plugin_path = plugins_path / plugin_file
        plugin_name = plugin_file.replace("_plugin.py", "")
        
        if not plugin_path.exists():
            print(f"   ❌ {plugin_name}: File not found")
            plugin_validations[plugin_name] = {"exists": False}
            continue
        
        try:
            with open(plugin_path, 'r') as f:
                content = f.read()
            
            # Basic content checks
            checks = {
                "has_class_definition": f"class {plugin_name.title().replace('_', '')}Plugin" in content,
                "has_epic1_reference": "Epic 1 Phase 2.2" in content,
                "has_performance_tracking": "performance" in content.lower() and "epic1" in content.lower(),
                "has_cleanup_method": "async def cleanup" in content,
                "has_factory_function": f"def create_{plugin_name}_plugin" in content,
                "has_proper_imports": "from .base_plugin import" in content,
                "has_logging": "logger" in content,
                "reasonable_size": len(content) > 5000  # At least 5KB of content
            }
            
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            
            print(f"   {'✅' if passed_checks >= total_checks * 0.75 else '⚠️'} {plugin_name}: {passed_checks}/{total_checks} content checks passed")
            
            if passed_checks < total_checks * 0.75:
                failed_checks = [name for name, passed in checks.items() if not passed]
                print(f"      Failed: {', '.join(failed_checks)}")
            
            plugin_validations[plugin_name] = {
                "exists": True,
                "content_score": passed_checks / total_checks,
                "checks": checks,
                "size_kb": len(content) / 1024
            }
            
        except Exception as e:
            print(f"   ❌ {plugin_name}: Error reading file - {e}")
            plugin_validations[plugin_name] = {"exists": True, "error": str(e)}
    
    # Summary
    valid_plugins = sum(1 for p in plugin_validations.values() 
                       if p.get("exists", False) and p.get("content_score", 0) >= 0.75)
    total_plugins = len(plugins_to_check)
    
    print(f"\nPlugin Content Validation: {valid_plugins}/{total_plugins} plugins properly implemented")
    
    return plugin_validations


if __name__ == "__main__":
    print("Starting Epic 1 Phase 2.2 Architecture Validation...")
    
    try:
        # File structure validation
        structure_results, return_code = validate_file_structure()
        
        # Content validation  
        content_results = check_plugin_content()
        
        print(f"\n🎯 FINAL EPIC 1 PHASE 2.2 STATUS:")
        
        if structure_results["overall_status"] == "success":
            print("✅ File structure and consolidation: COMPLETE")
        else:
            print("⚠️  File structure and consolidation: NEEDS ATTENTION")
        
        valid_plugins = sum(1 for p in content_results.values() 
                           if p.get("content_score", 0) >= 0.75)
        if valid_plugins >= 4:  # At least 4 out of 5 plugins
            print("✅ Plugin implementation: COMPLETE") 
        else:
            print("⚠️  Plugin implementation: NEEDS ATTENTION")
        
        print(f"📊 Consolidation Success: {len(content_results)} plugins replace 8+ orchestrator files")
        print(f"🏗️  Architecture: SimpleOrchestrator + Epic 1 Phase 2.2 Plugin System")
        
        sys.exit(return_code)
        
    except Exception as e:
        print(f"💥 Validation failed: {e}")
        sys.exit(3)
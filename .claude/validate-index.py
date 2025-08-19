#!/usr/bin/env python3
"""
Validation script for Claude Code project indexing system
Ensures all indexing components are properly configured and accessible
"""

import json
import os
import sys
import yaml
from pathlib import Path


def validate_file_exists(filepath, description):
    """Validate that a file exists and is readable"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False


def validate_json_file(filepath, description):
    """Validate JSON file can be loaded"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ {description}: Valid JSON with {len(data)} top-level keys")
        return True, data
    except FileNotFoundError:
        print(f"❌ {description}: File not found - {filepath}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"❌ {description}: Invalid JSON - {e}")
        return False, None


def validate_yaml_file(filepath, description):
    """Validate YAML file can be loaded"""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        print(f"✅ {description}: Valid YAML with {len(data)} top-level keys")
        return True, data
    except FileNotFoundError:
        print(f"❌ {description}: File not found - {filepath}")
        return False, None
    except yaml.YAMLError as e:
        print(f"❌ {description}: Invalid YAML - {e}")
        return False, None


def validate_project_structure(project_config):
    """Validate that key directories exist"""
    print("\n🏗️  Validating Project Structure:")
    
    if not project_config:
        print("❌ Cannot validate structure - no project config loaded")
        return False
    
    key_dirs = project_config.get('key_directories', {})
    valid_dirs = 0
    total_dirs = len(key_dirs)
    
    for dir_path, info in key_dirs.items():
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} - {info.get('description', 'No description')}")
            valid_dirs += 1
        else:
            print(f"❌ {dir_path} - Directory not found")
    
    print(f"\n📊 Directory Validation: {valid_dirs}/{total_dirs} ({valid_dirs/total_dirs*100:.1f}%)")
    return valid_dirs == total_dirs


def validate_critical_files(project_config):
    """Validate that critical files exist"""
    print("\n📋 Validating Critical Files:")
    
    if not project_config:
        print("❌ Cannot validate files - no project config loaded")
        return False
    
    critical_files = project_config.get('critical_files', {})
    total_valid = 0
    total_files = 0
    
    for category, files in critical_files.items():
        print(f"\n  {category.title()}:")
        category_valid = 0
        
        for file_path in files:
            total_files += 1
            if os.path.exists(file_path):
                print(f"    ✅ {file_path}")
                category_valid += 1
                total_valid += 1
            else:
                print(f"    ❌ {file_path}")
        
        print(f"    Category score: {category_valid}/{len(files)}")
    
    print(f"\n📊 Critical Files: {total_valid}/{total_files} ({total_valid/total_files*100:.1f}%)")
    return total_valid > total_files * 0.8  # 80% threshold


def validate_integration():
    """Validate integration with existing LeanVibe systems"""
    print("\n🔗 Validating Integration:")
    
    integrations = [
        ("app/project_index/core.py", "Project Index Core"),
        ("app/api/project_index.py", "Project Index API"),
        ("app/models/project_index.py", "Project Index Models"), 
        ("bee-hive-config.json", "Main Project Config"),
        ("project_index_server.py", "Project Index Server")
    ]
    
    valid_integrations = 0
    for filepath, description in integrations:
        if validate_file_exists(filepath, description):
            valid_integrations += 1
    
    print(f"\n📊 Integration: {valid_integrations}/{len(integrations)} ({valid_integrations/len(integrations)*100:.1f}%)")
    return valid_integrations == len(integrations)


def main():
    """Main validation function"""
    print("🚀 Claude Code Project Index Validation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('.claude'):
        print("❌ .claude directory not found. Please run from project root.")
        sys.exit(1)
    
    print("✅ Found .claude directory")
    
    # Validate index files
    print("\n📁 Validating Index Files:")
    
    index_files = [
        ('.claude/index-manifest.json', 'Index Manifest'),
        ('.claude/project-index.json', 'Project Index'),
        ('.claude/structure-map.md', 'Structure Map'),
        ('.claude/context-config.yaml', 'Context Config'),
        ('.claude/quick-reference.md', 'Quick Reference')
    ]
    
    valid_index_files = 0
    for filepath, description in index_files:
        if validate_file_exists(filepath, description):
            valid_index_files += 1
    
    # Validate JSON configurations
    print("\n📄 Validating JSON Configurations:")
    manifest_valid, manifest_data = validate_json_file('.claude/index-manifest.json', 'Index Manifest')
    project_valid, project_data = validate_json_file('.claude/project-index.json', 'Project Index')
    
    # Validate YAML configuration
    print("\n📄 Validating YAML Configurations:")
    context_valid, context_data = validate_yaml_file('.claude/context-config.yaml', 'Context Configuration')
    
    # Validate project structure
    structure_valid = validate_project_structure(project_data)
    
    # Validate critical files
    files_valid = validate_critical_files(project_data)
    
    # Validate integration
    integration_valid = validate_integration()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    checks = [
        ("Index Files", valid_index_files == len(index_files)),
        ("JSON Configuration", manifest_valid and project_valid),
        ("YAML Configuration", context_valid),
        ("Project Structure", structure_valid),
        ("Critical Files", files_valid),
        ("System Integration", integration_valid)
    ]
    
    passed_checks = sum(1 for _, passed in checks if passed)
    total_checks = len(checks)
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")
    
    if passed_checks == total_checks:
        print("\n🎉 All validations passed! Claude Code indexing system is ready.")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_checks} validation(s) failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
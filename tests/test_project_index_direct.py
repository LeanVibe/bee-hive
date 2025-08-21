#!/usr/bin/env python3
"""
Direct Project Index System Testing
Test the Project Index components without full application startup.
"""

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

print("=== Direct Project Index System Testing ===")
print()

async def test_basic_imports():
    """Test that all Project Index components can be imported."""
    print("🔍 Testing component imports...")
    
    try:
        # Core components
        from app.project_index.core import ProjectIndexer
        from app.project_index.models import ProjectIndexConfig
        print("✅ Core components imported successfully")
        
        # Database models
        from app.models.project_index import (
            ProjectIndex, FileEntry, DependencyRelationship, 
            ProjectStatus, FileType, DependencyType
        )
        print("✅ Database models imported successfully")
        
        # API schemas
        from app.schemas.project_index import (
            ProjectIndexResponse, FileEntryResponse, 
            DependencyRelationshipResponse
        )
        print("✅ API schemas imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_model_creation():
    """Test basic model creation and validation."""
    print("\n🏗️  Testing model creation...")
    
    try:
        from app.project_index.models import ProjectIndexConfig, AnalysisConfiguration
        
        # Test ProjectIndexConfig creation
        config = ProjectIndexConfig(
            project_name="test_project",
            root_path="/tmp/test",
            enable_real_time_monitoring=False,
            enable_ml_analysis=False
        )
        print("✅ ProjectIndexConfig created successfully")
        
        # Test AnalysisConfiguration creation
        analysis_config = AnalysisConfiguration(
            force_reanalysis=False,
            custom_settings={}
        )
        print("✅ AnalysisConfiguration created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

async def test_schema_validation():
    """Test Pydantic schema validation."""
    print("\n📋 Testing schema validation...")
    
    try:
        from app.schemas.project_index import ProjectIndexCreate, FileEntryCreate
        from app.models.project_index import FileType
        
        # Test ProjectIndexCreate validation
        project_data = {
            "name": "Test Project",
            "root_path": "/tmp/test",
            "description": "A test project for validation"
        }
        project_schema = ProjectIndexCreate(**project_data)
        print("✅ ProjectIndexCreate validation successful")
        
        # Test FileEntryCreate validation
        file_data = {
            "project_id": str(uuid.uuid4()),
            "file_path": "/tmp/test/file.py",
            "relative_path": "file.py",
            "file_name": "file.py",
            "file_type": FileType.SOURCE,
            "language": "python"
        }
        file_schema = FileEntryCreate(**file_data)
        print("✅ FileEntryCreate validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

async def test_enum_values():
    """Test enum definitions and values."""
    print("\n🎯 Testing enum definitions...")
    
    try:
        from app.models.project_index import (
            ProjectStatus, FileType, DependencyType, 
            AnalysisSessionType, AnalysisStatus
        )
        
        # Test ProjectStatus
        assert ProjectStatus.ACTIVE.value == "active"
        assert ProjectStatus.ANALYZING.value == "analyzing"
        print("✅ ProjectStatus enum working correctly")
        
        # Test FileType
        assert FileType.SOURCE.value == "source"
        assert FileType.CONFIG.value == "config"
        print("✅ FileType enum working correctly")
        
        # Test DependencyType
        assert DependencyType.IMPORT.value == "import"
        assert DependencyType.REFERENCES.value == "references"
        print("✅ DependencyType enum working correctly")
        
        # Test AnalysisSessionType
        assert AnalysisSessionType.FULL_ANALYSIS.value == "full_analysis"
        assert AnalysisSessionType.INCREMENTAL.value == "incremental"
        print("✅ AnalysisSessionType enum working correctly")
        
        # Test AnalysisStatus
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.COMPLETED.value == "completed"
        print("✅ AnalysisStatus enum working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Enum testing failed: {e}")
        return False

async def test_utility_functions():
    """Test utility functions from Project Index utils."""
    print("\n🔧 Testing utility functions...")
    
    try:
        from app.project_index.utils import PathUtils, FileUtils, HashUtils
        
        # Test with a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test Python file\nprint('Hello, World!')\n")
            temp_path = Path(f.name)
        
        try:
            # Test PathUtils
            normalized_path = PathUtils.normalize_path(str(temp_path))
            print("✅ PathUtils.normalize_path working")
            
            # Test FileUtils
            file_info = FileUtils.get_file_info(temp_path)
            assert isinstance(file_info, dict)
            assert 'is_binary' in file_info
            print("✅ FileUtils.get_file_info working")
            
            # Test HashUtils
            file_hash = HashUtils.hash_file(temp_path)
            assert len(file_hash) == 64  # SHA256 hash length
            print("✅ HashUtils.hash_file working")
            
        finally:
            # Clean up
            temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions testing failed: {e}")
        return False

async def test_analyzer_basic():
    """Test basic CodeAnalyzer functionality."""
    print("\n🔍 Testing CodeAnalyzer basics...")
    
    try:
        from app.project_index.analyzer import CodeAnalyzer
        from app.project_index.models import AnalysisConfiguration
        
        # Create analyzer with basic config
        config = AnalysisConfiguration()
        analyzer = CodeAnalyzer(config=config)
        print("✅ CodeAnalyzer instantiated successfully")
        
        # Test language detection with a temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("# Test Python file\nimport os\nprint('Hello')\n")
            temp_path = Path(f.name)
        
        try:
            language = analyzer.detect_language(temp_path)
            assert language == "python"
            print("✅ Language detection working correctly")
            
        finally:
            temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ CodeAnalyzer testing failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Starting comprehensive Project Index system testing...\n")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation), 
        ("Schema Validation", test_schema_validation),
        ("Enum Values", test_enum_values),
        ("Utility Functions", test_utility_functions),
        ("CodeAnalyzer Basics", test_analyzer_basic),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 PROJECT INDEX TESTING SUMMARY")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - Project Index system is functional!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing framework crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
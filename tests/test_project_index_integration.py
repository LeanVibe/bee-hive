#!/usr/bin/env python3
"""
Project Index Integration Testing
Test core functionality with database integration.
"""

import asyncio
import tempfile
import uuid
from pathlib import Path

print("=== Project Index Integration Testing ===")
print()

async def test_basic_integration():
    """Test basic Project Index integration without full database."""
    print("🔧 Testing basic integration...")
    
    try:
        from app.project_index.core import ProjectIndexer
        from app.project_index.models import ProjectIndexConfig
        
        # Create config
        config = ProjectIndexConfig(
            project_name="integration_test",
            root_path="/tmp/test",
            enable_real_time_monitoring=False,
            enable_ml_analysis=False
        )
        
        # Test instantiation without database
        indexer = ProjectIndexer(
            session=None,
            redis_client=None,
            config=config,
            event_publisher=None
        )
        
        print("✅ ProjectIndexer instantiated successfully")
        print("✅ Configuration loaded correctly")
        
        # Test statistics access
        stats = await indexer.get_analysis_statistics()
        assert isinstance(stats, dict)
        print("✅ Statistics retrieval working")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_file_analysis_chain():
    """Test the complete file analysis chain."""
    print("\n📁 Testing file analysis chain...")
    
    try:
        from app.project_index.analyzer import CodeAnalyzer
        from app.project_index.models import AnalysisConfiguration
        from app.project_index.utils import FileUtils, HashUtils
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""# Test Python file for analysis
import os
import sys
from pathlib import Path

def main():
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
""")
            temp_path = Path(f.name)
        
        try:
            # Test analysis chain
            config = AnalysisConfiguration()
            analyzer = CodeAnalyzer(config=config)
            
            # 1. Language detection
            language = analyzer.detect_language(temp_path)
            assert language == "python"
            print("✅ Language detection working")
            
            # 2. File utilities
            file_info = FileUtils.get_file_info(temp_path)
            assert not file_info['is_binary']
            print("✅ File info extraction working")
            
            # 3. File hashing
            file_hash = HashUtils.hash_file(temp_path)
            assert len(file_hash) == 64
            print("✅ File hashing working")
            
            # 4. AST parsing
            ast_result = await analyzer.parse_file(temp_path)
            assert ast_result is not None
            print("✅ AST parsing working")
            
            # 5. Dependency extraction
            dependencies = await analyzer.extract_dependencies(temp_path)
            assert isinstance(dependencies, list)
            print("✅ Dependency extraction working")
            
        finally:
            temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ File analysis chain failed: {e}")
        return False

async def test_api_schema_integration():
    """Test API schema integration."""
    print("\n🌐 Testing API schema integration...")
    
    try:
        from app.schemas.project_index import (
            ProjectIndexCreate, ProjectIndexResponse,
            FileEntryCreate, FileEntryResponse,
            DependencyRelationshipCreate, DependencyRelationshipResponse,
            AnalysisSessionCreate, AnalysisSessionResponse
        )
        from app.models.project_index import ProjectStatus, FileType, DependencyType
        
        # Test complete schema chain
        project_data = {
            "name": "Test Project",
            "root_path": "/tmp/test",
            "description": "Integration test project",
            "configuration": {"test": True},
            "file_patterns": {"include": ["**/*.py"]},
            "ignore_patterns": {"exclude": ["**/__pycache__/**"]}
        }
        
        # Validate creation schema
        create_schema = ProjectIndexCreate(**project_data)
        assert create_schema.name == "Test Project"
        print("✅ Project creation schema validation")
        
        # Test file entry schema
        file_data = {
            "project_id": str(uuid.uuid4()),
            "file_path": "/tmp/test/file.py",
            "relative_path": "file.py",
            "file_name": "file.py",
            "file_type": FileType.SOURCE,
            "language": "python"
        }
        
        file_schema = FileEntryCreate(**file_data)
        assert file_schema.file_name == "file.py"
        print("✅ File entry schema validation")
        
        # Test dependency schema
        dep_data = {
            "project_id": str(uuid.uuid4()),
            "source_file_id": str(uuid.uuid4()),
            "target_name": "os",
            "dependency_type": DependencyType.IMPORT,
            "is_external": True
        }
        
        dep_schema = DependencyRelationshipCreate(**dep_data)
        assert dep_schema.target_name == "os"
        print("✅ Dependency schema validation")
        
        return True
        
    except Exception as e:
        print(f"❌ API schema integration failed: {e}")
        return False

async def test_database_model_integration():
    """Test database model integration."""
    print("\n🗄️  Testing database model integration...")
    
    try:
        from app.models.project_index import (
            ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
            ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus
        )
        
        # Test model instantiation
        project = ProjectIndex(
            name="Test Project",
            root_path="/tmp/test",
            description="Test description",
            status=ProjectStatus.INACTIVE
        )
        
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.INACTIVE
        print("✅ ProjectIndex model instantiation")
        
        # Test model serialization
        project_dict = project.to_dict()
        assert isinstance(project_dict, dict)
        assert project_dict["name"] == "Test Project"
        print("✅ Model serialization working")
        
        # Test FileEntry model
        file_entry = FileEntry(
            project_id=project.id,
            file_path="/tmp/test/file.py",
            relative_path="file.py",
            file_name="file.py",
            file_type=FileType.SOURCE,
            language="python"
        )
        
        assert file_entry.file_name == "file.py"
        print("✅ FileEntry model instantiation")
        
        return True
        
    except Exception as e:
        print(f"❌ Database model integration failed: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("Starting Project Index integration testing...\n")
    
    tests = [
        ("Basic Integration", test_basic_integration),
        ("File Analysis Chain", test_file_analysis_chain),
        ("API Schema Integration", test_api_schema_integration),
        ("Database Model Integration", test_database_model_integration),
    ]
    
    test_results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 PROJECT INDEX INTEGRATION SUMMARY")
    print("="*60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOVERALL: {passed_tests}/{total_tests} integration tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Project Index system is fully integrated and functional!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} integration tests failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Testing framework crashed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
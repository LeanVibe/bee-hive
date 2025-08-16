#!/usr/bin/env python3
"""
Project Index Enablement Script for Bee-Hive
Enable and initialize Project Index system for the current project.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

print("🚀 LeanVibe Agent Hive - Project Index Enablement")
print("="*60)
print()

async def validate_environment():
    """Validate that the environment is ready for Project Index."""
    print("🔍 Validating environment...")
    
    issues = []
    
    try:
        # Test imports
        from app.project_index.core import ProjectIndexer
        from app.project_index.models import ProjectIndexConfig
        from app.schemas.project_index import ProjectIndexCreate
        print("✅ Project Index components importable")
        
    except ImportError as e:
        issues.append(f"Import error: {e}")
        print(f"❌ Import failed: {e}")
    
    # Check project structure
    required_paths = [
        "app/",
        "app/project_index/",
        "app/api/",
        "migrations/",
        "tests/"
    ]
    
    for path in required_paths:
        if Path(path).exists():
            print(f"✅ {path} exists")
        else:
            issues.append(f"Missing required path: {path}")
            print(f"❌ Missing: {path}")
    
    # Check migration file
    migration_file = Path("migrations/versions/022_add_project_index_system.py")
    if migration_file.exists():
        print("✅ Project Index database migration exists")
    else:
        issues.append("Missing Project Index database migration")
        print("❌ Missing database migration")
    
    return len(issues) == 0, issues

async def create_configuration():
    """Create optimal configuration for bee-hive project."""
    print("\n⚙️  Creating Project Index configuration...")
    
    project_root = Path.cwd()
    
    config_data = {
        "project_name": "bee-hive",
        "root_path": str(project_root),
        "enable_real_time_monitoring": True,
        "enable_ml_analysis": False,  # Disable for performance
        "cache_enabled": True,
        "incremental_updates": True,
        "events_enabled": True,
        
        # Analysis settings
        "analysis_config": {
            "languages": ["python", "javascript", "typescript", "yaml", "json"],
            "analysis_depth": 3,
            "include_tests": True,
            "include_documentation": True
        },
        
        # Performance settings
        "analysis_batch_size": 25,
        "max_concurrent_analyses": 3,
        
        # File patterns
        "file_patterns": {
            "include": [
                "**/*.py",
                "**/*.js", 
                "**/*.ts",
                "**/*.tsx",
                "**/*.md",
                "**/*.yml",
                "**/*.yaml",
                "**/*.json",
                "**/*.toml",
                "**/*.cfg",
                "**/*.ini"
            ]
        },
        
        # Ignore patterns
        "ignore_patterns": {
            "exclude": [
                "**/__pycache__/**",
                "**/node_modules/**",
                "**/.git/**",
                "**/.venv/**",
                "**/venv/**",
                "**/*.pyc",
                "**/*.pyo",
                "**/build/**",
                "**/dist/**",
                "**/.pytest_cache/**",
                "**/.coverage/**",
                "**/htmlcov/**",
                "**/.mypy_cache/**",
                "**/logs/**",
                "**/checkpoints/**",
                "**/workspaces/**"
            ]
        },
        
        # Cache configuration
        "cache_config": {
            "max_memory_mb": 500,
            "enable_compression": True,
            "compression_threshold": 1024
        },
        
        # Monitoring configuration
        "monitoring_config": {
            "debounce_seconds": 2.0,
            "max_file_size_mb": 10
        }
    }
    
    try:
        from app.project_index.models import ProjectIndexConfig
        config = ProjectIndexConfig(**config_data)
        print("✅ Configuration created successfully")
        return config
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return None

async def simulate_project_creation(config):
    """Simulate project creation without database dependency."""
    print("\n🏗️  Simulating project creation...")
    
    try:
        # Test project data creation
        project_data = {
            "name": "LeanVibe Agent Hive 2.0",
            "description": "Multi-Agent Orchestration System for Autonomous Software Development",
            "root_path": str(Path.cwd()),
            "git_repository_url": "https://github.com/leanvibe/agent-hive.git",
            "git_branch": "main",
            "configuration": config.model_dump() if config else {},
            "analysis_settings": {
                "auto_analysis": True,
                "real_time_updates": True,
                "dependency_tracking": True
            },
            "file_patterns": {
                "include": [
                    "**/*.py", "**/*.js", "**/*.ts", "**/*.md", "**/*.yml", "**/*.yaml", "**/*.json"
                ]
            },
            "ignore_patterns": {
                "exclude": [
                    "**/__pycache__/**", "**/node_modules/**", "**/.git/**", "**/.venv/**"
                ]
            }
        }
        
        # Validate with Pydantic schema
        from app.schemas.project_index import ProjectIndexCreate
        schema = ProjectIndexCreate(**project_data)
        
        print("✅ Project data validation successful")
        print(f"   Project name: {schema.name}")
        print(f"   Root path: {schema.root_path}")
        print(f"   File patterns: {len(schema.file_patterns.get('include', []))} include patterns")
        print(f"   Ignore patterns: {len(schema.ignore_patterns.get('exclude', []))} exclude patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Project creation simulation failed: {e}")
        return False

async def test_file_analysis_capability():
    """Test file analysis capabilities on current project."""
    print("\n📁 Testing file analysis capabilities...")
    
    try:
        from app.project_index.analyzer import CodeAnalyzer
        from app.project_index.models import AnalysisConfiguration
        from app.project_index.utils import FileUtils, HashUtils
        
        # Find a Python file to test with
        python_files = list(Path("app").glob("**/*.py"))
        if not python_files:
            print("⚠️  No Python files found for testing")
            return True
        
        test_file = python_files[0]
        print(f"   Testing with file: {test_file}")
        
        # Create analyzer
        analysis_config = AnalysisConfiguration()
        analyzer = CodeAnalyzer(config=analysis_config)
        
        # Test language detection
        language = analyzer.detect_language(test_file)
        print(f"✅ Language detection: {language}")
        
        # Test file utilities
        file_info = FileUtils.get_file_info(test_file)
        print(f"✅ File info: Binary={file_info['is_binary']}, Size={test_file.stat().st_size} bytes")
        
        # Test file hashing
        file_hash = HashUtils.hash_file(test_file)
        print(f"✅ File hash: {file_hash[:16]}...")
        
        # Test AST parsing
        ast_result = await analyzer.parse_file(test_file)
        if ast_result:
            print(f"✅ AST parsing: {ast_result.get('line_count', 0)} lines")
        
        # Test dependency extraction
        dependencies = await analyzer.extract_dependencies(test_file)
        print(f"✅ Dependencies found: {len(dependencies)}")
        
        return True
        
    except Exception as e:
        print(f"❌ File analysis test failed: {e}")
        return False

async def create_usage_examples():
    """Create usage examples for the enabled Project Index."""
    print("\n📝 Creating usage examples...")
    
    examples = {
        "api_examples.py": '''
"""
Project Index API Usage Examples
"""

import asyncio
import httpx
from pathlib import Path

# Example 1: Create project via API
async def create_project_via_api():
    async with httpx.AsyncClient() as client:
        project_data = {
            "name": "LeanVibe Agent Hive 2.0",
            "root_path": str(Path.cwd()),
            "description": "Multi-Agent System",
            "git_repository_url": "https://github.com/leanvibe/agent-hive.git",
            "git_branch": "main",
            "file_patterns": {
                "include": ["**/*.py", "**/*.js", "**/*.ts"]
            }
        }
        
        response = await client.post(
            "http://localhost:8000/api/project-index/create",
            json=project_data
        )
        
        if response.status_code == 200:
            project = response.json()["data"]
            print(f"✅ Project created: {project['id']}")
            return project["id"]
        else:
            print(f"❌ Failed to create project: {response.text}")
            return None

# Example 2: Get project information
async def get_project_info(project_id):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://localhost:8000/api/project-index/{project_id}"
        )
        
        if response.status_code == 200:
            project = response.json()["data"]
            print(f"Project: {project['name']}")
            print(f"Files: {project['file_count']}")
            print(f"Dependencies: {project['dependency_count']}")
        else:
            print(f"❌ Failed to get project: {response.text}")

# Example 3: Trigger analysis
async def trigger_analysis(project_id):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://localhost:8000/api/project-index/{project_id}/analyze",
            json={
                "analysis_type": "full",
                "force": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()["data"]
            print(f"✅ Analysis started: {result['analysis_session_id']}")
        else:
            print(f"❌ Failed to start analysis: {response.text}")

# Run examples
async def main():
    print("🔍 Project Index API Examples")
    print("Note: Make sure the server is running on localhost:8000")
    
    # Uncomment to run (requires running server):
    # project_id = await create_project_via_api()
    # if project_id:
    #     await get_project_info(project_id)
    #     await trigger_analysis(project_id)

if __name__ == "__main__":
    asyncio.run(main())
''',
        
        "direct_usage.py": '''
"""
Direct Project Index Usage Examples
"""

import asyncio
from pathlib import Path
from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig

async def direct_usage_example():
    """Example of using Project Index directly."""
    
    # Create configuration
    config = ProjectIndexConfig(
        project_name="bee-hive",
        root_path=str(Path.cwd()),
        enable_real_time_monitoring=False,  # Disable for demo
        enable_ml_analysis=False,
        cache_enabled=True
    )
    
    # Note: This requires database and Redis to be running
    # For demo purposes, we show the code structure
    
    print("Direct Project Index Usage Example:")
    print("1. Create ProjectIndexer with configuration")
    print("2. Create project index")
    print("3. Analyze project files")
    print("4. Get statistics and results")
    
    # Code structure (requires database):
    # async with ProjectIndexer(config=config) as indexer:
    #     project = await indexer.create_project(
    #         name="LeanVibe Agent Hive 2.0",
    #         root_path=str(Path.cwd()),
    #         description="Multi-Agent System"
    #     )
    #     
    #     result = await indexer.analyze_project(str(project.id))
    #     print(f"Analyzed {result.files_processed} files")
    #     
    #     stats = await indexer.get_analysis_statistics()
    #     print(f"Statistics: {stats}")

if __name__ == "__main__":
    asyncio.run(direct_usage_example())
''',
        
        "websocket_example.html": '''
<!DOCTYPE html>
<html>
<head>
    <title>Project Index WebSocket Example</title>
</head>
<body>
    <h1>Project Index Real-time Updates</h1>
    <div id="status">Connecting...</div>
    <div id="messages"></div>

    <script>
        const ws = new WebSocket('ws://localhost:8000/api/project-index/ws?token=demo');
        const statusDiv = document.getElementById('status');
        const messagesDiv = document.getElementById('messages');

        ws.onopen = () => {
            statusDiv.textContent = 'Connected ✅';
            
            // Subscribe to events
            ws.send(JSON.stringify({
                action: 'subscribe',
                event_types: ['analysis_progress', 'file_change', 'dependency_changed'],
                project_id: 'your-project-id'  // Replace with actual project ID
            }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const messageDiv = document.createElement('div');
            
            switch(data.type) {
                case 'analysis_progress':
                    messageDiv.innerHTML = `📊 Analysis: ${data.progress_percentage}% complete`;
                    break;
                case 'file_change':
                    messageDiv.innerHTML = `📁 File changed: ${data.file_path}`;
                    break;
                case 'dependency_changed':
                    messageDiv.innerHTML = `🔗 Dependency: ${data.dependency_details.target_name}`;
                    break;
                default:
                    messageDiv.innerHTML = `📨 ${data.type}: ${JSON.stringify(data)}`;
            }
            
            messagesDiv.appendChild(messageDiv);
        };

        ws.onclose = () => {
            statusDiv.textContent = 'Disconnected ❌';
        };

        ws.onerror = (error) => {
            statusDiv.textContent = 'Error ⚠️';
            console.error('WebSocket error:', error);
        };
    </script>
</body>
</html>
'''
    }
    
    examples_dir = Path("project_index_examples")
    examples_dir.mkdir(exist_ok=True)
    
    for filename, content in examples.items():
        example_file = examples_dir / filename
        with open(example_file, 'w') as f:
            f.write(content)
        print(f"✅ Created: {example_file}")
    
    return True

async def main():
    """Main enablement process."""
    
    print("Starting Project Index enablement for bee-hive...\n")
    
    # Step 1: Environment validation
    env_valid, issues = await validate_environment()
    if not env_valid:
        print(f"\n❌ Environment validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease resolve these issues before enabling Project Index.")
        return False
    
    print("✅ Environment validation passed\n")
    
    # Step 2: Configuration creation
    config = await create_configuration()
    if not config:
        print("❌ Configuration creation failed")
        return False
    
    # Step 3: Project creation simulation
    project_ok = await simulate_project_creation(config)
    if not project_ok:
        print("❌ Project creation simulation failed")
        return False
    
    # Step 4: File analysis testing
    analysis_ok = await test_file_analysis_capability()
    if not analysis_ok:
        print("❌ File analysis testing failed")
        return False
    
    # Step 5: Create usage examples
    examples_ok = await create_usage_examples()
    if not examples_ok:
        print("❌ Usage examples creation failed")
        return False
    
    # Success summary
    print("\n" + "="*60)
    print("🎉 PROJECT INDEX ENABLEMENT SUCCESSFUL!")
    print("="*60)
    print()
    print("✅ Environment validated and ready")
    print("✅ Configuration created and tested")
    print("✅ Project creation validated")
    print("✅ File analysis capabilities confirmed")
    print("✅ Usage examples created")
    print()
    print("📋 Next Steps:")
    print("1. Start PostgreSQL and Redis services")
    print("2. Run: alembic upgrade head")
    print("3. Start the FastAPI server")
    print("4. Use examples in project_index_examples/")
    print()
    print("📚 Documentation:")
    print("- Setup Guide: PROJECT_INDEX_ENABLEMENT_GUIDE.md")
    print("- API Examples: project_index_examples/api_examples.py")
    print("- Direct Usage: project_index_examples/direct_usage.py")
    print("- WebSocket Demo: project_index_examples/websocket_example.html")
    print()
    print("🚀 Project Index is ready to enhance your development workflow!")
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Enablement interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Enablement script crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
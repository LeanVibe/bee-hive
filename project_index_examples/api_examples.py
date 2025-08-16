
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

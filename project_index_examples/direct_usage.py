
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

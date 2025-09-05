#!/usr/bin/env python3
"""
Debug FastAPI routes to understand the API v2 routing issue.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up test environment
os.environ["TESTING"] = "true"
os.environ["SKIP_STARTUP_INIT"] = "true"

def debug_routes():
    """Debug all FastAPI routes."""
    try:
        from app.main import create_app
        
        app = create_app()
        
        print("🔍 FastAPI Routes Debug")
        print("=" * 50)
        
        # List all routes
        for route in app.routes:
            print(f"Route: {route.path} [{', '.join(route.methods)}]")
            if hasattr(route, 'name'):
                print(f"  Name: {route.name}")
            if hasattr(route, 'dependencies'):
                print(f"  Dependencies: {len(route.dependencies)}")
            print()
        
        print(f"\nTotal routes: {len(app.routes)}")
        
        # Check specifically for v2 routes
        v2_routes = [route for route in app.routes if '/api/v2' in route.path]
        print(f"v2 API routes: {len(v2_routes)}")
        
        for route in v2_routes:
            print(f"  v2: {route.path} [{', '.join(route.methods)}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Route debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_routes()
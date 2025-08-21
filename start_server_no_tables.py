#!/usr/bin/env python3
"""
Start server without database table creation for quick validation
Uses existing tables only, doesn't attempt to create new ones
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def main():
    """Start server with database table creation disabled."""
    print("🚀 Starting LeanVibe Agent Hive 2.0 - No Table Creation Mode")
    print("=" * 60)
    
    # Override database initialization to skip table creation
    os.environ["DEBUG"] = "false"  # This prevents table creation in database.py
    
    try:
        # Test core connectivity first
        print("🧪 Testing core systems...")
        
        # Database
        import asyncpg
        conn = await asyncpg.connect(
            host='localhost', port=15432, user='leanvibe_user',
            password='leanvibe_secure_pass', database='leanvibe_agent_hive'
        )
        await conn.execute('SELECT 1')
        await conn.close()
        print("✅ Database connection verified")
        
        # Redis
        import redis
        r = redis.Redis(host='localhost', port=16379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis connection verified")
        
        # Import app (this may still trigger some initialization)
        print("📦 Loading application...")
        from app.main import app
        
        print(f"✅ Application loaded successfully")
        print(f"   Routes: {len(app.routes)}")
        
        # Start server
        import uvicorn
        
        print("\n🌟 Starting server...")
        print("   Server: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Health: http://localhost:8000/health")
        print("\n🔧 Test CLI: python3 -m app.cli.main status")
        print("\nPress Ctrl+C to stop")
        
        # Configure and run server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0", 
            port=8000,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
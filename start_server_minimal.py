#!/usr/bin/env python3
"""
Minimal server startup script that bypasses vector type issues
Starts the server with essential functionality only
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def start_minimal_server():
    """Start server with minimal database requirements."""
    print("🚀 Starting LeanVibe Agent Hive 2.0 - Minimal Mode")
    print("=" * 60)
    
    # Set environment variable to skip problematic table creation
    os.environ["SKIP_VECTOR_TABLES"] = "true"
    os.environ["MINIMAL_STARTUP"] = "true"
    os.environ["DEBUG"] = "false"  # Skip table creation
    
    try:
        # Import and configure
        from app.core.config import settings
        print(f"✅ Configuration loaded: {settings.APP_NAME}")
        
        # Test database connectivity first
        import asyncpg
        print("🧪 Testing database connectivity...")
        
        conn = await asyncpg.connect(
            host='localhost',
            port=15432,
            user='leanvibe_user',
            password='leanvibe_secure_pass',
            database='leanvibe_agent_hive'
        )
        await conn.execute('SELECT 1')
        await conn.close()
        print("✅ Database connection verified")
        
        # Test Redis connectivity
        import redis
        r = redis.Redis(host='localhost', port=16379, db=0, decode_responses=True)
        r.ping()
        print("✅ Redis connection verified")
        
        # Import FastAPI app
        from app.main import app
        print(f"✅ FastAPI app loaded with {len(app.routes)} routes")
        
        # Start server manually without database table creation
        import uvicorn
        print("🌟 Starting server on http://localhost:8000")
        print("   API: http://localhost:8000/docs")
        print("   Health: http://localhost:8000/health")
        print("\n🔧 To test CLI: python3 -m app.cli.main status")
        
        # Start uvicorn server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(start_minimal_server())
    sys.exit(exit_code)
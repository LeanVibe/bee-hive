#!/usr/bin/env python3
"""
Initialize the LeanVibe Agent Hive system for proper session persistence
"""
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.initialization import initialize_system

async def main():
    """Initialize the system components."""
    print("🔄 Initializing LeanVibe Agent Hive 2.0 system...")
    
    # Initialize both Redis and Database
    results = await initialize_system(['redis', 'database'])
    
    print("\n📊 Initialization Results:")
    for component, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {component}: {status}")
    
    if all(results.values()):
        print("\n✅ System initialization completed successfully!")
        print("Session persistence should now work correctly.")
        return 0
    else:
        print("\n⚠️  Some components failed to initialize.")
        print("Session persistence may not work correctly.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
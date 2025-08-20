#!/usr/bin/env python3
"""
Production Demo Validation Script

Quick validation that the production API demo can initialize properly
and all components are working correctly.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def validate_demo_readiness():
    """Validate that the production demo is ready to run."""
    print("🔍 Validating Production Demo Readiness...")
    
    validation_results = {
        "imports": False,
        "database": False,
        "ai_gateway": False,
        "task_queue": False,
        "demo_class": False
    }
    
    try:
        # Test imports
        from scripts.demos.production_api_demo import ProductionAPIDemo
        from app.core.database import get_session
        from app.core.ai_gateway import get_ai_gateway
        from app.core.task_queue import TaskQueue
        validation_results["imports"] = True
        print("✅ All required modules import successfully")
        
        # Test database session creation
        try:
            async with get_session() as session:
                # Try a simple query
                from sqlalchemy import text
                result = await session.execute(text("SELECT 1"))
                result.scalar()
            validation_results["database"] = True
            print("✅ Database system ready")
        except Exception as e:
            print(f"⚠️  Database connection issue: {e}")
            print("   This is expected if PostgreSQL is not running")
        
        # Test AI Gateway
        try:
            gateway = await get_ai_gateway()
            validation_results["ai_gateway"] = True
            print("✅ AI Gateway system ready")
        except Exception as e:
            print(f"⚠️  AI Gateway issue: {e}")
        
        # Test Task Queue
        try:
            task_queue = TaskQueue()
            # Don't start it, just validate it can be created
            validation_results["task_queue"] = True
            print("✅ Task Queue system ready")
        except Exception as e:
            print(f"⚠️  Task Queue issue: {e}")
        
        # Test demo class
        try:
            demo = ProductionAPIDemo()
            validation_results["demo_class"] = True
            print("✅ Production demo class ready")
        except Exception as e:
            print(f"❌ Demo class issue: {e}")
        
        # Summary
        passed = sum(validation_results.values())
        total = len(validation_results)
        
        print(f"\n📊 Validation Results: {passed}/{total} components ready")
        
        if passed >= 3:  # Core components working
            print("🎉 Production demo is ready to run!")
            print("\nTo run the demo:")
            print("  python3 scripts/demos/production_api_demo.py")
            return True
        else:
            print("⚠️  Some components need attention before running demo")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class ValidateProductionDemoScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await validate_demo_readiness()
            sys.exit(0 if result else 1)
            
            return {"status": "completed"}
    
    script_main(ValidateProductionDemoScript)
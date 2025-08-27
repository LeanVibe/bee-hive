#!/usr/bin/env python3
"""
Quick test of Business Analytics functionality for Epic 5

This script tests the core Business Intelligence components to ensure
the Executive Dashboard and API endpoints are working correctly.
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.business_intelligence.executive_dashboard import ExecutiveDashboard


async def test_executive_dashboard():
    """Test the Executive Dashboard functionality."""
    print("🚀 Testing Executive Dashboard...")
    
    try:
        dashboard = ExecutiveDashboard()
        print("✅ ExecutiveDashboard instance created successfully")
        
        # Test get_current_metrics (this will use test data since no real DB data)
        print("\n📊 Testing business metrics collection...")
        metrics = await dashboard.get_current_metrics()
        
        print(f"✅ Metrics collected successfully:")
        print(f"   - Total Agents: {metrics.total_agents}")
        print(f"   - Active Agents: {metrics.active_agents}")
        print(f"   - Total Tasks: {metrics.total_tasks}")
        print(f"   - Success Rate: {metrics.success_rate}%")
        print(f"   - System Uptime: {metrics.system_uptime}%")
        print(f"   - Timestamp: {metrics.timestamp}")
        
        # Test dashboard data
        print("\n📈 Testing dashboard data aggregation...")
        dashboard_data = await dashboard.get_dashboard_data()
        
        print("✅ Dashboard data generated successfully:")
        print(f"   - Status: {dashboard_data['status']}")
        print(f"   - Health Status: {dashboard_data['health_status']}")
        print(f"   - Metrics Count: {len(dashboard_data['metrics'])}")
        print(f"   - Alerts Count: {len(dashboard_data['alerts'])}")
        
        # Test alerts
        print("\n🚨 Testing alerts system...")
        alerts = await dashboard.get_alerts()
        print(f"✅ Alerts retrieved: {len(alerts)} alerts")
        
        return True
        
    except Exception as e:
        print(f"❌ Executive Dashboard test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_business_analytics_import():
    """Test that business analytics modules can be imported."""
    print("📦 Testing Business Analytics imports...")
    
    try:
        from app.api.business_analytics import router
        print("✅ Business Analytics API router imported successfully")
        
        from app.models.business_intelligence import BusinessMetric, UserSession
        print("✅ Business Intelligence models imported successfully")
        
        from app.core.business_intelligence import ExecutiveDashboard, BusinessMetrics
        print("✅ Business Intelligence core components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all business analytics tests."""
    print("🎯 Epic 5: Business Intelligence & Analytics Engine - Test Suite")
    print("=" * 70)
    
    tests = [
        ("Import Test", test_business_analytics_import()),
        ("Executive Dashboard", test_executive_dashboard())
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🏆 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Business Analytics system is operational.")
        return 0
    else:
        print("⚠️ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
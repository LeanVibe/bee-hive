#!/usr/bin/env python3
"""
Quick Dashboard Validation
Tests the essential dashboard functionality quickly
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any

async def test_dashboard_endpoints():
    """Test essential dashboard endpoints"""
    base_url = "http://localhost:8000"
    results = {}
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Dashboard HTML page
        try:
            async with session.get(f"{base_url}/dashboard/simple") as response:
                html_content = await response.text()
                results["dashboard_html"] = {
                    "status": response.status,
                    "has_agent_activities": "agent-activities" in html_content,
                    "has_coordination_dashboard": "Coordination Dashboard" in html_content,
                    "has_agent_panel": "Agent Activities" in html_content
                }
        except Exception as e:
            results["dashboard_html"] = {"error": str(e)}
        
        # Test 2: Live data API
        try:
            async with session.get(f"{base_url}/dashboard/api/live-data") as response:
                data = await response.json()
                results["live_data_api"] = {
                    "status": response.status,
                    "has_metrics": "metrics" in data,
                    "has_agent_activities": "agent_activities" in data,
                    "agent_count": len(data.get("agent_activities", []))
                }
        except Exception as e:
            results["live_data_api"] = {"error": str(e)}
        
        # Test 3: Health check
        try:
            async with session.get(f"{base_url}/health") as response:
                health_data = await response.json()
                results["health_check"] = {
                    "status": response.status,
                    "system_status": health_data.get("status")
                }
        except Exception as e:
            results["health_check"] = {"error": str(e)}
    
    return results

async def main():
    print("🚀 Quick Dashboard Validation")
    print("=" * 50)
    
    start_time = time.time()
    results = await test_dashboard_endpoints()
    duration = time.time() - start_time
    
    # Print results
    for test_name, result in results.items():
        print(f"\n📊 {test_name.replace('_', ' ').title()}:")
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            for key, value in result.items():
                status_emoji = "✅" if (
                    (key == "status" and value == 200) or
                    (key.startswith("has_") and value) or
                    (key == "system_status" and value == "healthy")
                ) else "⚠️" if key == "status" and value != 200 else "📋"
                print(f"  {status_emoji} {key}: {value}")
    
    print(f"\n⏱️ Total validation time: {duration:.2f}s")
    
    # Summary
    dashboard_working = (
        results.get("dashboard_html", {}).get("status") == 200 and
        results.get("dashboard_html", {}).get("has_agent_activities", False)
    )
    api_working = results.get("live_data_api", {}).get("status") == 200
    system_healthy = results.get("health_check", {}).get("system_status") == "healthy"
    
    print(f"\n🎯 Dashboard Status: {'✅ WORKING' if dashboard_working else '❌ ISSUES'}")
    print(f"🎯 API Status: {'✅ WORKING' if api_working else '❌ ISSUES'}")
    print(f"🎯 System Status: {'✅ HEALTHY' if system_healthy else '❌ ISSUES'}")
    
    if dashboard_working and api_working and system_healthy:
        print("\n🎉 All essential dashboard components are working!")
        return True
    else:
        print("\n⚠️ Some issues detected that need attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Simple API v2 Endpoints Validation
Testing Infrastructure Specialist Agent Mission

Direct validation of Epic C API v2 endpoints without pytest overhead.
"""

import asyncio
import sys
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

async def test_api_v2_endpoints():
    """Test API v2 endpoints directly without pytest."""
    results = []
    base_url = "http://localhost:8000"
    
    console.print("🔧 Testing Infrastructure Specialist Agent - API v2 Validation")
    console.print(f"Testing endpoints at: {base_url}")
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        
        # Test 1: Health check
        try:
            response = await client.get(f"{base_url}/health")
            health_ok = response.status_code == 200
            results.append({
                "endpoint": "GET /health",
                "status": "✅ PASS" if health_ok else "❌ FAIL",
                "details": f"Status: {response.status_code}"
            })
        except Exception as e:
            results.append({
                "endpoint": "GET /health",
                "status": "❌ ERROR",
                "details": f"Connection failed: {str(e)}"
            })
            console.print(f"[red]❌ Server not running at {base_url}[/red]")
            return False
            
        # Test 2: API v2 agents list
        try:
            response = await client.get(f"{base_url}/api/v2/agents")
            agents_ok = response.status_code == 200
            if agents_ok:
                data = response.json()
                total = data.get("total", 0)
                details = f"Status: {response.status_code}, Total agents: {total}"
            else:
                details = f"Status: {response.status_code}, Error: {response.text[:100]}"
            results.append({
                "endpoint": "GET /api/v2/agents",
                "status": "✅ PASS" if agents_ok else "❌ FAIL", 
                "details": details
            })
        except Exception as e:
            results.append({
                "endpoint": "GET /api/v2/agents",
                "status": "❌ ERROR",
                "details": str(e)
            })
            
        # Test 3: API v2 tasks list
        try:
            response = await client.get(f"{base_url}/api/v2/tasks")
            tasks_ok = response.status_code == 200
            if tasks_ok:
                data = response.json()
                total = data.get("total", 0)
                details = f"Status: {response.status_code}, Total tasks: {total}"
            else:
                details = f"Status: {response.status_code}, Error: {response.text[:100]}"
            results.append({
                "endpoint": "GET /api/v2/tasks",
                "status": "✅ PASS" if tasks_ok else "❌ FAIL",
                "details": details
            })
        except Exception as e:
            results.append({
                "endpoint": "GET /api/v2/tasks",
                "status": "❌ ERROR",
                "details": str(e)
            })
            
        # Test 4: Create agent
        try:
            agent_data = {
                "role": "backend_developer",
                "agent_type": "claude_code"
            }
            response = await client.post(f"{base_url}/api/v2/agents", json=agent_data)
            create_ok = response.status_code == 201
            if create_ok:
                data = response.json()
                agent_id = data.get("id", "")[:8]
                details = f"Status: {response.status_code}, Agent ID: {agent_id}..."
            else:
                details = f"Status: {response.status_code}, Error: {response.text[:100]}"
            results.append({
                "endpoint": "POST /api/v2/agents",
                "status": "✅ PASS" if create_ok else "❌ FAIL",
                "details": details
            })
        except Exception as e:
            results.append({
                "endpoint": "POST /api/v2/agents", 
                "status": "❌ ERROR",
                "details": f"Error: {str(e)[:50]}..."
            })
    
    # Display results
    table = Table(title="🧪 API v2 Endpoints Validation Results")
    table.add_column("Endpoint", style="cyan", no_wrap=True)
    table.add_column("Status", style="green", justify="center")
    table.add_column("Details", style="white")
    
    for result in results:
        table.add_row(
            result["endpoint"],
            result["status"],
            result["details"]
        )
    
    console.print("\n")
    console.print(table)
    
    # Summary
    total_tests = len(results)
    passed_tests = len([r for r in results if "✅ PASS" in r["status"]])
    
    if passed_tests == total_tests:
        console.print(f"\n[green]🎉 All {total_tests} API v2 endpoints are working! Epic C Phase C.1 validated.[/green]")
        return True
    else:
        console.print(f"\n[red]❌ {total_tests - passed_tests}/{total_tests} endpoints failed. Epic C issues remain.[/red]")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_api_v2_endpoints())
    sys.exit(0 if success else 1)
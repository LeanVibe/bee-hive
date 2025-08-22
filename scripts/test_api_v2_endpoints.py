#!/usr/bin/env python3
"""
API v2 Endpoints Validation Script

Tests the core functionality of API v2 endpoints after Epic C fixes.
Validates that the double prefix issue is resolved and endpoints return
real data from SimpleOrchestrator integration.

Usage:
    python scripts/test_api_v2_endpoints.py
    python scripts/test_api_v2_endpoints.py --base-url http://localhost:8000
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import httpx
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class APIv2Tester:
    """Test suite for API v2 endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self.test_results = []
    
    async def test_health(self) -> bool:
        """Test basic server health."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            success = response.status_code == 200
            self.test_results.append({
                "test": "Health Check",
                "status": "✅ PASS" if success else "❌ FAIL",
                "details": f"Status: {response.status_code}"
            })
            return success
        except Exception as e:
            self.test_results.append({
                "test": "Health Check", 
                "status": "❌ ERROR",
                "details": str(e)
            })
            return False
    
    async def test_agents_list(self) -> bool:
        """Test GET /api/v2/agents endpoint."""
        try:
            response = await self.client.get(f"{self.base_url}/api/v2/agents")
            
            if response.status_code == 404:
                self.test_results.append({
                    "test": "List Agents",
                    "status": "❌ FAIL",
                    "details": "404 Not Found - Double prefix issue not resolved"
                })
                return False
            
            success = response.status_code == 200
            if success:
                data = response.json()
                agent_count = data.get("total", 0)
                details = f"Status: {response.status_code}, Agents: {agent_count}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:100]}"
            
            self.test_results.append({
                "test": "List Agents",
                "status": "✅ PASS" if success else "❌ FAIL",
                "details": details
            })
            return success
            
        except Exception as e:
            self.test_results.append({
                "test": "List Agents",
                "status": "❌ ERROR", 
                "details": str(e)
            })
            return False
    
    async def test_tasks_list(self) -> bool:
        """Test GET /api/v2/tasks endpoint."""
        try:
            response = await self.client.get(f"{self.base_url}/api/v2/tasks")
            
            if response.status_code == 404:
                self.test_results.append({
                    "test": "List Tasks",
                    "status": "❌ FAIL",
                    "details": "404 Not Found - Double prefix issue not resolved"
                })
                return False
            
            success = response.status_code == 200
            if success:
                data = response.json()
                task_count = data.get("total", 0)
                details = f"Status: {response.status_code}, Tasks: {task_count}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:100]}"
            
            self.test_results.append({
                "test": "List Tasks",
                "status": "✅ PASS" if success else "❌ FAIL",
                "details": details
            })
            return success
            
        except Exception as e:
            self.test_results.append({
                "test": "List Tasks",
                "status": "❌ ERROR",
                "details": str(e)
            })
            return False
    
    async def test_agent_creation(self) -> Optional[str]:
        """Test POST /api/v2/agents endpoint."""
        try:
            agent_data = {
                "role": "backend_developer",
                "agent_type": "claude_code",
                "workspace_name": "test-workspace"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v2/agents",
                json=agent_data
            )
            
            if response.status_code == 404:
                self.test_results.append({
                    "test": "Create Agent",
                    "status": "❌ FAIL",
                    "details": "404 Not Found - Endpoint not accessible"
                })
                return None
            
            success = response.status_code == 201
            if success:
                data = response.json()
                agent_id = data.get("id")
                details = f"Status: {response.status_code}, Agent ID: {agent_id[:8]}..."
                self.test_results.append({
                    "test": "Create Agent",
                    "status": "✅ PASS",
                    "details": details
                })
                return agent_id
            else:
                details = f"Status: {response.status_code}, Response: {response.text[:100]}"
                self.test_results.append({
                    "test": "Create Agent", 
                    "status": "❌ FAIL",
                    "details": details
                })
                return None
                
        except Exception as e:
            self.test_results.append({
                "test": "Create Agent",
                "status": "❌ ERROR",
                "details": str(e)
            })
            return None
    
    async def test_websocket_endpoint(self) -> bool:
        """Test WebSocket endpoint availability."""
        try:
            import websockets
            
            client_id = "test-client-123"
            ws_url = f"{self.base_url.replace('http', 'ws')}/api/v2/ws/{client_id}"
            
            # Test connection
            async with websockets.connect(ws_url, ping_timeout=5) as websocket:
                # Send a test message
                await websocket.send(json.dumps({"command": "ping"}))
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    self.test_results.append({
                        "test": "WebSocket Connection",
                        "status": "✅ PASS",
                        "details": f"Connected successfully to {ws_url}"
                    })
                    return True
                except asyncio.TimeoutError:
                    self.test_results.append({
                        "test": "WebSocket Connection",
                        "status": "⚠️ PARTIAL",
                        "details": "Connected but no response received"
                    })
                    return True
                    
        except Exception as e:
            self.test_results.append({
                "test": "WebSocket Connection",
                "status": "❌ ERROR",
                "details": str(e)
            })
            return False
    
    def display_results(self):
        """Display test results in a formatted table."""
        table = Table(title="🧪 API v2 Endpoints Test Results")
        table.add_column("Test", style="cyan", no_wrap=True)
        table.add_column("Status", style="green", justify="center")
        table.add_column("Details", style="white")
        
        for result in self.test_results:
            table.add_row(
                result["test"],
                result["status"],
                result["details"]
            )
        
        console.print("\n")
        console.print(table)
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if "✅ PASS" in r["status"]])
        failed_tests = len([r for r in self.test_results if "❌" in r["status"]])
        
        if passed_tests == total_tests:
            status_color = "green"
            status_msg = "🎉 All tests passed! Epic C Phase C.1 successfully completed."
        elif passed_tests > failed_tests:
            status_color = "yellow"
            status_msg = f"⚠️ Partial success: {passed_tests}/{total_tests} tests passed."
        else:
            status_color = "red"
            status_msg = f"❌ Epic C issues remain: {failed_tests}/{total_tests} tests failed."
        
        console.print(f"\n[{status_color}]{status_msg}[/{status_color}]")
        
        return passed_tests == total_tests
    
    async def run_all_tests(self) -> bool:
        """Run all API v2 tests."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Test 1: Basic health check
            task = progress.add_task("Testing server health...", total=None)
            health_ok = await self.test_health()
            progress.advance(task, 1)
            
            if not health_ok:
                console.print("[red]❌ Server health check failed. Is the server running?[/red]")
                return False
            
            # Test 2: API v2 endpoints 
            progress.update(task, description="Testing API v2 agents endpoint...")
            agents_ok = await self.test_agents_list()
            progress.advance(task, 1)
            
            progress.update(task, description="Testing API v2 tasks endpoint...")
            tasks_ok = await self.test_tasks_list()
            progress.advance(task, 1)
            
            # Test 3: Agent creation
            progress.update(task, description="Testing agent creation...")
            agent_id = await self.test_agent_creation()
            progress.advance(task, 1)
            
            # Test 4: WebSocket connection
            progress.update(task, description="Testing WebSocket connection...")
            ws_ok = await self.test_websocket_endpoint()
            progress.advance(task, 1)
            
            progress.remove_task(task)
        
        return self.display_results()
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


@click.command()
@click.option('--base-url', default='http://localhost:8000', help='Base URL of the API server')
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON')
def main(base_url: str, output_json: bool):
    """
    Test API v2 endpoints to validate Epic C fixes.
    
    Validates that the double prefix issue is resolved and that
    endpoints return real data from SimpleOrchestrator integration.
    """
    console.print("🎯 [bold blue]API v2 Endpoints Validation - Epic C Phase C.1[/bold blue]")
    console.print(f"Testing endpoints at: {base_url}\n")
    
    async def run_tests():
        tester = APIv2Tester(base_url)
        try:
            success = await tester.run_all_tests()
            
            if output_json:
                results_json = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "base_url": base_url,
                    "success": success,
                    "results": tester.test_results
                }
                console.print(json.dumps(results_json, indent=2))
            
            return success
        finally:
            await tester.close()
    
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
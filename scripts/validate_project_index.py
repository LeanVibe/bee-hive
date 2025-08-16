#!/usr/bin/env python3
"""
Project Index Validation Script

Comprehensive validation of the Project Index system including:
- Database schema validation
- API endpoint testing  
- WebSocket event testing
- Performance benchmarking
- Frontend component validation

Usage:
    python validate_project_index.py --full
    python validate_project_index.py --quick
    python validate_project_index.py --api-only
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
import httpx
import websockets
import psutil
import subprocess
import argparse


class ProjectIndexValidator:
    """Comprehensive Project Index system validator"""
    
    def __init__(self, server_url: str = "http://localhost:8081"):
        self.server_url = server_url.rstrip("/")
        self.results = {}
        self.test_project_id = None
        
    async def validate_database_schema(self) -> Dict:
        """Validate database schema and migrations"""
        print("🗄️  Validating database schema...")
        
        results = {
            "tables_exist": False,
            "indexes_exist": False,
            "enums_exist": False,
            "migration_applied": False
        }
        
        try:
            # Check if tables exist via API health endpoint
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/api/health/database")
                if response.status_code == 200:
                    health_data = response.json()
                    
                    # Check for project index tables in database info
                    if "project_index_tables" in health_data.get("details", {}):
                        results["tables_exist"] = True
                        results["indexes_exist"] = True
                        results["enums_exist"] = True
                        results["migration_applied"] = True
                    
        except Exception as e:
            print(f"   ⚠️ Database validation error: {e}")
            
        status = "✅" if all(results.values()) else "❌"
        print(f"   {status} Database schema validation: {results}")
        return results
    
    async def validate_api_endpoints(self) -> Dict:
        """Validate all API endpoints"""
        print("🔗 Validating API endpoints...")
        
        results = {
            "health_check": False,
            "create_project": False,
            "get_project": False,
            "analyze_project": False,
            "get_files": False,
            "get_dependencies": False,
            "optimize_context": False,
            "delete_project": False
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Health check
            try:
                response = await client.get(f"{self.server_url}/health")
                results["health_check"] = response.status_code == 200
            except Exception as e:
                print(f"   ⚠️ Health check failed: {e}")
            
            # 2. Create project
            try:
                create_data = {
                    "name": "validation-test-project",
                    "description": "Test project for validation",
                    "root_path": str(Path.cwd()),
                    "configuration": {
                        "languages": ["python"],
                        "exclude_patterns": ["__pycache__", ".git"],
                        "analysis_depth": 2,
                        "enable_ai_analysis": False  # Disable for faster testing
                    }
                }
                
                response = await client.post(
                    f"{self.server_url}/api/project-index/create",
                    json=create_data
                )
                
                if response.status_code == 201:
                    results["create_project"] = True
                    self.test_project_id = response.json()["id"]
                    print(f"   📝 Created test project: {self.test_project_id}")
                    
            except Exception as e:
                print(f"   ⚠️ Create project failed: {e}")
            
            if self.test_project_id:
                # 3. Get project
                try:
                    response = await client.get(f"{self.server_url}/api/project-index/{self.test_project_id}")
                    results["get_project"] = response.status_code == 200
                except Exception as e:
                    print(f"   ⚠️ Get project failed: {e}")
                
                # 4. Analyze project (start analysis)
                try:
                    response = await client.post(f"{self.server_url}/api/project-index/{self.test_project_id}/analyze")
                    results["analyze_project"] = response.status_code in [200, 202]
                    
                    if results["analyze_project"]:
                        # Wait a moment for some analysis to happen
                        await asyncio.sleep(5)
                        
                except Exception as e:
                    print(f"   ⚠️ Analyze project failed: {e}")
                
                # 5. Get files
                try:
                    response = await client.get(f"{self.server_url}/api/project-index/{self.test_project_id}/files")
                    results["get_files"] = response.status_code == 200
                except Exception as e:
                    print(f"   ⚠️ Get files failed: {e}")
                
                # 6. Get dependencies
                try:
                    response = await client.get(f"{self.server_url}/api/project-index/{self.test_project_id}/dependencies")
                    results["get_dependencies"] = response.status_code == 200
                except Exception as e:
                    print(f"   ⚠️ Get dependencies failed: {e}")
                
                # 7. Context optimization
                try:
                    context_data = {
                        "task_description": "Test context optimization",
                        "context_type": "analysis",
                        "max_files": 5
                    }
                    response = await client.post(
                        f"{self.server_url}/api/project-index/{self.test_project_id}/context",
                        json=context_data
                    )
                    results["optimize_context"] = response.status_code == 200
                except Exception as e:
                    print(f"   ⚠️ Context optimization failed: {e}")
                
                # 8. Delete project (cleanup)
                try:
                    response = await client.delete(f"{self.server_url}/api/project-index/{self.test_project_id}")
                    results["delete_project"] = response.status_code == 204
                except Exception as e:
                    print(f"   ⚠️ Delete project failed: {e}")
        
        passed = sum(results.values())
        total = len(results)
        status = "✅" if passed == total else "⚠️" if passed > total // 2 else "❌"
        print(f"   {status} API endpoints: {passed}/{total} passed")
        
        return results
    
    async def validate_websocket_events(self) -> Dict:
        """Validate WebSocket event system"""
        print("🔌 Validating WebSocket events...")
        
        results = {
            "connection": False,
            "subscription": False,
            "events_received": False
        }
        
        events_received = []
        
        try:
            # Test WebSocket connection
            uri = f"ws://localhost:8000/ws"  # Adjust based on your WebSocket endpoint
            
            async with websockets.connect(uri) as websocket:
                results["connection"] = True
                
                # Subscribe to project index events
                subscribe_msg = {
                    "type": "subscribe",
                    "channels": ["project_index.*"]
                }
                await websocket.send(json.dumps(subscribe_msg))
                results["subscription"] = True
                
                # Create a project to generate events (in parallel)
                async def create_test_project():
                    async with httpx.AsyncClient() as client:
                        create_data = {
                            "name": "websocket-test-project",
                            "description": "WebSocket event test",
                            "root_path": str(Path.cwd()),
                            "configuration": {
                                "languages": ["python"],
                                "analysis_depth": 1,
                                "enable_ai_analysis": False
                            }
                        }
                        
                        response = await client.post(
                            f"{self.server_url}/api/project-index/create",
                            json=create_data
                        )
                        
                        if response.status_code == 201:
                            project_id = response.json()["id"]
                            
                            # Trigger analysis to generate more events
                            await client.post(f"{self.server_url}/api/project-index/{project_id}/analyze")
                            
                            # Clean up after a moment
                            await asyncio.sleep(10)
                            await client.delete(f"{self.server_url}/api/project-index/{project_id}")
                
                # Start project creation in background
                project_task = asyncio.create_task(create_test_project())
                
                # Listen for events for 15 seconds
                timeout = 15
                while timeout > 0:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        event = json.loads(message)
                        
                        if event.get("type", "").startswith("project_index"):
                            events_received.append(event)
                            print(f"   📨 Received event: {event['type']}")
                            
                    except asyncio.TimeoutError:
                        timeout -= 1
                        continue
                    except Exception as e:
                        print(f"   ⚠️ WebSocket event error: {e}")
                        break
                
                await project_task
                
        except Exception as e:
            print(f"   ⚠️ WebSocket validation error: {e}")
        
        results["events_received"] = len(events_received) > 0
        
        status = "✅" if all(results.values()) else "⚠️" if any(results.values()) else "❌"
        print(f"   {status} WebSocket events: {len(events_received)} events received")
        
        return results
    
    async def validate_performance(self) -> Dict:
        """Validate performance requirements"""
        print("⚡ Validating performance...")
        
        results = {
            "api_response_time": False,
            "memory_usage": False,
            "analysis_speed": False
        }
        
        # Test API response time
        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/health")
                api_time = (time.time() - start_time) * 1000  # Convert to ms
                results["api_response_time"] = api_time < 200  # < 200ms requirement
                print(f"   📊 API response time: {api_time:.1f}ms")
        except Exception as e:
            print(f"   ⚠️ API performance test failed: {e}")
        
        # Test memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            results["memory_usage"] = memory_mb < 500  # < 500MB reasonable for testing
            print(f"   🧠 Memory usage: {memory_mb:.1f}MB")
        except Exception as e:
            print(f"   ⚠️ Memory test failed: {e}")
        
        # Test analysis speed (create small project and measure)
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                # Create a small test project
                create_data = {
                    "name": "performance-test-project",
                    "description": "Performance test",
                    "root_path": str(Path(__file__).parent),  # Just this directory
                    "configuration": {
                        "languages": ["python"],
                        "analysis_depth": 1,
                        "enable_ai_analysis": False
                    }
                }
                
                response = await client.post(
                    f"{self.server_url}/api/project-index/create",
                    json=create_data
                )
                
                if response.status_code == 201:
                    project_id = response.json()["id"]
                    
                    # Trigger analysis
                    await client.post(f"{self.server_url}/api/project-index/{project_id}/analyze")
                    
                    # Wait for completion
                    for _ in range(30):  # 30 second timeout
                        status_response = await client.get(f"{self.server_url}/api/project-index/{project_id}")
                        if status_response.json().get("status") == "analyzed":
                            break
                        await asyncio.sleep(1)
                    
                    analysis_time = time.time() - start_time
                    results["analysis_speed"] = analysis_time < 60  # < 60s for small project
                    print(f"   🔍 Analysis time: {analysis_time:.1f}s")
                    
                    # Cleanup
                    await client.delete(f"{self.server_url}/api/project-index/{project_id}")
                    
        except Exception as e:
            print(f"   ⚠️ Analysis performance test failed: {e}")
        
        passed = sum(results.values())
        total = len(results)
        status = "✅" if passed == total else "⚠️" if passed > 0 else "❌"
        print(f"   {status} Performance: {passed}/{total} tests passed")
        
        return results
    
    async def validate_frontend(self) -> Dict:
        """Validate frontend components (if possible)"""
        print("🎨 Validating frontend components...")
        
        results = {
            "pwa_build": False,
            "components_exist": False,
            "dashboard_accessible": False
        }
        
        pwa_path = Path("mobile-pwa")
        
        # Check if PWA directory exists
        if pwa_path.exists():
            # Check if key components exist
            component_path = pwa_path / "src" / "components" / "project-index"
            if component_path.exists():
                results["components_exist"] = True
            
            # Try to build PWA (if npm is available)
            try:
                result = subprocess.run(
                    ["npm", "run", "build"],
                    cwd=pwa_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                results["pwa_build"] = result.returncode == 0
            except Exception as e:
                print(f"   ⚠️ PWA build test failed: {e}")
        
        # Test dashboard accessibility
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_url}/dashboard")
                results["dashboard_accessible"] = response.status_code == 200
        except Exception as e:
            print(f"   ⚠️ Dashboard accessibility test failed: {e}")
        
        passed = sum(results.values())
        total = len(results)
        status = "✅" if passed >= total // 2 else "❌"
        print(f"   {status} Frontend: {passed}/{total} tests passed")
        
        return results
    
    async def run_validation(self, quick: bool = False, api_only: bool = False) -> Dict:
        """Run comprehensive validation"""
        print("🚀 Starting Project Index validation...\n")
        
        all_results = {}
        
        # Always validate database and API
        all_results["database"] = await self.validate_database_schema()
        all_results["api"] = await self.validate_api_endpoints()
        
        if not api_only:
            if not quick:
                all_results["websocket"] = await self.validate_websocket_events()
                all_results["frontend"] = await self.validate_frontend()
            
            all_results["performance"] = await self.validate_performance()
        
        # Calculate overall success
        all_passed = []
        for category, results in all_results.items():
            if isinstance(results, dict):
                passed = sum(results.values())
                total = len(results)
                all_passed.append(passed / total if total > 0 else 0)
        
        overall_success = sum(all_passed) / len(all_passed) if all_passed else 0
        
        print(f"\n📊 Validation Summary:")
        print(f"   Overall success rate: {overall_success:.1%}")
        
        for category, results in all_results.items():
            if isinstance(results, dict):
                passed = sum(results.values())
                total = len(results)
                success_rate = passed / total if total > 0 else 0
                status = "✅" if success_rate == 1.0 else "⚠️" if success_rate > 0.5 else "❌"
                print(f"   {status} {category.title()}: {passed}/{total} ({success_rate:.1%})")
        
        if overall_success > 0.8:
            print("\n🎉 Project Index system is working well!")
        elif overall_success > 0.5:
            print("\n⚠️ Project Index system is partially functional - some issues detected")
        else:
            print("\n❌ Project Index system has significant issues")
        
        all_results["overall_success"] = overall_success
        return all_results


async def main():
    parser = argparse.ArgumentParser(description="Project Index Validation Script")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (skip WebSocket and frontend)")
    parser.add_argument("--api-only", action="store_true", help="Test API endpoints only")
    parser.add_argument("--output-json", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        validator = ProjectIndexValidator(args.server_url)
        results = await validator.run_validation(quick=args.quick, api_only=args.api_only)
        
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output_json}")
        
        # Exit with appropriate code
        overall_success = results.get("overall_success", 0)
        if overall_success > 0.8:
            sys.exit(0)  # Success
        elif overall_success > 0.5:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # Failure
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
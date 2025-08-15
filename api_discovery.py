#!/usr/bin/env python3
"""
API Route Discovery for LeanVibe Agent Hive 2.0

Discovers and documents all available API endpoints without requiring 
heavy initialization.
"""

import os
from unittest.mock import patch, AsyncMock

def setup_test_environment():
    """Setup test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["SKIP_STARTUP_INIT"] = "true"
    os.environ["DISABLE_OBSERVABILITY_MIDDLEWARE"] = "true"
    os.environ["DISABLE_SECURITY_MIDDLEWARE"] = "true"

def discover_routes():
    """Discover all routes in the FastAPI application."""
    print("🔍 Discovering API routes...")
    
    # Mock the lifespan to avoid startup
    with patch('app.main.lifespan') as mock_lifespan:
        mock_lifespan.return_value = AsyncMock()
        
        from app.main import create_app
        app = create_app()
    
    routes_info = []
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            routes_info.append({
                'path': route.path,
                'methods': list(route.methods),
                'name': getattr(route, 'name', 'unknown')
            })
    
    # Analyze routes by category
    categories = {
        'health': [r for r in routes_info if 'health' in r['path'].lower()],
        'api_v1': [r for r in routes_info if r['path'].startswith('/api/v1')],
        'dashboard': [r for r in routes_info if '/dashboard' in r['path']],
        'websocket': [r for r in routes_info if '/ws' in r['path'] or 'websocket' in r['path'].lower()],
        'admin': [r for r in routes_info if '/admin' in r['path']],
        'metrics': [r for r in routes_info if 'metric' in r['path'].lower()],
        'other_api': [r for r in routes_info if r['path'].startswith('/api') and not r['path'].startswith('/api/v1')]
    }
    
    print(f"\n📊 Route Discovery Results:")
    print(f"Total routes: {len(routes_info)}")
    
    for category, routes in categories.items():
        if routes:
            print(f"\n{category.upper()} Routes ({len(routes)}):")
            for route in routes[:5]:  # Show first 5 of each category
                methods = ", ".join(route['methods'])
                print(f"  {methods:15} {route['path']}")
            if len(routes) > 5:
                print(f"  ... and {len(routes) - 5} more")
    
    # Find key endpoints we want to test
    key_endpoints = {
        'health': [r for r in routes_info if r['path'] == '/health'],
        'status': [r for r in routes_info if r['path'] == '/status'],
        'metrics': [r for r in routes_info if r['path'] == '/metrics'],
        'dashboard_live': [r for r in routes_info if 'live-data' in r['path']],
        'agents': [r for r in routes_info if '/agents' in r['path'] and 'GET' in r['methods']],
    }
    
    print(f"\n🎯 Key Endpoints for Testing:")
    for endpoint_type, routes in key_endpoints.items():
        if routes:
            route = routes[0]  # Take first match
            methods = ", ".join(route['methods'])
            print(f"  {endpoint_type:15} {methods:15} {route['path']}")
        else:
            print(f"  {endpoint_type:15} {'NOT FOUND':15}")
    
    return routes_info, categories

def test_openapi_schema():
    """Test if OpenAPI schema is available."""
    print("\n🔍 Testing OpenAPI Schema...")
    
    from fastapi.testclient import TestClient
    
    # Create minimal app for testing
    with patch('app.main.lifespan') as mock_lifespan:
        mock_lifespan.return_value = AsyncMock()
        
        from app.main import create_app
        app = create_app()
    
    client = TestClient(app)
    
    try:
        # Mock dependencies to avoid errors
        with patch('app.core.redis.get_redis') as mock_redis:
            mock_redis.return_value = AsyncMock()
            mock_redis.return_value.ping.return_value = True
            
            with patch('app.core.database.get_async_session') as mock_db:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalar.return_value = 1
                mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_db.return_value.__aexit__ = AsyncMock(return_value=None)
                
                response = client.get("/openapi.json")
    
        if response.status_code == 200:
            schema = response.json()
            paths = list(schema["paths"].keys())
            print(f"✅ OpenAPI schema available with {len(paths)} documented paths")
            
            # Show some example paths
            print(f"Sample documented paths:")
            for path in paths[:10]:
                print(f"  {path}")
            if len(paths) > 10:
                print(f"  ... and {len(paths) - 10} more")
            
            return paths
        else:
            print(f"❌ OpenAPI schema not available: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ OpenAPI test failed: {e}")
        return []

def test_basic_endpoints():
    """Test basic endpoints with minimal setup."""
    print("\n🧪 Testing Basic Endpoints...")
    
    from fastapi.testclient import TestClient
    
    with patch('app.main.lifespan') as mock_lifespan:
        mock_lifespan.return_value = AsyncMock()
        
        from app.main import create_app
        app = create_app()
    
    client = TestClient(app)
    
    # Test basic endpoints that should work without dependencies
    endpoints_to_test = [
        "/health",
        "/status", 
        "/metrics",
        "/docs",
        "/openapi.json"
    ]
    
    results = {}
    
    for endpoint in endpoints_to_test:
        try:
            # Mock all dependencies
            with patch('app.core.redis.get_redis') as mock_redis:
                mock_redis.return_value = AsyncMock()
                mock_redis.return_value.ping.return_value = True
                
                with patch('app.core.database.get_async_session') as mock_db:
                    mock_session = AsyncMock()
                    mock_session.execute.return_value.scalar.return_value = 1
                    mock_db.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                    mock_db.return_value.__aexit__ = AsyncMock(return_value=None)
                    
                    with patch('app.core.agent_spawner.get_active_agents_status') as mock_agents:
                        mock_agents.return_value = [{"id": "test-agent", "status": "active"}]
                        
                        with patch('app.core.prometheus_exporter.get_prometheus_exporter') as mock_prometheus:
                            mock_exporter = AsyncMock()
                            mock_exporter.generate_metrics.return_value = "# Test metrics\ntest_metric 1"
                            mock_prometheus.return_value = mock_exporter
                            
                            response = client.get(endpoint)
                            
                            results[endpoint] = {
                                'status_code': response.status_code,
                                'content_type': response.headers.get('content-type', 'unknown'),
                                'working': response.status_code < 400
                            }
                            
                            status_emoji = "✅" if response.status_code < 400 else "❌"
                            print(f"  {status_emoji} {endpoint:20} HTTP {response.status_code}")
                            
        except Exception as e:
            results[endpoint] = {
                'status_code': 'error',
                'error': str(e)[:100],
                'working': False
            }
            print(f"  ❌ {endpoint:20} ERROR: {str(e)[:50]}")
    
    working_count = sum(1 for r in results.values() if r.get('working', False))
    total_count = len(results)
    
    print(f"\n📊 Endpoint Test Summary: {working_count}/{total_count} working")
    
    return results

def main():
    """Main discovery and testing function."""
    print("🚀 LeanVibe Agent Hive 2.0 - API Discovery")
    print("=" * 60)
    
    # Setup test environment
    setup_test_environment()
    
    # Discover all routes
    routes, categories = discover_routes()
    
    # Test OpenAPI schema
    openapi_paths = test_openapi_schema()
    
    # Test basic endpoints
    endpoint_results = test_basic_endpoints()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DISCOVERY SUMMARY")
    print("=" * 60)
    
    print(f"✅ Total routes discovered: {len(routes)}")
    print(f"✅ API v1 routes: {len(categories['api_v1'])}")
    print(f"✅ Dashboard routes: {len(categories['dashboard'])}")
    print(f"✅ Health endpoints: {len(categories['health'])}")
    print(f"✅ WebSocket routes: {len(categories['websocket'])}")
    print(f"✅ OpenAPI paths: {len(openapi_paths)}")
    
    working_endpoints = sum(1 for r in endpoint_results.values() if r.get('working', False))
    print(f"✅ Working endpoints tested: {working_endpoints}/{len(endpoint_results)}")
    
    print("\n🎯 Next Steps for API Testing:")
    print("1. Create comprehensive tests for working endpoints")
    print("2. Test dashboard API endpoints with mock data")
    print("3. Test WebSocket connections")
    print("4. Test authentication/authorization endpoints")
    print("5. Performance testing of key endpoints")
    
    print("\n🔧 To run full API testing:")
    print("  python -m pytest tests/test_api_endpoints.py -v")

if __name__ == "__main__":
    main()